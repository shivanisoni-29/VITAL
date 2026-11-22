# VITAL storage layer.
# Everything is stored as small JSON documents in one SQLite table (Python stdlib, no extra install).
# If Firebase credentials are configured, every write is also mirrored to Firestore.

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = os.getenv("DATABASE_BACKEND", "sqlite").lower()          # sqlite | memory | firebase
DB_PATH = ":memory:" if BACKEND == "memory" else str(HERE / os.getenv("SQLITE_DB_PATH", "vital.db"))


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def norm_ts(value):
    """Turn any ISO timestamp (e.g. JS toISOString) into our UTC format so strings sort correctly."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds")


# ---------- optional Firebase mirror ----------
_firebase = None
_creds = os.getenv("FIREBASE_CREDENTIALS_PATH", "").strip()
if _creds and os.path.exists(_creds):
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        if not firebase_admin._apps:
            project = os.getenv("FIREBASE_PROJECT_ID", "").strip()
            firebase_admin.initialize_app(credentials.Certificate(_creds), {"projectId": project} if project else None)
        _firebase = firestore.client()
        print("Firebase Firestore mirror enabled.")
    except Exception as e:
        print("Firebase unavailable, using local database only:", e)


def is_firebase_active():
    return _firebase is not None


# ---------- generic document store ----------
_lock = threading.RLock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.execute("""CREATE TABLE IF NOT EXISTS docs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, coll TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL,
    UNIQUE (coll, id))""")
_conn.commit()


def put(coll, doc):
    with _lock:
        _conn.execute("INSERT INTO docs (coll, id, data) VALUES (?, ?, ?) "
                      "ON CONFLICT (coll, id) DO UPDATE SET data = excluded.data",
                      (coll, doc["id"], json.dumps(doc)))
        _conn.commit()
    if _firebase:
        try:
            _firebase.collection(coll).document(doc["id"]).set(doc)
        except Exception as e:
            print("Firebase write failed:", e)
    return doc


def get(coll, doc_id):
    with _lock:
        row = _conn.execute("SELECT data FROM docs WHERE coll = ? AND id = ?", (coll, doc_id)).fetchone()
    return json.loads(row[0]) if row else None


def all_docs(coll):
    with _lock:
        rows = _conn.execute("SELECT data FROM docs WHERE coll = ? ORDER BY seq", (coll,)).fetchall()
    return [json.loads(r[0]) for r in rows]


def remove(coll, doc_id):
    with _lock:
        _conn.execute("DELETE FROM docs WHERE coll = ? AND id = ?", (coll, doc_id))
        _conn.commit()
    if _firebase:
        try:
            _firebase.collection(coll).document(doc_id).delete()
        except Exception as e:
            print("Firebase delete failed:", e)


# ---------- fixed setup data (rooms / cameras) and first-run demo data ----------
rooms = [
    {"id": "ROOM001", "name": "Oak Hall - Room 101", "camera_id": "CAM001"},
    {"id": "ROOM002", "name": "Maple Suite - Room 102", "camera_id": "CAM002"},
]
cameras = [
    {"id": "CAM001", "room_id": "ROOM001", "device_index": 0},
    {"id": "CAM002", "room_id": "ROOM002", "device_index": 1},
]


def _seed():
    prefs = {"prefFall": True, "prefSound": True, "prefSms": False}
    for cid, name, email in (("C001", "Sarah Jenkins", "caretaker1@vital.local"),
                             ("C002", "Alex Caregiver", "caregiver@example.com")):
        put("caretakers", {"id": cid, "name": name, "email": email, "password": "password123",
                           "room_id": "ROOM001", "preferences": dict(prefs), "created_at": now_iso()})
    for eid, name, age, gender, phone, blood, h, w, dis, notes in (
        ("E001", "Eleanor Vance", 78, "Female", "+91 98111 22331", "B+", "162", "60",
         "Mild arthritis", "History of hip injury, careful when walking"),
        ("E002", "Arthur Pendelton", 82, "Male", "+91 98111 22332", "A+", "170", "68",
         "Mild dementia", "Needs assistance standing"),
    ):
        put("elders", {"id": eid, "name": name, "age": age, "gender": gender, "room_id": "ROOM001",
                       "phone": phone, "emergency_contact": phone, "blood": blood, "height": h, "weight": w,
                       "diseases": dis, "notes": notes, "photo": "", "active": True, "created_at": now_iso()})


if not all_docs("caretakers"):
    _seed()

# Activities that were still "open" when the server last stopped can't be timed any more: close them.
for _log in all_docs("activity_history"):
    if _log.get("end_time") is None:
        _log["end_time"] = _log["start_time"]
        put("activity_history", _log)


# ---------- caretakers ----------
def get_caretaker_by_email(email):
    email = (email or "").strip().lower()
    return next((c for c in all_docs("caretakers") if c["email"].lower() == email), None)


def get_caretaker_by_id(cid):
    return get("caretakers", cid)


def save_caretaker(user):
    return put("caretakers", user)


# ---------- elders ----------
def get_elders_by_room(room_id):
    return [e for e in all_docs("elders") if e["room_id"] == room_id]


def get_elder(elder_id):
    return get("elders", elder_id)


def add_elder(data):
    ids = [int(e["id"][1:]) for e in all_docs("elders") if e["id"][1:].isdigit()]
    elder = {
        "id": data.get("id") or f"E{max(ids, default=0) + 1:03d}",
        "name": data.get("name", "New Patient"), "age": data.get("age", 75),
        "gender": data.get("gender", "Not specified"), "room_id": data.get("room_id", "ROOM001"),
        "phone": data.get("phone", ""), "emergency_contact": data.get("emergency_contact", ""),
        "blood": data.get("blood", "Not specified"), "height": str(data.get("height", "")),
        "weight": str(data.get("weight", "")), "diseases": data.get("diseases", ""),
        "notes": data.get("notes", ""), "photo": data.get("photo", ""),
        "active": data.get("active", True), "created_at": now_iso(),
    }
    return put("elders", elder)


def update_elder(elder_id, data):
    elder = get_elder(elder_id)
    if not elder:
        return None
    elder.update({k: v for k, v in data.items() if k != "id"})
    return put("elders", elder)


def delete_elder(elder_id):
    remove("elders", elder_id)
    current_activities.pop(elder_id, None)
    _open_logs.pop(elder_id, None)


# ---------- activities (live state + history log) ----------
current_activities = {}   # elder_id -> latest live reading (kept in memory only)
_open_logs = {}           # elder_id -> history entry that has not ended yet


def get_current_activity(elder_id):
    if elder_id not in current_activities:
        elder = get_elder(elder_id)
        current_activities[elder_id] = {
            "elder_id": elder_id, "room_id": elder["room_id"] if elder else "ROOM001", "activity": "Sitting",
            "current_activity": "Sitting", "confidence": 0.95, "updated_at": now_iso()}
    return current_activities[elder_id]


def close_activity(elder_id):
    """End the running history entry (person left the picture / monitoring stopped)."""
    log = _open_logs.pop(elder_id, None)
    if log:
        log["end_time"] = now_iso()
        put("activity_history", log)


def update_activity(elder_id, room_id, activity, confidence):
    """Update the live reading. A new history entry is written only when the activity changes,
    and the previous entry is closed so it gets a duration."""
    now = now_iso()
    act = {"elder_id": elder_id, "room_id": room_id, "activity": activity, "current_activity": activity,
           "confidence": float(confidence), "updated_at": now}
    current_activities[elder_id] = act
    open_log = _open_logs.get(elder_id)
    if not open_log or open_log["activity"] != activity:
        close_activity(elder_id)
        log = {"id": f"act_{uuid.uuid4().hex[:8]}", "elder_id": elder_id, "room_id": room_id,
               "activity": activity, "confidence": float(confidence),
               "start_time": now, "end_time": None, "created_at": now}
        _open_logs[elder_id] = put("activity_history", log)
    return act


def activity_items(elder_id=None, start=None, end=None, limit=500):
    """Activity history for the UI, newest first. start/end are ISO times (start inclusive, end exclusive)."""
    start, end = norm_ts(start), norm_ts(end)
    names = {e["id"]: e["name"] for e in all_docs("elders")}
    now = datetime.now(timezone.utc)
    items = []
    for log in reversed(all_docs("activity_history")):
        if (elder_id and log["elder_id"] != elder_id) or (start and log["start_time"] < start) \
                or (end and log["start_time"] >= end):
            continue
        began = datetime.fromisoformat(log["start_time"])
        ended = datetime.fromisoformat(log["end_time"]) if log["end_time"] else now
        items.append({**log, "patient_name": names.get(log["elder_id"], "Patient"),
                      "duration_seconds": max(0, int((ended - began).total_seconds())),
                      "ongoing": log["end_time"] is None,
                      "status": "Fall detected" if log["activity"] == "Falling" else "Recorded"})
        if len(items) >= limit:
            break
    return items


def day_bounds(date_str):
    """UTC start/end of a YYYY-MM-DD date (used when the client sends no explicit range)."""
    day = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    return day.isoformat(), (day + timedelta(days=1)).isoformat()


# ---------- alerts ----------
def add_alert(elder_id, room_id, caretaker_id, elder_name="Patient", alert_type="FALL_DETECTED"):
    now = now_iso()
    return put("alerts", {
        "id": str(uuid.uuid4()), "elder_id": elder_id, "elder_name": elder_name, "room_id": room_id,
        "caretaker_id": caretaker_id, "alert_type": alert_type, "activity": "Falling", "severity": "CRITICAL",
        "status": "ACTIVE", "timestamp": now, "created_at": now, "is_read": False, "is_acknowledged": False,
        "acknowledged_at": None, "tracking_id": None})


def list_alerts():
    return list(reversed(all_docs("alerts")))      # newest first


def ack_alert(alert_id):
    alert = get("alerts", alert_id)
    if not alert:
        return None
    alert.update(status="ACKNOWLEDGED", is_acknowledged=True, is_read=True, acknowledged_at=now_iso())
    return put("alerts", alert)


# ---------- reports ----------
def summarize(elder_id, start=None, end=None):
    """Plain-text daily summary built from the real activity history."""
    items = activity_items(elder_id, start, end, limit=100000)
    falls = sum(1 for i in items if i["status"] == "Fall detected")
    if not items:
        return "No activity was recorded for this day.", 0, 0
    seconds = {}
    for i in items:
        seconds[i["activity"]] = seconds.get(i["activity"], 0) + i["duration_seconds"]
    top = max(seconds, key=seconds.get)
    text = (f"{len(items)} activity record(s); most time spent {top.lower()}. "
            + (f"{falls} fall alert(s) recorded - please review." if falls else "No falls recorded."))
    return text, len(items), falls


def add_report(data):
    pid = data.get("patient_id") or data.get("elder_id") or "E001"
    elder = get_elder(pid)
    rdate = data.get("report_date") or now_iso()[:10]
    start, end = data.get("start"), data.get("end")
    if not (start and end):
        start, end = day_bounds(rdate)
    summary, count, falls = summarize(pid, start, end)
    return put("reports", {
        "id": f"rep_{uuid.uuid4().hex[:8]}", "patient_id": pid,
        "patient_name": data.get("patient_name") or (elder["name"] if elder else "Patient"),
        "report_date": rdate, "summary": data.get("summary") or summary,
        "activity_count": count, "fall_count": falls,
        "created_by": data.get("created_by", "C001"), "created_at": now_iso()})


def list_reports():
    return list(reversed(all_docs("reports")))


def delete_report(report_id):
    remove("reports", report_id)
