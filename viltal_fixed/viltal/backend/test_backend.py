"""VITAL backend tests.  Run:  pip install pytest httpx   then   pytest -q
Uses a fake YOLO model, an in-memory database and no camera, so it needs no weights or hardware."""
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["DATABASE_BACKEND"] = "memory"
os.environ["DISABLE_STARTUP_CAMERA"] = "1"
os.environ.pop("TWILIO_ACCOUNT_SID", None)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as backend
import db
import model
from vital_infer import CLASSES, ActivityFilter

client = TestClient(backend.app)
UPRIGHT, LYING = [100, 50, 160, 250], [50, 200, 300, 260]       # tall box / wide box
JPEG = cv2.imencode(".jpg", np.zeros((360, 640, 3), np.uint8))[1].tobytes()


class _T:                                   # tensor stand-in for the fake model output
    def __init__(self, a): self.a = np.array(a)
    def int(self): return _T(self.a.astype(int))
    def cpu(self): return self
    def numpy(self): return self.a
    def tolist(self): return self.a.tolist()


class _Boxes:
    def __init__(self, people):
        self.id = _T([p[0] for p in people])
        self.xyxy, self.cls = _T([p[3] for p in people]), _T([CLASSES.index(p[1]) for p in people])
        self.conf, self.n = _T([p[2] for p in people]), len(people)
    def __len__(self): return self.n


class FakeYolo:
    names = dict(enumerate(CLASSES))
    people = []                             # [(track_id, class_name, confidence, box)] returned for the next frame

    def track(self, frame, **kw):
        return [type("R", (), {"boxes": _Boxes(self.people) if self.people else None})]


fake = FakeYolo()


@pytest.fixture(autouse=True)
def fresh():
    db._conn.execute("DELETE FROM docs")
    db._conn.commit()
    db._seed()
    db.current_activities.clear()
    db._open_logs.clear()
    backend.state.clear()
    backend.fall_state.clear()
    model._filters.clear()
    model.model = fake
    fake.people = []
    yield


def login(email="caregiver@example.com"):
    r = client.post("/api/caretakers/login", json={"email": email, "password": "password123"})
    assert r.status_code == 200
    return {"Authorization": "Bearer " + r.json()["access_token"]}


def frame(h, people, elder_id=None):
    fake.people = people
    url = "/api/cameras/CAM001/frame" + (f"?elder_id={elder_id}" if elder_id else "")
    r = client.post(url, headers=h, content=JPEG)
    assert r.status_code == 200, r.text
    return r.json()


# ---------- system & auth ----------
def test_health_and_status():
    assert client.get("/health").json()["model_loaded"] is True
    s = client.get("/api/system/status").json()
    assert s["status"] == "online" and s["database"] == "memory" and s["sms"] is False


def test_auth():
    assert client.get("/api/elders").status_code == 401
    assert client.get("/api/elders", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.post("/api/caretakers/login", json={"email": "caregiver@example.com", "password": "x"}).status_code == 401
    h = login()
    assert client.get("/api/caretakers/me", headers=h).json()["email"] == "caregiver@example.com"
    token = h["Authorization"][7:]
    assert client.get("/api/elders?token=" + token).status_code == 200          # query token (used by <img> stream)
    demo = client.post("/api/caretakers/demo-login", json={"email": "caregiver@example.com"}).json()
    assert demo["caretaker"]["id"] == "C002" and "password" not in demo["caretaker"]
    assert backend.user_from_token(token)["id"] == "C002" and backend.user_from_token("junk") is None


def test_profile_and_preferences_persist():
    h = login()
    assert client.patch("/api/caretakers/me", headers=h, json={"name": "Alex B"}).json()["name"] == "Alex B"
    assert client.put("/api/caretakers/me/preferences", headers=h, json={"prefSms": True}).status_code == 200
    assert client.patch("/api/caretakers/me/preferences", headers=h, json={"prefSound": False}).status_code == 200
    me = client.get("/api/caretakers/me", headers=h).json()
    assert me["name"] == "Alex B" and me["preferences"] == {"prefFall": True, "prefSound": False, "prefSms": True}


# ---------- patients ----------
def test_elder_crud_and_unique_ids():
    h = login()
    assert client.post("/api/elders", headers=h, json={}).status_code == 422
    e3 = client.post("/api/elders", headers=h, json={"name": "Meera", "age": 70}).json()
    assert e3["id"] == "E003"
    assert client.patch("/api/elders/E003", headers=h, json={"blood": "O+"}).json()["blood"] == "O+"
    client.delete("/api/elders/E001", headers=h)                                 # delete an early one...
    assert client.post("/api/elders", headers=h, json={"name": "Next"}).json()["id"] == "E004"   # ...no id clash
    assert client.get("/api/elders/E001", headers=h).status_code == 404
    assert len(client.get("/api/elders", headers=h).json()) == 3


# ---------- activity logs ----------
def test_activity_log_is_generated_from_camera_frames():
    h = login()
    frame(h, [(1, "Standing", 0.9, UPRIGHT)], "E002")
    time.sleep(1.1)
    frame(h, [(1, "Standing", 0.9, UPRIGHT)], "E002")                            # same activity: no new entry
    for _ in range(3):                                                            # the filter smooths over ~1.5s of frames
        frame(h, [(1, "Sitting", 0.8, UPRIGHT)], "E002")                         # changed: new entry, old one closed
    logs = client.get("/api/activities?elder_id=E002", headers=h).json()
    assert [l["activity"] for l in logs] == ["Sitting", "Standing"]              # newest first
    sitting, standing = logs
    assert sitting["ongoing"] and sitting["end_time"] is None
    assert not standing["ongoing"] and standing["duration_seconds"] >= 1
    assert standing["patient_name"] == "Arthur Pendelton" and standing["status"] == "Recorded"
    assert client.get("/api/activities?elder_id=E001", headers=h).json() == []   # selected patient got the log
    cur = client.get("/api/elders/E002/activity/current", headers=h).json()
    assert cur["activity"] == "Sitting"


def test_activity_date_range_filter():
    h = login()
    frame(h, [(1, "Walking", 0.9, UPRIGHT)])
    assert len(client.get("/api/activities", headers=h, params={"start": "2000-01-01T00:00:00.000Z", "end": "2001-01-01T00:00:00.000Z"}).json()) == 0
    assert len(client.get("/api/activities", headers=h, params={"start": "2000-01-01T00:00:00.000Z", "end": "2100-01-01T00:00:00Z"}).json()) == 1
    hist = client.get("/api/elders/E001/activity/history", headers=h).json()
    assert hist["total"] == 1 and hist["items"][0]["activity"] == "Walking"


def test_person_leaving_closes_activity(monkeypatch):
    h = login()
    frame(h, [(1, "Sitting", 0.9, UPRIGHT)])
    monkeypatch.setattr(backend, "TRACK_GRACE_S", 0)
    frame(h, [])                                                                  # nobody visible any more
    log = client.get("/api/activities", headers=h).json()[0]
    assert log["end_time"] is not None and not log["ongoing"]


def test_two_people_map_to_two_patients():
    h = login()
    r = frame(h, [(7, "Sitting", 0.9, UPRIGHT), (3, "Walking", 0.9, UPRIGHT)])
    assert {t["elder_id"] for t in r["tracks"]} == {"E001", "E002"}
    assert len(client.get("/api/activities", headers=h).json()) == 2


# ---------- falls & alerts ----------
def test_fall_raises_exactly_one_alert_and_logs_it(monkeypatch):
    h = login()
    for _ in range(6):
        frame(h, [(1, "Standing", 0.9, UPRIGHT)], "E001")
    sent = []
    monkeypatch.setattr(backend, "broadcast_alerts", lambda a: _record(sent, a))
    for _ in range(20):
        frame(h, [(1, "Falling", 0.9, LYING)], "E001")
    alerts = client.get("/api/alerts", headers=h).json()
    assert len(alerts) == 1 and alerts[0]["status"] == "ACTIVE" and alerts[0]["elder_id"] == "E001"
    pushed = [a for batch in sent for a in batch]
    assert len(pushed) == 1 and pushed[0]["alert_type"] == "FALL_DETECTED"       # pushed to the dashboard once
    assert client.get("/api/activities?elder_id=E001", headers=h).json()[0]["status"] == "Fall detected"
    acked = client.patch(f"/api/alerts/{alerts[0]['id']}/acknowledge", headers=h).json()
    assert acked["status"] == "ACKNOWLEDGED" and acked["is_acknowledged"]
    assert client.patch("/api/alerts/nope/acknowledge", headers=h).status_code == 404


async def _record(sent, alerts):
    sent.append(alerts)


def test_manual_test_alert():
    h = login()
    r = client.post("/api/alerts/test", headers=h, json={"elder_id": "E002"})
    assert r.status_code == 201 and r.json()["elder_name"] == "Arthur Pendelton"
    assert client.get("/api/activities?elder_id=E002", headers=h).json()[0]["activity"] == "Falling"


def test_broadcast_reaches_websocket_clients():
    class WS:
        def __init__(self): self.msgs = []
        async def send_text(self, m): self.msgs.append(m)
    ws = WS()
    backend.active_sockets["C002"].append(ws)
    asyncio.run(backend.broadcast_alerts([{"id": "a1"}]))
    backend.active_sockets.clear()
    assert '"event": "alert"' in ws.msgs[0] and len(ws.msgs) == 2


def test_sms_sent_only_when_configured_and_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "send_sms", lambda to, body: calls.append((to, body)))
    alert = {"caretaker_id": "C002", "elder_id": "E001", "message": "FALL"}
    backend.sms_alert(alert)
    assert calls == []                                                           # pref off
    c = db.get_caretaker_by_id("C002")
    c["preferences"]["prefSms"] = True
    db.save_caretaker(c)
    backend.sms_alert(alert)
    assert calls == [("+91 98111 22331", "FALL")]


# ---------- reports ----------
def test_report_summary_uses_real_activity():
    h = login()
    frame(h, [(1, "Sleeping", 0.9, LYING)], "E001")
    day = time.strftime("%Y-%m-%d", time.gmtime())
    r = client.post("/api/reports", headers=h, json={"patient_id": "E001", "report_date": day}).json()
    assert r["patient_name"] == "Eleanor Vance" and r["activity_count"] == 1 and "sleeping" in r["summary"]
    empty = client.post("/api/reports", headers=h, json={"patient_id": "E002", "report_date": day}).json()
    assert empty["activity_count"] == 0
    assert len(client.get("/api/reports", headers=h).json()) == 2
    assert client.delete(f"/api/reports/{r['id']}", headers=h).json()["deleted"]
    assert len(client.get("/api/reports", headers=h).json()) == 1


# ---------- cameras & monitoring ----------
def test_monitoring_lifecycle_and_status():
    h = login()
    st = client.get("/api/rooms/ROOM001/monitoring/status", headers=h).json()
    assert st["is_active"] is False and st["model_available"] is True and st["fps"] == 0.0
    assert client.get("/api/cameras", headers=h).json()[0]["status"] == "OFFLINE"
    assert client.post("/api/rooms/ROOM001/monitoring/start", headers=h, json={"camera_source": "browser"}).json()["status"] == "started"
    r = frame(h, [(1, "Walking", 0.9, UPRIGHT)])
    assert r["detections"] == 1 and r["activity"] == "Walking"
    for _ in range(3):
        frame(h, [(1, "Walking", 0.9, UPRIGHT)])
    st = client.get("/api/rooms/ROOM001/monitoring/status", headers=h).json()
    assert st["is_active"] and st["inference_count"] == 4 and st["fps"] > 0 and st["active_tracks_count"] == 1
    assert client.get("/api/rooms/ROOM001/tracking", headers=h).json()["active_tracks"][0]["last_activity"] == "Walking"
    assert client.get("/api/cameras", headers=h).json()[0]["status"] == "ONLINE"
    client.post("/api/rooms/ROOM001/monitoring/stop", headers=h)
    assert client.get("/api/rooms/ROOM001/tracking", headers=h).json()["active_tracks"] == []
    assert client.get("/api/activities", headers=h).json()[0]["ongoing"] is False


def test_frame_upload_validation_and_snapshot():
    h = login()
    assert client.post("/api/cameras/CAM001/frame", headers=h, content=b"").status_code == 400
    assert client.post("/api/cameras/CAM001/frame", headers=h, content=b"not-a-jpeg").status_code == 400
    snap = client.get("/api/cameras/CAM001/snapshot?elder_id=E001", headers=h)   # placeholder picture, no camera yet
    assert snap.headers["content-type"] == "image/jpeg" and snap.content[:2] == b"\xff\xd8"
    frame(h, [(1, "Walking", 0.9, UPRIGHT)])
    assert client.get("/api/cameras/CAM001/snapshot", headers=h).content[:2] == b"\xff\xd8"


# ---------- model filter ----------
def test_activity_filter_confirms_fall_once():
    flt = ActivityFilter()
    for i in range(10):
        label, alert = flt.update(1, i * 0.04, "Standing", 0.92, [200, 100, 300, 400])
    assert label == "Standing" and not alert
    alerts = 0
    for i in range(10, 40):
        label, alert = flt.update(1, i * 0.04, "Falling", 0.88, [150, 300, 450, 400])
        alerts += alert
    assert label == "Falling" and alerts == 1


def test_sitting_down_is_not_a_fall():
    flt = ActivityFilter()
    for i in range(30):                                                          # bent torso, still upright box
        label, alert = flt.update(1, i * 0.04, "Falling" if i % 3 == 0 else "Sitting", 0.7, [200, 100, 300, 300])
        assert not alert


# ---------- storage ----------
def test_sqlite_persists_between_restarts(tmp_path):
    env = {**os.environ, "DATABASE_BACKEND": "sqlite", "SQLITE_DB_PATH": str(tmp_path / "t.db")}
    here = str(Path(__file__).resolve().parent)
    write = "import db; db.add_elder({'name': 'Saved'}); db.update_activity('E001', 'ROOM001', 'Walking', .9)"
    read = "import db; print([e['name'] for e in db.all_docs('elders')], len(db.activity_items()), db.activity_items()[0]['ongoing'])"
    for code in (write, read):
        out = subprocess.run([sys.executable, "-c", code], cwd=here, env=env, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
    assert "'Saved'" in out.stdout and " 1 False" in out.stdout                  # open entry closed on restart
