# VITAL - elderly activity monitoring backend (FastAPI).
# Run: python app.py   ->   open http://127.0.0.1:8000/dashboard

import asyncio
import base64
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:  # load backend/.env before db/model read their settings
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import cv2
import jwt
import numpy as np
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

import db
import model

app = FastAPI(title="VITAL App")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

JWT_SECRET = os.getenv("JWT_SECRET_KEY", "vital-student-secret-key-super-secure-32bytes")
TOKEN_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
DASHBOARD = Path(__file__).resolve().parent.parent / "vital_care_dashboard.html"
TRACK_GRACE_S = 2.5          # keep showing a person for this long after a missed detection


class RoomState:
    """Live, in-memory monitoring state of one room."""
    def __init__(self):
        self.frame = None                 # latest annotated frame
        self.tracks = []                  # people currently visible
        self.last_seen = 0.0              # time of the last frame with a detection
        self.seen = {}                    # elder_id -> time last detected (to close finished activities)
        self.monitoring = False
        self.frames = 0
        self.stamps = deque(maxlen=30)    # recent frame times -> real FPS
        self.last_inference = None
        self.stop = threading.Event()     # asks the server-webcam thread to stop
        self.thread = None


state = defaultdict(RoomState)
fall_state = {}                           # elder_id -> True while a fall alert is already raised
active_sockets = defaultdict(list)        # caretaker_id -> websockets
LOOP = None                               # main event loop (lets worker threads broadcast)


# ---------- auth ----------
def make_token(user):
    exp = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_MINUTES)
    return jwt.encode({"sub": user["id"], "email": user["email"], "room_id": user["room_id"], "exp": exp},
                      JWT_SECRET, algorithm="HS256")


def user_from_token(token):
    try:
        return db.get_caretaker_by_id(jwt.decode(token, JWT_SECRET, algorithms=["HS256"]).get("sub"))
    except jwt.PyJWTError:
        return None


def get_current_user(request: Request):
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Missing auth token")
    user = user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user


api = APIRouter(dependencies=[Depends(get_current_user)])     # every route on `api` needs a valid login


def public(user):
    return {k: user.get(k) for k in ("id", "name", "email", "room_id")} | {"preferences": user.get("preferences", {})}


def login_response(user):
    return {"access_token": make_token(user), "token_type": "bearer", "caretaker": public(user)}


async def body_of(request):
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ---------- alerts: websocket + optional SMS ----------
async def broadcast_alerts(alerts):
    for alert in alerts:
        for event, data in (("alert", {"alert": alert}), ("fall_alert", alert)):
            msg = json.dumps({"event": event, "data": data}, default=str)
            for sockets in list(active_sockets.values()):
                for ws in list(sockets):
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        sockets.remove(ws)


def sms_enabled():
    return all(os.getenv(k) for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM"))


def send_sms(to, body):
    """Optional: text the patient's contact number through Twilio (needs TWILIO_* in .env)."""
    sid, token = os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"]
    req = urllib.request.Request(
        f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json", method="POST",
        data=urllib.parse.urlencode({"To": re.sub(r"[^\d+]", "", to), "From": os.environ["TWILIO_FROM"], "Body": body}).encode())
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{sid}:{token}".encode()).decode())
    urllib.request.urlopen(req, timeout=8).read()


def sms_alert(alert):
    try:
        caretaker, elder = db.get_caretaker_by_id(alert["caretaker_id"]), db.get_elder(alert["elder_id"])
        phone = elder and (elder.get("phone") or elder.get("emergency_contact"))
        if caretaker and caretaker.get("preferences", {}).get("prefSms") and phone:
            send_sms(phone, alert["message"])
    except Exception as e:  # an SMS problem must never stop the alert itself
        print("SMS failed:", e)


def raise_alert(elder_id, elder_name, room_id, user_id):
    """Store a fall alert and return the payload sent to the dashboard."""
    alert = db.add_alert(elder_id, room_id, user_id, elder_name=elder_name)
    payload = {"id": alert["id"], "elder_id": elder_id, "elder_name": elder_name, "room_id": room_id,
               "caretaker_id": user_id, "alert_type": "FALL_DETECTED", "activity": "Falling",
               "message": f"EMERGENCY: Fall detected for {elder_name} in Room {room_id}!",
               "timestamp": alert["timestamp"]}
    if sms_enabled():
        threading.Thread(target=sms_alert, args=(payload,), daemon=True).start()
    return payload


# ---------- detection pipeline ----------
def process_detections(room_id, detections, user_id="C001", preferred_elder=None):
    """Match detected people to patients, update activity + history, raise fall alerts.
    Returns (tracks, new_alerts)."""
    room, now, alerts, tracks, seen = state[room_id], time.time(), [], [], set()
    elders = db.get_elders_by_room(room_id)
    if preferred_elder:                                   # the patient selected on the dashboard goes first
        elders.sort(key=lambda e: e["id"] != preferred_elder)
    for i, det in enumerate(sorted(detections, key=lambda d: d["track_id"])):
        elder = elders[i] if i < len(elders) else None
        elder_id = elder["id"] if elder else f"{room_id}-P{i + 1}"
        name = elder["name"] if elder else f"Patient {i + 1}"
        act, conf = det["activity"], det["confidence"]
        tracks.append({"tracking_id": f"T{det['track_id']}", "room_id": room_id, "elder_id": elder_id,
                       "elder_name": name, "is_verified": True, "confidence": round(conf, 2),
                       "last_activity": act, "box": det.get("box", []), "last_seen": db.now_iso()})
        seen.add(elder_id)
        room.seen[elder_id] = now
        db.update_activity(elder_id, room_id, act, conf)       # writes history only when the activity changes
        if act != "Falling":
            fall_state[elder_id] = False
        elif not fall_state.get(elder_id):                    # one alert per fall, not one per frame
            fall_state[elder_id] = True
            alerts.append(raise_alert(elder_id, name, room_id, user_id))
    for elder_id, last in list(room.seen.items()):            # person gone for a while -> end their activity
        if elder_id not in seen and now - last >= TRACK_GRACE_S:
            db.close_activity(elder_id)
            del room.seen[elder_id]
    if detections:
        room.tracks, room.last_seen = tracks, now
    elif now - room.last_seen >= TRACK_GRACE_S:
        room.tracks = []
    return room.tracks, alerts


def analyze(room_id, frame, user_id="C001", elder_id=None):
    """Run the model on one frame and process the result. Returns (tracks, alerts, detection_count)."""
    room = state[room_id]
    room.frames += 1
    room.stamps.append(time.time())
    room.last_inference = db.now_iso()
    detections, room.frame = model.predict_annotated(frame, conf_threshold=0.30, room_id=room_id)
    tracks, alerts = process_detections(room_id, detections, user_id, elder_id)
    return tracks, alerts, len(detections)


def device_camera_worker(room_id, device_index):
    """Background thread: read the server computer's webcam and analyse every frame."""
    room = state[room_id]
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW if os.name == "nt" else 0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print(f"Device camera {device_index} not accessible. Ready for browser webcam stream.")
        room.monitoring = False
        return
    room.monitoring = True
    print(f"Hardware webcam connected for Room {room_id}.")
    fails = 0
    try:
        while not room.stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                fails += 1
                if fails > 50:
                    print("Device camera stopped returning frames.")
                    break
                time.sleep(0.1)
                continue
            fails = 0
            try:
                _, alerts, _ = analyze(room_id, frame)
            except Exception as e:      # keep the camera thread alive if one frame fails
                print("Frame processing error:", e)
                time.sleep(0.5)
                continue
            if alerts and LOOP:
                asyncio.run_coroutine_threadsafe(broadcast_alerts(alerts), LOOP)
            time.sleep(0.04)
    finally:
        cap.release()
        room.monitoring = False


def start_device_camera(room_id):
    room = state[room_id]
    room.stop.clear()
    if not (room.thread and room.thread.is_alive()):
        cam = next((c for c in db.cameras if c["room_id"] == room_id), None)
        room.thread = threading.Thread(target=device_camera_worker, args=(room_id, cam["device_index"] if cam else 0), daemon=True)
        room.thread.start()


@app.on_event("startup")
async def startup_event():
    global LOOP
    LOOP = asyncio.get_running_loop()
    if os.getenv("DISABLE_STARTUP_CAMERA", "").lower() in ("1", "true"):
        print("Startup hardware camera disabled via DISABLE_STARTUP_CAMERA.")
        return
    start_device_camera("ROOM001")


def current_fps(room):
    s = room.stamps
    return round((len(s) - 1) / (s[-1] - s[0]), 1) if len(s) > 1 and s[-1] > s[0] and time.time() - s[-1] < 3 else 0.0


def camera_room(camera_id, user):
    cam = next((c for c in db.cameras if c["id"] == camera_id), None)
    return cam["room_id"] if cam else user.get("room_id", "ROOM001")


def placeholder_frame(elder_id=None):
    """Shown when no camera is feeding the room."""
    elder = db.get_elder(elder_id) if elder_id else None
    act = db.get_current_activity(elder_id)["activity"] if elder else "Sitting"
    fall = act == "Falling"
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (30, 36, 30)
    color = (0, 0, 240) if fall else (70, 200, 90)
    bx, by, bw, bh = (230, 220, 180, 70) if fall else (270, 140, 100, 150)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, 2)
    cv2.putText(img, f"{elder['name'] if elder else 'Patient'} [{act}]", (bx, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(img, f"NO LIVE CAMERA - DEMO VIEW | {datetime.now():%Y-%m-%d %H:%M:%S}", (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return img


# ---------- system & dashboard ----------
@app.get("/")
async def root(request: Request):
    if "text/html" in request.headers.get("accept", "") and DASHBOARD.exists():
        return FileResponse(DASHBOARD, media_type="text/html")
    return {"app": "VITAL Elderly Monitoring System", "version": "1.0.0", "docs": "/docs"}


@app.get("/dashboard")
async def dashboard():
    if not DASHBOARD.exists():
        raise HTTPException(status_code=404, detail="vital_care_dashboard.html not found")
    return FileResponse(DASHBOARD, media_type="text/html")


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": db.now_iso(), "model_loaded": model.is_ready()}


@app.get("/api/system/status")
def system_status():
    return {"status": "online", "timestamp": db.now_iso(), "model_loaded": model.is_ready(), "sms": sms_enabled(),
            "database": ("firebase+" if db.is_firebase_active() else "") + ("memory" if db.BACKEND == "memory" else "sqlite")}


# ---------- caretakers & auth ----------
@app.post("/api/caretakers/login")
async def login(request: Request):
    body = await body_of(request)
    user = db.get_caretaker_by_email(body.get("email"))
    if not user or user.get("password") != body.get("password"):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return login_response(user)


@app.post("/api/caretakers/demo-login")
async def demo_login(request: Request):
    body = await body_of(request)
    return login_response(db.get_caretaker_by_email(body.get("email")) or db.get_caretaker_by_id("C001"))


@api.get("/api/caretakers/me")
def get_me(user: dict = Depends(get_current_user)):
    return public(user)


@api.patch("/api/caretakers/me")
async def update_me(request: Request, user: dict = Depends(get_current_user)):
    body = await body_of(request)
    user.update({k: body[k] for k in ("name", "email") if body.get(k)})
    db.save_caretaker(user)
    return public(user)


@api.put("/api/caretakers/me/preferences")
@api.patch("/api/caretakers/me/preferences")
async def update_preferences(request: Request, user: dict = Depends(get_current_user)):
    body = await body_of(request)
    user.setdefault("preferences", {}).update(body)
    db.save_caretaker(user)
    return {"message": "Preferences updated", "preferences": user["preferences"]}


# ---------- elders (patients) ----------
@api.get("/api/elders")
def list_elders():
    return db.all_docs("elders")


@api.get("/api/elders/{elder_id}")
def get_elder(elder_id: str):
    elder = db.get_elder(elder_id)
    if not elder:
        raise HTTPException(status_code=404, detail="Elder not found")
    return elder


@api.post("/api/elders", status_code=201)
async def create_elder(request: Request):
    body = await body_of(request)
    if not body.get("name"):
        raise HTTPException(status_code=422, detail="Name is required")
    return db.add_elder(body)


@api.patch("/api/elders/{elder_id}")
async def edit_elder(elder_id: str, request: Request):
    elder = db.update_elder(elder_id, await body_of(request))
    if not elder:
        raise HTTPException(status_code=404, detail="Elder not found")
    return elder


@api.delete("/api/elders/{elder_id}")
def remove_elder(elder_id: str):
    db.delete_elder(elder_id)
    return {"message": "Elder deleted successfully.", "deleted": True}


@api.get("/api/elders/{elder_id}/activity/current")
def get_elder_activity(elder_id: str):
    return db.get_current_activity(elder_id)


@api.get("/api/elders/{elder_id}/activity/history")
def get_elder_activity_history(elder_id: str, start: str = None, end: str = None, limit: int = 500):
    items = db.activity_items(elder_id, start, end, limit)
    return {"elder_id": elder_id, "total": len(items), "page": 1, "page_size": limit, "items": items}


# ---------- activity logs & alerts ----------
@api.get("/api/activities")
def list_activities(elder_id: str = None, start: str = None, end: str = None, limit: int = 500):
    """Activity history, newest first. Optional filters: elder_id, start/end (ISO times)."""
    return db.activity_items(elder_id, start, end, limit)


@api.post("/api/activities")
async def record_activity(request: Request, user: dict = Depends(get_current_user)):
    body = await body_of(request)
    if not db.get_elder(body.get("elder_id")):
        raise HTTPException(status_code=404, detail="Elder not found")
    return db.update_activity(body["elder_id"], user["room_id"], body.get("activity", "Sitting"), body.get("confidence", 0.95))


@api.get("/api/alerts")
def list_alerts():
    return db.list_alerts()


@api.post("/api/alerts/test", status_code=201)
async def test_alert(request: Request, user: dict = Depends(get_current_user)):
    body = await body_of(request)
    elder_id = body.get("elder_id") or "E001"
    elder = db.get_elder(elder_id) or {"name": "Patient"}
    payload = raise_alert(elder_id, elder["name"], user["room_id"], user["id"])
    db.update_activity(elder_id, user["room_id"], "Falling", 0.99)
    await broadcast_alerts([payload])
    return db.get("alerts", payload["id"])


@api.patch("/api/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str):
    alert = db.ack_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    fall_state[alert["elder_id"]] = False
    return alert


# ---------- reports ----------
@api.get("/api/reports")
def list_reports():
    return db.list_reports()


@api.post("/api/reports", status_code=201)
async def create_report(request: Request, user: dict = Depends(get_current_user)):
    body = await body_of(request)
    return db.add_report({**body, "created_by": user["id"]})


@api.delete("/api/reports/{report_id}")
def remove_report(report_id: str):
    db.delete_report(report_id)
    return {"message": "Report deleted successfully.", "deleted": True}


# ---------- cameras & video ----------
@api.get("/api/cameras")
def list_cameras():
    return [{**c, "status": "ONLINE" if state[c["room_id"]].monitoring else "OFFLINE", "fps": current_fps(state[c["room_id"]])}
            for c in db.cameras]


@api.post("/api/cameras/{camera_id}/frame")
async def receive_browser_frame(camera_id: str, request: Request, elder_id: str = None, user: dict = Depends(get_current_user)):
    """The dashboard's browser webcam posts JPEG frames here. elder_id = patient selected on screen."""
    room_id = camera_room(camera_id, user)
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty frame")
    frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    state[room_id].stop.set()               # browser upload takes over from the server webcam
    state[room_id].monitoring = True
    tracks, alerts, count = await asyncio.to_thread(analyze, room_id, frame, user["id"], elder_id)
    await broadcast_alerts(alerts)
    return {"accepted": True, "camera_id": camera_id, "room_id": room_id, "detections": count,
            "activity": tracks[0]["last_activity"] if tracks else None, "tracks": tracks}


@api.get("/api/cameras/{camera_id}/snapshot")
def get_snapshot(camera_id: str, elder_id: str = None, user: dict = Depends(get_current_user)):
    frame = state[camera_room(camera_id, user)].frame
    _, jpg = cv2.imencode(".jpg", frame if frame is not None else placeholder_frame(elder_id))
    return Response(content=jpg.tobytes(), media_type="image/jpeg")


@api.get("/api/cameras/{camera_id}/stream")
async def get_stream(camera_id: str, elder_id: str = None, user: dict = Depends(get_current_user)):
    room = state[camera_room(camera_id, user)]

    async def frames():
        while True:
            frame = room.frame if room.monitoring and room.frame is not None else placeholder_frame(elder_id)
            _, jpg = cv2.imencode(".jpg", frame)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            await asyncio.sleep(0.08)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


# ---------- rooms & monitoring ----------
@api.get("/api/rooms/{room_id}/tracking")
def get_room_tracking(room_id: str):
    return {"room_id": room_id, "active_tracks": state[room_id].tracks}


@api.get("/api/rooms/{room_id}/monitoring/status")
def get_monitoring_status(room_id: str):
    room = state[room_id]
    cam = next((c for c in db.cameras if c["room_id"] == room_id), None)
    return {"room_id": room_id, "is_active": room.monitoring, "monitoring_mode": "MULTIPLE_ELDERS",
            "camera_id": cam["id"] if cam else "CAM001", "camera_status": "ONLINE" if room.monitoring else "OFFLINE",
            "active_tracks_count": len(room.tracks), "elders_count": len(db.get_elders_by_room(room_id)),
            "fps": current_fps(room), "inference_count": room.frames, "latest_detection_count": len(room.tracks),
            "last_inference_at": room.last_inference, "model_available": model.is_ready()}


@api.post("/api/rooms/{room_id}/monitoring/start")
async def start_monitoring(room_id: str, request: Request):
    body = await body_of(request)
    state[room_id].monitoring = True
    if body.get("camera_source", "device") == "device":
        start_device_camera(room_id)
    cam = next((c for c in db.cameras if c["room_id"] == room_id), None)
    return {"message": f"Monitoring started for Room {room_id}", "room_id": room_id,
            "camera_id": cam["id"] if cam else "CAM001", "status": "started"}


@api.post("/api/rooms/{room_id}/monitoring/stop")
def stop_monitoring(room_id: str):
    room = state[room_id]
    room.monitoring, room.tracks = False, []
    room.stop.set()
    for elder_id in list(room.seen):        # finish the running activity entries
        db.close_activity(elder_id)
    room.seen.clear()
    return {"message": f"Monitoring stopped for Room {room_id}", "room_id": room_id, "status": "stopped"}


app.include_router(api)      # (after all `@api` routes are defined)


# ---------- real-time notifications ----------
@app.websocket("/ws/notifications")
async def websocket_notifications(websocket: WebSocket, token: str = None):
    user = user_from_token(token) if token else None
    if not user:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    active_sockets[user["id"]].append(websocket)
    try:
        while True:
            if await websocket.receive_text() == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in active_sockets[user["id"]]:
            active_sockets[user["id"]].remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)), reload=True)
