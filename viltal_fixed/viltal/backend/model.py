# VITAL activity model: YOLOv11 + ByteTrack, with vital_infer.ActivityFilter turning noisy
# per-frame predictions into stable activity labels and confirmed fall alerts.

import os
import threading
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
# Keep Ultralytics settings inside the project (avoids permission errors on Windows)
os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE / "Ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

from vital_infer import ActivityFilter, CLASSES  # noqa: E402

COLORS = {"Walking": (0, 200, 0), "Standing": (255, 160, 0), "Sitting": (0, 160, 255),
          "Sleeping": (180, 80, 255), "Falling": (0, 0, 255)}

model = None
model_path_used = None
_filters = {}                 # room_id -> ActivityFilter
_lock = threading.Lock()      # the tracker keeps state, so only one frame is processed at a time

for _p in (os.getenv("YOLO_MODEL_PATH"), HERE / "models" / "best.pt"):
    if _p and Path(_p).exists():
        try:
            from ultralytics import YOLO
            model, model_path_used = YOLO(str(_p)), str(_p)
            print(f"[VITAL] Loaded model from {_p}")
            break
        except Exception as e:
            print(f"[VITAL] Failed to load model weights from {_p}: {e}")
if model is None:
    print("[VITAL WARNING] No usable model weights found - inference unavailable.")


def is_ready():
    return model is not None


def predict_annotated(frame, conf_threshold=0.30, room_id="ROOM001"):
    """Returns (detections, annotated_frame). Each detection has box, activity, raw_activity,
    confidence, track_id and new_fall_alert (True on exactly one frame per confirmed fall)."""
    if model is None or frame is None or frame.size == 0:
        return [], frame
    annotated, detections, t = frame.copy(), [], time.time()
    flt = _filters.setdefault(room_id, ActivityFilter())
    scale = max(frame.shape[1] / 640.0, 1.0)
    thick = max(2, int(round(2 * scale)))
    try:
        with _lock:
            try:
                result = model.track(frame, persist=True, conf=conf_threshold, imgsz=640,
                                     agnostic_nms=True, tracker="bytetrack.yaml", verbose=False)[0]
            except Exception:  # tracker can fail on the first frame: fall back to plain detection
                result = model.predict(source=frame, conf=conf_threshold, imgsz=640,
                                       agnostic_nms=True, verbose=False)[0]
            boxes = result.boxes
            if boxes is not None and len(boxes):
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else list(range(1, len(boxes) + 1))
                for box, tid, cid, conf in zip(boxes.xyxy.cpu().numpy(), ids,
                                               boxes.cls.int().cpu().tolist(), boxes.conf.cpu().tolist()):
                    raw = CLASSES[cid] if cid < len(CLASSES) else str(model.names.get(cid, "Unknown"))
                    label, new_alert = flt.update(tid, t, raw, conf, box)
                    x1, y1, x2, y2 = map(int, box)
                    color = COLORS.get(label, (255, 255, 255))
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick + 2 if label == "Falling" else thick)
                    cv2.putText(annotated, f"#{tid} {label} ({conf:.2f})", (x1, max(y1 - int(8 * scale), int(20 * scale))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65 * scale, color, thick)
                    if label == "Falling":
                        cv2.putText(annotated, "FALL DETECTED", (x1, min(y2 + int(24 * scale), annotated.shape[0] - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75 * scale, (0, 0, 255), thick + 1)
                    detections.append({"box": [x1, y1, x2, y2], "activity": label, "raw_activity": raw,
                                       "confidence": float(conf), "track_id": int(tid),
                                       "new_fall_alert": bool(new_alert)})
            flt.cleanup(t)
    except Exception as e:
        print(f"[VITAL] Inference error: {e}")
        return [], frame
    return detections, annotated
