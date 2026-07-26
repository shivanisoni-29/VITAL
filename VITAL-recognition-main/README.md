# Vital IMG - Patient Monitoring System

A webcam-based patient activity monitoring system using MediaPipe pose
detection. Detects Standing, Sitting, Walking, Running, Jumping, Sleeping,
and Falls in real time, with an automatic SOS alert if a fall isn't
recovered from within a configurable time window.

## Features

- Real-time pose tracking (MediaPipe)
- Scale-invariant activity classification (movement/angle thresholds are
  relative to the person's torso length, so it works at any camera
  distance/resolution)
- Fall detection via a state-transition rule (Standing -> horizontal = fall,
  Standing -> Sitting -> horizontal = normal sleeping) with a confirmation
  window before triggering
- SOS alert (local alarm sound) if a fall isn't recovered from in time
- Debug overlay (press `d`) showing live threshold values for calibration
- Works with a laptop webcam or a phone camera (via the IP Webcam app)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux

pip install -r requirement.txt
```

## Run

```bash
python main.py
```

Controls:
- `q` - quit
- `d` - toggle the debug/calibration panel
- `c` - clear an active SOS alert

## Project structure

```
detector/
    pose_detector.py     # MediaPipe pose wrapper
activity_classifier.py   # Consolidated activity + fall + jump logic
sos_alert.py              # SOS alarm handler
utils/
    angles.py             # Angle calculation helper
    draw.py
main.py                   # Entry point / camera loop
```

## Notes

- `sos_alert.py`'s `_notify()` is a stub - plug in your own SMS/call/push
  notification service (e.g. Twilio) with your own credentials.
- Camera source (laptop webcam vs phone IP Webcam stream) is set via the
  `CAMERA_SOURCE` variable in `main.py`.