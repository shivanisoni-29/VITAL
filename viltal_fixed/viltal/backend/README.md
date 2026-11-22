# VITAL - Elderly Activity Monitor

Human Activity Recognition web app: YOLOv11 detects Walking / Standing / Sitting / Sleeping / Falling,
FastAPI serves the data, and `../vital_care_dashboard.html` is the dashboard.

## Run
```bash
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:8000/dashboard (demo login: `caregiver@example.com` / `password123`).

## Files
| File | Purpose |
|---|---|
| `app.py` | API routes, camera/detection pipeline, fall alerts (WebSocket + optional SMS) |
| `db.py` | SQLite storage (patients, activity history, alerts, reports); optional Firebase mirror |
| `model.py` | YOLO + tracking, draws the boxes on the live stream |
| `vital_infer.py` | Temporal filter that turns noisy frames into stable activities and confirmed falls (also a standalone webcam/video CLI) |
| `test_backend.py` | `pip install pytest httpx` then `pytest` |

Settings live in `.env` (see `.env.example`). Data is saved in `vital.db` next to `app.py`.
