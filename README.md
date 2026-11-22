# VITAL - Elderly Activity Monitor & Fall Detection

Human Activity Recognition web application using YOLOv11 to detect activities (Walking, Standing, Sitting, Sleeping, Falling), featuring a FastAPI backend and an interactive care dashboard.

## Project Structure
- `viltal_fixed/`
  - `viltal/`
    - `backend/` - FastAPI backend, SQLite database, inference pipeline, and model weights (`models/best.pt`)
    - `training/` - Google Colab training notebook (`VITAL_colab_training_FIXED.ipynb`)
    - `vital_care_dashboard.html` - Interactive caregiver dashboard

## Getting Started

### 1. Setup Backend
Navigate to the backend directory and install dependencies:
```bash
cd viltal_fixed/viltal/backend
pip install -r requirements.txt
```

### 2. Run Application
```bash
python app.py
```
Open [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard) (Demo login: `caregiver@example.com` / `password123`).

## Backend Components
| Component | Description |
|---|---|
| `app.py` | FastAPI application, camera & detection pipeline, WebSocket and SMS fall alerts |
| `db.py` | SQLite storage for patients, activity history, alerts, reports (optional Firebase mirror) |
| `model.py` | YOLOv11 detection and tracking pipeline |
| `vital_infer.py` | Temporal filter stabilizing activity detections and confirming falls |
| `models/best.pt` | Trained YOLO model weights |
| `test_backend.py` | Test suite for backend APIs and services |
