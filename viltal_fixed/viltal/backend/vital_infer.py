"""
VITAL inference helper: turns noisy per-frame YOLO predictions into stable activities.

WHY: a detector only sees ONE frame. During sit->stand, a bent torso looks like a fall for a
few frames, so raw output flickers to "Falling". A real fall is a SEQUENCE:
    upright person  ->  quickly lying (wide box)  ->  stays down
This filter checks that sequence per tracked person and raises ONE fall alert (not one per frame).

Usage (laptop or Colab):
    python vital_infer.py --weights best.pt --source 0                 # webcam
    python vital_infer.py --weights best.pt --source video.mp4 --save out.mp4

Needs: pip install ultralytics   (the filter class itself is pure Python)
"""
import os
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import dataclass
import statistics

# Set local YOLO settings dir to prevent Windows permission errors in sandbox
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent))

CLASSES = ["Walking", "Standing", "Sitting", "Sleeping", "Falling"]


@dataclass
class FilterConfig:
    fps: float = 25.0           # camera / video fps (used to convert seconds to frames)
    window_s: float = 1.5       # smoothing window for the activity label
    history_s: float = 4.0      # how far back we look for "was this person upright?"
    fall_conf: float = 0.60     # a frame only counts as "falling" if its confidence >= this
    fall_ratio: float = 0.60    # >= 60% of frames in the window must say Falling
    min_frames: int = 6         # need at least this many frames in the window to decide
    min_aspect: float = 1.00    # width/height of box to count as "lying" (1.0 = wider than tall)
    upright_aspect: float = 0.80  # width/height <= this means upright (standing/sitting/walking)
    hold_s: float = 3.0         # keep showing "Falling" this long after the alert
    cooldown_s: float = 8.0     # never alert twice for the same person within this time
    forget_s: float = 3.0       # drop a track that has not been seen for this long


class _Track:
    def __init__(self):
        self.hist = deque()            # (t, cls, conf, aspect)
        self.alert_until = -1e9
        self.last_alert = -1e9
        self.armed = True              # re-armed when the person is seen upright again
        self.last_nonfall = "Standing"
        self.last_seen = 0.0


class ActivityFilter:
    def __init__(self, cfg: FilterConfig = None):
        self.cfg = cfg or FilterConfig()
        self.tracks = defaultdict(_Track)

    def update(self, track_id, t, cls_name, conf, box_xyxy):
        """One detection of one person at time t (seconds).
        Returns (label, new_fall_alert: bool). new_fall_alert is True on ONE frame only."""
        c = self.cfg
        x1, y1, x2, y2 = box_xyxy
        aspect = max(x2 - x1, 1e-6) / max(y2 - y1, 1e-6)
        st = self.tracks[track_id]
        st.last_seen = t
        st.hist.append((t, cls_name, float(conf), aspect))
        while st.hist and t - st.hist[0][0] > c.history_s:
            st.hist.popleft()
        win = [h for h in st.hist if t - h[0] <= c.window_s]
        n = len(win)

        med_aspect = statistics.median(h[3] for h in win)
        fall_frames = sum(1 for h in win if h[1] == "Falling" and h[2] >= c.fall_conf)
        upright_seen = any(h[3] <= c.upright_aspect for h in st.hist)

        # person is clearly upright again -> allow a future alert
        if n >= c.min_frames and med_aspect <= c.upright_aspect and t > st.alert_until:
            st.armed = True

        new_alert = False
        if (n >= c.min_frames
                and fall_frames / n >= c.fall_ratio       # Falling predicted consistently, not a flicker
                and med_aspect >= c.min_aspect            # body is now horizontal
                and upright_seen                          # ...and was upright a moment ago
                and st.armed
                and t - st.last_alert >= c.cooldown_s):
            new_alert = True
            st.last_alert = t
            st.alert_until = t + c.hold_s
            st.armed = False

        if t <= st.alert_until:
            return "Falling", new_alert

        # Not a confirmed fall: best NON-falling class by summed confidence in the window
        score = defaultdict(float)
        for _, cl, cf, _a in win:
            if cl != "Falling":
                score[cl] += cf
        if score:
            label = max(score, key=score.get)
            st.last_nonfall = label
        elif med_aspect >= c.min_aspect:
            label = "Sleeping"                            # lying, but no fall sequence seen
        else:
            label = st.last_nonfall
        return label, False

    def cleanup(self, t):
        for k in [k for k, s in self.tracks.items() if t - s.last_seen > self.cfg.forget_s]:
            del self.tracks[k]


def run(weights, source, conf=0.30, imgsz=640, fall_conf=0.60, save=None, show=True, min_aspect=1.0):
    import cv2
    from ultralytics import YOLO
    model = YOLO(weights)
    cap = cv2.VideoCapture(0 if str(source) == "0" else str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    flt = ActivityFilter(FilterConfig(fps=fps, fall_conf=fall_conf, min_aspect=min_aspect))
    writer = None
    frame_i = 0
    COL = {"Walking": (0, 200, 0), "Standing": (255, 160, 0), "Sitting": (0, 160, 255),
           "Sleeping": (180, 80, 255), "Falling": (0, 0, 255)}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_i / fps
        frame_i += 1
        # agnostic_nms=True -> ONE class per person; track() gives a stable ID per person
        r = model.track(frame, persist=True, conf=conf, imgsz=imgsz, agnostic_nms=True,
                        tracker="bytetrack.yaml", verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else list(range(1, len(r.boxes) + 1))
            for box, tid, cl, cf in zip(r.boxes.xyxy.cpu().numpy(), ids,
                                        r.boxes.cls.int().cpu().tolist(), r.boxes.conf.cpu().tolist()):
                label, alert = flt.update(tid, t, CLASSES[cl], cf, box)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), COL[label], 3 if label == "Falling" else 2)
                cv2.putText(frame, f"#{tid} {label}", (x1, max(y1 - 8, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL[label], 2)
                if alert:
                    print(f"[ALERT] fall confirmed for person #{tid} at {t:.1f}s")
        flt.cleanup(t)
        if save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        if show:
            cv2.imshow("VITAL", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="0")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--fall-conf", type=float, default=0.60)
    ap.add_argument("--min-aspect", type=float, default=1.0,
                    help="width/height of box that counts as lying. Lower (0.8) for overhead/angled cameras")
    ap.add_argument("--save", default=None)
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    run(a.weights, a.source, a.conf, 640, a.fall_conf, a.save, not a.no_show, a.min_aspect)
