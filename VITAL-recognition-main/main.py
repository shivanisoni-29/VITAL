import cv2
import threading
import requests
from detector.pose_detector import PoseDetector
from activity_classifier import ActivityClassifier
from sos_alert import SOSAlert


class VideoStream:
    """
    Reads frames from the source in a background thread and always keeps
    only the LATEST frame. Without this, cv2.VideoCapture.read() pulls
    frames in the order they arrived - if processing (pose detection +
    classification) is even slightly slower than the incoming stream, a
    backlog builds up and the video you see keeps falling further and
    further behind in "slow motion". This discards the backlog every time
    instead of queuing it.
    """

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        # ask for the smallest OS-level buffer possible (works on some
        # backends, ignored on others - the thread below is the real fix)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return self.ret, None
            return self.ret, self.frame.copy()

    def isOpened(self):
        return self.cap.isOpened()

    def set(self, *args, **kwargs):
        # resolution for a phone stream is controlled inside the IP Webcam
        # app itself, not via cv2 - keep this as a harmless no-op so the
        # rest of main.py doesn't need to change
        pass

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()


def main():

    # ---- Camera source ----
    # Laptop webcam (currently active):
    CAMERA_SOURCE = 0
    #
    # Android phone via "IP Webcam" app (Play Store):
    # 1. Open the app, tap "Start server"
    # 2. Note the IP shown on screen, e.g. http://192.168.1.5:8080
    # 3. Phone and laptop must be on the SAME WiFi network
    # 4. Uncomment the line below and put your IP in, keep "/video" at the end
    # CAMERA_SOURCE = "http://192.168.1.5:8080/video"

    cap = VideoStream(CAMERA_SOURCE)

    if not cap.isOpened():
        print("Camera not found!")
        return

    # Note: for a phone stream, resolution is set INSIDE the IP Webcam app
    # (Video preferences -> Resolution), not via cap.set() here. These two
    # lines only affect an actual USB/built-in camera.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Patient Monitoring", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Patient Monitoring", 1280, 720)

    pose_detector = PoseDetector()
    classifier = ActivityClassifier()
    sos = SOSAlert()
    last_activity = None
    ELDER_ID = "ELDER001"
    show_debug = False  # press 'd' to toggle the calibration panel on/off

    while True:

        success, frame = cap.read()
        if not success:
            break

        frame, landmarks = pose_detector.detect_pose(frame)

        activity = classifier.update(landmarks)

        if activity != last_activity:
            try:
               requests.post(
            "http://127.0.0.1:5000/activity",
            json={
                "elderId": ELDER_ID,
                "activity": activity
                  },
                  timeout=2
                )
               print("Sent to backend:", activity)
               last_activity = activity
            except Exception as e:
                print("Backend Error:", e)


        # fire SOS exactly once, the frame the 5-second recovery window expires
        if classifier.should_trigger_sos():
            sos.trigger()

        # ---- everything below scales with the ACTUAL frame size ----
        # Fixed pixel sizes (font scale, rectangle sizes) only look right at
        # one specific resolution. If the camera returns a different size
        # than expected, text/panels drawn at fixed sizes look oversized or
        # tiny. Scaling relative to frame width fixes this for any camera.
        h, w = frame.shape[:2]
        scale = w / 1280.0
        scale = max(0.35, min(scale, 2.0))  # keep it sane at extreme resolutions

        for idx, x, y in landmarks:
            cv2.circle(frame, (x, y), max(2, int(4 * scale)), (0, 255, 0), -1)
            cv2.putText(
                frame, str(idx), (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (255, 0, 0), 1
            )

        cv2.putText(
            frame, "Patient Monitoring System", (int(20 * scale), int(40 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (0, 255, 0), max(1, int(2 * scale))
        )

        color = (0, 0, 255) if activity == "Fall Detected" else (0, 140, 255) if activity == "Checking Fall..." else (0, 0, 255) if activity == "Jump" else (255, 200, 0)
        cv2.putText(
            frame, f"Activity : {activity}", (int(20 * scale), int(75 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65 * scale, color, max(1, int(2 * scale))
        )

        # while a fall is pending recovery, show a countdown to SOS
        remaining = classifier.seconds_since_fall()
        if remaining is not None and not sos.active:
            secs_left = max(0, classifier.fall_recovery_seconds - remaining)
            cv2.putText(
                frame, f"No movement - SOS in {secs_left:0.1f}s",
                (int(20 * scale), int(105 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65 * scale, (0, 140, 255), max(1, int(2 * scale))
            )

        if sos.active:
            bar_h = int(55 * scale)
            cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 0, 255), -1)
            cv2.putText(
                frame, "SOS - EMERGENCY - PATIENT NOT RESPONDING",
                (int(20 * scale), int(bar_h * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (255, 255, 255), max(1, int(2 * scale))
            )

        # ---- DEBUG PANEL: live numbers, for threshold calibration ----
        # Off by default now - press 'd' to show/hide. Move around / walk /
        # sit / fall / jump and watch these values to calibrate thresholds.
        if show_debug and classifier.debug:
            dbg = classifier.debug
            lines = [
                f"torso_px: {dbg['torso_px']}",
                f"movement_ratio: {dbg['movement_ratio']}  avg: {dbg['avg_movement_ratio']}",
                f"hip_mv: {dbg['hip_movement_ratio']}  ankle_mv: {dbg['ankle_movement_ratio']}",
                f"recent_max_movement_ratio (fall): {dbg['recent_max_movement_ratio']}",
                f"knee_angle: {dbg['knee_angle']}  hip_angle: {dbg['hip_angle']}",
                f"body_diff_ratio: {dbg['body_diff_ratio']}  jump_rise_ratio: {dbg['jump_rise_ratio']}",
                f"walk_frames: {dbg['walk_frames']}  run_frames: {dbg['run_frames']}",
                f"fall_state: {dbg['fall_state']}",
            ]
            font_scale = 0.42 * scale
            line_h = int(20 * scale)
            panel_w = int(560 * scale)
            panel_h = line_h * len(lines) + int(15 * scale)
            y0 = h - panel_h

            overlay = frame.copy()
            cv2.rectangle(overlay, (10, y0), (10 + panel_w, h - 5), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

            for i, line in enumerate(lines):
                cv2.putText(
                    frame, line, (18, y0 + line_h * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1
                )

        cv2.imshow("Patient Monitoring", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            show_debug = not show_debug
        if key == ord("c") and sos.active:
            # manual acknowledge/clear of an active SOS (e.g. caretaker arrived)
            sos.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()