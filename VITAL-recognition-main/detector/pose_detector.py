import cv2
import mediapipe as mp


class PoseDetector:

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_pose(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb)

        landmarks = []

        if results.pose_landmarks:

            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

            h, w, _ = frame.shape

            for idx, lm in enumerate(results.pose_landmarks.landmark):

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                landmarks.append((idx, cx, cy))

        return frame, landmarks