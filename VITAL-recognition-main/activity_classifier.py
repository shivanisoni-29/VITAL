import math
import time
from collections import deque, Counter
 
 
def calculate_angle(a, b, c):
    """Angle at point b, given points a, b, c as (x, y)."""
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle
 
 
class ActivityClassifier:
 
    def __init__(
        self,
        movement_window=5,             # frames to average hip speed over
        confirm_frames=7,              # frames for majority-vote confirmation
        walk_confirm=5,                # consecutive "step" frames to confirm walking
        run_confirm=4,                 # consecutive "sprint" frames to confirm running
        recovery_confirm_frames=5,     # consecutive recovered frames to clear a fall
        horizontal_confirm_frames=3,   # consecutive frames to trust a horizontal-body reading
 
        # ---- scale-invariant thresholds, as a RATIO of torso length ----
        low_movement_ratio=0.05,       # below this avg movement/torso => "not moving"
        walk_movement_ratio=0.10,      # per-frame hip movement/torso => a walk step
        run_movement_ratio=0.22,       # per-frame hip movement/torso => running
        fall_movement_ratio=0.30,      # movement spike/torso => possible fall
        jump_rise_ratio=0.10,          # hip rise above baseline/torso => airborne
        horizontal_body_ratio=0.35,    # shoulder-hip vertical gap/torso => lying down
 
        sitting_hip_angle=(60, 120),   # shoulder-hip-knee angle range for sitting
        standing_hip_angle=150,        # shoulder-hip-knee angle above this => standing
 
        fall_window=10,                # frames to look back for a movement spike
        fall_recovery_seconds=5,       # seconds to wait for recovery before SOS
 
        jump_window=6,                 # frames considered for a jump rise-then-return
        jump_display_seconds=1.0,      # how long "Jump" stays shown after landing
 
        min_torso_px=40,               # safety floor so ratios don't blow up if the
                                        # person is very small/far in frame
    ):
        self.movement_window = movement_window
        self.confirm_frames = confirm_frames
        self.walk_confirm = walk_confirm
        self.run_confirm = run_confirm
        self.recovery_confirm_frames = recovery_confirm_frames
        self.horizontal_confirm_frames = horizontal_confirm_frames
 
        self.low_movement_ratio = low_movement_ratio
        self.walk_movement_ratio = walk_movement_ratio
        self.run_movement_ratio = run_movement_ratio
        self.fall_movement_ratio = fall_movement_ratio
        self.jump_rise_ratio = jump_rise_ratio
        self.horizontal_body_ratio = horizontal_body_ratio
 
        self.sitting_hip_angle = sitting_hip_angle
        self.standing_hip_angle = standing_hip_angle
 
        self.fall_recovery_seconds = fall_recovery_seconds
        self.fall_movement_history = deque(maxlen=fall_window)
 
        self.jump_display_seconds = jump_display_seconds
        self.min_torso_px = min_torso_px
 
        self.prev_hip = None
        self.prev_left_ankle = None
        self.prev_right_ankle = None
        self.hip_speed_history = deque(maxlen=movement_window)
 
        self.walk_frames = 0
        self.run_frames = 0
 
        self.activity_history = deque(maxlen=confirm_frames)
        self.normal_confirmed = "Unknown"   # confirmed non-fall activity
        self.confirmed_activity = "Unknown" # final displayed activity
        self.debug = {}
 
        # -- torso-length smoothing (the scale reference) --
        self.torso_length_ema = None
 
        # -- jump tracking --
        self.hip_y_baseline = None
        self.jump_display_until = 0
        self._was_airborne = False
 
        # -- fall state machine: "normal" -> "pending" -> "confirmed" --
        self.fall_state = "normal"
        self.fall_pending_since = None
        self.sos_triggered = False
        self._sos_flag_this_frame = False
        self._recovery_streak = 0
 
        # -- transition tracking: was the last upright state Standing (fall)
        #    or Sitting (normal lying-down), right before going horizontal? --
        self.last_upright_activity = "Unknown"
        self._was_horizontal = False
        self._horizontal_is_fall = False
        self._horizontal_streak = 0
 
    # ---------- helpers ----------
 
    @staticmethod
    def _midpoint(p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
 
    def _avg_movement(self):
        if not self.hip_speed_history:
            return 0
        return sum(self.hip_speed_history) / len(self.hip_speed_history)
 
    # ---------- main entry point ----------
 
    def update(self, landmarks):
        """
        landmarks: list of (idx, x, y) as produced by PoseDetector.
        Returns the confirmed (de-noised) activity string.
        """
        now = time.time()
 
        if len(landmarks) < 29:
            # Full pose isn't visible - very common right after a fall,
            # since a person lying down often has their legs/feet out of
            # frame or self-occluded. We must NOT freeze the fall timer
            # here, or a real fall can silently never reach SOS.
            #
            # Safety-first behaviour: if we can't see the pose well enough
            # to confirm the person got up, assume they have NOT recovered.
            self._sos_flag_this_frame = False
 
            if self.fall_state == "pending":
                self._recovery_streak = 0
                elapsed = now - self.fall_pending_since if self.fall_pending_since else 0
                if elapsed >= self.fall_recovery_seconds:
                    self.fall_state = "confirmed"
                    self.sos_triggered = True
                    self._sos_flag_this_frame = True
                    self.confirmed_activity = "Fall Detected"
                else:
                    self.confirmed_activity = "Checking Fall..."
            elif self.fall_state == "confirmed":
                # Don't reset recovery progress here - if the person is in
                # the middle of standing back up, fast motion can briefly
                # drop landmark tracking. Losing a frame or two shouldn't
                # wipe out recovery progress already made on good frames.
                self.confirmed_activity = "Fall Detected"
            # if fall_state == "normal", we simply don't have enough info
            # this frame - keep showing whatever was last confirmed
 
            return self.confirmed_activity
 
        self._sos_flag_this_frame = False
 
        pts = {idx: (x, y) for idx, x, y in landmarks}
 
        left_hip, right_hip = pts[23], pts[24]
        left_knee, right_knee = pts[25], pts[26]
        left_ankle, right_ankle = pts[27], pts[28]
        left_shoulder, right_shoulder = pts[11], pts[12]
 
        hip = self._midpoint(left_hip, right_hip)
        shoulder = self._midpoint(left_shoulder, right_shoulder)
 
        # ---- scale reference: torso length, smoothed ----
        raw_torso = math.dist(shoulder, hip)
        if self.torso_length_ema is None:
            self.torso_length_ema = raw_torso
        else:
            # smooth quickly enough to track real scale changes (person walks
            # toward/away from camera) but not so fast that a single noisy
            # frame throws off every threshold
            self.torso_length_ema = 0.9 * self.torso_length_ema + 0.1 * raw_torso
        torso = max(self.torso_length_ema, self.min_torso_px)
 
        # ---- movement (hip displacement between frames), as a ratio of torso ----
        if self.prev_hip is None:
            movement_px = 0
        else:
            movement_px = math.dist(hip, self.prev_hip)
        self.prev_hip = hip
        hip_movement = movement_px / torso
 
        # ---- ankle movement: legs swing noticeably during a normal gait
        # even when the hip barely translates in the image (e.g. walking
        # straight toward/away from the camera, where perspective makes
        # hip displacement small). Using the larger of hip vs ankle
        # movement catches walking regardless of direction relative to
        # the camera.
        if self.prev_left_ankle is None:
            ankle_movement = 0
        else:
            left_ankle_move = math.dist(left_ankle, self.prev_left_ankle) / torso
            right_ankle_move = math.dist(right_ankle, self.prev_right_ankle) / torso
            ankle_movement = max(left_ankle_move, right_ankle_move)
        self.prev_left_ankle = left_ankle
        self.prev_right_ankle = right_ankle
 
        movement = max(hip_movement, ankle_movement)
        self.hip_speed_history.append(movement)
        avg_movement = self._avg_movement()
 
        # ---- walking / running counters ----
        if movement > self.run_movement_ratio:
            self.run_frames += 1
        else:
            self.run_frames = max(0, self.run_frames - 1)
 
        if movement > self.walk_movement_ratio:
            self.walk_frames += 1
        else:
            self.walk_frames = max(0, self.walk_frames - 1)
 
        is_running = self.run_frames >= self.run_confirm
        is_walking = self.walk_frames >= self.walk_confirm
 
        # ---- angles ----
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        knee_angle = (left_knee_angle + right_knee_angle) / 2
 
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        hip_angle = (left_hip_angle + right_hip_angle) / 2
 
        # ---- horizontal body check (used by both sleeping and fall) ----
        # Debounced over a few consecutive frames: fast movement causes
        # brief pose-estimation glitches (motion blur) where a single frame
        # can misreport the body as horizontal even while standing. Only
        # trust it once it's been true for `horizontal_confirm_frames` in
        # a row.
        body_diff_px = abs(shoulder[1] - hip[1])
        body_diff_ratio = body_diff_px / torso
        raw_horizontal = body_diff_ratio < self.horizontal_body_ratio
        self._horizontal_streak = self._horizontal_streak + 1 if raw_horizontal else 0
        is_horizontal_body = self._horizontal_streak >= self.horizontal_confirm_frames
        is_sleeping = is_horizontal_body
 
        # ---- transition-based fall detection ----
        # Standing/Walking/Running -> directly horizontal = a FALL.
        # Standing -> Sitting -> horizontal = normal lying down = Sleeping.
        # Decided once, right at the moment the body first goes horizontal
        # (the transition edge), then held for as long as the body stays down.
        if is_horizontal_body and not self._was_horizontal:
            self._horizontal_is_fall = self.last_upright_activity != "Sitting"
        if not is_horizontal_body:
            if self.normal_confirmed in ("Standing", "Sitting", "Walking", "Running"):
                self.last_upright_activity = self.normal_confirmed
            self._horizontal_is_fall = False
        self._was_horizontal = is_horizontal_body
 
        # ---- fall spike: track recent movement too, as a secondary signal ----
        # (e.g. a hard fall directly onto the floor with no clean Standing/
        # Sitting history yet, such as right after the program starts)
        self.fall_movement_history.append(movement)
        recent_max_movement = max(self.fall_movement_history)
        is_fall_spike = is_horizontal_body and (
            self._horizontal_is_fall or recent_max_movement > self.fall_movement_ratio
        )
 
        # ---- jump: hip rises sharply above its resting baseline, then returns ----
        # y grows DOWNWARD in image coords, so "rising" = decreasing y.
        if self.hip_y_baseline is None:
            self.hip_y_baseline = hip[1]
 
        rise_px = self.hip_y_baseline - hip[1]
        rise_ratio = rise_px / torso
        is_airborne = rise_ratio > self.jump_rise_ratio
 
        if not is_airborne:
            # only adapt the "ground level" baseline while NOT mid-jump
            self.hip_y_baseline = 0.9 * self.hip_y_baseline + 0.1 * hip[1]
 
        if not is_airborne and self._was_airborne:
            self.jump_display_until = now + self.jump_display_seconds
        self._was_airborne = is_airborne
 
        is_jump = now < self.jump_display_until
 
        # ---- sitting: hip angle in sitting range AND low movement ----
        is_sitting = (
            self.sitting_hip_angle[0] <= hip_angle <= self.sitting_hip_angle[1]
            and avg_movement < self.low_movement_ratio
        )
 
        # ---- standing: hip angle high AND low movement ----
        is_standing = (
            hip_angle > self.standing_hip_angle
            and avg_movement < self.low_movement_ratio
        )
 
        # ---- resolve NORMAL candidate by priority (fall handled separately below) ----
        if is_jump:
            candidate = "Jump"
        elif is_running:
            candidate = "Running"
        elif is_walking:
            candidate = "Walking"
        elif is_sleeping:
            candidate = "Sleeping"
        elif is_sitting:
            candidate = "Sitting"
        elif is_standing:
            candidate = "Standing"
        else:
            candidate = self.normal_confirmed
 
        # ---- hysteresis: majority vote over last `confirm_frames` frames ----
        self.activity_history.append(candidate)
        most_common, count = Counter(self.activity_history).most_common(1)[0]
        if count >= max(3, self.confirm_frames // 2 + 1):
            self.normal_confirmed = most_common
 
        # ---- fall state machine: normal -> pending (checking) -> confirmed (sticky) ----
        # A person can't be "recovered" while their body is still measured
        # as horizontal - even if a leg angle momentarily lands in the
        # sitting range while lying flat.
        #
        # Walking/Running are already hysteresis-confirmed strong signals
        # (they needed several consecutive high-movement frames to become
        # true in the first place) -> treat them as an immediate recovery,
        # rather than making the person wait through a second streak on
        # top of that. Standing/Sitting are weaker (angle + stillness only)
        # so they still need a short streak to avoid noise.
        strong_recovery = (is_walking or is_running) and not is_horizontal_body
        weak_recovery = (is_standing or is_sitting) and not is_horizontal_body
 
        if strong_recovery:
            self._recovery_streak = self.recovery_confirm_frames
        elif weak_recovery:
            self._recovery_streak += 1
        else:
            self._recovery_streak = 0
 
        recovery_confirmed = self._recovery_streak >= self.recovery_confirm_frames
 
        if self.fall_state == "normal":
            if is_fall_spike:
                self.fall_state = "pending"
                self.fall_pending_since = now
                self._recovery_streak = 0
 
        elif self.fall_state == "pending":
            if recovery_confirmed:
                self.fall_state = "normal"
                self.fall_pending_since = None
            else:
                elapsed = now - self.fall_pending_since
                if elapsed >= self.fall_recovery_seconds:
                    self.fall_state = "confirmed"
                    self.sos_triggered = True
                    self._sos_flag_this_frame = True
 
        elif self.fall_state == "confirmed":
            if recovery_confirmed:
                self.fall_state = "normal"
                self.fall_pending_since = None
                self.sos_triggered = False
 
        # ---- final displayed activity ----
        if self.fall_state == "confirmed":
            self.confirmed_activity = "Fall Detected"
        elif self.fall_state == "pending":
            self.confirmed_activity = "Checking Fall..."
        else:
            self.confirmed_activity = self.normal_confirmed
 
        # ---- debug snapshot (for calibrating ratios against your real camera) ----
        self.debug = {
            "torso_px": round(torso, 1),
            "movement_ratio": round(movement, 3),
            "hip_movement_ratio": round(hip_movement, 3),
            "ankle_movement_ratio": round(ankle_movement, 3),
            "avg_movement_ratio": round(avg_movement, 3),
            "recent_max_movement_ratio": round(recent_max_movement, 3),
            "knee_angle": round(knee_angle, 1),
            "hip_angle": round(hip_angle, 1),
            "body_diff_ratio": round(body_diff_ratio, 3),
            "jump_rise_ratio": round(rise_ratio, 3),
            "walk_frames": self.walk_frames,
            "run_frames": self.run_frames,
            "fall_state": self.fall_state,
        }
 
        return self.confirmed_activity
 
    def seconds_since_fall(self):
        """Seconds since a pending fall started, or None if no fall is pending."""
        if self.fall_pending_since is None:
            return None
        return time.time() - self.fall_pending_since
 
    def should_trigger_sos(self):
        """
        Returns True exactly once, on the frame the SOS condition first
        fires (fall + no recovery within fall_recovery_seconds). Call this
        once per frame in your main loop right after update().
        """
        return self._sos_flag_this_frame