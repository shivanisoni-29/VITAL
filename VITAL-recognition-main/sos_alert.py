import time
import threading
 
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False
 
 
class SOSAlert:
 
    def __init__(self, cooldown_seconds=30, alarm_duration=10):
        self.cooldown_seconds = cooldown_seconds
        self.alarm_duration = alarm_duration
        self.last_triggered = 0
        self._alarm_thread = None
        self._stop_alarm = threading.Event()
        self.active = False
 
    def trigger(self):
        now = time.time()
        # avoid re-triggering repeatedly while one SOS is already active
        if now - self.last_triggered < self.cooldown_seconds:
            return
        self.last_triggered = now
        self.active = True
 
        print("SOS TRIGGERED - fall not recovered within timeout window!")
        self._notify()
        self._start_alarm_sound()
 
    def clear(self):
        """Call this once the emergency is manually acknowledged/resolved."""
        self.active = False
        self._stop_alarm.set()
 
    def _notify(self):
        # TODO: plug in your real alert channel here, for example:
        #
        # import requests
        # requests.post("https://api.twilio.com/.../Messages.json", data={...})
        #
        # or an email / push notification / webhook call to a
        # caretaker's phone. Left blank on purpose - needs your own
        # account credentials.
        pass
 
    def _start_alarm_sound(self):
        if self._alarm_thread and self._alarm_thread.is_alive():
            return
        self._stop_alarm.clear()
        self._alarm_thread = threading.Thread(target=self._alarm_loop, daemon=True)
        self._alarm_thread.start()
 
    def _alarm_loop(self):
        end_time = time.time() + self.alarm_duration
        while time.time() < end_time and not self._stop_alarm.is_set():
            if HAS_WINSOUND:
                winsound.Beep(1500, 400)
            else:
                # cross-platform fallback: terminal bell
                print("\a", end="", flush=True)
            time.sleep(0.3)
        self.active = False