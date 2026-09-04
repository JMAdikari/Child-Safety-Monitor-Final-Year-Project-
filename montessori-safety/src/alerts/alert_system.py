"""
Real-time Alert System.
Delivers alerts through: audible alarm, dashboard notification, SMS.
"""

import time
import math
import threading
import json
from datetime import datetime


# The same alarm for every activity — one recognisable sound, sustained long
# enough to be noticed across a room. Total length is roughly 3.3 seconds.
ALARM_TONE_HZ = 1000
ALARM_BEEP_MS = 400
ALARM_GAP_MS = 150
ALARM_REPEATS = 6


class AlertSystem:
    """
    Manages alert generation, cooldown, and multi-channel delivery.
    """
    
    def __init__(self, cooldown_seconds=4, enable_sound=True,
                 enable_sms=False, twilio_config=None, spatial_radius=0):
        """
        Args:
            cooldown_seconds: Minimum time between alerts for the same person+activity
            enable_sound: Play audible alarm
            enable_sms: Send SMS via Twilio
            twilio_config: dict with 'account_sid', 'auth_token', 'from_number', 'to_number'
            spatial_radius: Pixels within which a repeat alert is treated as the
                same event even if the track ID changed. 0 disables it.
        """
        self.cooldown_seconds = cooldown_seconds
        self.enable_sound = enable_sound
        self.enable_sms = enable_sms

        # Track last alert time per person+activity to prevent spam
        self.last_alert_time = {}  # Key: "track_id_activity" -> timestamp

        # A second cooldown keyed on where the alert happened, which survives
        # ByteTrack renumbering a child mid-event.
        #
        # Off by default (radius 0). It was built to stop one fall alarming
        # twice under two track numbers, but testing showed that suppression
        # working against the more important case: a child who falls, is
        # renumbered, and is still on the floor needs the second alarm, not
        # silence. Kept switchable rather than deleted so the effect can still
        # be measured.
        self.last_alert_place = []      # (x, y, activity, timestamp)
        self.spatial_radius = spatial_radius
        
        # Alert log
        self.alert_log = []
        
        # WebSocket clients (for dashboard)
        self.ws_clients = []
        
        # Twilio setup
        self.twilio_client = None
        if enable_sms and twilio_config:
            try:
                from twilio.rest import Client
                self.twilio_client = Client(
                    twilio_config['account_sid'],
                    twilio_config['auth_token']
                )
                self.twilio_from = twilio_config['from_number']
                self.twilio_to = twilio_config['to_number']
            except Exception as e:
                print(f"Twilio init failed: {e}. SMS disabled.")
                self.enable_sms = False
    
    def trigger_alert(self, activity_type, confidence, person_id,
                      location=None, frame=None, timestamp=None):
        """
        Process a detected dangerous activity and send alerts.

        Args:
            activity_type: One of 'fall', 'climb', 'fight', 'leaving', 'danger_zone'
            confidence: Detection confidence (0-1)
            person_id: Track ID of the person
            location: (x, y) pixel coordinates
            frame: Current video frame (for screenshot in alert)
            timestamp: Video time in seconds. Omit for a live camera, where the
                wall clock and video time are the same thing.

        Returns:
            True if alert was sent, False if suppressed by cooldown
        """
        current_time = self._now(timestamp)

        if self.in_cooldown(person_id, activity_type, timestamp):
            return False

        if self._place_in_cooldown(activity_type, location, current_time):
            return False

        # Create alert object
        alert = {
            'timestamp': datetime.now().isoformat(),
            'activity': activity_type,
            'confidence': round(confidence, 3),
            'person_id': person_id,
            'location': location,
            'message': self._format_message(activity_type, person_id, confidence)
        }

        # Log it
        self.alert_log.append(alert)
        self.last_alert_time[f"{person_id}_{activity_type}"] = current_time
        if location is not None and self.spatial_radius:
            self.last_alert_place.append(
                (location[0], location[1], activity_type, current_time))
        
        # Print to console
        print(f"\n{'='*50}")
        print(f"  ALERT: {alert['message']}")
        print(f"  Confidence: {confidence:.1%} | Person #{person_id}")
        print(f"  Time: {alert['timestamp']}")
        print(f"{'='*50}\n")
        
        # Send through channels (in separate threads to not block pipeline)
        if self.enable_sound:
            threading.Thread(target=self._play_alarm, daemon=True).start()
        
        if self.enable_sms:
            threading.Thread(target=self._send_sms, args=(alert,), daemon=True).start()
        
        # Broadcast to dashboard via WebSocket
        self._broadcast_to_dashboard(alert)
        
        return True
    
    # --- Phase A verification (NEXT_PHASE_PLAN §4-A) ---
    def _now(self, timestamp=None):
        """Video time when the caller supplies it, otherwise the wall clock.

        Recorded video is processed far slower than real time — around 1 fps on
        CPU — so a wall-clock cooldown of 10s would only cover about 2s of
        footage and barely suppress anything.
        """
        return time.time() if timestamp is None else timestamp

    def in_cooldown(self, person_id, activity_type, timestamp=None):
        """Query the cooldown without consuming it, so AlertVerifier can report it."""
        alert_key = f"{person_id}_{activity_type}"
        if alert_key not in self.last_alert_time:
            return False
        elapsed = self._now(timestamp) - self.last_alert_time[alert_key]
        return elapsed < self.cooldown_seconds

    def _place_in_cooldown(self, activity_type, location, current_time):
        """True when this activity already alerted nearby, whatever the track ID."""
        if not self.spatial_radius or location is None:
            return False

        # Drop expired entries so the list cannot grow across a long session
        self.last_alert_place = [
            p for p in self.last_alert_place
            if current_time - p[3] < self.cooldown_seconds
        ]

        x, y = location[0], location[1]
        for px, py, act, _ in self.last_alert_place:
            if act == activity_type and math.hypot(x - px, y - py) <= self.spatial_radius:
                return True
        return False
    # --- end Phase A verification ---

    def _format_message(self, activity_type, person_id, confidence):
        """Create human-readable alert message."""
        messages = {
            'fall': f"FALL DETECTED — Child #{person_id} may have fallen!",
            'climb': f"CLIMBING DETECTED — Child #{person_id} is climbing on furniture!",
            'fight': f"FIGHTING DETECTED — Physical conflict involving Child #{person_id}!",
            'leaving': f"EXIT ALERT — Child #{person_id} is leaving the classroom!",
            'danger_zone': f"DANGER ZONE — Child #{person_id} entered a restricted area!",
        }
        return messages.get(activity_type, f"Unknown activity: {activity_type}")
    
    def _play_alarm(self):
        # A single 500ms beep was easy to miss, and one incident may raise only
        # one alert, so the tone repeats for a few seconds instead. Runs in a
        # daemon thread, so the pipeline is not held up by it.
        try:
            import winsound
            for i in range(ALARM_REPEATS):
                winsound.Beep(ALARM_TONE_HZ, ALARM_BEEP_MS)
                if i < ALARM_REPEATS - 1:
                    time.sleep(ALARM_GAP_MS / 1000)
        except Exception:
            pass
    
    def _send_sms(self, alert):
        """Send SMS via Twilio."""
        if self.twilio_client:
            try:
                message = self.twilio_client.messages.create(
                    body=f"[SAFETY ALERT] {alert['message']} ({alert['timestamp']})",
                    from_=self.twilio_from,
                    to=self.twilio_to
                )
                print(f"  SMS sent: {message.sid}")
            except Exception as e:
                print(f"  SMS failed: {e}")
    
    def _broadcast_to_dashboard(self, alert):
        """Send alert to dashboard via WebSocket (implemented in dashboard module)."""
        # This will be connected to Flask-SocketIO in the dashboard
        pass
    
    def get_recent_alerts(self, count=20):
        """Get most recent alerts."""
        return self.alert_log[-count:]
    
    def save_log(self, path="logs/alert_log.json"):
        """Save alert log to file."""
        with open(path, 'w') as f:
            json.dump(self.alert_log, f, indent=2)