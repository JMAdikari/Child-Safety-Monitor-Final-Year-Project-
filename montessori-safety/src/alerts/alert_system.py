"""
Real-time Alert System.
Delivers alerts through: audible alarm, dashboard notification, SMS.
"""

import time
import threading
import json
from datetime import datetime


class AlertSystem:
    """
    Manages alert generation, cooldown, and multi-channel delivery.
    """
    
    def __init__(self, cooldown_seconds=10, enable_sound=True, 
                 enable_sms=False, twilio_config=None):
        """
        Args:
            cooldown_seconds: Minimum time between alerts for the same person+activity
            enable_sound: Play audible alarm
            enable_sms: Send SMS via Twilio
            twilio_config: dict with 'account_sid', 'auth_token', 'from_number', 'to_number'
        """
        self.cooldown_seconds = cooldown_seconds
        self.enable_sound = enable_sound
        self.enable_sms = enable_sms
        
        # Track last alert time per person+activity to prevent spam
        self.last_alert_time = {}  # Key: "track_id_activity" -> timestamp
        
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
                      location=None, frame=None):
        """
        Process a detected dangerous activity and send alerts.
        
        Args:
            activity_type: One of 'fall', 'climb', 'fight', 'leaving', 'danger_zone'
            confidence: Detection confidence (0-1)
            person_id: Track ID of the person
            location: (x, y) pixel coordinates
            frame: Current video frame (for screenshot in alert)
            
        Returns:
            True if alert was sent, False if suppressed by cooldown
        """
        # Check cooldown
        alert_key = f"{person_id}_{activity_type}"
        current_time = time.time()
        
        if alert_key in self.last_alert_time:
            elapsed = current_time - self.last_alert_time[alert_key]
            if elapsed < self.cooldown_seconds:
                return False  # Still in cooldown
        
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
        self.last_alert_time[alert_key] = current_time
        
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
        """Play audible alarm sound."""
        try:
            import os
            # Use system beep as fallback
            os.system('echo -e "\a"')
            # For a proper alarm sound, place an alarm.wav in the project:
            # from playsound import playsound
            # playsound('assets/alarm.wav')
        except Exception as e:
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