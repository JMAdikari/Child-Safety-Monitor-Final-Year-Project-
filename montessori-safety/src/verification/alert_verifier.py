"""Deterministic rules deciding whether a model prediction becomes an alert.

Section 4, Phase A of NEXT_PHASE_PLAN.md. No LLM involved — every rule here is
individually switchable so its contribution can be measured separately.
"""

from collections import defaultdict, deque

NORMAL = 'normal'

# Emitted by the pipeline while a track's frame buffer fills. Not a prediction,
# so counting it as a threshold rejection would inflate that rule's share of
# the ablation stats by ~15 entries per new track.
PENDING = 'pending'

# Minimum confidence before a prediction is treated as a real detection.
#
# Measured by sweeping the Phase 2 model over the 6,253-window test set
# (5,995 normal / 64 fall / 194 climb):
#
#   fall   0.70 -> recall 0.438, 184 false alarms  (3.1% of normal frames)
#   climb  0.70 -> recall 0.572, 535 false alarms  (8.9%)
#          0.90 -> recall 0.340, 232 false alarms  (3.9%)
#
# Climb is held to 0.90 deliberately: it produced roughly three times the false
# alarms of fall at the same bar, and it is the less dangerous behaviour to miss.
# The cost is real — recall falls from 0.572 to 0.340.
#
# Fall stays at 0.70. Note the model is poorly calibrated for falls — median
# confidence on a genuine fall is only 0.36 — so raising this bar trades recall
# away quickly on the class where a miss matters most.
DEFAULT_THRESHOLD = {'fall': 0.70, 'climb': 0.90}

# Used when a class is not named above, and when a caller passes a bare number
FALLBACK_THRESHOLD = 0.70

# How long an activity must persist before it counts as a real event.
#
# Both a frame count and a wall-duration are required, and the stricter of the
# two governs. On a 30 fps camera they agree; on 60 fps the frame count is
# reached in half the time, so the duration keeps the rule honest.
#
# The values differ per class because the two behaviours differ in length.
# Measured across the 201 annotated events in ground_truth.csv:
#
#   fall   median 2.0s, but 22% run under 1.0s  -> a 1.0s rule would lose 21
#   climb  median 4.4s, only  5% run under 1.0s -> a 1.0s rule loses just 5
#
# Falls are therefore given a short requirement: they are brief, and missing
# one is the worst outcome the system can produce. Climbs are sustained, so a
# longer requirement costs little and suppresses far more noise.
#
# Note these are annotated event lengths, not detection lengths — the model may
# only track part of an event, so real losses run higher. Tune by measurement.
MIN_EVENT_FRAMES = {'fall': 12, 'climb': 30}
MIN_EVENT_DURATION = {'fall': 0.4, 'climb': 1.0}

# Detection drops out mid-event, especially during falls when the body is at an
# odd angle. Requiring an unbroken run would reject genuine events, so a gap
# this long is tolerated before the event is considered finished.
EVENT_GAP_TOLERANCE = 0.4


class AlertVerifier:

    def __init__(self, threshold=DEFAULT_THRESHOLD,
                 min_frames=None, min_duration=None,
                 gap_tolerance=EVENT_GAP_TOLERANCE,
                 use_threshold=True, use_duration=True,
                 use_geometry_veto=True, use_climb_veto=True, use_cooldown=True,
                 alert_system=None, rule_detector=None):
        # The pipeline passes --threshold through unconditionally and it defaults
        # to None, so None must mean "use the per-class values", not "no value"
        self.threshold = dict(DEFAULT_THRESHOLD) if threshold is None else threshold

        # A scalar applies to every class; a dict sets each class separately
        self.min_frames = min_frames if min_frames is not None else dict(MIN_EVENT_FRAMES)
        self.min_duration = (min_duration if min_duration is not None
                             else dict(MIN_EVENT_DURATION))
        self.gap_tolerance = gap_tolerance

        self.use_threshold = use_threshold
        self.use_duration = use_duration
        self.use_geometry_veto = use_geometry_veto
        self.use_climb_veto = use_climb_veto
        self.use_cooldown = use_cooldown

        self.alert_system = alert_system
        self.rule_detector = rule_detector

        # One open event per track: what it is, when it started, how many frames
        # it has been seen for, and when it was last seen. Cleared when the
        # activity changes or the gap tolerance is exceeded.
        self.events = {}

        # Per-rule rejection counts, so one pass over the test set produces the
        # whole ablation table instead of one pass per configuration
        self.rejections = defaultdict(int)
        self.alerts_passed = 0

        # Warm-up frames are counted under 'not_dangerous' alongside genuine
        # normal ones, because they are equally not a rule rejection. Tracking
        # them separately as well lets the dashboard funnel tell the two apart
        # without changing what the ablation table reads.
        self.warmup_frames = 0

        # Video time advances only when the caller supplies a timestamp. Falling
        # back to a frame counter keeps the frame rule working when it does not.
        self._frame_counter = defaultdict(int)

    def _threshold_for(self, predicted):
        # A bare number applies to every class; a dict sets each one separately
        if isinstance(self.threshold, dict):
            return self.threshold.get(predicted, FALLBACK_THRESHOLD)
        return self.threshold

    def should_alert(self, track_id, predicted, confidence, activity_features=None,
                     timestamp=None):
        """timestamp is video time in seconds (frame_idx / fps), not wall clock.

        Wall clock would break on recorded video: a 30-second clip takes minutes
        to process on CPU, so a one-second rule would never be satisfied.
        """
        checks = {}

        self._frame_counter[track_id] += 1
        if timestamp is None:
            timestamp = self._frame_counter[track_id] / 30.0   # assume 30 fps

        if predicted in (NORMAL, PENDING) or predicted is None:
            # A normal frame ends whatever event was open for this child
            self.events.pop(track_id, None)
            if predicted == PENDING:
                self.warmup_frames += 1
            return self._reject('not_dangerous', checks)

        thr = self._threshold_for(predicted)

        checks['threshold'] = (not self.use_threshold) or confidence >= thr
        if not checks['threshold']:
            return self._reject('threshold', checks)

        event = self._track_event(track_id, predicted, timestamp)
        progress = self._progress(predicted, event)

        checks['duration'] = (not self.use_duration) or self._event_is_long_enough(predicted, event)
        if not checks['duration']:
            return self._reject('duration', checks, progress)

        checks['geometry'] = ((not self.use_geometry_veto)
                              or not self._contradicted(track_id, predicted, activity_features))
        if not checks['geometry']:
            return self._reject('geometry', checks, progress)

        checks['cooldown'] = ((not self.use_cooldown)
                              or not self._in_cooldown(track_id, predicted, timestamp))
        if not checks['cooldown']:
            return self._reject('cooldown', checks, progress)

        self.alerts_passed += 1
        return {'alert': True, 'rejected_by': None, 'checks': checks,
                'progress': progress}

    def _reject(self, reason, checks, progress=None):
        self.rejections[reason] += 1
        return {'alert': False, 'rejected_by': reason, 'checks': checks,
                'progress': progress}

    def _progress(self, predicted, event):
        """How far this event has got towards the duration rule.

        The dashboard shows this so a track waiting on the rule reads as
        "18 of 30 frames" rather than as an unexplained silence.
        """
        return {
            'frames': event['frames'],
            'need_frames': self._per_class(self.min_frames, predicted, 1),
            'seconds': round(event['last_seen'] - event['start'], 2),
            'need_seconds': self._per_class(self.min_duration, predicted, 0.0),
        }

    def _track_event(self, track_id, predicted, timestamp):
        """Open, extend or restart this track's event, and return it."""
        event = self.events.get(track_id)

        # A different activity, or too long since the last sighting, means the
        # previous event is over and this is the start of a new one
        restart = (event is None
                   or event['activity'] != predicted
                   or timestamp - event['last_seen'] > self.gap_tolerance)

        if restart:
            event = {'activity': predicted, 'start': timestamp,
                     'last_seen': timestamp, 'frames': 1}
        else:
            event['last_seen'] = timestamp
            event['frames'] += 1

        self.events[track_id] = event
        return event

    def _event_is_long_enough(self, predicted, event):
        """Both the frame count and the elapsed duration must be satisfied."""
        need_frames = self._per_class(self.min_frames, predicted, 1)
        need_seconds = self._per_class(self.min_duration, predicted, 0.0)

        elapsed = event['last_seen'] - event['start']
        return event['frames'] >= need_frames and elapsed >= need_seconds

    @staticmethod
    def _per_class(setting, predicted, default):
        if isinstance(setting, dict):
            return setting.get(predicted, default)
        return setting

    def _contradicted(self, track_id, predicted, activity_features):
        if self.rule_detector is None or activity_features is None:
            return False
        # The climb veto costs about 1 in 10 real climbs for a 20% cut in false
        # ones — a far weaker trade than the fall veto, so it is separable
        return self.rule_detector.contradicts(
            track_id, predicted, activity_features,
            use_climb_veto=self.use_climb_veto)

    def _in_cooldown(self, track_id, predicted, timestamp=None):
        if self.alert_system is None:
            return False
        return self.alert_system.in_cooldown(track_id, predicted, timestamp)

    def cleanup(self, active_track_ids):
        """Drop state for children who have left, so memory does not grow."""
        active = set(active_track_ids)
        for tid in [t for t in self.events if t not in active]:
            del self.events[tid]
        for tid in [t for t in self._frame_counter if t not in active]:
            del self._frame_counter[tid]

    def config(self):
        """The rules actually in force, for the dashboard to display.

        Read from the live objects rather than restated in the page, so that
        running with --no-climb-veto or --min-duration shows what is really
        happening instead of the defaults.
        """
        cfg = {
            'threshold': {'value': self.threshold, 'on': self.use_threshold},
            'duration': {'frames': self.min_frames, 'seconds': self.min_duration,
                         'gap_tolerance': self.gap_tolerance,
                         'on': self.use_duration},
            'geometry': {'on': self.use_geometry_veto,
                         'climb_veto_on': self.use_climb_veto},
            'cooldown': {'on': self.use_cooldown},
        }
        if self.rule_detector is not None:
            cfg['geometry'].update(self.rule_detector.config())
        if self.alert_system is not None:
            cfg['cooldown'].update({
                'seconds': self.alert_system.cooldown_seconds,
                'spatial_radius': self.alert_system.spatial_radius,
            })
        return cfg

    def stats(self):
        total = self.alerts_passed + sum(self.rejections.values())
        return {
            'candidates': total,
            'alerts': self.alerts_passed,
            'rejected': dict(self.rejections),
            # Split out so the funnel can separate "nothing was happening" from
            # "the buffer had not filled yet" — neither is a rule rejection
            'warmup': self.warmup_frames,
            'normal': self.rejections.get('not_dangerous', 0) - self.warmup_frames,
        }

    def print_stats(self):
        s = self.stats()
        print(f"\n  Verifier — {s['candidates']} predictions, {s['alerts']} alerts")
        for reason, n in sorted(s['rejected'].items(), key=lambda kv: -kv[1]):
            print(f"    rejected by {reason}: {n}")
