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

# Placeholder until the threshold sweep (A2) runs on the retrained model
DEFAULT_THRESHOLD = 0.70

# Counted in predictions, and the live pipeline predicts once per frame — so
# N is frames here, not stride-8 windows as during extraction
DEFAULT_CONSECUTIVE = 3


class AlertVerifier:

    def __init__(self, threshold=DEFAULT_THRESHOLD,
                 consecutive_required=DEFAULT_CONSECUTIVE,
                 use_threshold=True, use_consecutive=True,
                 use_geometry_veto=True, use_cooldown=True,
                 alert_system=None, rule_detector=None):
        self.threshold = threshold
        self.consecutive_required = max(1, consecutive_required)

        self.use_threshold = use_threshold
        self.use_consecutive = use_consecutive
        self.use_geometry_veto = use_geometry_veto
        self.use_cooldown = use_cooldown

        self.alert_system = alert_system
        self.rule_detector = rule_detector

        self.history = defaultdict(
            lambda: deque(maxlen=self.consecutive_required)
        )

        # Per-rule rejection counts, so one pass over the test set produces the
        # whole ablation table instead of one pass per configuration
        self.rejections = defaultdict(int)
        self.alerts_passed = 0

    def _threshold_for(self, predicted):
        if isinstance(self.threshold, dict):
            return self.threshold.get(predicted, DEFAULT_THRESHOLD)
        return self.threshold

    def should_alert(self, track_id, predicted, confidence, activity_features=None):
        checks = {}

        self.history[track_id].append((predicted, confidence))

        if predicted in (NORMAL, PENDING) or predicted is None:
            return self._reject('not_dangerous', checks)

        thr = self._threshold_for(predicted)

        checks['threshold'] = (not self.use_threshold) or confidence >= thr
        if not checks['threshold']:
            return self._reject('threshold', checks)

        checks['consecutive'] = ((not self.use_consecutive)
                                 or self._consecutive_confirmed(track_id, predicted, thr))
        if not checks['consecutive']:
            return self._reject('consecutive', checks)

        checks['geometry'] = ((not self.use_geometry_veto)
                              or not self._contradicted(track_id, predicted, activity_features))
        if not checks['geometry']:
            return self._reject('geometry', checks)

        checks['cooldown'] = (not self.use_cooldown) or not self._in_cooldown(track_id, predicted)
        if not checks['cooldown']:
            return self._reject('cooldown', checks)

        self.alerts_passed += 1
        return {'alert': True, 'rejected_by': None, 'checks': checks}

    def _reject(self, reason, checks):
        self.rejections[reason] += 1
        return {'alert': False, 'rejected_by': reason, 'checks': checks}

    def _consecutive_confirmed(self, track_id, predicted, threshold):
        hist = self.history[track_id]
        if len(hist) < self.consecutive_required:
            return False
        # Only hold history to the threshold while that rule is active, or
        # disabling it in an ablation would leave it silently enforced here
        if self.use_threshold:
            return all(cls == predicted and conf >= threshold for cls, conf in hist)
        return all(cls == predicted for cls, _ in hist)

    def _contradicted(self, track_id, predicted, activity_features):
        if self.rule_detector is None or activity_features is None:
            return False
        return self.rule_detector.contradicts(track_id, predicted, activity_features)

    def _in_cooldown(self, track_id, predicted):
        if self.alert_system is None:
            return False
        return self.alert_system.in_cooldown(track_id, predicted)

    def cleanup(self, active_track_ids):
        active = set(active_track_ids)
        for tid in [t for t in self.history if t not in active]:
            del self.history[tid]

    def stats(self):
        total = self.alerts_passed + sum(self.rejections.values())
        return {
            'candidates': total,
            'alerts': self.alerts_passed,
            'rejected': dict(self.rejections),
        }

    def print_stats(self):
        s = self.stats()
        print(f"\n  Verifier — {s['candidates']} predictions, {s['alerts']} alerts")
        for reason, n in sorted(s['rejected'].items(), key=lambda kv: -kv[1]):
            print(f"    rejected by {reason}: {n}")
