"""Tests for the Phase A alert rules. Run: python src/verification/test_alert_verifier.py

A wrongly-rejecting rule produces no alerts and no error, so these check the
reject path as carefully as the accept path.
"""

import sys

sys.path.insert(0, '.')

from src.verification.alert_verifier import AlertVerifier
from src.classification.rule_based_detector import RuleBasedDetector
from src.alerts.alert_system import AlertSystem

FALLEN = {'aspect_ratio': 0.5, 'body_angle': 75.0, 'min_ankle_y': 400.0,
          'mid_hip_y': 380.0, 'mid_shoulder_y': 300.0, 'valid_keypoints': 12}

UPRIGHT = {'aspect_ratio': 2.2, 'body_angle': 10.0, 'min_ankle_y': 400.0,
           'mid_hip_y': 300.0, 'mid_shoulder_y': 220.0, 'valid_keypoints': 12}

FEW_KEYPOINTS = dict(FALLEN, valid_keypoints=2)

passed = failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


def fresh(**kwargs):
    opts = dict(threshold=0.70, consecutive_required=3,
                alert_system=AlertSystem(enable_sound=False),
                rule_detector=RuleBasedDetector())
    opts.update(kwargs)
    return AlertVerifier(**opts)


def feed(verifier, track, label, conf, features, times):
    """Send the same prediction repeatedly, return the last decision."""
    d = None
    for _ in range(times):
        d = verifier.should_alert(track, label, conf, features)
    return d


# --- basic gating -----------------------------------------------------------

def test_normal_never_alerts():
    v = fresh()
    d = feed(v, 1, 'normal', 0.99, FALLEN, 5)
    check("normal never alerts", d['alert'] is False and d['rejected_by'] == 'not_dangerous')


def test_none_label_rejected():
    v = fresh()
    d = v.should_alert(1, None, 0.99, FALLEN)
    check("None label rejected", d['alert'] is False)


def test_pending_not_counted_as_threshold_rejection():
    """Warm-up frames must not inflate the threshold rule's ablation share."""
    v = fresh()
    for _ in range(15):
        v.should_alert(1, 'pending', 0.0, FALLEN)
    s = v.stats()
    check("pending is not a threshold rejection",
          s['rejected'].get('threshold', 0) == 0
          and s['rejected'].get('not_dangerous') == 15)


def test_pending_breaks_a_consecutive_run():
    v = fresh(consecutive_required=3)
    v.should_alert(1, 'fall', 0.85, FALLEN)
    v.should_alert(1, 'pending', 0.0, FALLEN)
    d = v.should_alert(1, 'fall', 0.85, FALLEN)
    check("pending breaks a consecutive run", d['alert'] is False)


def test_below_threshold_rejected():
    v = fresh()
    d = feed(v, 1, 'fall', 0.40, FALLEN, 5)
    check("below threshold rejected", d['rejected_by'] == 'threshold')


def test_at_threshold_accepted():
    v = fresh()
    d = feed(v, 1, 'fall', 0.70, FALLEN, 3)
    check("exactly at threshold accepted", d['alert'] is True)


# --- consecutive confirmation ----------------------------------------------

def test_consecutive_requires_n():
    v = fresh(consecutive_required=3)
    first = v.should_alert(1, 'fall', 0.85, FALLEN)
    second = v.should_alert(1, 'fall', 0.85, FALLEN)
    third = v.should_alert(1, 'fall', 0.85, FALLEN)
    check("consecutive: 1st rejected", first['rejected_by'] == 'consecutive')
    check("consecutive: 2nd rejected", second['rejected_by'] == 'consecutive')
    check("consecutive: 3rd alerts", third['alert'] is True)


def test_consecutive_broken_by_class_change():
    v = fresh(consecutive_required=3)
    v.should_alert(1, 'fall', 0.85, FALLEN)
    v.should_alert(1, 'climb', 0.85, FALLEN)
    d = v.should_alert(1, 'fall', 0.85, FALLEN)
    check("consecutive broken by a different class", d['alert'] is False)


def test_consecutive_broken_by_low_confidence():
    v = fresh(consecutive_required=3)
    v.should_alert(1, 'fall', 0.85, FALLEN)
    v.should_alert(1, 'fall', 0.50, FALLEN)   # dips below threshold
    d = v.should_alert(1, 'fall', 0.85, FALLEN)
    check("consecutive broken by a confidence dip", d['alert'] is False)


def test_consecutive_is_per_track():
    v = fresh(consecutive_required=3)
    for tid in (1, 2, 3):
        v.should_alert(tid, 'fall', 0.85, FALLEN)
    d = v.should_alert(1, 'fall', 0.85, FALLEN)
    check("consecutive counted per track, not globally", d['alert'] is False)


def test_consecutive_one_alerts_immediately():
    v = fresh(consecutive_required=1)
    d = v.should_alert(1, 'fall', 0.85, FALLEN)
    check("N=1 alerts on the first prediction", d['alert'] is True)


# --- geometry veto ----------------------------------------------------------

def test_geometry_vetoes_upright_fall():
    v = fresh()
    d = feed(v, 1, 'fall', 0.90, UPRIGHT, 3)
    check("upright body vetoes a predicted fall", d['rejected_by'] == 'geometry')


def test_geometry_allows_fallen_fall():
    v = fresh()
    d = feed(v, 1, 'fall', 0.90, FALLEN, 3)
    check("fallen pose does not veto a predicted fall", d['alert'] is True)


def test_geometry_cannot_judge_few_keypoints():
    v = fresh()
    d = feed(v, 1, 'fall', 0.90, FEW_KEYPOINTS, 3)
    check("too few keypoints does not veto", d['alert'] is True)


def test_climb_veto_passes_during_warmup():
    """The critical case — an incomplete baseline must never suppress alerts."""
    rules = RuleBasedDetector()
    v = fresh(rule_detector=rules)
    d = feed(v, 1, 'climb', 0.90, UPRIGHT, 3)
    check("climb passes while the baseline is still filling", d['alert'] is True)


def test_climb_veto_active_once_baseline_ready():
    rules = RuleBasedDetector()
    for _ in range(35):
        rules.update(9, UPRIGHT)          # feet consistently on the ground
    v = fresh(rule_detector=rules)
    d = feed(v, 9, 'climb', 0.90, UPRIGHT, 3)
    check("climb vetoed when feet are at ground level", d['rejected_by'] == 'geometry')


def test_climb_allowed_when_feet_raised():
    rules = RuleBasedDetector()
    for _ in range(35):
        rules.update(9, UPRIGHT)
    raised = dict(UPRIGHT, min_ankle_y=330.0)   # smaller y = higher in frame
    v = fresh(rule_detector=rules)
    d = feed(v, 9, 'climb', 0.90, raised, 3)
    check("climb allowed when feet are raised", d['alert'] is True)


# --- cooldown ---------------------------------------------------------------

def test_cooldown_blocks_repeat():
    alerts = AlertSystem(enable_sound=False)
    v = fresh(alert_system=alerts)
    feed(v, 1, 'fall', 0.90, FALLEN, 3)
    alerts.trigger_alert('fall', 0.9, 1)          # the pipeline would do this
    d = v.should_alert(1, 'fall', 0.90, FALLEN)
    check("cooldown blocks a repeat alert", d['rejected_by'] == 'cooldown')


def test_cooldown_is_per_activity():
    alerts = AlertSystem(enable_sound=False)
    alerts.trigger_alert('fall', 0.9, 1)
    check("fall cooldown does not block climb", alerts.in_cooldown(1, 'climb') is False)


def test_cooldown_is_per_track():
    alerts = AlertSystem(enable_sound=False)
    alerts.trigger_alert('fall', 0.9, 1)
    check("cooldown for one child does not block another",
          alerts.in_cooldown(2, 'fall') is False)


# --- rule toggles, for the ablation table -----------------------------------

def test_disable_consecutive():
    v = fresh(use_consecutive=False)
    d = v.should_alert(1, 'fall', 0.85, FALLEN)
    check("consecutive disabled -> alerts immediately", d['alert'] is True)


def test_disable_threshold():
    v = fresh(use_threshold=False, consecutive_required=1)
    d = v.should_alert(1, 'fall', 0.01, FALLEN)
    check("threshold disabled -> low confidence alerts", d['alert'] is True)


def test_disable_geometry():
    v = fresh(use_geometry_veto=False, consecutive_required=1)
    d = v.should_alert(1, 'fall', 0.90, UPRIGHT)
    check("geometry disabled -> upright fall alerts", d['alert'] is True)


def test_disable_cooldown():
    alerts = AlertSystem(enable_sound=False)
    v = fresh(alert_system=alerts, use_cooldown=False, consecutive_required=1)
    v.should_alert(1, 'fall', 0.90, FALLEN)
    alerts.trigger_alert('fall', 0.9, 1)
    d = v.should_alert(1, 'fall', 0.90, FALLEN)
    check("cooldown disabled -> repeat alerts allowed", d['alert'] is True)


def test_all_rules_disabled():
    v = fresh(use_threshold=False, use_consecutive=False,
              use_geometry_veto=False, use_cooldown=False)
    d = v.should_alert(1, 'fall', 0.01, UPRIGHT)
    check("all rules off -> any non-normal alerts", d['alert'] is True)


# --- per-class thresholds ---------------------------------------------------

def test_per_class_threshold():
    v = fresh(threshold={'fall': 0.60, 'climb': 0.90}, consecutive_required=1)
    fall = v.should_alert(1, 'fall', 0.65, FALLEN)
    climb = v.should_alert(2, 'climb', 0.65, UPRIGHT)
    check("per-class threshold: fall passes at 0.65", fall['alert'] is True)
    check("per-class threshold: climb rejected at 0.65", climb['rejected_by'] == 'threshold')


# --- bookkeeping ------------------------------------------------------------

def test_cleanup_drops_history():
    v = fresh()
    v.should_alert(1, 'fall', 0.85, FALLEN)
    v.should_alert(2, 'fall', 0.85, FALLEN)
    v.cleanup([2])
    check("cleanup removes departed tracks", 1 not in v.history and 2 in v.history)


def test_stats_account_for_every_call():
    v = fresh()
    for _ in range(5):
        v.should_alert(1, 'fall', 0.85, FALLEN)
    s = v.stats()
    total = s['alerts'] + sum(s['rejected'].values())
    check("stats account for every prediction", total == 5 and s['candidates'] == 5)


def test_rejection_reasons_recorded():
    v = fresh()
    v.should_alert(1, 'fall', 0.10, FALLEN)      # threshold
    v.should_alert(2, 'normal', 0.99, FALLEN)    # not_dangerous
    s = v.stats()
    check("rejection reasons recorded separately",
          s['rejected'].get('threshold') == 1 and s['rejected'].get('not_dangerous') == 1)


if __name__ == '__main__':
    print("\nPhase A alert rules\n")
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print(f"\n  {passed} passed, {failed} failed\n")
    sys.exit(1 if failed else 0)
