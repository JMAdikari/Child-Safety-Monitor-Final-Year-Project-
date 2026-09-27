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

# Crouching: hip close to the feet, so hip_above_feet = (400-380)/80 = 0.25
CROUCHED = {'aspect_ratio': 1.2, 'body_angle': 10.0, 'min_ankle_y': 400.0,
            'mid_hip_y': 380.0, 'mid_shoulder_y': 300.0, 'valid_keypoints': 12}

# Bolt upright at 3 deg, below the 8 deg fall veto threshold. UPRIGHT sits at
# 10 deg and is deliberately left alone by the veto.
VERTICAL = dict(UPRIGHT, body_angle=3.0)

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
    # 3 frames / 0.0s keeps most tests focused on one rule at a time; the
    # duration tests below override these
    opts = dict(threshold=0.70, min_frames=3, min_duration=0.0,
                alert_system=AlertSystem(enable_sound=False),
                rule_detector=RuleBasedDetector())
    opts.update(kwargs)
    return AlertVerifier(**opts)


def feed(verifier, track, label, conf, features, times, fps=30.0, t0=0.0):
    """Send the same prediction repeatedly at a steady frame rate."""
    d = None
    for i in range(times):
        d = verifier.should_alert(track, label, conf, features,
                                  timestamp=t0 + i / fps)
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


def test_pending_ends_the_event():
    v = fresh(min_frames=3)
    v.should_alert(1, 'fall', 0.85, FALLEN)
    v.should_alert(1, 'pending', 0.0, FALLEN)
    d = v.should_alert(1, 'fall', 0.85, FALLEN)
    check("pending ends the open event", d['alert'] is False)


def test_below_threshold_rejected():
    v = fresh()
    d = feed(v, 1, 'fall', 0.40, FALLEN, 5)
    check("below threshold rejected", d['rejected_by'] == 'threshold')


def test_at_threshold_accepted():
    v = fresh()
    d = feed(v, 1, 'fall', 0.70, FALLEN, 3)
    check("exactly at threshold accepted", d['alert'] is True)


# --- event duration: frame count -------------------------------------------

def test_frames_requires_n():
    v = fresh(min_frames=3)
    a = v.should_alert(1, 'fall', 0.85, FALLEN, timestamp=0.00)
    b = v.should_alert(1, 'fall', 0.85, FALLEN, timestamp=0.03)
    c = v.should_alert(1, 'fall', 0.85, FALLEN, timestamp=0.07)
    check("frames: 1st rejected", a['rejected_by'] == 'duration')
    check("frames: 2nd rejected", b['rejected_by'] == 'duration')
    check("frames: 3rd alerts", c['alert'] is True)


def test_event_restarts_on_class_change():
    v = fresh(min_frames=3)
    v.should_alert(1, 'fall', 0.85, FALLEN, timestamp=0.00)
    v.should_alert(1, 'climb', 0.85, FALLEN, timestamp=0.03)
    d = v.should_alert(1, 'fall', 0.85, FALLEN, timestamp=0.07)
    check("a different activity restarts the event", d['alert'] is False)


def test_normal_frame_ends_the_event():
    v = fresh(min_frames=3)
    v.should_alert(1, 'fall', 0.85, FALLEN, timestamp=0.00)
    v.should_alert(1, 'fall', 0.85, FALLEN, timestamp=0.03)
    v.should_alert(1, 'normal', 0.99, FALLEN, timestamp=0.07)
    d = v.should_alert(1, 'fall', 0.85, FALLEN, timestamp=0.10)
    check("a normal frame ends the event", d['alert'] is False)


def test_events_are_per_track():
    v = fresh(min_frames=3)
    for tid in (1, 2, 3):
        v.should_alert(tid, 'fall', 0.85, FALLEN, timestamp=0.00)
    d = v.should_alert(1, 'fall', 0.85, FALLEN, timestamp=0.03)
    check("events counted per track, not globally", d['alert'] is False)


# --- event duration: elapsed time ------------------------------------------

def test_duration_requires_elapsed_time():
    """30 frames at 30fps spans 0.97s, so a 2.0s rule must still reject."""
    v = fresh(min_frames=1, min_duration=2.0)
    d = feed(v, 1, 'fall', 0.85, FALLEN, 30)
    check("duration rejects when too little time has passed",
          d['rejected_by'] == 'duration')


def test_duration_alerts_once_satisfied():
    v = fresh(min_frames=1, min_duration=0.4)
    d = feed(v, 1, 'fall', 0.85, FALLEN, 20)     # 0.63s
    check("duration alerts once the time is reached", d['alert'] is True)


def test_frame_rate_independence():
    """The same real duration should behave the same at 30 and 60 fps."""
    a = feed(fresh(min_frames=1, min_duration=0.5), 1, 'fall', 0.85, FALLEN, 20, fps=30)
    b = feed(fresh(min_frames=1, min_duration=0.5), 1, 'fall', 0.85, FALLEN, 40, fps=60)
    check("0.5s rule behaves the same at 30 and 60 fps",
          a['alert'] is True and b['alert'] is True)


def test_frames_and_duration_both_required():
    """Plenty of elapsed time, but too few frames — the stricter rule governs."""
    v = fresh(min_frames=20, min_duration=0.1)
    d = feed(v, 1, 'fall', 0.85, FALLEN, 5, fps=5)   # 0.8s, only 5 frames
    check("frame count still applies when the duration is met",
          d['rejected_by'] == 'duration')


def test_gap_within_tolerance_continues_event():
    v = fresh(min_frames=3, min_duration=0.0, gap_tolerance=0.4)
    v.should_alert(1, 'fall', 0.85, FALLEN, timestamp=0.0)
    v.should_alert(1, 'fall', 0.85, FALLEN, timestamp=0.2)   # 0.2s gap, tolerated
    d = v.should_alert(1, 'fall', 0.85, FALLEN, timestamp=0.4)
    check("a short detection gap does not restart the event", d['alert'] is True)


def test_gap_beyond_tolerance_restarts_event():
    v = fresh(min_frames=3, min_duration=0.0, gap_tolerance=0.4)
    v.should_alert(1, 'fall', 0.85, FALLEN, timestamp=0.0)
    v.should_alert(1, 'fall', 0.85, FALLEN, timestamp=0.1)
    d = v.should_alert(1, 'fall', 0.85, FALLEN, timestamp=2.0)   # 1.9s gap
    check("a long detection gap restarts the event", d['alert'] is False)


# --- per-class settings -----------------------------------------------------

def test_per_class_duration():
    """Falls are brief and safety-critical; climbs are sustained and noisier."""
    v = fresh(min_frames=1, min_duration={'fall': 0.4, 'climb': 1.0})
    fall = feed(v, 1, 'fall', 0.85, FALLEN, 20)        # 0.63s
    climb = feed(v, 2, 'climb', 0.85, UPRIGHT, 20)     # 0.63s
    check("per-class: fall passes at 0.63s", fall['alert'] is True)
    check("per-class: climb rejected at 0.63s", climb['rejected_by'] == 'duration')


def test_per_class_frames():
    v = fresh(min_frames={'fall': 3, 'climb': 30}, min_duration=0.0)
    fall = feed(v, 1, 'fall', 0.85, FALLEN, 5)
    climb = feed(v, 2, 'climb', 0.85, UPRIGHT, 5)
    check("per-class frames: fall passes at 5", fall['alert'] is True)
    check("per-class frames: climb rejected at 5", climb['rejected_by'] == 'duration')


# --- geometry veto ----------------------------------------------------------

def test_geometry_vetoes_never_tilted_fall():
    """The veto reads the peak angle over recent frames, so history must be fed."""
    rules = RuleBasedDetector()
    for _ in range(10):
        rules.update(1, VERTICAL)         # 3 deg throughout, never tilts
    v = fresh(rule_detector=rules)
    d = feed(v, 1, 'fall', 0.90, VERTICAL, 3)
    check("a body that never tilted vetoes a predicted fall",
          d['rejected_by'] == 'geometry')


def test_geometry_allows_a_mildly_tilted_fall():
    """10 deg is above the 8 deg bar — the veto only catches truly vertical bodies."""
    rules = RuleBasedDetector()
    for _ in range(10):
        rules.update(1, UPRIGHT)
    v = fresh(rule_detector=rules)
    d = feed(v, 1, 'fall', 0.90, UPRIGHT, 3)
    check("a mildly tilted body is not vetoed", d['alert'] is True)


def test_geometry_allows_fall_that_tilted_briefly():
    """One steep frame in the window is enough — that is the whole point."""
    rules = RuleBasedDetector()
    for _ in range(9):
        rules.update(1, VERTICAL)         # mostly bolt upright
    rules.update(1, FALLEN)               # one frame at 75 deg
    v = fresh(rule_detector=rules)
    d = feed(v, 1, 'fall', 0.90, VERTICAL, 3)
    check("a brief tilt in the window prevents the veto", d['alert'] is True)


def test_fall_veto_passes_with_too_little_history():
    rules = RuleBasedDetector()
    rules.update(1, VERTICAL)             # only 1 frame, below the minimum
    v = fresh(rule_detector=rules)
    d = feed(v, 1, 'fall', 0.90, VERTICAL, 3)
    check("fall veto stays quiet until it has enough history", d['alert'] is True)


def test_geometry_allows_fallen_fall():
    v = fresh()
    d = feed(v, 1, 'fall', 0.90, FALLEN, 3)
    check("fallen pose does not veto a predicted fall", d['alert'] is True)


def test_geometry_cannot_judge_few_keypoints():
    v = fresh()
    d = feed(v, 1, 'fall', 0.90, FEW_KEYPOINTS, 3)
    check("too few keypoints does not veto", d['alert'] is True)


def test_climb_vetoed_when_posture_is_compact():
    """hip_above_feet = (400-380)/80 = 0.25, below the 0.30 threshold."""
    v = fresh()
    d = feed(v, 9, 'climb', 0.95, CROUCHED, 3)
    check("a compact posture vetoes a predicted climb",
          d['rejected_by'] == 'geometry')


def test_climb_allowed_when_posture_is_extended():
    """hip_above_feet = (400-300)/80 = 1.25, well clear of the threshold."""
    v = fresh()
    d = feed(v, 9, 'climb', 0.95, UPRIGHT, 3)
    check("an extended posture allows a predicted climb", d['alert'] is True)


def test_climb_veto_can_be_disabled_separately():
    """It costs ~1 in 10 real climbs, so it is switchable without the fall veto."""
    v = fresh(use_climb_veto=False)
    d = feed(v, 9, 'climb', 0.95, CROUCHED, 3)
    check("use_climb_veto=False leaves the fall veto intact", d['alert'] is True)


def test_climb_veto_passes_when_torso_unmeasurable():
    flat = dict(CROUCHED, mid_shoulder_y=300.0, mid_hip_y=300.0)   # torso = 0
    v = fresh()
    d = feed(v, 9, 'climb', 0.95, flat, 3)
    check("unmeasurable torso does not veto", d['alert'] is True)


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
    alerts = AlertSystem(enable_sound=False, spatial_radius=0)
    alerts.trigger_alert('fall', 0.9, 1, location=(100, 100))
    check("cooldown for one child does not block another",
          alerts.in_cooldown(2, 'fall') is False)


def test_cooldown_uses_video_time_when_given():
    """Wall clock would barely suppress anything on video processed at ~1 fps."""
    alerts = AlertSystem(enable_sound=False, cooldown_seconds=10, spatial_radius=0)
    alerts.trigger_alert('fall', 0.9, 1, location=(100, 100), timestamp=0.0)
    check("still in cooldown 5s of video later",
          alerts.in_cooldown(1, 'fall', timestamp=5.0) is True)
    check("out of cooldown 11s of video later",
          alerts.in_cooldown(1, 'fall', timestamp=11.0) is False)


def test_spatial_cooldown_survives_id_change():
    """ByteTrack renumbers a child mid-fall; the location does not move."""
    alerts = AlertSystem(enable_sound=False, spatial_radius=150)
    first = alerts.trigger_alert('fall', 0.9, 7, location=(400, 300), timestamp=0.0)
    # same place, same activity, but the track was renumbered
    second = alerts.trigger_alert('fall', 0.9, 12, location=(410, 305), timestamp=1.0)
    check("first alert fires", first is True)
    check("renumbered track nearby is suppressed", second is False)


def test_spatial_cooldown_allows_a_different_place():
    alerts = AlertSystem(enable_sound=False, spatial_radius=150)
    alerts.trigger_alert('fall', 0.9, 7, location=(400, 300), timestamp=0.0)
    far = alerts.trigger_alert('fall', 0.9, 12, location=(900, 300), timestamp=1.0)
    check("a fall across the room still alerts", far is True)


def test_spatial_cooldown_is_per_activity():
    alerts = AlertSystem(enable_sound=False, spatial_radius=150)
    alerts.trigger_alert('fall', 0.9, 7, location=(400, 300), timestamp=0.0)
    climb = alerts.trigger_alert('climb', 0.9, 7, location=(400, 300), timestamp=1.0)
    check("a climb in the same place still alerts", climb is True)


def test_spatial_cooldown_expires():
    alerts = AlertSystem(enable_sound=False, cooldown_seconds=10, spatial_radius=150)
    alerts.trigger_alert('fall', 0.9, 7, location=(400, 300), timestamp=0.0)
    later = alerts.trigger_alert('fall', 0.9, 12, location=(400, 300), timestamp=15.0)
    check("same place alerts again once the cooldown expires", later is True)


def test_spatial_cooldown_can_be_disabled():
    alerts = AlertSystem(enable_sound=False, spatial_radius=0)
    alerts.trigger_alert('fall', 0.9, 7, location=(400, 300), timestamp=0.0)
    second = alerts.trigger_alert('fall', 0.9, 12, location=(400, 300), timestamp=1.0)
    check("radius 0 disables the spatial cooldown", second is True)


# --- rule toggles, for the ablation table -----------------------------------

def test_disable_duration():
    v = fresh(use_duration=False)
    d = v.should_alert(1, 'fall', 0.85, FALLEN)
    check("duration disabled -> alerts immediately", d['alert'] is True)


def test_disable_threshold():
    v = fresh(use_threshold=False, min_frames=1)
    d = v.should_alert(1, 'fall', 0.01, FALLEN)
    check("threshold disabled -> low confidence alerts", d['alert'] is True)


def test_disable_geometry():
    v = fresh(use_geometry_veto=False, min_frames=1)
    d = v.should_alert(1, 'fall', 0.90, UPRIGHT)
    check("geometry disabled -> upright fall alerts", d['alert'] is True)


def test_disable_cooldown():
    alerts = AlertSystem(enable_sound=False)
    v = fresh(alert_system=alerts, use_cooldown=False, min_frames=1)
    v.should_alert(1, 'fall', 0.90, FALLEN)
    alerts.trigger_alert('fall', 0.9, 1)
    d = v.should_alert(1, 'fall', 0.90, FALLEN)
    check("cooldown disabled -> repeat alerts allowed", d['alert'] is True)


def test_all_rules_disabled():
    v = fresh(use_threshold=False, use_duration=False,
              use_geometry_veto=False, use_cooldown=False)
    d = v.should_alert(1, 'fall', 0.01, UPRIGHT)
    check("all rules off -> any non-normal alerts", d['alert'] is True)


# --- per-class thresholds ---------------------------------------------------

def test_per_class_threshold():
    v = fresh(threshold={'fall': 0.60, 'climb': 0.90}, min_frames=1)
    fall = v.should_alert(1, 'fall', 0.65, FALLEN)
    climb = v.should_alert(2, 'climb', 0.65, UPRIGHT)
    check("per-class threshold: fall passes at 0.65", fall['alert'] is True)
    check("per-class threshold: climb rejected at 0.65", climb['rejected_by'] == 'threshold')


# --- bookkeeping ------------------------------------------------------------

def test_cleanup_drops_departed_tracks():
    v = fresh()
    v.should_alert(1, 'fall', 0.85, FALLEN)
    v.should_alert(2, 'fall', 0.85, FALLEN)
    v.cleanup([2])
    check("cleanup removes departed tracks", 1 not in v.events and 2 in v.events)


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
