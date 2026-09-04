"""Dashboard server. Runs the pipeline in a background thread and serves it.

Usage:
    python src/dashboard/app.py --source "data/test data/ft1.mp4"
    python src/dashboard/app.py --source 0
    python src/dashboard/app.py --source 0 --threshold 0.8 --min-duration 1.0

Test clips live in data/test data/ — ft* fall, ct* climb, nt* normal.
"""

import os
import sys
import time
import argparse
import threading

import cv2
from flask import Flask, Response, jsonify, render_template

sys.path.insert(0, '.')

from src.pipeline.main_pipeline import run as run_pipeline

JPEG_QUALITY = 70

# How many past alerts the feed carries. Generous enough to be the session's
# full history in practice, bounded so a long noisy run cannot bloat the poll.
ALERT_FEED_LIMIT = 300


class SharedState:
    """One frame slot, not a queue — the browser only ever wants the newest."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None
        self._stats = {'people': 0, 'frames': 0, 'fps': 0.0, 'uptime_sec': 0.0,
                       'alerts': [], 'verifier': None, 'rules': None,
                       'tracks': [], 'video_time_sec': None}
        self.running = False
        self.finished = False
        self.error = None

    def update(self, frame, stats):
        ok, buf = cv2.imencode('.jpg', frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
            self._stats = stats

    def jpeg(self):
        with self._lock:
            return self._jpeg

    def stats(self):
        with self._lock:
            s = dict(self._stats)
        s['running'] = self.running
        s['finished'] = self.finished
        s['error'] = self.error
        return s


state = SharedState()
stop_flag = threading.Event()
app = Flask(__name__)


def pipeline_thread(opts):
    state.running = True
    try:
        run_pipeline(
            source=opts.source,
            model_path=opts.model,
            zones_path=opts.zones,
            alert_threshold=opts.threshold,
            use_rules=not opts.no_rules,
            display=False,              # the browser is the display
            save_path=opts.save,
            use_verifier=not opts.no_verifier,
            min_frames=opts.min_frames,
            min_duration=opts.min_duration,
            rules_mode=opts.rules_mode,
            use_climb_veto=not opts.no_climb_veto,
            on_frame=state.update,
            enable_sound=not opts.no_sound,
            stop_flag=stop_flag,
        )
    except Exception as e:
        state.error = f"{type(e).__name__}: {e}"
        print(f"\n[ERROR] Pipeline stopped: {state.error}")
    finally:
        state.running = False
        state.finished = True


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/state')
def api_state():
    s = state.stats()
    alerts = list(reversed(s.get('alerts', [])))
    # The page polls once a second, so an unbounded list would keep growing the
    # payload for the whole session. The true total is sent alongside, so the
    # header count stays honest even when the list itself is trimmed.
    s['alert_total'] = len(alerts)
    s['alerts'] = alerts[:ALERT_FEED_LIMIT]
    return jsonify(s)


@app.route('/api/stop', methods=['POST'])
def api_stop():
    stop_flag.set()
    return jsonify({'stopping': True})


def mjpeg():
    boundary = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    while True:
        buf = state.jpeg()
        if buf is not None:
            yield boundary + buf + b'\r\n'
        # Pace the stream — the pipeline produces far fewer frames than this
        time.sleep(0.03)
        if state.finished and state.jpeg() is None:
            break


@app.route('/video_feed')
def video_feed():
    return Response(mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def main():
    parser = argparse.ArgumentParser(description='Child safety monitor dashboard')
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--model', type=str,
                        default='models/saved/child_cnn_lstm_best.pth')
    parser.add_argument('--zones', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--min-frames', type=int, default=None)
    parser.add_argument('--min-duration', type=float, default=None)
    parser.add_argument('--rules-mode', choices=['veto', 'detector', 'both'],
                        default='veto')
    parser.add_argument('--no-rules', action='store_true')
    parser.add_argument('--no-verifier', action='store_true')
    # Kept in step with main_pipeline.py, so a demo can switch a rule off and
    # the "rules in force" panel reflects it
    parser.add_argument('--no-climb-veto', action='store_true')
    parser.add_argument('--no-sound', action='store_true', default=True,
                        help='Alarm is off by default here; --sound re-enables it')
    parser.add_argument('--sound', dest='no_sound', action='store_false')
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--port', type=int, default=5000)
    opts = parser.parse_args()

    if not os.path.exists(opts.model):
        print(f"[ERROR] Model not found: {opts.model}")
        sys.exit(1)

    threading.Thread(target=pipeline_thread, args=(opts,), daemon=True).start()

    print(f"\n  Dashboard: http://localhost:{opts.port}\n")
    app.run(host='0.0.0.0', port=opts.port, threaded=True,
            debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
