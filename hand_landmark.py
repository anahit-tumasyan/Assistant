#!/usr/bin/env python3
"""
hand_landmark.py - control the mouse with one hand in front of the webcam.
(MediaPipe 0.10+ Tasks API)

Hold your hand up with the palm towards the camera:

  all five fingers open        move the cursor
  four fingers up, thumb in    freeze the cursor - reposition your hand freely
  fist                         hold the left button and drag; open hand drops
  only the thumb up            right click
  only the index finger up     left click
  index + middle (V sign)      long press - hold the button, move to drag
  index + middle + ring        scroll: move your hand up or down
  thumb + index (L shape)      zoom: spread them to zoom in, pinch to zoom out

The cursor follows the centre of your palm and moves relative to where it
already is (like a trackpad), so curling fingers to make a gesture doesn't
move it. What makes it feel smooth:

  * a "1 Euro" filter on the palm position - it smooths hard while your hand
    is nearly still (no shivering cursor) and hardly at all while it moves
    (no lag), so even very slow, careful movements come through;
  * pointer acceleration that changes gradually with speed: slow = precise,
    fast = across the screen;
  * a background thread that glides the pointer ~100 times a second towards
    where it should be, instead of jumping 30 times a second with the camera;
  * while a new pose is being confirmed (say, you're curling fingers for a
    click) the cursor holds still, so it doesn't slide off your target;
  * short tracking hiccups (motion blur, edge of the frame) are bridged
    instead of resetting everything or dropping what you were dragging.

A pose has to be held for a few frames before it counts, so passing through
other poses on the way from one gesture to the next doesn't fire stray clicks.

Run it on its own with `python hand_landmark.py` (q or Esc quits).
run_all.py uses the same code together with the voice assistant.
"""

import math
import os
import sys
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0
FLIP_FRAME     = True    # mirror the picture, so moving right moves right

# Hand speeds are in camera-frame widths per second.
SENSITIVITY    = 3.0     # screen widths moved per frame width of hand movement
SLOW_SPEED     = 0.24    # below this speed → precision mode (scales down to SLOW_FACTOR)
SLOW_FACTOR    = 0.35    # sensitivity multiplier when moving very slowly
FAST_SPEED     = 0.90    # above this speed → full acceleration
FAST_FACTOR    = 1.25    # sensitivity multiplier for fast movements
STILL_RADIUS   = 0.003   # hand jitter smaller than this (frame widths) never moves the cursor

FILTER_MIN_CUTOFF = 0.8  # smoothing while still: lower = steadier but laggier (Hz)
FILTER_BETA       = 25.0 # how fast smoothing eases off with speed: higher = less lag
FILTER_D_CUTOFF   = 2.0  # smoothing of the speed estimate itself (Hz)

OUTPUT_HZ      = 100     # pointer updates per second (the camera gives ~30)
GLIDE_TIME     = 0.035   # seconds the pointer takes to catch up: lower = snappier, higher = silkier

HAND_LOST_GRACE = 0.25   # seconds the hand may vanish before we let go of it

CLICK_COOLDOWN = 0.30    # min seconds between any click events
STABLE_FRAMES  = 3       # frames a pose must be held before it counts

SCROLL_SENS    = 40      # multiply normalised delta → scroll ticks
SCROLL_DEAD    = 0.003   # min delta to accumulate scroll

ZOOM_THRESH    = 0.025   # normalised dist-change to trigger one zoom step
ZOOM_KEY       = "command" if sys.platform == "darwin" else "ctrl"

THUMB_DIST_THRESH = 0.07  # distance thumb-tip → index-MCP; > = extended

pyautogui.FAILSAFE = False  # the cursor visits the screen corners a lot
pyautogui.PAUSE    = 0      # no 0.1 s sleep after every mouse call

# ── Skeleton overlay ──────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# OpenCV can only draw plain ASCII text - anything else shows up as "???".
GESTURE_INFO = {
    "move":   ("MOVE - tracking cursor",           (100, 220, 255)),
    "freeze": ("FREEZE - reposition hand freely",  (120, 120, 255)),
    "drag":   ("DRAG - holding left button",       (255, 180,   0)),
    "rclick": ("RIGHT CLICK",                      (  0, 100, 255)),
    "lclick": ("LEFT CLICK",                       (  0, 255, 100)),
    "press":  ("LONG PRESS - hold, move to drag",  (  0, 200, 200)),
    "scroll": ("SCROLL - move hand up / down",     (200, 100, 255)),
    "zoom":   ("ZOOM - spread / pinch",            (255, 200,   0)),
    "none":   ("Unrecognised pose",                (100, 100, 100)),
}

GUIDE = [
    ("move",   "ALL FINGERS OPEN",   "Move cursor"),
    ("freeze", "THUMB IN, 4 UP",     "Freeze cursor"),
    ("drag",   "FIST",               "Drag (hold left button)"),
    ("rclick", "ONLY THUMB UP",      "Right click"),
    ("lclick", "ONLY INDEX UP",      "Left click"),
    ("press",  "INDEX + MIDDLE (V)", "Long press / drag"),
    ("scroll", "3 FINGERS UP",       "Scroll up / down"),
    ("zoom",   "THUMB + INDEX",      "Zoom in / out"),
]

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Hand geometry ─────────────────────────────────────────────────────────────

def _thumb_extended(lm):
    """Thumb extended = tip is far from index MCP (works for any direction)."""
    dx = lm[4].x - lm[5].x
    dy = lm[4].y - lm[5].y
    return math.sqrt(dx*dx + dy*dy) > THUMB_DIST_THRESH


def _finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def classify(lm):
    t = _thumb_extended(lm)
    i = _finger_up(lm, 8,  6)
    m = _finger_up(lm, 12, 10)
    r = _finger_up(lm, 16, 14)
    p = _finger_up(lm, 20, 18)

    if   not t and not i and not m and not r and not p: return "drag"    # fist
    elif     t and     i and     m and     r and     p: return "move"    # all open
    elif not t and     i and     m and     r and     p: return "freeze"  # 4 up, no thumb
    elif     t and not i and not m and not r and not p: return "rclick"  # thumb only
    elif not t and     i and not m and not r and not p: return "lclick"  # index only
    elif not t and     i and     m and not r and not p: return "press"   # V-sign
    elif not t and     i and     m and     r and not p: return "scroll"  # 3 fingers
    elif     t and     i and not m and not r and not p: return "zoom"    # thumb+index
    return "none"


def palm_center(lm):
    """Stable anchor: average of wrist + all 4 finger MCPs."""
    xs = [lm[k].x for k in (0, 5, 9, 13, 17)]
    ys = [lm[k].y for k in (0, 5, 9, 13, 17)]
    return sum(xs) / 5, sum(ys) / 5


def tip_dist(lm, a, b):
    dx = lm[a].x - lm[b].x
    dy = lm[a].y - lm[b].y
    return math.sqrt(dx*dx + dy*dy)


def pointer_gain(speed):
    """Screen distance per unit of hand distance at a given hand speed.

    Slow movements are scaled down for precise aiming and fast ones up to
    cross the screen. The curve has no steps, so the cursor never lurches
    when you speed up or slow down.
    """
    if speed <= SLOW_SPEED:
        factor = SLOW_FACTOR + (1 - SLOW_FACTOR) * speed / SLOW_SPEED
    elif speed >= FAST_SPEED:
        factor = FAST_FACTOR
    else:
        factor = 1 + (FAST_FACTOR - 1) * (speed - SLOW_SPEED) / (FAST_SPEED - SLOW_SPEED)
    return SENSITIVITY * factor


class OneEuroFilter:
    """The "1 Euro filter" (Casiez, Roussel & Vogel, CHI 2012) for a 2-D point.

    A low-pass filter whose cut-off frequency rises with speed: it smooths
    hard while the hand is nearly still, so tracking noise doesn't make the
    cursor shiver, and hardly at all while the hand moves, so the cursor
    keeps up without lag.
    """

    def __init__(self, min_cutoff=None, beta=None, d_cutoff=None):
        self.min_cutoff = FILTER_MIN_CUTOFF if min_cutoff is None else min_cutoff
        self.beta = FILTER_BETA if beta is None else beta
        self.d_cutoff = FILTER_D_CUTOFF if d_cutoff is None else d_cutoff
        self.reset()

    def reset(self):
        self.pos = None           # filtered (x, y)
        self.vel = (0.0, 0.0)     # filtered velocity, per second
        self.t = None

    @property
    def speed(self):
        return math.hypot(*self.vel)

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, y, t):
        if self.pos is None:
            self.pos, self.t = (x, y), t
            return self.pos
        dt = min(max(t - self.t, 1e-3), 0.5)
        (px, py), (vx, vy) = self.pos, self.vel
        a_d = self._alpha(self.d_cutoff, dt)
        vx += a_d * ((x - px) / dt - vx)
        vy += a_d * ((y - py) / dt - vy)
        a = self._alpha(self.min_cutoff + self.beta * math.hypot(vx, vy), dt)
        self.pos = (px + a * (x - px), py + a * (y - py))
        self.vel, self.t = (vx, vy), t
        return self.pos

# ── Gestures → mouse ──────────────────────────────────────────────────────────

class Pointer:
    """Moves the real mouse pointer, smoothly.

    The camera only delivers ~30 hand positions a second, and jumping the
    pointer straight to each one looks choppy. With glide=True a background
    thread eases it towards the latest target about OUTPUT_HZ times a second
    (catching up within roughly GLIDE_TIME), so it slides instead. Buttons
    go through here too, so moves become drag events while a button is held
    (macOS ignores plain moves then, and nothing would get dragged).
    """

    def __init__(self, mouse, glide=True):
        self.mouse = mouse
        self.button_held = False
        self._target = None     # where the pointer should end up (float pixels)
        self._pos = None        # where we last put it (float pixels)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        if glide:
            self._thread = threading.Thread(target=self._glide_loop, daemon=True)
            self._thread.start()

    def move_to(self, x, y):
        with self._lock:
            if self._pos is None:
                self._pos = tuple(float(v) for v in self.mouse.position())
            self._target = (x, y)
        if self._thread is None:      # no gliding: go straight there
            self._step(1.0)

    def settle(self):
        """Finish the current glide right now (e.g. just before a click)."""
        self._step(1.0)

    def forget(self):
        """Stop steering; the next move starts from wherever the pointer is then."""
        with self._lock:
            self._target = self._pos = None

    def press(self):
        self.settle()
        with self._lock:
            self.mouse.mouseDown()
            self.button_held = True

    def release(self):
        with self._lock:
            if self.button_held:
                self.mouse.mouseUp()
            self.button_held = False

    def click(self, button="left"):
        self.settle()
        with self._lock:
            self.mouse.click(button=button)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def _glide_loop(self):
        period = 1.0 / OUTPUT_HZ
        last = time.monotonic()
        while not self._stop.wait(max(0.0, last + period - time.monotonic())):
            now = time.monotonic()
            # Ease by elapsed time, so the glide feels the same even when an
            # update runs late (each move takes ~10 ms on macOS).
            self._step(1.0 - math.exp(-(now - last) / GLIDE_TIME))
            last = now

    def _step(self, share):
        with self._lock:
            if self._target is None or self._pos is None:
                return
            (tx, ty), (px, py) = self._target, self._pos
            nx, ny = px + (tx - px) * share, py + (ty - py) * share
            if abs(tx - nx) < 0.5 and abs(ty - ny) < 0.5:
                nx, ny = tx, ty
            self._pos = (nx, ny)
            new = (round(nx), round(ny))
            if new == (round(px), round(py)):
                return
            if self.button_held:
                self.mouse.dragTo(*new, button="left", mouseDownUp=False, _pause=False)
            else:
                self.mouse.moveTo(*new, _pause=False)


MOVING_GESTURES = ("move", "drag", "press")


class GestureController:
    """Turns a stream of hand landmarks into mouse actions."""

    def __init__(self, screen_size=None, mouse=pyautogui, glide=True):
        self.mouse = mouse
        self.screen_w, self.screen_h = screen_size or mouse.size()
        self.pointer = Pointer(mouse, glide=glide)
        self.filter = OneEuroFilter()
        self.cursor_x = self.cursor_y = 0.5
        self.dragging = False    # left button held by the fist
        self.pressing = False    # left button held by the V sign
        self.last_click = 0.0
        self.last_seen = None    # time of the last frame with a hand in it
        self._reset_tracking()

    def _reset_tracking(self):
        self.filter.reset()                    # re-seeds on the next hand frame
        self.prev_x = self.prev_y = 0.0
        self.anchor = None                     # palm position after the still-zone
        self.gesture = "none"                  # the debounced gesture
        self.prev_gesture = "none"
        self._candidate, self._candidate_frames = "none", 0
        self._held_dx = self._held_dy = 0.0    # movement held back while a pose settles
        self.scroll_acc = 0.0
        self.zoom_prev = None
        self.zoom_acc = 0.0

    def _debounce(self, raw):
        if raw == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate, self._candidate_frames = raw, 1
        if self._candidate_frames >= STABLE_FRAMES:
            self.gesture = raw
        return self.gesture

    def _sync_cursor(self):
        """Continue from wherever the mouse is now (you may have used the
        trackpad while your hand was out of view)."""
        try:
            x, y = self.mouse.position()
        except Exception:
            return
        self.cursor_x = min(max(x / self.screen_w, 0.0), 1.0)
        self.cursor_y = min(max(y / self.screen_h, 0.0), 1.0)

    def update(self, lm, now=None):
        """Process one new set of landmarks; returns the active gesture.
        `now` is the frame time in seconds (any steadily increasing clock)."""
        now = time.monotonic() if now is None else now
        self.last_seen = now

        # ── Palm centre → 1 Euro filter → still-zone ──────────────────────
        fx, fy = self._steady_palm(lm, now)

        gesture = self._debounce(classify(lm))
        changed = gesture != self.prev_gesture
        settling = self._candidate != gesture  # another pose is being confirmed

        # ── Held buttons: fist = drag, V = long press ─────────────────────
        if gesture == "drag" and not self.dragging:
            self.pointer.press()
            self.dragging = True
        elif gesture != "drag" and self.dragging:
            self.pointer.release()
            self.dragging = False

        if gesture == "press" and not self.pressing:
            self.pointer.press()
            self.pressing = True
        elif gesture != "press" and self.pressing:
            self.pointer.release()
            self.pressing = False

        # ── Act on the gesture ─────────────────────────────────────────────
        dx, dy = fx - self.prev_x, fy - self.prev_y
        if gesture in MOVING_GESTURES:
            if changed:
                # Just switched pose: the change of hand shape isn't a movement.
                self._held_dx = self._held_dy = 0.0
            elif settling:
                # A new pose is on its way (maybe a click): keep the cursor on
                # target, but remember the movement in case it was a flicker...
                self._held_dx += dx
                self._held_dy += dy
            else:
                # ...and catch up on it if it was.
                self._move_cursor(dx + self._held_dx, dy + self._held_dy)
                self._held_dx = self._held_dy = 0.0
        else:
            self._held_dx = self._held_dy = 0.0
            if gesture in ("lclick", "rclick"):
                # fire once when the pose starts, not every frame it is held
                if changed and (now - self.last_click) > CLICK_COOLDOWN:
                    self.pointer.click("left" if gesture == "lclick" else "right")
                    self.last_click = now
            elif gesture == "scroll":
                self._scroll(dy, changed)
            elif gesture == "zoom":
                self._zoom(lm, changed)
            # "freeze" and "none": the cursor stays where it is

        # ── Always advance prev position (no jump when moving resumes) ─────
        self.prev_x, self.prev_y = fx, fy
        self.prev_gesture = gesture
        return gesture

    def _steady_palm(self, lm, now):
        """Filtered palm position that ignores jitter around a resting point.

        Works like a loose joint: the point only moves once the hand has
        moved more than STILL_RADIUS away from it, and from then on it
        follows continuously. A resting hand never moves the cursor, yet
        slow movements come through smoothly - not in jumps.
        """
        fx, fy = self.filter(*palm_center(lm), now)
        if self.anchor is None:                # hand just appeared: no jump
            self.anchor = (fx, fy)
            self.prev_x, self.prev_y = fx, fy
            self._sync_cursor()
            return self.anchor
        ax, ay = self.anchor
        dist = math.hypot(fx - ax, fy - ay)
        if dist > STILL_RADIUS:
            pull = (dist - STILL_RADIUS) / dist
            self.anchor = (ax + (fx - ax) * pull, ay + (fy - ay) * pull)
        return self.anchor

    def _move_cursor(self, dx, dy):
        gain = pointer_gain(self.filter.speed)
        self.cursor_x = min(max(self.cursor_x + dx * gain, 0.0), 1.0)
        self.cursor_y = min(max(self.cursor_y + dy * gain, 0.0), 1.0)
        self.pointer.move_to(self.cursor_x * (self.screen_w - 1),
                             self.cursor_y * (self.screen_h - 1))

    def _scroll(self, dy, changed):
        if changed:
            self.scroll_acc = 0.0
        elif abs(dy) > SCROLL_DEAD:
            self.scroll_acc += dy
        ticks = int(self.scroll_acc * SCROLL_SENS)
        if ticks:
            self.mouse.scroll(-ticks)     # hand up → page scrolls up (positive)
            self.scroll_acc -= ticks / SCROLL_SENS

    def _zoom(self, lm, changed):
        d = tip_dist(lm, 4, 8)
        if changed or self.zoom_prev is None:
            self.zoom_acc = 0.0
        else:
            self.zoom_acc += d - self.zoom_prev
        self.zoom_prev = d
        if abs(self.zoom_acc) >= ZOOM_THRESH:
            # spread → zoom in, pinch → zoom out
            self.mouse.hotkey(ZOOM_KEY, "=" if self.zoom_acc > 0 else "-")
            self.zoom_acc = 0.0

    def hand_missing(self, now=None):
        """No hand in this frame. Short gaps (motion blur, the edge of the
        frame) are bridged - nothing moves and a drag stays held. Only after
        HAND_LOST_GRACE seconds without a hand do we let go."""
        if self.last_seen is None:
            return                              # already let go
        now = time.monotonic() if now is None else now
        if now - self.last_seen > HAND_LOST_GRACE:
            self.hand_lost()

    def hand_lost(self):
        """Hand gone: let go of any held button and start fresh."""
        self.release()
        self.pointer.settle()
        self.pointer.forget()
        self.last_seen = None
        self._reset_tracking()

    def release(self):
        if self.dragging or self.pressing:
            self.pointer.release()
        self.dragging = self.pressing = False

    def close(self):
        """Let go of the buttons and stop the glide thread."""
        self.release()
        self.pointer.close()

# ── MediaPipe ─────────────────────────────────────────────────────────────────

def ensure_model(path=MODEL_PATH):
    """Download the hand landmark model the first time."""
    if not os.path.exists(path):
        print("Downloading the hand landmark model (~8 MB)...")
        partial = path + ".part"
        urllib.request.urlretrieve(MODEL_URL, partial)
        os.replace(partial, path)   # never leave a half-downloaded model behind
        print("Download complete.")
    return path


class HandTracker:
    """MediaPipe's hand landmarker in live-stream mode (results arrive on
    MediaPipe's own thread; latest() hands back the newest one)."""

    def __init__(self, model_path=MODEL_PATH):
        self._lock = threading.Lock()
        self._landmarks = None
        self._result_ts = -1
        self._last_ts = 0
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model(model_path)),
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.50,
            result_callback=self._on_result,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def _on_result(self, result, _image, timestamp_ms):
        with self._lock:
            self._landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
            self._result_ts = timestamp_ms

    def submit(self, frame_bgr):
        # Real time stamps (MediaPipe needs them strictly increasing).
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._landmarker.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

    def latest(self):
        """(landmarks or None, id of the result) - the id changes with every new result."""
        with self._lock:
            return self._landmarks, self._result_ts

    def close(self):
        self._landmarker.close()

# ── Drawing ───────────────────────────────────────────────────────────────────

_ASCII = {"—": "-", "–": "-", "…": "...", "→": "->", "←": "<-", "✓": "OK",
          "“": '"', "”": '"', "‘": "'", "’": "'"}


def fit_text(text, max_width, scale, thickness=1):
    """Plain-ASCII version of text, cut with '...' to fit max_width pixels."""
    for fancy, plain in _ASCII.items():
        text = text.replace(fancy, plain)
    text = text.encode("ascii", "ignore").decode()
    if cv2.getTextSize(text, FONT, scale, thickness)[0][0] <= max_width:
        return text
    while text and cv2.getTextSize(text + "...", FONT, scale, thickness)[0][0] > max_width:
        text = text[:-1]
    return text + "..."


def _status_color(status):
    s = status.lower()
    if "listening" in s:
        return (100, 220, 255)
    if "yes or no" in s:
        return (0, 200, 255)
    if any(word in s for word in ("error", "blocked", "off", "failed", "cancel")):
        return (90, 90, 255)
    return (0, 255, 100)


def draw_ui(frame, gesture, lm, status=None):
    h, w = frame.shape[:2]

    if lm:
        for a, b in HAND_CONNECTIONS:
            p1 = (int(lm[a].x * w), int(lm[a].y * h))
            p2 = (int(lm[b].x * w), int(lm[b].y * h))
            cv2.line(frame, p1, p2, (0, 200, 200), 2)
        for pt in lm:
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 4, (200, 200, 200), -1)

    # Top bar: what the hand is doing
    cv2.rectangle(frame, (0, 0), (w, 46), (20, 20, 20), -1)
    if lm is None:
        label, color = "Show your hand to the camera", (0, 100, 255)
    else:
        label, color = GESTURE_INFO.get(gesture, GESTURE_INFO["none"])
    cv2.putText(frame, fit_text(label, w - 16, 0.75, 2), (8, 31), FONT, 0.75, color, 2)

    # Second bar: voice assistant status (run_all.py)
    if status:
        cv2.rectangle(frame, (0, 46), (w, 74), (40, 40, 40), -1)
        cv2.putText(frame, fit_text(status, w - 16, 0.5), (8, 66), FONT, 0.5,
                    _status_color(status), 1)

    # Gesture guide, with the active gesture highlighted
    gx = 10
    gy = h - len(GUIDE) * 24 - 18
    cv2.rectangle(frame, (gx-4, gy-8), (gx+370, gy + len(GUIDE)*24 + 4), (20,20,20), -1)
    cv2.rectangle(frame, (gx-4, gy-8), (gx+370, gy + len(GUIDE)*24 + 4), (70,70,70),  1)
    for i, (key, pose, action) in enumerate(GUIDE):
        active = lm is not None and key == gesture
        row_color = (0, 255, 100) if active else (155, 155, 155)
        y = gy + i*24 + 18
        cv2.putText(frame, pose, (gx, y), FONT, 0.42, row_color, 1)
        cv2.putText(frame, action, (gx + 190, y), FONT, 0.42, row_color, 1)

# ── Main loop ─────────────────────────────────────────────────────────────────

def run(status_text=None, window_title="Hand Gesture Control  [q to quit]"):
    """Open the camera and control the mouse until q / Esc is pressed.

    status_text: optional function returning a line to show under the
    gesture bar (run_all.py uses it for the voice assistant's status).
    Must be called from the main thread (OpenCV windows require it on macOS).
    """
    tracker = HandTracker()
    controller = GestureController()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        tracker.close()
        sys.exit("Cannot open the camera. Allow camera access for your terminal app in "
                 "System Settings > Privacy & Security > Camera, then try again.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Hand gesture control started. Press q (or Esc) in the camera window to quit.")
    gesture = "none"
    last_result = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("The camera stopped sending pictures.")
                break
            if FLIP_FRAME:
                frame = cv2.flip(frame, 1)

            tracker.submit(frame)
            lm, result_id = tracker.latest()
            if result_id != last_result:          # act on each new detection once
                last_result = result_id
                now = result_id / 1000.0          # MediaPipe time stamp, in seconds
                if lm is None:
                    controller.hand_missing(now)
                else:
                    gesture = controller.update(lm, now)

            draw_ui(frame, gesture, lm, status_text() if status_text else None)
            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        # Never leave the mouse button stuck down.
        controller.close()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Hand gesture control stopped.")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
