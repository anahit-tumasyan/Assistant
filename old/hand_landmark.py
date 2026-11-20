#!/usr/bin/env python3
"""
Hand Gesture Control — mediapipe 0.10+ (Tasks API)

Gesture map:
  ALL 5 FINGERS OPEN          → Move cursor  (relative + adaptive sensitivity)
  4 FINGERS UP, THUMB DOWN    → Freeze cursor  (reposition hand freely, no jump)
  FIST  (all closed)          → Drag  — left mouse held; open hand = drop
  ONLY THUMB UP               → Right click
  ONLY INDEX UP               → Left click
  INDEX + MIDDLE  (V-sign)    → Double click
  INDEX + MIDDLE + RING       → Scroll up / down  (move hand up or down)
  THUMB + INDEX               → Zoom in / out  (spread = zoom in, pinch = zoom out)

Cursor tracks the palm centre, so changing which fingers are up/down
does NOT move the cursor — no jitter during gesture transitions.
"""

import os, sys, math, time, urllib.request
from collections import deque

import cv2, pyautogui, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Auto-download model ───────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model (~9 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# ── Config ────────────────────────────────────────────────────────────────────
SW, SH         = pyautogui.size()
FLIP_FRAME     = True

SENSITIVITY    = 3.0     # base cursor sensitivity
HAND_SMOOTH    = 0.50    # EMA smoothing (higher = more responsive, less smooth)
DEAD_ZONE      = 0.004   # min hand-delta to register movement (filters tremor)

SLOW_THRESH    = 0.008   # hand speed below this → precision mode
SLOW_FACTOR    = 0.35    # sensitivity multiplier in precision mode
FAST_THRESH    = 0.030   # hand speed above this → fast mode
FAST_FACTOR    = 1.25    # sensitivity multiplier in fast mode

CLICK_COOLDOWN = 0.30    # min seconds between any click events

SCROLL_SENS    = 40      # multiply normalised delta → scroll ticks
SCROLL_DEAD    = 0.003   # min delta to accumulate scroll

ZOOM_THRESH    = 0.025   # normalised dist-change to trigger one zoom step

THUMB_DIST_THRESH = 0.07  # distance thumb-tip → index-MCP; > = extended

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

# ── Skeleton overlay ──────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ── Gesture colours ───────────────────────────────────────────────────────────
GESTURE_INFO = {
    "move":   ("MOVE — tracking cursor",           (100, 220, 255)),
    "freeze": ("FREEZE — reposition hand freely",  (120, 120, 255)),
    "drag":   ("DRAG — hold to move item",         (255, 180,   0)),
    "rclick": ("RIGHT CLICK",                      (  0, 100, 255)),
    "lclick": ("LEFT CLICK",                       (  0, 255, 100)),
    "dclick": ("LONG PRESS + DRAG — hold then move", (  0, 200, 200)),
    "scroll": ("SCROLL — move hand up / down",     (200, 100, 255)),
    "zoom":   ("ZOOM — spread / pinch",            (255, 200,   0)),
    "none":   ("Unrecognised pose",                (100, 100, 100)),
}

GUIDE = [
    ("ALL FINGERS OPEN",    "Move cursor"),
    ("THUMB DOWN, 4 UP",    "Freeze cursor"),
    ("FIST (all closed)",   "Drag (hold left button)"),
    ("ONLY THUMB UP",       "Right click"),
    ("ONLY INDEX UP",       "Left click"),
    ("INDEX + MIDDLE (V)",  "Long press + drag"),
    ("3 FINGERS UP",        "Scroll up / down"),
    ("THUMB + INDEX",       "Zoom in / out"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

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
    elif not t and     i and     m and not r and not p: return "dclick"  # V-sign
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


def adaptive_sens(avg_spd):
    if avg_spd < SLOW_THRESH:
        t = avg_spd / SLOW_THRESH
        return SENSITIVITY * (SLOW_FACTOR + (1 - SLOW_FACTOR) * t)
    if avg_spd > FAST_THRESH:
        return SENSITIVITY * FAST_FACTOR
    return SENSITIVITY

# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_ui(frame, gesture, lm):
    h, w = frame.shape[:2]

    if lm:
        for a, b in HAND_CONNECTIONS:
            p1 = (int(lm[a].x * w), int(lm[a].y * h))
            p2 = (int(lm[b].x * w), int(lm[b].y * h))
            cv2.line(frame, p1, p2, (0, 200, 200), 2)
        for pt in lm:
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 4, (200, 200, 200), -1)

    label, color = GESTURE_INFO.get(gesture, GESTURE_INFO["none"])
    cv2.rectangle(frame, (0, 0), (w, 46), (20, 20, 20), -1)
    cv2.putText(frame, f"  {label}", (8, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    gx = 10
    gy = h - len(GUIDE) * 24 - 18
    cv2.rectangle(frame, (gx-4, gy-8), (gx+360, gy + len(GUIDE)*24 + 4), (20,20,20), -1)
    cv2.rectangle(frame, (gx-4, gy-8), (gx+360, gy + len(GUIDE)*24 + 4), (70,70,70),  1)
    for i, (g, act) in enumerate(GUIDE):
        cv2.putText(frame, f"{g:<26}{act}", (gx, gy + i*24 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (155, 155, 155), 1)

# ── Global state ──────────────────────────────────────────────────────────────
_cx, _cy = pyautogui.position()
cursor_x, cursor_y = _cx / SW, _cy / SH    # start from actual cursor position
smooth_x = smooth_y = None                  # None → reinitialise on next hand frame
prev_x   = prev_y   = 0.5

speed_buf    = deque(maxlen=6)
prev_gesture = "none"
last_click   = 0.0
is_dragging  = False
v_pressing   = False
scroll_acc   = 0.0
zoom_prev    = None
zoom_acc     = 0.0
latest_lm    = None

# ── MediaPipe setup ───────────────────────────────────────────────────────────
def _on_result(result, _img, _ts):
    global latest_lm
    latest_lm = result.hand_landmarks[0] if result.hand_landmarks else None

base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
lm_opts = mp_vision.HandLandmarkerOptions(
    base_options=base_opts,
    running_mode=mp_vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.60,
    min_hand_presence_confidence=0.60,
    min_tracking_confidence=0.50,
    result_callback=_on_result,
)
landmarker = mp_vision.HandLandmarker.create_from_options(lm_opts)

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera. Grant access in System Settings → Privacy → Camera.")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Hand Gesture Control started.  Press 'q' to quit.")
ts = 0

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ok, frame = cap.read()
    if not ok:
        break

    if FLIP_FRAME:
        frame = cv2.flip(frame, 1)

    ts += 33
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarker.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

    now = time.time()
    lm  = latest_lm

    if lm is not None:
        # ── Palm centre → EMA smooth ───────────────────────────────────────
        px, py = palm_center(lm)
        if smooth_x is None:          # first frame after hand appears
            smooth_x = px;  smooth_y = py
            prev_x   = px;  prev_y   = py
        else:
            smooth_x = HAND_SMOOTH * smooth_x + (1 - HAND_SMOOTH) * px
            smooth_y = HAND_SMOOTH * smooth_y + (1 - HAND_SMOOTH) * py

        gesture = classify(lm)
        changed = (gesture != prev_gesture)

        # ── Drag transition (fist) ────────────────────────────────────────
        if gesture == "drag" and not is_dragging:
            pyautogui.mouseDown()
            is_dragging = True
        elif gesture != "drag" and is_dragging:
            pyautogui.mouseUp()
            is_dragging = False

        # ── V long-press transition ────────────────────────────────────────
        if gesture == "dclick" and not v_pressing:
            pyautogui.mouseDown()
            v_pressing = True
        elif gesture != "dclick" and v_pressing:
            pyautogui.mouseUp()
            v_pressing = False

        # ── Cursor movement (move, drag, and V long-press) ─────────────────
        if gesture in ("move", "drag", "dclick"):
            dx   = smooth_x - prev_x
            dy   = smooth_y - prev_y
            dist = math.sqrt(dx*dx + dy*dy)
            speed_buf.append(dist)
            avg  = sum(speed_buf) / max(len(speed_buf), 1)
            sens = adaptive_sens(avg)
            if dist > DEAD_ZONE:
                cursor_x = max(0.0, min(1.0, cursor_x + dx * sens))
                cursor_y = max(0.0, min(1.0, cursor_y + dy * sens))
            pyautogui.moveTo(int(cursor_x * SW), int(cursor_y * SH))

        # ── Freeze ─────────────────────────────────────────────────────────
        elif gesture == "freeze":
            pass    # cursor stays; prev_x/y updated below so no jump on resume

        # ── Single clicks  (fire once on gesture entry) ────────────────────
        elif gesture == "rclick":
            if changed and (now - last_click) > CLICK_COOLDOWN:
                pyautogui.click(button="right")
                last_click = now

        elif gesture == "lclick":
            if changed and (now - last_click) > CLICK_COOLDOWN:
                pyautogui.click(button="left")
                last_click = now

        # ── Scroll ─────────────────────────────────────────────────────────
        elif gesture == "scroll":
            if changed:
                scroll_acc = 0.0
            else:
                dy = smooth_y - prev_y
                if abs(dy) > SCROLL_DEAD:
                    scroll_acc += dy
            ticks = int(scroll_acc * SCROLL_SENS)
            if ticks:
                pyautogui.scroll(-ticks)     # hand up → page scrolls up (positive)
                scroll_acc -= ticks / SCROLL_SENS

        # ── Zoom ───────────────────────────────────────────────────────────
        elif gesture == "zoom":
            d = tip_dist(lm, 4, 8)
            if changed:
                zoom_prev = d
                zoom_acc  = 0.0
            else:
                if zoom_prev is not None:
                    zoom_acc += d - zoom_prev
            zoom_prev = d
            if abs(zoom_acc) >= ZOOM_THRESH:
                if zoom_acc > 0:
                    pyautogui.hotkey("command", "=")   # spread → zoom in
                else:
                    pyautogui.hotkey("command", "-")   # pinch  → zoom out
                zoom_acc = 0.0

        # ── Always advance prev position ───────────────────────────────────
        prev_x = smooth_x
        prev_y = smooth_y
        prev_gesture = gesture

        draw_ui(frame, gesture, lm)

    else:
        # ── No hand detected ───────────────────────────────────────────────
        if is_dragging:
            pyautogui.mouseUp()
            is_dragging = False
        if v_pressing:
            pyautogui.mouseUp()
            v_pressing = False
        prev_gesture = "none"
        smooth_x = smooth_y = None      # will reinitialise when hand reappears

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (20, 20, 20), -1)
        cv2.putText(frame, "  Show your hand to the camera", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 100, 255), 2)
        draw_ui(frame, "none", None)

    cv2.imshow("Hand Gesture Control  [q to quit]", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
if is_dragging:
    pyautogui.mouseUp()
if v_pressing:
    pyautogui.mouseUp()
landmarker.close()
cap.release()
cv2.destroyAllWindows()
print("Hand Gesture Control ended.")
