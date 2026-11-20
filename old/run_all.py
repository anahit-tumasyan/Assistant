#!/usr/bin/env python3
"""
run_all.py  —  Runs hand gesture control + voice assistant together.

  Main thread : hand landmark (OpenCV window — must be on main thread on macOS)
  Thread 2    : voice listen → Gemma3 predict → execute command
"""

import os
import sys
import math
import time
import threading
import urllib.request
from collections import deque

import cv2
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Voice imports ────────────────────────────────────────────────────────────
from speech_module import SpeechAssistant
from Predict_command import TerminalAssistant
from Run_command import run_command

# ============================================================
# SHARED STATE  (written by voice thread, read by overlay)
# ============================================================
voice_status = "Voice: starting…"
voice_lock   = threading.Lock()

def set_voice_status(msg: str):
    global voice_status
    with voice_lock:
        voice_status = msg

def get_voice_status() -> str:
    with voice_lock:
        return voice_status

# ============================================================
# VOICE THREAD
# ============================================================

def voice_loop():
    api_key         = os.environ.get("ELEVENLABS_API_KEY", "")
    vosk_model_path = os.path.abspath("Includes/model")
    gemma_path      = os.path.abspath("Includes/gemma-3-1b-it")

    set_voice_status("Voice: loading speech module…")
    try:
        speech = SpeechAssistant(api_key=api_key, model_path=vosk_model_path)
    except Exception as e:
        set_voice_status(f"Voice: speech init failed — {e}")
        return

    set_voice_status("Voice: loading Gemma 3…")
    try:
        terminal = TerminalAssistant(model_path=gemma_path, device="mps")
    except Exception as e:
        set_voice_status(f"Voice: Gemma init failed — {e}")
        return

    set_voice_status("Voice: ready  |  say a command")

    while True:
        try:
            set_voice_status("Voice: listening…")
            text = speech.speech_to_text(duration=5)

            if not text or not text.strip():
                set_voice_status("Voice: ready  |  say a command")
                continue

            set_voice_status(f'Voice: heard "{text[:40]}"')
            time.sleep(0.4)

            set_voice_status("Voice: thinking…")
            cmd = terminal.predict_command(text)
            set_voice_status(f"Voice: running → {cmd[:50]}")

            output = run_command(cmd)
            short  = output.strip().splitlines()[0][:60] if output.strip() else "done"
            set_voice_status(f"Voice: ✓ {short}")
            time.sleep(2)
            set_voice_status("Voice: ready  |  say a command")

        except Exception as e:
            set_voice_status(f"Voice: error — {e}")
            time.sleep(3)
            set_voice_status("Voice: ready  |  say a command")

# ============================================================
# HAND LANDMARK SETUP
# ============================================================

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model (~9 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Cursor
SMOOTHING = 0.40
DEAD_ZONE = 0.002

# Gestures
PINCH_CLICK_LO = 0.035
PINCH_CLICK_HI = 0.075
PINCH_DOUBLE   = 0.035
PINCH_DRAG_LO  = 0.08
PINCH_DRAG_HI  = 0.14
FIST_THRESHOLD = 0.28

# Swipe
SWIPE_THRESHOLD  = 0.020
SWIPE_MIN_FRAMES = 5
SWIPE_COOLDOWN   = 1.5
SWIPE_WARMUP     = 12

pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()

# ── Hand state ───────────────────────────────────────────────
smoothed_x = smoothed_y = 0.5
last_raw_x = last_raw_y = None
last_swipe_time  = 0.0
click_down       = False
drag_mode        = False
latest_landmarks = None
hand_frame_count = 0
x_vel_buf        = deque(maxlen=SWIPE_MIN_FRAMES)

# ── Helpers ──────────────────────────────────────────────────

def norm_dist(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def detect_gesture(lm):
    pinch    = norm_dist(lm[4], lm[8])
    avg_dist = sum(norm_dist(lm[0], lm[i]) for i in (4,8,12,16,20)) / 5
    if avg_dist < FIST_THRESHOLD:              return "fist"
    if pinch <= PINCH_DOUBLE:                  return "double_click"
    if PINCH_CLICK_LO <= pinch <= PINCH_CLICK_HI: return "click"
    if PINCH_DRAG_LO  <  pinch <  PINCH_DRAG_HI:  return "drag"
    return None

def draw_skeleton(frame, lm, w, h):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame,
                 (int(lm[a].x*w), int(lm[a].y*h)),
                 (int(lm[b].x*w), int(lm[b].y*h)),
                 (0, 210, 210), 2)
    for i, pt in enumerate(lm):
        color = (0, 255, 80) if i in (4, 8) else (220, 220, 220)
        cv2.circle(frame, (int(pt.x*w), int(pt.y*h)),
                   7 if i in (4, 8) else 4, color, -1)

def draw_hand_guide(frame, gesture, drag_mode):
    lines = [
        ("OPEN HAND",     "Move cursor",       gesture is None),
        ("PINCH med",     "Left click",        gesture == "click"),
        ("PINCH tight",   "Double click",      gesture == "double_click"),
        ("PINCH wide",    "Drag / grab",       gesture == "drag" or drag_mode),
        ("FIST",          "Spotlight",         gesture == "fist"),
        ("SWIPE L/R",     "Switch desktop",    False),
    ]
    h = frame.shape[0]
    bx, by = 10, h - len(lines)*26 - 20
    cv2.rectangle(frame, (bx-4, by-10), (bx+300, by+len(lines)*26+4), (20,20,20), -1)
    cv2.rectangle(frame, (bx-4, by-10), (bx+300, by+len(lines)*26+4), (70,70,70), 1)
    for i, (gname, action, active) in enumerate(lines):
        color  = (0, 255, 100) if active else (150, 150, 150)
        prefix = "▶ " if active else "  "
        cv2.putText(frame, f"{prefix}{gname:<13} {action}",
                    (bx, by+i*26+16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)

def draw_top_bar(frame, gesture, stable, drag_mode, voice_msg):
    W = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (W, 48), (20, 20, 20), -1)

    # Hand status (left)
    hand_label = {
        "click":        "LEFT CLICK",
        "double_click": "DOUBLE CLICK",
        "drag":         "DRAG (holding)",
        "fist":         "FIST / SPOTLIGHT",
    }.get(gesture, "DRAG ACTIVE" if drag_mode else ("STABLE" if stable else "Moving"))
    hand_color = {
        "click":        (0, 255, 100),
        "double_click": (0, 200, 255),
        "drag":         (255, 180, 0),
        "fist":         (255, 80, 200),
    }.get(gesture, (160,160,160) if stable else (100,220,255))
    cv2.putText(frame, f"Hand: {hand_label}", (8, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, hand_color, 2)

    # Voice status (right)
    voice_color = (100, 220, 255) if "listening" in voice_msg else \
                  (0, 255, 100)   if "ready" in voice_msg else \
                  (255, 200, 0)
    cv2.putText(frame, voice_msg, (W//2, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, voice_color, 1)

def result_callback(result, output_image, timestamp_ms):
    global latest_landmarks
    latest_landmarks = result.hand_landmarks[0] if result.hand_landmarks else None

# ── Build landmarker ─────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
    result_callback=result_callback,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

# ============================================================
# MAIN
# ============================================================

def main():
    global smoothed_x, smoothed_y, last_raw_x, last_raw_y
    global last_swipe_time, click_down, drag_mode
    global hand_frame_count, x_vel_buf

    # Start voice thread
    t = threading.Thread(target=voice_loop, daemon=True)
    t.start()

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera. Grant access in System Settings → Privacy → Camera.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Running. Press 'q' in the camera window to quit.")
    timestamp_ms = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        timestamp_ms += 33

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, timestamp_ms)

        lm = latest_landmarks

        if lm is not None:
            hand_frame_count += 1
            draw_skeleton(frame, lm, w, h)

            raw_x, raw_y = lm[9].x, lm[9].y

            if last_raw_x is None:          # first frame — seed without jump
                last_raw_x, last_raw_y = raw_x, raw_y
                smoothed_x, smoothed_y = raw_x, raw_y

            delta  = math.sqrt((raw_x - last_raw_x)**2 + (raw_y - last_raw_y)**2)
            stable = delta <= DEAD_ZONE

            if not stable:
                smoothed_x = smoothed_x * (1 - SMOOTHING) + raw_x * SMOOTHING
                smoothed_y = smoothed_y * (1 - SMOOTHING) + raw_y * SMOOTHING

            pyautogui.moveTo(int(smoothed_x * SCREEN_W),
                             int(smoothed_y * SCREEN_H), duration=0)

            gesture = detect_gesture(lm)

            if gesture == "fist":
                if not click_down:
                    click_down = True
                    pyautogui.hotkey("cmd", "space")
            elif gesture == "double_click":
                if not click_down:
                    click_down = True
                    pyautogui.doubleClick()
            elif gesture == "click":
                if not click_down:
                    click_down = True
                    pyautogui.click()
            elif gesture == "drag":
                if not drag_mode:
                    drag_mode = True; click_down = True
                    pyautogui.mouseDown()
            else:
                if drag_mode:
                    drag_mode = False
                    pyautogui.mouseUp()
                click_down = False

            # Swipe
            x_vel_buf.append(raw_x - last_raw_x)
            now = time.time()
            if (hand_frame_count > SWIPE_WARMUP
                    and (now - last_swipe_time) > SWIPE_COOLDOWN
                    and len(x_vel_buf) == SWIPE_MIN_FRAMES):
                all_left  = all(v < -SWIPE_THRESHOLD for v in x_vel_buf)
                all_right = all(v >  SWIPE_THRESHOLD for v in x_vel_buf)
                if all_left or all_right:
                    pyautogui.hotkey("ctrl", "left" if all_left else "right")
                    last_swipe_time = now
                    x_vel_buf.clear()

            last_raw_x, last_raw_y = raw_x, raw_y

            draw_top_bar(frame, gesture, stable, drag_mode, get_voice_status())
            draw_hand_guide(frame, gesture, drag_mode)

        else:
            if drag_mode:
                drag_mode = False; pyautogui.mouseUp()
            click_down = False
            last_raw_x = last_raw_y = None
            hand_frame_count = 0
            x_vel_buf.clear()

            cv2.rectangle(frame, (0, 0), (w, 48), (20, 20, 20), -1)
            cv2.putText(frame, "  Show your hand to the camera",
                        (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            cv2.putText(frame, get_voice_status(),
                        (w//2, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 220, 255), 1)
            draw_hand_guide(frame, None, False)

        cv2.imshow("Assistant  [q = quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Bye.")


if __name__ == "__main__":
    main()