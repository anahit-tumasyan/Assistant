import os
import random
import statistics
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hand_landmark as hl  # noqa: E402

FRAME = 1 / 30   # the webcam's frame interval


class Point:
    def __init__(self, x, y):
        self.x, self.y, self.z = x, y, 0.0


def make_hand(thumb=True, index=True, middle=True, ring=True, pinky=True,
              dx=0.0, dy=0.0, thumb_tip=None):
    """21 landmarks of an upright hand, palm to the camera."""
    pts = [None] * 21
    pts[0] = (0.50, 0.80)
    pts[1], pts[2], pts[3] = (0.43, 0.75), (0.38, 0.68), (0.35, 0.62)
    pts[4] = thumb_tip or ((0.32, 0.57) if thumb else (0.47, 0.64))
    fingers = [((5, 6, 7, 8), 0.45, index), ((9, 10, 11, 12), 0.50, middle),
               ((13, 14, 15, 16), 0.55, ring), ((17, 18, 19, 20), 0.60, pinky)]
    for (mcp, pip, dip, tip), x, up in fingers:
        pts[mcp] = (x, 0.60)
        pts[pip] = (x, 0.50)
        pts[dip] = (x, 0.45) if up else (x, 0.55)
        pts[tip] = (x, 0.40) if up else (x, 0.56)
    return [Point(x + dx, y + dy) for x, y in pts]


def jittery(points, rng, sigma=0.002):
    """Add tracking noise like MediaPipe's (about a pixel or two)."""
    return [Point(p.x + rng.gauss(0, sigma), p.y + rng.gauss(0, sigma)) for p in points]


OPEN = dict()
FIST = dict(thumb=False, index=False, middle=False, ring=False, pinky=False)
INDEX = dict(thumb=False, middle=False, ring=False, pinky=False)
THUMB = dict(index=False, middle=False, ring=False, pinky=False)
V_SIGN = dict(thumb=False, ring=False, pinky=False)
THREE = dict(thumb=False, pinky=False)
FOUR = dict(thumb=False)
L_SHAPE = dict(middle=False, ring=False, pinky=False)


class FakeMouse:
    def __init__(self):
        self.calls = []

    def size(self):
        return (1000, 600)

    def position(self):
        return (500, 300)

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record

    def names(self):
        return [name for name, _, _ in self.calls]

    def args(self, name):
        return [args for call, args, _ in self.calls if call == name]


class ClassifyTests(unittest.TestCase):
    def test_every_pose(self):
        expected = {"move": OPEN, "drag": FIST, "lclick": INDEX, "rclick": THUMB,
                    "press": V_SIGN, "scroll": THREE, "freeze": FOUR, "zoom": L_SHAPE}
        for gesture, pose in expected.items():
            self.assertEqual(hl.classify(make_hand(**pose)), gesture)


class ControllerTestCase(unittest.TestCase):
    def setUp(self):
        self.mouse = FakeMouse()
        # glide=False: pointer moves happen right away, so tests are exact
        self.ctrl = hl.GestureController(mouse=self.mouse, glide=False)
        self.t = 10.0

    def feed_points(self, points):
        self.t += FRAME
        return self.ctrl.update(points, now=self.t)

    def feed(self, pose, frames=1, **kw):
        for _ in range(frames):
            gesture = self.feed_points(make_hand(**pose, **kw))
        return gesture


class GestureTests(ControllerTestCase):
    def test_click_needs_a_steady_pose_and_fires_once(self):
        self.feed(INDEX, hl.STABLE_FRAMES - 1)
        self.assertNotIn("click", self.mouse.names())
        self.feed(INDEX, 5)
        self.assertEqual(self.mouse.names().count("click"), 1)

    def test_passing_through_other_poses_fires_nothing(self):
        self.feed(OPEN, 3)
        self.feed(V_SIGN, hl.STABLE_FRAMES - 1)   # on the way to the index finger
        self.feed(INDEX, 3)
        self.assertNotIn("mouseDown", self.mouse.names())
        self.assertEqual(self.mouse.names().count("click"), 1)

    def test_fist_drags_with_real_drag_events(self):
        self.feed(FIST, 3)
        self.assertEqual(self.mouse.names().count("mouseDown"), 1)
        for step in range(1, 4):
            self.feed(FIST, dx=0.03 * step)
        drags = [kw for name, _, kw in self.mouse.calls if name == "dragTo"]
        self.assertTrue(drags)
        self.assertNotIn("moveTo", self.mouse.names())
        self.assertFalse(drags[0]["mouseDownUp"])
        self.feed(OPEN, 3, dx=0.09)
        self.assertEqual(self.mouse.names().count("mouseUp"), 1)

    def test_open_hand_moves_from_the_current_mouse_position(self):
        self.feed(OPEN, 3)
        self.feed(OPEN, dx=0.05)
        moves = self.mouse.args("moveTo")
        self.assertTrue(moves)
        self.assertGreater(moves[-1][0], 500)

    def test_losing_the_hand_releases_the_button(self):
        self.feed(FIST, 3)
        self.ctrl.hand_lost()
        self.assertEqual(self.mouse.names().count("mouseUp"), 1)
        self.ctrl.hand_lost()   # nothing held any more -> no extra mouseUp
        self.assertEqual(self.mouse.names().count("mouseUp"), 1)

    def test_three_fingers_moving_up_scroll_up(self):
        self.feed(THREE, 3)
        for step in range(1, 6):
            self.feed(THREE, dy=-0.02 * step)
        scrolls = [args[0] for args in self.mouse.args("scroll")]
        self.assertTrue(scrolls)
        self.assertTrue(all(ticks > 0 for ticks in scrolls))

    def test_spreading_thumb_and_index_zooms_in(self):
        self.feed(L_SHAPE, 3)
        for step in range(1, 5):
            self.feed(L_SHAPE, thumb_tip=(0.32 - 0.015 * step, 0.57 + 0.01 * step))
        self.assertIn(("hotkey", (hl.ZOOM_KEY, "="), {}), self.mouse.calls)


class SmoothnessTests(ControllerTestCase):
    def test_a_resting_hand_never_moves_the_cursor(self):
        rng = random.Random(1)
        for _ in range(90):                         # 3 s of jittery tracking
            self.feed_points(jittery(make_hand(), rng))
        self.assertNotIn("moveTo", self.mouse.names())

    def test_slow_careful_movement_is_smooth(self):
        rng = random.Random(2)
        for i in range(90):                         # 0.05 frame widths a second, for 3 s
            self.feed_points(jittery(make_hand(dx=0.05 * i * FRAME), rng))
        xs = [args[0] for args in self.mouse.args("moveTo")]
        self.assertGreater(len(xs), 60)             # moves on nearly every frame...
        steps = [b - a for a, b in zip(xs, xs[1:])]
        self.assertLess(max(abs(s) for s in steps), 12)   # ...in small steps, never a lurch
        self.assertGreater(xs[-1] - xs[0], 100)     # ...and really gets somewhere

    def test_cursor_stays_on_target_while_a_click_pose_forms(self):
        self.feed(OPEN, 3)
        moves = len(self.mouse.args("moveTo"))
        # curling the fingers shifts the palm centre a little
        self.feed(INDEX, hl.STABLE_FRAMES, dx=0.01, dy=0.006)
        self.assertEqual(len(self.mouse.args("moveTo")), moves)
        self.assertEqual(self.mouse.names().count("click"), 1)

    def test_a_flicker_does_not_lose_movement(self):
        self.feed(OPEN, 3)
        for i in range(1, 6):
            self.feed(OPEN, dx=0.02 * i)
        steady = self.mouse.args("moveTo")[-1][0]
        self.setUp()
        self.feed(OPEN, 3)
        for i in range(1, 6):   # same movement, but one frame is misread as another pose
            self.feed(FOUR if i == 3 else OPEN, dx=0.02 * i)
        self.assertAlmostEqual(self.mouse.args("moveTo")[-1][0], steady, delta=3)

    def test_short_tracking_gaps_keep_a_drag_going(self):
        self.feed(FIST, 3)
        self.t += 0.1
        self.ctrl.hand_missing(self.t)              # a blurry frame or two
        self.assertNotIn("mouseUp", self.mouse.names())
        self.feed(FIST, 2, dx=0.02)
        self.assertNotIn("mouseUp", self.mouse.names())
        self.t += hl.HAND_LOST_GRACE + 0.05
        self.ctrl.hand_missing(self.t)              # really gone
        self.assertEqual(self.mouse.names().count("mouseUp"), 1)


class OneEuroFilterTests(unittest.TestCase):
    def test_calms_jitter(self):
        rng, f = random.Random(3), hl.OneEuroFilter()
        xs = [f(0.5 + rng.gauss(0, 0.002), 0.5, i * FRAME)[0] for i in range(150)]
        self.assertLess(statistics.pstdev(xs[30:]), 0.001)   # input jitter is 0.002

    def test_keeps_up_with_fast_movement(self):
        f = hl.OneEuroFilter()
        for i in range(30):                         # one frame width per second
            x = f(0.2 + i * FRAME, 0.5, i * FRAME)[0]
        self.assertGreater(x, 0.2 + 29 * FRAME - 0.02)


class PointerTests(unittest.TestCase):
    def test_glides_in_small_steps_and_arrives(self):
        mouse = FakeMouse()
        pointer = hl.Pointer(mouse, glide=True)
        try:
            pointer.move_to(800.0, 300.0)           # from (500, 300)
            deadline = time.monotonic() + 2
            while (not mouse.args("moveTo") or mouse.args("moveTo")[-1] != (800, 300)) \
                    and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            pointer.close()
        xs = [args[0] for args in mouse.args("moveTo")]
        self.assertEqual(xs[-1], 800)
        self.assertGreater(len(xs), 5)              # glided, didn't jump
        self.assertLess(xs[0], 650)                 # the first step goes only part of the way

    def test_moves_become_drags_while_a_button_is_held(self):
        mouse = FakeMouse()
        pointer = hl.Pointer(mouse, glide=False)
        pointer.press()
        pointer.move_to(600.0, 300.0)
        pointer.release()
        self.assertEqual(mouse.names(), ["mouseDown", "dragTo", "mouseUp"])


class OverlayTextTests(unittest.TestCase):
    def test_text_is_plain_ascii(self):
        self.assertEqual(hl.fit_text("Voice: ✓ done — ok…", 2000, 0.5),
                         "Voice: OK done - ok...")

    def test_long_text_is_cut_to_fit(self):
        text = hl.fit_text("x" * 500, 200, 0.5)
        self.assertTrue(text.endswith("..."))
        self.assertLessEqual(hl.cv2.getTextSize(text, hl.FONT, 0.5, 1)[0][0], 200)


if __name__ == "__main__":
    unittest.main()
