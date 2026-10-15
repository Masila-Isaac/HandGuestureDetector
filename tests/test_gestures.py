import unittest

from gesture_presenter.gestures import Gesture, GestureDebouncer, classify, map_to_slide
from gesture_presenter.presenter import Presenter


class ClassifyTests(unittest.TestCase):
    def test_slide_gestures_need_hand_above_line(self):
        self.assertEqual(classify([1, 0, 0, 0, 0], True), Gesture.PREVIOUS)
        self.assertEqual(classify([0, 0, 0, 0, 1], True), Gesture.NEXT)
        self.assertEqual(classify([0, 0, 0, 0, 1], False), Gesture.NONE)

    def test_draw_and_pointer_work_at_any_height(self):
        for above in (True, False):   # regression: used to work only below the line
            self.assertEqual(classify([0, 1, 0, 0, 0], above), Gesture.DRAW)
            self.assertEqual(classify([0, 1, 1, 0, 0], above), Gesture.POINTER)

    def test_unknown_is_none(self):
        self.assertEqual(classify([1, 1, 1, 1, 1], True), Gesture.NONE)


class DebounceTests(unittest.TestCase):
    def test_needs_consecutive_frames(self):
        d = GestureDebouncer(3)
        out = [d.update(g) for g in (Gesture.NEXT, Gesture.NEXT, Gesture.NONE, Gesture.NEXT, Gesture.NEXT, Gesture.NEXT)]
        self.assertEqual(out[1], Gesture.NONE)
        self.assertEqual(out[-1], Gesture.NEXT)


class PresenterTests(unittest.TestCase):
    def test_navigation_bounds_and_cooldown(self):
        p = Presenter(3, cooldown=1.0)
        self.assertEqual(p.handle(Gesture.NEXT, now=0), "Next slide")
        self.assertIsNone(p.handle(Gesture.NEXT, now=0.5))      # cooling down
        self.assertEqual(p.handle(Gesture.NEXT, now=1.5), "Next slide")
        self.assertIsNone(p.handle(Gesture.NEXT, now=3))        # last slide
        self.assertEqual(p.index, 2)

    def test_draw_undo_clear_and_per_slide_strokes(self):
        p = Presenter(2, cooldown=0)
        p.handle(Gesture.DRAW, (0.1, 0.1), now=0)
        p.handle(Gesture.DRAW, (0.2, 0.2), now=1)
        p.handle(Gesture.NONE, None, now=2)
        p.handle(Gesture.DRAW, (0.5, 0.5), now=3)               # second stroke
        self.assertEqual(len(p.strokes), 2)
        p.handle(Gesture.UNDO, now=4)
        self.assertEqual(len(p.strokes), 1)
        p.handle(Gesture.NEXT, now=5)
        self.assertEqual(p.strokes, [])                         # other slide is clean
        p.handle(Gesture.PREVIOUS, now=6)
        self.assertEqual(len(p.strokes), 1)                     # ...and drawings are kept
        p.handle(Gesture.CLEAR, now=7)
        self.assertEqual(p.strokes, [])

    def test_undo_on_empty_does_not_crash(self):
        p = Presenter(1, cooldown=0)
        self.assertIsNone(p.handle(Gesture.UNDO, now=0))


class MappingTests(unittest.TestCase):
    def test_clamped(self):
        self.assertEqual(map_to_slide(0, 0, 960, 540, (0.5, 1), (0.3, 0.7)), (0.0, 0.0))
        self.assertEqual(map_to_slide(960, 540, 960, 540, (0.5, 1), (0.3, 0.7)), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
