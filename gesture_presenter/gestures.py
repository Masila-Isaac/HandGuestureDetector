"""Pure gesture logic (no camera, no OpenCV) so it can be unit-tested."""
from enum import Enum

import numpy as np


class Gesture(Enum):
    NONE = "none"
    PREVIOUS = "previous slide"
    NEXT = "next slide"
    POINTER = "pointer"
    DRAW = "draw"
    UNDO = "undo"
    CLEAR = "clear"


# finger order from cvzone: thumb, index, middle, ring, pinky
SLIDE_GESTURES = {            # only while the hand is above the green line
    (1, 0, 0, 0, 0): Gesture.PREVIOUS,
    (0, 0, 0, 0, 1): Gesture.NEXT,
}
ANYWHERE_GESTURES = {         # work at any hand height
    (0, 1, 1, 0, 0): Gesture.POINTER,
    (0, 1, 0, 0, 0): Gesture.DRAW,
    (0, 1, 0, 0, 1): Gesture.UNDO,
    (0, 1, 1, 1, 0): Gesture.CLEAR,
}
DISCRETE = {Gesture.PREVIOUS, Gesture.NEXT, Gesture.UNDO, Gesture.CLEAR}


def classify(fingers, hand_above_line: bool) -> Gesture:
    key = tuple(int(f) for f in fingers)
    if key in ANYWHERE_GESTURES:
        return ANYWHERE_GESTURES[key]
    if hand_above_line and key in SLIDE_GESTURES:
        return SLIDE_GESTURES[key]
    return Gesture.NONE


class GestureDebouncer:
    """A gesture only counts once it has been seen for `frames` frames in a row."""

    def __init__(self, frames: int = 3):
        self.frames = max(1, frames)
        self._candidate = Gesture.NONE
        self._count = 0
        self.stable = Gesture.NONE

    def update(self, raw: Gesture) -> Gesture:
        if raw == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = raw, 1
        if self._count >= self.frames:
            self.stable = raw
        return self.stable


def map_to_slide(x, y, frame_w, frame_h, active_x, active_y):
    """Camera pixel -> normalised slide coordinate (0..1, clamped)."""
    u = np.interp(x / frame_w, active_x, (0.0, 1.0))
    v = np.interp(y / frame_h, active_y, (0.0, 1.0))
    return float(u), float(v)
