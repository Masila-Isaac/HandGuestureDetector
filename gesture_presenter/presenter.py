"""Presentation state: current slide, per-slide annotations, action cooldown."""
import time

from .gestures import DISCRETE, Gesture


class Presenter:
    def __init__(self, n_slides: int, cooldown: float = 0.8):
        self.n = n_slides
        self.index = 0
        self.cooldown = cooldown
        self._strokes = {i: [] for i in range(n_slides)}   # normalised (x, y) points
        self._drawing = False
        self._last_action = float("-inf")

    @property
    def strokes(self):
        return self._strokes[self.index]

    #  actions (also used by keyboard shortcuts)  
    def next(self):
        if self.index < self.n - 1:
            self.index += 1
            self._drawing = False
            return "Next slide"

    def previous(self):
        if self.index > 0:
            self.index -= 1
            self._drawing = False
            return "Previous slide"

    def undo(self):
        if self.strokes:
            self.strokes.pop()
            self._drawing = False
            return "Undo"

    def clear(self):
        if self.strokes:
            self.strokes.clear()
            self._drawing = False
            return "Cleared"

    #   gesture handling  
    def handle(self, gesture: Gesture, point=None, now=None):
        """Apply one frame of input. Returns a short message if something happened."""
        now = time.monotonic() if now is None else now

        if gesture is Gesture.DRAW and point is not None:
            if not self._drawing:
                self.strokes.append([])
                self._drawing = True
            self.strokes[-1].append(point)
        else:
            self._drawing = False

        if gesture in DISCRETE and now - self._last_action >= self.cooldown:
            action = {Gesture.NEXT: self.next, Gesture.PREVIOUS: self.previous,
                      Gesture.UNDO: self.undo, Gesture.CLEAR: self.clear}[gesture]()
            if action:
                self._last_action = now
                return action
        return None
