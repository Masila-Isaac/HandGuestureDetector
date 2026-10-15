"""Everything drawn on screen: strokes, laser pointer, webcam inset, HUD."""
import cv2
import numpy as np

INK = (50, 50, 215)          # BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX
HELP = [("Thumb (hand high)", "previous slide"),
        ("Pinky (hand high)", "next slide"),
        ("Index finger", "draw"),
        ("Index + middle", "laser pointer"),
        ("Index + pinky", "undo last stroke"),
        ("Index + middle + ring", "clear slide"),
        ("Keys", "n/p  u  c  f  h  q")]


def _px(pt, w, h):
    return int(pt[0] * w), int(pt[1] * h)


def draw_strokes(view, strokes):
    h, w = view.shape[:2]
    for s in strokes:
        pts = np.array([_px(p, w, h) for p in s], np.int32)
        if len(pts) == 1:
            cv2.circle(view, tuple(pts[0]), 4, INK, -1, cv2.LINE_AA)
        else:
            cv2.polylines(view, [pts], False, INK, 6, cv2.LINE_AA)


def draw_pointer(view, point):
    h, w = view.shape[:2]
    c = _px(point, w, h)
    glow = view.copy()
    cv2.circle(glow, c, 28, (80, 80, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.35, view, 0.65, 0, dst=view)
    cv2.circle(view, c, 10, (40, 40, 230), -1, cv2.LINE_AA)
    cv2.circle(view, c, 10, (255, 255, 255), 2, cv2.LINE_AA)


def _shade(view, x0, y0, x1, y1, alpha=0.55):
    roi = view[y0:y1, x0:x1]
    cv2.addWeighted(np.zeros_like(roi), alpha, roi, 1 - alpha, 0, dst=roi)


def draw_hud(view, index, total, gesture_label, toast=None, show_help=False):
    h, w = view.shape[:2]
    _shade(view, 0, h - 40, w, h)
    cv2.putText(view, f"{index + 1} / {total}", (16, h - 13), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    bar_x0, bar_x1 = 110, w - 260
    cv2.line(view, (bar_x0, h - 20), (bar_x1, h - 20), (110, 110, 110), 4, cv2.LINE_AA)
    frac = (index + 1) / total
    cv2.line(view, (bar_x0, h - 20), (int(bar_x0 + (bar_x1 - bar_x0) * frac), h - 20),
             (160, 200, 120), 4, cv2.LINE_AA)
    if gesture_label:
        cv2.putText(view, gesture_label, (w - 240, h - 13), FONT, 0.6, (200, 230, 255), 2, cv2.LINE_AA)
    if toast:
        (tw, th), _ = cv2.getTextSize(toast, FONT, 1.0, 2)
        x, y = (w - tw) // 2, 70
        _shade(view, x - 20, y - th - 18, x + tw + 20, y + 18, 0.7)
        cv2.putText(view, toast, (x, y), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    if show_help:
        _shade(view, 40, 100, 640, 120 + 36 * len(HELP), 0.75)
        for i, (left, right) in enumerate(HELP):
            cv2.putText(view, left, (60, 140 + 36 * i), FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(view, right, (340, 140 + 36 * i), FONT, 0.65, (200, 230, 255), 1, cv2.LINE_AA)


def draw_inset(view, cam_frame, scale):
    h, w = view.shape[:2]
    iw = int(w * scale)
    ih = int(iw * cam_frame.shape[0] / cam_frame.shape[1])
    small = cv2.resize(cam_frame, (iw, ih))
    view[8:8 + ih, w - iw - 8:w - 8] = small
    cv2.rectangle(view, (w - iw - 8, 8), (w - 8, 8 + ih), (255, 255, 255), 2)
