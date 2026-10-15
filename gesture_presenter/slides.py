import os
import re

import cv2
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_slide_paths(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Slides folder not found: {folder}")
    names = sorted((n for n in os.listdir(folder) if os.path.splitext(n)[1].lower() in EXTS),
                   key=_natural_key)
    if not names:
        raise FileNotFoundError(f"No slide images (png/jpg/...) in: {folder}")
    return [os.path.join(folder, n) for n in names]


def fit(img, w, h):
    """Resize keeping aspect ratio, centred on a dark canvas (no stretching)."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    resized = cv2.resize(img, (int(iw * s), int(ih * s)), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 20, np.uint8)
    y, x = (h - resized.shape[0]) // 2, (w - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def load_slides(folder, w, h):
    slides = []
    for path in list_slide_paths(folder):
        img = cv2.imread(path)
        if img is None:
            print(f"[warn] could not read {path}, skipping")
            continue
        slides.append(fit(img, w, h))
    if not slides:
        raise RuntimeError("No readable slides.")
    return slides
