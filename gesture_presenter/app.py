import time

import cv2
from cvzone.HandTrackingModule import HandDetector

from .gestures import Gesture, GestureDebouncer, classify, map_to_slide
from .presenter import Presenter
from .render import draw_hud, draw_inset, draw_pointer, draw_strokes
from .slides import load_slides

WIN = "Presentation"
LEFT = {ord("p"), ord("a"), 2424832, 65361, 63234}
RIGHT = {ord("n"), ord("d"), ord(" "), 2555904, 65363, 63235}


def run(cfg):
    slides = load_slides(cfg.slides_dir, cfg.width, cfg.height)
    cap = cv2.VideoCapture(cfg.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cfg.camera} (try --camera 1)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)

    detector = HandDetector(detectionCon=cfg.detection_confidence, maxHands=1)
    presenter = Presenter(len(slides), cfg.cooldown)
    debounce = GestureDebouncer(cfg.stable_frames)
    toast, toast_until, show_help, fullscreen = None, 0.0, False, cfg.fullscreen
    smooth = None

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, cfg.width, cfg.height)
    if fullscreen:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def notify(msg):
        nonlocal toast, toast_until
        if msg:
            toast, toast_until = msg, time.monotonic() + 1.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[error] camera returned no frame")
            break
        frame = cv2.flip(frame, 1)                       # mirror, like a selfie
        frame = cv2.resize(frame, (cfg.width, cfg.height))
        fh, fw = frame.shape[:2]
        line_y = int(cfg.gesture_line * fh)
        cv2.line(frame, (0, line_y), (fw, line_y), (0, 255, 0), 3)

        hands, frame = detector.findHands(frame, flipType=False)
        raw, point = Gesture.NONE, None
        if hands:
            hand = hands[0]
            cx, cy = hand["center"]
            tip = hand["lmList"][8]
            raw = classify(detector.fingersUp(hand), cy <= line_y)
            point = map_to_slide(tip[0], tip[1], fw, fh, cfg.active_x, cfg.active_y)
            if smooth is None:                           # ease out fingertip jitter
                smooth = point
            smooth = tuple(s + cfg.smoothing * (p - s) for s, p in zip(smooth, point))
            point = smooth
        else:
            smooth = None

        gesture = debounce.update(raw)
        notify(presenter.handle(gesture, point))

        view = slides[presenter.index].copy()
        draw_strokes(view, presenter.strokes)
        if gesture in (Gesture.POINTER, Gesture.DRAW) and point:
            draw_pointer(view, point)
        draw_inset(view, frame, cfg.inset_scale)
        draw_hud(view, presenter.index, len(slides),
                 gesture.value if gesture is not Gesture.NONE else "",
                 toast if time.monotonic() < toast_until else None, show_help)

        cv2.imshow(WIN, view)
        if cfg.show_webcam_window:
            cv2.imshow("Webcam", frame)

        key = cv2.waitKeyEx(1)
        if key in (ord("q"), 27):
            break
        elif key in RIGHT:
            notify(presenter.next())
        elif key in LEFT:
            notify(presenter.previous())
        elif key == ord("u"):
            notify(presenter.undo())
        elif key == ord("c"):
            notify(presenter.clear())
        elif key == ord("h"):
            show_help = not show_help
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()
