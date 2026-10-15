import argparse

from .app import run
from .config import Config


def main():
    d = Config()
    p = argparse.ArgumentParser(prog="gesture_presenter",
                                description="Control slides with hand gestures.")
    p.add_argument("--slides", default=d.slides_dir, help="folder of slide images")
    p.add_argument("--camera", type=int, default=d.camera)
    p.add_argument("--width", type=int, default=d.width)
    p.add_argument("--height", type=int, default=d.height)
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--webcam-window", action="store_true", help="also show the full webcam view")
    a = p.parse_args()
    run(Config(slides_dir=a.slides, camera=a.camera, width=a.width, height=a.height,
               fullscreen=a.fullscreen, show_webcam_window=a.webcam_window))


if __name__ == "__main__":
    main()
