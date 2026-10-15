from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    slides_dir: str = str(ROOT / "slides")
    camera: int = 0
    width: int = 1280              # presentation window size (16:9)
    height: int = 720
    detection_confidence: float = 0.8
    gesture_line: float = 350 / 540        # fraction of frame height; hand ABOVE it = slide gestures
    active_x: tuple = (0.5, 1.0)           # part of the camera frame that maps to the slide
    active_y: tuple = (150 / 540, 390 / 540)
    cooldown: float = 0.8          # seconds between slide-change / undo actions
    stable_frames: int = 3         # frames a gesture must hold before it counts (kills flicker)
    smoothing: float = 0.5         
    inset_scale: float = 0.2      
    fullscreen: bool = False
    show_webcam_window: bool = False
