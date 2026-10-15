"""Render tools/deck/deck.html -> ../slides/NN_name.png (1920x1080).
Needs:  pip install playwright && playwright install chromium
Edit the text in deck/deck.html, then run:  python tools/build_slides.py
"""
import shutil
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "slides"
NAMES = ["title", "team", "what-is-inclusive-education", "principles", "benefits",
         "barriers", "classroom-strategies", "tech-for-inclusion", "global-commitments", "thank-you"]

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir()
with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1920, "height": 1080})
    page.goto((HERE / "deck" / "deck.html").as_uri())
    page.evaluate("document.fonts.ready")
    page.wait_for_timeout(500)
    for i, name in enumerate(NAMES, 1):
        page.locator(f"#s{i}").screenshot(path=str(OUT / f"{i:02d}_{name}.png"))
        print("wrote", f"{i:02d}_{name}.png")
    b.close()
