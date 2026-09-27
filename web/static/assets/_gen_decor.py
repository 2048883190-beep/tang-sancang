"""Generate original decorative textures for contest UI (local assets)."""
from __future__ import annotations

import math
import os
import random

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

OUT = os.path.dirname(os.path.abspath(__file__))


def paper_texture(path: str, w: int = 1400, h: int = 900) -> None:
    img = Image.new("RGB", (w, h), (247, 243, 234))
    px = img.load()
    rng = random.Random(42)
    for y in range(h):
        for x in range(w):
            n = rng.randint(-10, 10)
            streak = int(4 * math.sin(x * 0.07 + y * 0.02))
            r = 247 + n + streak // 3
            g = 243 + n
            b = 234 + n - streak // 4
            px[x, y] = (
                max(220, min(255, r)),
                max(215, min(250, g)),
                max(205, min(240, b)),
            )
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for i in range(80):
        a = int(18 * (i / 80))
        d.ellipse([i * 4, i * 3, w - i * 4, h - i * 3], outline=(176, 138, 60, a))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    img.save(path, quality=82, optimize=True)


def medical_still(path: str, w: int = 1000, h: int = 700) -> None:
    img = Image.new("RGB", (w, h), (236, 244, 242))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(h * 0.62), w, h], fill=(232, 224, 210))
    d.rounded_rectangle(
        [120, 90, 420, 520],
        radius=18,
        fill=(250, 248, 242),
        outline=(180, 170, 150),
        width=2,
    )
    for y in range(140, 480, 36):
        d.line([150, y, 390, y], fill=(210, 200, 185), width=2)
    d.arc([480, 160, 780, 460], 20, 340, fill=(61, 107, 102), width=10)
    d.ellipse([720, 300, 820, 400], outline=(61, 107, 102), width=8)
    d.ellipse([500, 120, 560, 180], outline=(166, 61, 47), width=6)
    d.ellipse([600, 110, 660, 170], outline=(166, 61, 47), width=6)
    light = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ld = ImageDraw.Draw(light)
    ld.ellipse([-80, -100, 420, 360], fill=(255, 255, 255, 55))
    img = Image.alpha_composite(img.convert("RGBA"), light).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    ImageEnhance.Color(img).enhance(0.92).save(path, quality=85, optimize=True)


def herb_corner(path: str, w: int = 900, h: int = 600) -> None:
    img = Image.new("RGB", (w, h), (248, 245, 236))
    d = ImageDraw.Draw(img)
    greens = [(90, 120, 95), (70, 105, 85), (110, 130, 100)]
    for i, color in enumerate(greens):
        base_x = 120 + i * 220
        d.line([base_x, h - 40, base_x + 30, 80], fill=color, width=3)
        for j in range(8):
            y = 120 + j * 50
            spread = 40 + j * 6
            d.ellipse([base_x - spread, y, base_x + 20, y + 28], outline=color, width=2)
            d.ellipse(
                [base_x + 10, y + 10, base_x + spread + 30, y + 38],
                outline=color,
                width=2,
            )
    d.ellipse([720, 60, 860, 200], outline=(166, 61, 47), width=3)
    d.ellipse([735, 75, 845, 185], outline=(166, 61, 47), width=1)
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img.save(path, quality=85, optimize=True)


def main() -> None:
    paper_texture(os.path.join(OUT, "paper.jpg"))
    medical_still(os.path.join(OUT, "medical.jpg"))
    herb_corner(os.path.join(OUT, "herb.jpg"))
    for name in ("paper.jpg", "medical.jpg", "herb.jpg"):
        p = os.path.join(OUT, name)
        print(name, os.path.getsize(p))


if __name__ == "__main__":
    main()
