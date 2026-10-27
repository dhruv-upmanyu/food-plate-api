"""Crop and square helpers shared by training-data building and inference (kept free of heavy imports)."""
from __future__ import annotations

from PIL import Image, ImageOps

SIZE = 256
MARGIN = 0.06     # extra context around a box, fraction of its size
PAD_COLOR = (124, 116, 104)


def square(img: Image.Image, size: int = SIZE) -> Image.Image:
    """Pad to a square (whole dish kept, nothing cropped away) and resize. Used for training and inference alike."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), PAD_COLOR)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas.resize((size, size), Image.BICUBIC)


def crop_box(img: Image.Image, box: list[float], margin: float = MARGIN) -> Image.Image:
    """box: normalised xyxy; returns the region with a small margin."""
    w, h = img.size
    x1, y1, x2, y2 = sorted(box[0::2])[0], sorted(box[1::2])[0], sorted(box[0::2])[1], sorted(box[1::2])[1]
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    left, top = min(max(0.0, (x1 - mx) * w), w - 2), min(max(0.0, (y1 - my) * h), h - 2)
    right, bottom = max(min(float(w), (x2 + mx) * w), left + 2), max(min(float(h), (y2 + my) * h), top + 2)
    return img.crop((left, top, right, bottom))
