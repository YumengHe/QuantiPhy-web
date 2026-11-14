"""One drawing style for every object overlay the site shows (ours, PhysMind, Thinking with Spatial Code).

Colours are the paper palette, read from site.css (--c-*). Every object of one image has its own colour, taken in a
fixed order with the asked object first (object_colours); in one comparison an object keeps its colour across methods
when the names match. Each object: a light fill of its colour, its outline a little darker, and its name in dark text
on a tag tinted with the colour (never pale text). Images are BGR uint8 (OpenCV).
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WEB = Path(__file__).resolve().parents[1]
_CSS = (WEB / "site.css").read_text()
CSS_COLOURS = dict(re.findall(r"--c-([a-z]+):\s*(#[0-9A-Fa-f]{6})", _CSS))
TEXT = re.search(r"--text:\s*(#[0-9A-Fa-f]{6})", _CSS).group(1)
# the order objects take the palette in: the asked object blue, then orange, green, ...
OBJECT_ORDER = ["blue", "orange", "green", "purple", "yellow", "teal", "red", "lavender", "olive", "pink", "sky", "grey"]
OBJECTS = [CSS_COLOURS[c] for c in OBJECT_ORDER]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FILL = 0.28          # weight of the object colour in its fill
TAG_TINT = 0.45      # weight of the object colour in its label tag (rest white)
EDGE = 0.75          # an outline is the object colour scaled by this (a little darker than the fill)


def name_key(name: str) -> str:
    """An object's name as compared across methods: lower case, no article, letters and digits only."""
    n = name.lower().strip()
    n = re.sub(r"^(the|a|an) ", "", n)
    return re.sub(r"[^a-z0-9]", "", n)


def object_colours(ordered: list[str], *others: list[str]) -> dict[str, str]:
    """name key -> colour. `ordered`: one method's objects, asked object first; `others`: the names other methods
    drew in the same comparison, in their own order. A name met again keeps its colour; a new name takes the next
    colour. More objects than colours wrap around."""
    out: dict[str, str] = {}
    for names in (ordered, *others):
        for n in names:
            k = name_key(n)
            if k not in out:
                out[k] = OBJECTS[len(out) % len(OBJECTS)]
    return out


def bgr(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def edge(hex_colour: str) -> tuple[int, int, int]:
    """The outline shade of an object colour (BGR)."""
    return tuple(int(round(EDGE * c)) for c in bgr(hex_colour))


def tint(hex_colour: str, w: float = TAG_TINT) -> tuple[int, int, int]:
    return tuple(int(round(w * c + (1 - w) * 255)) for c in bgr(hex_colour))


def font_px(image: np.ndarray) -> int:
    return max(12, int(round(image.shape[1] / 66)))


def fill_and_outline(out: np.ndarray, mask: np.ndarray, colour: str, box: bool = False, width: int = 2) -> None:
    """Light fill, darker outline (and, with box, the bounding box) of one object's mask, in place."""
    c = np.array(bgr(colour), float)
    sel = mask > 0
    out[sel] = ((1 - FILL) * out[sel] + FILL * c).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, edge(colour), width, cv2.LINE_AA)
    if box:
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), edge(colour), 1, cv2.LINE_AA)


def tags(out: np.ndarray, items: list[tuple[str, int, int | None, str | None]], size: int | None = None) -> np.ndarray:
    """Draws labels: (text, x, y of the tag's bottom-left or None for under the previous tag, colour or None for a
    white tag). Tags stay inside the image; one that would cover a tag already placed moves a tag's height up or down
    if that is free. Returns the image (BGR)."""
    size = size or font_px(out)
    font = ImageFont.truetype(FONT, size)
    pad = max(2, size // 5)
    im = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(im)
    h_img, w_img = out.shape[:2]
    placed: list[tuple[int, int, int, int]] = []
    for text, x, y, colour in items:
        l, t, r, b = dr.textbbox((0, 0), text, font=font)
        w, h = r - l + 2 * pad, (b - t) + 2 * pad
        bx = int(min(max(0, x), w_img - w - 1))
        if y is None:  # a column: under the tag placed before
            by = int(min(placed[-1][3] + 2 if placed else 2, h_img - h - 1))
        else:
            by = int(min(max(0, y - h), h_img - h - 1))
        x0, y0 = bx, by
        # next to its place: as drawn, else a tag's height up or down, else as drawn over the other tag
        for dy in (0, -(h + 1), h + 1, -2 * (h + 1), 2 * (h + 1)):
            cy = int(min(max(0, by + dy), h_img - h - 1))
            if not any(bx < p[2] and p[0] < bx + w and cy < p[3] and p[1] < cy + h for p in placed):
                y0 = cy
                break
        fill = tint(colour) if colour else (255, 255, 255)
        rim = edge(colour) if colour else bgr(CSS_COLOURS["grey"])
        dr.rectangle((x0, y0, x0 + w, y0 + h), fill=fill[::-1], outline=rim[::-1], width=1)
        dr.text((x0 + pad - l, y0 + pad - t), text, font=font, fill=bgr(TEXT)[::-1])
        placed.append((x0, y0, x0 + w, y0 + h))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def time_tag(t: float) -> tuple[str, int, int, None]:
    """The frame's time, top-left, dark on a white tag."""
    return (f"t={t:.2f}s", 4, None, None)
