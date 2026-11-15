"""One fitted scene program drawn over a video frame in the site's style, with an in-figure legend.

Shared by tools/build_videos.py (the validation overlays, data/runtime/<tag>/), tools/build_wild.py (data/wild/) and
tools/build_qualitative.py (data/qualitative/<tag>/). Reads saved programs only: no fitting, no model calls.

Per object, in the colour order of overlay_style.object_colours (pipeline.palette, asked object first): the observed
track mask outline (dark with a white core), the program's occluded silhouette (1 px) and visible outline (2 px) in the
object's outline shade. Tags down the left edge: the unit, frame and time on a white tag, one tag per object tinted
with its colour (name and IoU on that frame), then one white tag per line style with a sample of the line, so every
mark has one meaning stated in the image.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WEB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB.parent))
sys.path.insert(0, str(WEB / "tools"))
import overlay_style as S  # noqa: E402

from pipeline import palette  # noqa: E402
from pipeline.data import Video  # noqa: E402
from pipeline.sceneprog.masks import ObservedMasks  # noqa: E402
from pipeline.sceneprog.metric import frame_record  # noqa: E402
from pipeline.sceneprog.poses import camera, poses_at  # noqa: E402
from pipeline.sceneprog.program import load  # noqa: E402
from pipeline.sceneprog.render import render  # noqa: E402
from pipeline.sceneprog.shapes import compile_scene  # noqa: E402

SAMPLE = S.bgr(S.CSS_COLOURS.get("grey", "#B9BEC4"))   # the neutral object colour of the legend's line samples
GLYPHS = (("observed", "track mask (observed)"), ("visible", "program, visible part"),
          ("occluded", "program, occluded part"))


def program_colours(program: dict, frame: int, theirs: list[str] = ()) -> dict[str, str]:
    """name key -> colour for one frame (the rule of the paper's figures): the program's bodies posed at the frame in
    program order with the asked object first, then other methods' names, then the bodies not posed at the frame."""
    posed = poses_at(program, frame / float(program["video"]["fps"]))
    bodies = [o["id"] for o in program["objects"] if o["kind"] == "body"]
    present = [b for b in bodies if posed.get(b) is not None]
    asked = (program.get("query") or {}).get("object")
    present.sort(key=lambda b: b != asked)
    return S.object_colours(present, list(theirs), [b for b in bodies if b not in present])


def _line(image: np.ndarray, kind: str, p0: tuple, p1: tuple, colour) -> None:
    if kind == "observed":
        cv2.line(image, p0, p1, S.bgr(S.TEXT), 3, cv2.LINE_AA)
        cv2.line(image, p0, p1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.line(image, p0, p1, palette.darker(colour), 2 if kind == "visible" else 1, cv2.LINE_AA)


def overlay(bgr: np.ndarray, rendered: dict, observed: dict, labels: dict, colours: dict[str, str],
            header: str) -> np.ndarray:
    """The marks of every object, then the tags: header, objects (tinted), line styles (white, with a sample)."""
    image = np.ascontiguousarray(bgr).copy()
    order = list(colours)
    ids = sorted(set(labels), key=lambda o: order.index(S.name_key(o)))
    for oid in ids:
        colour = S.bgr(colours[S.name_key(oid)])
        obs = observed.get(oid)
        if obs is not None and obs.any():
            cs, _ = cv2.findContours(obs.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image, cs, -1, S.bgr(S.TEXT), 3, cv2.LINE_AA)
            cv2.drawContours(image, cs, -1, (255, 255, 255), 1, cv2.LINE_AA)
        r = rendered.get(oid)
        if r is not None:
            cs, _ = cv2.findContours(r["silhouette"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image, cs, -1, palette.darker(colour), 1, cv2.LINE_AA)
            cs, _ = cv2.findContours(r["visible"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image, cs, -1, palette.darker(colour), 2, cv2.LINE_AA)
    size = S.font_px(image)
    font = ImageFont.truetype(S.FONT, size)
    pad = max(2, size // 5)
    sample_w = 2 * size
    im = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(im)
    rows = [(header, None, None)] + [(labels[o], colours[S.name_key(o)], None) for o in ids] + \
        [(text, None, kind) for kind, text in GLYPHS]
    y, samples = 2, []
    for text, colour, kind in rows:
        l, t, r, b = dr.textbbox((0, 0), text, font=font)
        indent = sample_w + pad if kind else 0
        w, h = r - l + 2 * pad + indent, (b - t) + 2 * pad
        x0 = 4
        fill = S.tint(colour) if colour else (255, 255, 255)
        rim = S.edge(colour) if colour else S.bgr(S.CSS_COLOURS["grey"])
        dr.rectangle((x0, y, x0 + w, y + h), fill=fill[::-1], outline=rim[::-1], width=1)
        dr.text((x0 + pad + indent - l, y + pad - t), text, font=font, fill=S.bgr(S.TEXT)[::-1])
        if kind:
            samples.append((kind, (x0 + pad, y + h // 2), (x0 + pad + sample_w, y + h // 2)))
        y += h + 2
    image = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    for kind, p0, p1 in samples:
        _line(image, kind, p0, p1, SAMPLE)
    return image


def draw(directory: Path, frame: int, out: Path, colours: dict[str, str] | None = None, meshes: dict | None = None,
         quality: int = 86) -> dict:
    """The program in `directory` at `frame` (the frames pipeline.sceneprog.evaluate drew), written to `out`.
    Returns the meshes so a caller drawing several frames of one program compiles them once."""
    program, scad = load(directory)
    observed = ObservedMasks(program["observations"]["tracks_tag"], program["unit"], program["split"])
    meshes = meshes if meshes is not None else compile_scene(program, scad)
    fps = float(program["video"]["fps"])
    cam = camera(program)
    if (cam["height"], cam["width"]) != observed.shape:
        raise ValueError(f"{directory}: camera image {cam['height']}x{cam['width']} differs from the masks' {observed.shape}")
    modelled = {o["id"]: o["track"] for o in program["objects"] if o["kind"] == "body" and o.get("track") is not None}
    colours = colours if colours is not None else program_colours(program, frame)
    r = render(program, meshes, frame / fps, cam)
    labels, obs = {}, {}
    for oid, name in modelled.items():
        o = observed.get(name, frame)
        obs[oid] = o
        if o is None:
            labels[oid] = f"{oid} untracked"
            continue
        others = [observed.get(n, frame) for n in observed.names() if n != name]
        rec = frame_record(r.get(oid), o, [x for x in others if x is not None])
        labels[oid] = f"{oid} IoU {rec['iou']:.2f}"
    video = Video(Path(program["video"]["path"]), fps=fps)
    try:
        image = overlay(cv2.cvtColor(video.frame(frame), cv2.COLOR_RGB2BGR), r, obs, labels, colours,
                        f"{program['unit']}  frame {frame}  t={frame / fps:.2f}s")
    finally:
        video.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise ValueError(f"{out}: not written")
    return meshes


def draw_unit(directory: Path, out_dir: Path) -> list[Path]:
    """Every frame pipeline.sceneprog.evaluate drew for this program (its overlay_f<frame>.jpg), redrawn into
    out_dir/overlay_f<frame>.jpg."""
    frames = sorted(int(p.stem.removeprefix("overlay_f")) for p in Path(directory).glob("overlay_f*.jpg"))
    if not frames:
        raise ValueError(f"{directory}: no overlay_f*.jpg")
    meshes, outs = None, []
    for f in frames:
        out = Path(out_dir) / f"overlay_f{f:04d}.jpg"
        meshes = draw(Path(directory), f, out, meshes=meshes)
        outs.append(out)
    return outs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Redraw one program's overlays (a check of the drawing).")
    ap.add_argument("directory", type=Path)
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args()
    print(draw_unit(a.directory, a.out_dir))
