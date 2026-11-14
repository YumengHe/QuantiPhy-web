"""PhysMind's and Thinking with Spatial Code's overlays in the site's style (tools/overlay_style.py).

Both runs drew their reconstructions onto video frames in their own colours (reconstructions/ of each result folder,
with index.json naming each drawn object and its colour). The geometry is taken from those images as drawn; nothing
is re-fitted or re-projected. For each image the video frame it was drawn on is decoded again, and every pixel is
explained as the frame blended with one of the drawn colours (p = frame + a (colour - frame), a in [0, 1]):

- PhysMind (posed meshes): the light fill (a about 0.25) with the outline next to it gives each object's projected
  mask; the image is redrawn on the clean frame with tools/overlay_style.py (fill, outline, box, name tag).
- Thinking with Spatial Code (3D box wireframes): the line pixels of each box keep their coverage a and take the
  object's new colour; the old name text (written just above each box) is replaced by a name tag at the same place.

Object colours: the palette without our method's blue, in order, by the object's position in the run's own drawn
list. Text is dark on tinted tags; the frame time sits top-left on a white tag.

  python -B web/tools/restyle_baselines.py

Writes, for the validation videos, data/physmind/<video>_f<frame>.jpg (and the colours in data/physmind/index.json)
and data/spatialcode/<video>_f<drawn_frame>.jpg with data/spatialcode/index.json; for the wild videos
data/physmind_wild/ and data/spatialcode_wild/, <run's video name>_f<frame>.jpg with index.json mapping our unit
names (wild.json) to the images.
"""
from __future__ import annotations

import csv
import datetime
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overlay_style as S  # noqa: E402

WEB = S.WEB
NAS = Path("/nas/yumenghe/QuantiPhy-data")
PM = NAS / "physmind_result_new"
SC = NAS / "spatial_code_result"
VAL_VIDEOS = WEB.parent / "data" / "QuantiPhy-validation" / "validation_videos"
WILD_VIDEOS = NAS / "videoscene"
WHITE = "#ffffff"


def video_frame(path: Path, k: int) -> np.ndarray:
    """Frame k (BGR) decoded by ffmpeg, as the runs decoded it (OpenCV renders HLG phone footage in other colours)."""
    w, h = (int(x) for x in subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(path)], capture_output=True, text=True, check=True).stdout.strip().split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{k})", "-vsync", "0",
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, check=True).stdout
    if len(raw) < w * h * 3:
        raise ValueError(f"{path}: no frame {k}")
    return np.frombuffer(raw[:w * h * 3], np.uint8).reshape(h, w, 3).copy()


def explain(img: np.ndarray, frame: np.ndarray, colours: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    """Per pixel: index of the drawn colour (len(colours) = white text, -1 = untouched frame) and its weight a.
    Also the share of clearly changed pixels left unexplained."""
    I, O = img.astype(np.float32), frame.astype(np.float32)
    D = I - O
    best_res = np.full(img.shape[:2], np.inf, np.float32)
    lab = np.full(img.shape[:2], -1, np.int32)
    alpha = np.zeros(img.shape[:2], np.float32)
    for k, col in enumerate(colours + [WHITE]):
        d = np.array(S.bgr(col), np.float32) - O
        n2 = (d * d).sum(2) + 1e-3
        a = np.clip((D * d).sum(2) / n2, 0, 1)
        res = np.sqrt(((D - a[..., None] * d) ** 2).sum(2))
        shift = a * np.sqrt(n2)
        ok = (a >= 0.1) & (shift > 22) & (res < 24 + 0.12 * shift) & (res < best_res)
        best_res[ok], lab[ok], alpha[ok] = res[ok], k, a[ok]
    changed = np.abs(D).sum(2) > 60
    return lab, alpha, float((changed & (lab < 0)).sum() / max(1, changed.sum()))


def physmind_masks(lab: np.ndarray, alpha: np.ndarray, n: int) -> list[np.ndarray]:
    """Each object's projected mesh mask: the core of its light fill (an opening removes the 1 px box and the name
    text), grown back over the fill and outline pixels next to it, holes filled, specks dropped."""
    out, cores = [], []
    k3 = np.ones((3, 3), np.uint8)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for k in range(n):
        drawn = (lab == k).astype(np.uint8)
        fill = ((lab == k) & (alpha < 0.5)).astype(np.uint8)
        core = cv2.morphologyEx(fill, cv2.MORPH_OPEN, k5)
        if core.sum() < 20:  # a thin object
            core = cv2.morphologyEx(fill, cv2.MORPH_OPEN, k3)
        nc, cc, st, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        if nc > 1:  # what survives of the text and the box is a few pixels; the mesh's fill is a blob
            area = st[1:, cv2.CC_STAT_AREA]
            keep = [i + 1 for i, a in enumerate(area) if a >= max(12, 0.08 * area.max())]
            core = np.isin(cc, keep).astype(np.uint8)
        cores.append(m := core)
        for _ in range(3):  # back over the opening's margin and the 2 px outline
            m = m | (drawn & cv2.dilate(m, k3))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5)
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        big = max((cv2.contourArea(c) for c in cnt), default=0)
        cnt = [c for c in cnt if cv2.contourArea(c) >= max(15, 0.03 * big)]
        m = np.zeros_like(fill)
        cv2.drawContours(m, cnt, -1, 1, -1)
        out.append(m)
    # the visible parts of the meshes are disjoint: a hole filled over another object's own pixels goes back to it
    near = [((lab == j) & (cv2.dilate(c, k5) > 0)) for j, c in enumerate(cores)]
    for k in range(n):
        others = np.zeros(lab.shape, bool)
        for j in range(n):
            if j != k:
                others |= near[j]
        out[k] = (out[k] > 0) & ~others
    return [m.astype(np.uint8) for m in out]


def text_and_lines(lab: np.ndarray, alpha: np.ndarray, k: int):
    """Spatial Code: the line pixels of colour k without its old name, and where the name sat (x, y of the bottom
    left of its text). The run wrote each name just above its box, starting at the box's left end: the name is the
    colour's top rows."""
    m = ((lab == k) & (alpha >= 0.2)).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)  # specks: misread pixels, anti-aliasing
    m[np.isin(cc, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] < 8])] = 0
    count = m.sum(1)
    rows = np.nonzero(count)[0]
    if not len(rows):
        return m.astype(bool), None
    # the run's text is 11-13 rows high at every image size: its body's fill, then a near-empty row under it or the
    # box's top edge (a row much fuller than the text's)
    top = int(rows[0])
    body = float(np.median(count[top:top + 8]))
    end = top + 12
    for r in range(top + 8, min(top + 15, m.shape[0])):
        if count[r] <= 0.3 * body or count[r] > 2.5 * body:
            end = r
            break
    cols = np.nonzero(m[top:end].any(0))[0]
    m[top:end] = 0
    at = (int(cols[0]), end)
    return m.astype(bool), at


def restyle(kind: str, src: Path, frame: np.ndarray, drawn: list[dict], t: float, dst: Path) -> dict:
    img = cv2.imread(str(src))
    if frame.shape[:2] != img.shape[:2]:
        frame = cv2.resize(frame, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    old = [d["color"] for d in drawn]
    lab, alpha, unexplained = explain(img, frame, old)
    new = [S.BASELINE[i % len(S.BASELINE)] for i in range(len(drawn))]
    out = frame.copy()
    labels = []
    if kind == "physmind":
        masks = physmind_masks(lab, alpha, len(drawn))
        for d, m, c in zip(drawn, masks, new):
            if m.any():
                S.fill_and_outline(out, m, c, box=True)
                x, y, _, _ = cv2.boundingRect(m)
                labels.append((d["label"], x, y - 2, c))
    else:
        for k, (d, c) in enumerate(zip(drawn, new)):
            lines, at = text_and_lines(lab, alpha, k)
            a = alpha[lines][:, None]
            out[lines] = ((1 - a) * out[lines] + a * np.array(S.bgr(c), np.float32)).astype(np.uint8)
            if at is not None:
                labels.append((d["label"], at[0], at[1] + 2, c))
    out = S.tags(out, [S.time_tag(t)] + labels)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 86]):
        raise ValueError(f"{dst}: not written")
    return {"unexplained": round(unexplained, 4), "labels": len(labels), "objects": len(drawn),
            "drawn": [{"label": d["label"], "color": c} for d, c in zip(drawn, new)]}


def wild_units() -> dict[str, str]:
    """Run question id (<run video>_q<k>) -> our unit, checked on the question text."""
    ours = {q["unit"]: q["q"].strip() for v in json.loads((WEB / "data" / "wild.json").read_text())["videos"]
            for q in v["questions"]}
    out = {}
    with open(WILD_VIDEOS / "questions.csv") as f:
        for r in csv.DictReader(f):
            name = Path(r["file"]).stem
            m = re.match(r"(\d+)_(robot|nonrobot)_", name)
            unit = f"{m.group(2)}_{int(m.group(1)):04d}_q{r['question_index']}"
            if ours.get(unit) != r["question"].strip():
                raise ValueError(f"{name} q{r['question_index']}: {r['question']!r} is not {unit}: {ours.get(unit)!r}")
            out[f"{name}_q{r['question_index']}"] = unit
    return out


def run(kind: str, index: dict, src_root: Path, videos: dict[str, Path], out_dir: Path, rel: str,
        name_of, units: dict[str, str] | None = None) -> dict:
    key = "physmind" if kind == "physmind" else "spatialcode"
    stats = defaultdict(int)
    worst = []
    for v in index["videos"]:
        for f in v["frames"]:
            fi = int(f["drawn_frame"] if kind == "spatialcode" else f["frame"])
            t = float(f["drawn_t"] if kind == "spatialcode" else f["t"])
            if units is not None:
                f["units"] = [units[q] for q in f["questions"]]
            if not f.get(key):
                continue
            name = name_of(v["video"], fi)
            rec = restyle(kind, src_root / Path(f[key]).name, video_frame(videos[v["video"]], fi), f["drawn"], t,
                          out_dir / name)
            f[key], f["drawn"] = f"{rel}/{name}", rec["drawn"]
            stats["images"] += 1
            worst.append((rec["unexplained"], name, rec["labels"], rec["objects"]))
    if units is not None:
        index["units"] = {u: v["video"] for v in index["videos"] for f in v["frames"] for u in f["units"]}
    index["generated"] = datetime.date.today().isoformat()
    index["style"] = "web/tools/restyle_baselines.py"
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    worst.sort(reverse=True)
    print(f"{rel}: {stats['images']} images; most unexplained {worst[:3]}")
    return index


def main() -> None:
    val = {p.stem: p for p in VAL_VIDEOS.glob("*.mp4")}
    wild = {p.stem: p for p in WILD_VIDEOS.glob("*/*.mp4")}
    units = wild_units()
    pm_val = json.loads((PM / "quantiphy" / "reconstructions" / "index.json").read_text())
    run("physmind", pm_val, PM / "quantiphy" / "reconstructions", val, WEB / "data" / "physmind", "data/physmind",
        lambda v, fi: f"{v}_f{fi}.jpg")
    sc_val = json.loads((SC / "quantiphy" / "reconstructions" / "index.json").read_text())
    run("spatialcode", sc_val, SC / "quantiphy" / "reconstructions", val, WEB / "data" / "spatialcode",
        "data/spatialcode", lambda v, fi: f"{v}_f{fi}.jpg")
    pm_wild = json.loads((PM / "videoscene" / "reconstructions" / "index.json").read_text())
    run("physmind", pm_wild, PM / "videoscene" / "reconstructions", wild, WEB / "data" / "physmind_wild",
        "data/physmind_wild", lambda v, fi: f"{v}_f{fi}.jpg", units)
    sc_wild = json.loads((SC / "videoscene" / "reconstructions" / "index.json").read_text())
    run("spatialcode", sc_wild, SC / "videoscene" / "reconstructions", wild, WEB / "data" / "spatialcode_wild",
        "data/spatialcode_wild", lambda v, fi: f"{v}_f{fi}.jpg", units)


if __name__ == "__main__":
    main()
