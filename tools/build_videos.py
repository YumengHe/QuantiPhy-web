"""Build QuantiPhy-web/data/videos.json (+ first-frame thumbnails) for videos.html.

Reads only existing outputs; never runs the pipeline.

  /home/yumenghe/miniconda3/envs/quantiphy/bin/python QuantiPhy-web/tools/build_videos.py

Inputs (relative to the QuantiPhy repo root):
  outputs/val_best_routed.csv                    validation predictions of the current best harness
  outputs/submission_v5.csv                      latest test submission (row-aligned with load_split('test'))
  outputs/scene_fit_eval/c_mesh2/<video>.json    validation scene fits
  outputs/scene_fit_eval/test_mesh/<video>.json  test scene fits
  data/QuantiPhy*/...mp4                         only to grab the first frame and duration
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pipeline.data import load_split, video_path  # noqa: E402
from pipeline.metrics import CATEGORIES, evaluate, mra_row  # noqa: E402

VAL_CSV = ROOT / "outputs" / "val_best_routed.csv"
TEST_CSV = ROOT / "outputs" / "submission_v5.csv"
SCENE_DIRS = {"val": ROOT / "outputs" / "scene_fit_eval" / "c_mesh2",
              "test": ROOT / "outputs" / "scene_fit_eval" / "test_mesh"}
OUT = WEB / "data" / "videos.json"
THUMBS = WEB / "data" / "thumbs"
THUMB_W = 240
THUMB_MAX_BYTES = 30_000


def num(x, sig=4):
    """Round to `sig` significant digits; NaN/inf/None -> None."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    if x == 0:
        return 0.0
    return float(f"{x:.{sig}g}")


def text(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)


def vec(v):
    return [num(a, 5) for a in v] if v is not None else None


def scene_fit(split: str, vid: str):
    p = SCENE_DIRS[split] / f"{vid}.json"
    if not p.exists():
        return None
    try:
        j = json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None
    pl = j.get("plane") or {}
    plane = {k: vec(pl.get(k)) for k in ("normal", "u1", "u2")} | {"dist": num(pl.get("dist"), 5)} if pl else None
    objs = []
    for name, o in (j.get("objects") or {}).items():
        objs.append({"name": name, "shape": o.get("shape"), "dims": {k: num(v) for k, v in (o.get("dims") or {}).items()},
                     "rms": num(o.get("rms"), 3), "p0": vec(o.get("p0")), "v0": vec(o.get("v0")), "a": vec(o.get("a")),
                     "yaw": num(o.get("yaw"), 5)})
    res = []
    for r in j.get("results") or []:
        res.append({"q": r.get("question"), "obj": r.get("object"), "pred": num(r.get("pred")),
                    "method": r.get("method"), "answer": num(r.get("answer")) if split == "val" else None,
                    "mra": num(r.get("mra"), 3) if split == "val" else None})
    return {"file": str(p.relative_to(ROOT)), "f": num(j.get("f"), 5), "plane": plane, "objects": objs, "results": res}


def video_meta(split: str, vid: str, make_thumb: bool):
    """(duration_s, width, height, thumb_rel_path) from the mp4; thumbnail from the first frame."""
    p = video_path(split, vid)
    if not p.exists():
        return None, None, None, None
    cap = cv2.VideoCapture(str(p))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    cap.release()
    dur = n / fps if fps > 0 and n > 0 else None
    thumb = None
    if ok:
        out = THUMBS / f"{vid}.jpg"
        if make_thumb or not out.exists():
            th = max(1, round(frame.shape[0] * THUMB_W / frame.shape[1]))
            small = cv2.resize(frame, (THUMB_W, th), interpolation=cv2.INTER_AREA)
            for q in (72, 60, 48, 36, 25):
                ok2, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])
                if ok2 and len(buf) <= THUMB_MAX_BYTES:
                    break
            out.write_bytes(buf.tobytes())
        thumb = str(out.relative_to(WEB))
    return num(dur, 4), w, h, thumb


def build_val(make_thumb: bool):
    df = pd.read_csv(VAL_CSV)
    df["mra"] = [mra_row(p, g) for p, g in zip(df["pred"], df["answer"])]
    ev = evaluate(df)
    print(f"val: n={ev['n']} macro={ev['average']:.4f} " + " ".join(f"{c}={ev[c]:.4f}" for c in CATEGORIES))
    videos = []
    for vid, g in df.groupby("video_id", sort=True):
        dur, w, h, thumb = video_meta("val", vid, make_thumb)
        first = g.iloc[0]
        qs = [{"cat": r.category, "type": r.inference_type, "q": r.question, "prior": text(r.prior),
               "depth": text(r.depth_info), "answer": num(r.answer), "geo": num(r.geo), "vlm": num(r.vlm),
               "pred": num(r.pred), "src": r.source, "mra": num(r.mra, 3)} for r in g.itertuples()]
        is3d = str(first.dim) == "3"
        videos.append({"id": vid, "source": first.video_source, "video_type": first.video_type, "fps": num(first.fps),
                       "dim": int(first.dim), "cats": sorted(set(g.category)), "duration": dur, "size": [w, h],
                       "thumb": thumb, "scene": scene_fit("val", vid) if is3d else None, "questions": qs})
    return {"file": str(VAL_CSV.relative_to(ROOT)), "macro": num(ev["average"], 6),
            "per_cat": {c: num(ev[c], 6) for c in CATEGORIES}, "videos": videos}


def build_test(make_thumb: bool):
    sub = pd.read_csv(TEST_CSV)
    ref = load_split("test").reset_index(drop=True)
    if len(sub) != len(ref):
        raise SystemExit(f"row count mismatch: {len(sub)} vs {len(ref)}")
    bad = (sub["video_id"].values != ref["video_id"].values) | (sub["question"].values != ref["question"].values)
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} rows of {TEST_CSV.name} do not align with load_split('test')")
    ref["pred"] = sub["parsed_value"].values
    videos = []
    n_fit = 0
    for vid, g in ref.groupby("video_id", sort=True):
        first = g.iloc[0]
        dur, w, h, thumb = video_meta("test", vid, make_thumb)
        is3d = str(first.dim) == "3"
        sc = scene_fit("test", vid) if is3d else None
        n_fit += sc is not None
        qs = [{"row": int(i) + 1, "cat": r.category, "type": r.inference_type, "q": r.question, "prior": text(r.prior),
               "depth": text(r.depth_info), "pred": num(r.pred)} for i, r in zip(g.index, g.itertuples())]
        videos.append({"id": vid, "source": first.video_source, "video_type": first.video_type, "fps": num(first.fps),
                       "dim": int(first.dim), "cats": sorted(set(g.category)), "duration": dur, "size": [w, h],
                       "thumb": thumb, "scene": sc, "questions": qs})
    n3d = sum(v["dim"] == 3 for v in videos)
    print(f"test: {len(ref)} questions, {len(videos)} videos, scene fit {n_fit}/{n3d} 3D videos")
    return {"file": str(TEST_CSV.relative_to(ROOT)), "videos": videos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rethumb", action="store_true", help="regenerate thumbnails that already exist")
    args = ap.parse_args()
    THUMBS.mkdir(parents=True, exist_ok=True)
    data = {"generated": date.today().isoformat(), "val": build_val(args.rethumb), "test": build_test(args.rethumb)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    sizes = [p.stat().st_size for p in THUMBS.glob("*.jpg")]
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB); {len(sizes)} thumbs, {sum(sizes) / 1e6:.2f} MB, max {max(sizes, default=0)} B")


if __name__ == "__main__":
    main()
