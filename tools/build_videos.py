"""Build QuantiPhy-web/data/videos.json (+ first-frame thumbnails) for videos.html.

Reads only existing outputs; never runs the pipeline.

  /home/yumenghe/miniconda3/envs/quantiphy/bin/python QuantiPhy-web/tools/build_videos.py

Inputs (relative to the QuantiPhy repo root):
  outputs/val_s2d_v2_routed.csv                  validation predictions of the current best harness (0.668; column s2d =
                                                 2D-scene value given to Qwen as a second hint)
  outputs/submission_v6.csv                      test predictions shown on the test tab (row-aligned with load_split('test'))
  outputs/scene_fit_eval/c_mesh2/<video>.json    validation scene fits
  outputs/scene2d_eval/s2d_v2/<video>.json       validation 2D layered scenes (+ <video>_bg.jpg, copied to data/scene2d/);
                                                 object inventory, scene graph, rejected layers, constraints, mask hulls
  outputs/scene_fit_eval/test_mesh/<video>.json  test scene fits
  data/QuantiPhy*/...mp4                         only to grab the first frame and duration

The page plays the original videos from media/val and media/test (git-ignored symlinks, see video_info).
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

VAL_CSV = ROOT / "outputs" / "val_s2d_v2_routed.csv"
TEST_CSV = ROOT / "outputs" / "submission_v6.csv"
SCENE_DIRS = {"val": ROOT / "outputs" / "scene_fit_eval" / "c_mesh2",
              "test": ROOT / "outputs" / "scene_fit_eval" / "test_mesh"}
SCENE2D_DIRS = {"val": ROOT / "outputs" / "scene2d_eval" / "s2d_v2"}   # the test tab stays on the submitted predictions
SCENE2D_WEB = WEB / "data" / "scene2d"
OUT = WEB / "data" / "videos.json"
THUMBS = WEB / "data" / "thumbs"
THUMB_W = 240
THUMB_MAX_BYTES = 30_000
HULL_KEYS = 12        # mask hulls kept per layer (evenly over its key frames), to bound videos.json
HULL_EPS = 1.0        # polygon simplification tolerance (px, cv2.approxPolyDP)


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


def inventory(inv):
    """Object inventory (pipeline.objects) as shown on the page: canonical name, aliases, where it is mentioned."""
    if not inv:
        return None
    out = []
    for o in inv.get("objects") or []:
        where = []
        for m in o.get("mentions") or []:
            w = text(m.get("where")) if isinstance(m, dict) else ""
            if w and w not in where:
                where.append(w)
        out.append({"name": text(o.get("name")), "aliases": [text(a) for a in o.get("aliases") or []],
                    "track": o.get("track"), "dyn": o.get("dyn"), "role": o.get("role"), "where": where})
    return out


def graph(sg):
    """Scene graph (pipeline.scenegraph) without the raw LLM text: support nodes, edges, per-object 2D info, same-instance checks."""
    if not sg:
        return None
    edges = [{"from": text(e.get("from")), "to": e.get("to"), "rel": text(e.get("rel")), "conf": num(e.get("conf"), 3),
              "t0": num(e.get("t_start"), 4), "t1": num(e.get("t_end"), 4)} for e in sg.get("edges") or []]
    support = [text(n.get("id")) for n in sg.get("nodes") or [] if n.get("type") == "support"]
    same = [{"a": text(x.get("a")), "b": text(x.get("b")), "same": x.get("same"), "conf": num(x.get("conf"), 3)}
            for x in sg.get("same_instance") or []]
    return {"source": text(sg.get("source")), "support": support, "edges": edges, "same": same,
            "objects_2d": sg.get("objects_2d") or None}


def hulls(L, fps):
    """At most HULL_KEYS mask hulls per layer as [[t, [[x, y], ...]], ...], simplified to HULL_EPS px."""
    H = L.get("hulls") or {}
    keys = sorted(H, key=lambda f: int(f))
    if len(keys) > HULL_KEYS:
        keys = [keys[round(i * (len(keys) - 1) / (HULL_KEYS - 1))] for i in range(HULL_KEYS)]
    t_of = dict(zip(L.get("frames") or [], L.get("t") or []))
    out = []
    for f in keys:
        poly = np.asarray(H[f].get("hull") or [], dtype=np.float32)
        if len(poly) < 3:
            continue
        poly = cv2.approxPolyDP(poly.reshape(-1, 1, 2), HULL_EPS, True).reshape(-1, 2)
        t = t_of.get(int(f), int(f) / fps if fps else None)
        out.append([num(t, 6), [[round(float(x), 1), round(float(y), 1)] for x, y in poly]])
    return out


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
    # 3D scene fits do not carry an inventory / scene graph yet; shown on the page once they do
    return {"file": str(p.relative_to(ROOT)), "f": num(j.get("f"), 5), "plane": plane, "objects": objs, "results": res,
            "inventory": inventory(j.get("inventory") or (j.get("objects_inventory"))),
            "scene_graph": graph(j.get("scene_graph"))}


def scene2d(split: str, vid: str):
    """Compact copy of the 2D layered scene (pipeline/scene2d.py docstring); pixel coordinates of that json's width/height."""
    d = SCENE2D_DIRS.get(split)
    p = d / f"{vid}.json" if d else None
    if p is None or not p.exists():
        return None
    try:
        j = json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None
    bg = None
    bgname = (j.get("background") or {}).get("image")
    if bgname and (d / bgname).exists():
        SCENE2D_WEB.mkdir(parents=True, exist_ok=True)
        out = SCENE2D_WEB / bgname
        out.write_bytes((d / bgname).read_bytes())
        bg = str(out.relative_to(WEB))
    sc = j.get("scale") or {}
    persp = sc.get("perspective") or {}
    scale = {"model": sc.get("model"), "m_per_px_ref": num(sc.get("m_per_px_ref"), 5), "y_ref": num(sc.get("y_ref"), 5),
             "k_per_px": num(sc.get("k_per_px"), 5), "degenerate": bool(sc.get("degenerate")),
             "persp_accepted": bool(persp.get("accepted")), "persp_reason": text(persp.get("reason")),
             "notes": [text(x) for x in sc.get("notes") or []]}
    pr = j.get("prior") or {}
    prior = {k: text(pr.get(k)) for k in ("text", "quantity", "object", "layer", "how")} | {"value_si": num(pr.get("value_si")),
                                                                                         "px": num(pr.get("px"))}

    def par(v):
        if isinstance(v, (list, tuple)):
            return [num(a, 6) for a in v]
        return v if isinstance(v, bool) else num(v, 6)

    layers = []
    for L in j.get("layers") or []:
        segs = [{"t0": num(s.get("t0"), 6), "t1": num(s.get("t1"), 6), "model": s.get("model"),
                 "rms": num(s.get("rms_px"), 3), "p": {k: par(v) for k, v in (s.get("params") or {}).items()}}
                for s in L.get("segments") or []]
        layers.append({"name": text(L.get("name")), "moving": bool(L.get("moving")),
                       "t": [num(t, 6) for t in L.get("t") or []],
                       "box": [[round(float(a), 1) for a in b] for b in L.get("box_px") or []],
                       "c": [[round(float(a), 1) for a in c] for c in L.get("center_px") or []],
                       "visible": [[num(a, 5) for a in iv] for iv in L.get("visible") or []],
                       "extent": {k: (text(v) if k == "src" else num(v, 4)) for k, v in (L.get("extent_px") or {}).items()},
                       "rms": num(L.get("rms_px"), 3), "flags": [text(f) for f in L.get("flags") or []], "segments": segs,
                       "orient": vec(L.get("orientation")), "hulls": hulls(L, j.get("fps")),
                       "constraint": {k: (L.get("constraint") or {}).get(k) for k in ("mode", "surface", "motion_plane", "applied", "note")}})
    return {"file": str(p.relative_to(ROOT)), "fps": num(j.get("fps")), "width": j.get("width"), "height": j.get("height"),
            "n_frames": j.get("n_frames"), "bg": bg, "prior": prior, "scale": scale,
            "gravity_dir": vec(j.get("gravity_dir")), "gravity_source": text(j.get("gravity_source")),
            "inventory": inventory(j.get("objects")), "scene_graph": graph(j.get("scene_graph")),
            "rejected": {text(k): text(v) for k, v in (j.get("rejected_layers") or {}).items()}, "layers": layers}


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


def fourcc(p: Path) -> str:
    cap = cv2.VideoCapture(str(p))
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    return "".join(chr((code >> (8 * k)) & 0xFF) for k in range(4)).strip("\x00").lower()


def video_info(split: str, vid: str) -> dict:
    """Page-relative URL of the original mp4 (media/<split> are git-ignored symlinks to the video dirs of video_path()).

    Browsers only decode H.264 here. All validation mp4s are MPEG-4 Part 2 ('mp4v'); the test set has the same
    video_id at higher resolution, usually in H.264, so the page plays that copy instead and says so.
    "video" is None when no browser-playable copy exists; "orig" is always the original file.
    """
    orig = f"media/{split}/{video_path(split, vid).name}"
    codec = fourcc(video_path(split, vid))
    info = {"orig": orig, "codec": codec, "video": orig if codec in ("avc1", "h264") else None, "video_from": split}
    if info["video"] is None and split == "val" and video_path("test", vid).exists():
        if fourcc(video_path("test", vid)) in ("avc1", "h264"):
            info.update(video=f"media/test/{video_path('test', vid).name}", video_from="test")
    return info


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
               "pred": num(r.pred), "src": r.source, "mra": num(r.mra, 3), "s2d": num(getattr(r, "s2d", None))}
              for r in g.itertuples()]
        is3d = str(first.dim) == "3"
        videos.append({"id": vid, "source": first.video_source, "video_type": first.video_type, "fps": num(first.fps),
                       "dim": int(first.dim), "cats": sorted(set(g.category)), "duration": dur, "size": [w, h],
                       "thumb": thumb, **video_info("val", vid),
                       "scene": scene_fit("val", vid) if is3d else None, "scene2d": None if is3d else scene2d("val", vid),
                       "questions": qs})
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
                       "thumb": thumb, **video_info("test", vid), "scene": sc, "questions": qs})
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
