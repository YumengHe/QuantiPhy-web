"""Build web/data/videos.json (+ first-frame thumbnails) for videos.html.

Reads only existing outputs; never runs the pipeline.

  python -B web/tools/build_videos.py                      # validation from the current code version's harness run
  python -B web/tools/build_videos.py --val-run <tag>      # a specific harness run tag
  python -B web/tools/build_videos.py --test-sub <csv> [--test-fit <dir>] [--test-s2d <dir>] [--test-route <csv>]
                                      [--test-audit2d <csv>] [--test-tracks <kind>]

Validation (inputs relative to the repo root, <tag> defaults to pipeline/VERSION):
  outputs/val_<tag>_answers.csv                       completed harness CSV
  outputs/harness/<tag>/answers/<video>.json           per-video answer records
  outputs/scene_fit_eval/h_<tag>/agent/, outputs/scene2d_eval/h_<tag>/agent/   saved independent group scenes
Runtime overlays are offline renderings written under data/runtime/<tag>/.

Test: without --test-sub the existing test block of videos.json is kept. With it, the predictions are row-aligned
with load_split('test'); per-video scenes are written to data/scenes/test/<video>.json and fetched when a card opens.

The page plays the original videos from media/val and media/test (git-ignored symlinks, see video_info).

Note: the "v0" keys below (and o.v0 / p.v0 in videos.js) are the initial-velocity field of the scene json written by
the pipeline, not a version label; do not rename or remove them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pipeline.data import cache_root, load_split, video_path  # noqa: E402
from pipeline.metrics import CATEGORIES, evaluate, mra_row  # noqa: E402
from pipeline.parse import parse_question  # noqa: E402

TEST_CSV: Path | None = None   # --test-sub
SCENE_DIRS: dict = {"val": None, "test": None}     # 3D scene-fit dirs (val: from the run tag, test: --test-fit)
SCENE2D_DIRS: dict = {"val": None, "test": None}   # 2D layered-scene dirs (val: from the run tag, test: --test-s2d)
SCENES_TEST = WEB / "data" / "scenes" / "test"   # one json per test video, fetched by the page when a card opens
BG_MAX_W = 960                                   # background images wider than this are shrunk (test is up to 2560 px)
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


def track_intervals(split: str, vid: str) -> dict:
    """{track name: [[t0, t1], ...]} — the frames where the track cache (outputs/tracks[_test]/<video>.json) has a box,
    merged into runs of consecutive frames, in seconds (frame / fps of the cache)."""
    p = cache_root(TRACKS_KIND.get(split, "tracks"), split) / f"{vid}.json"
    if not p.exists():
        return {}
    try:
        j = json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return {}
    fps = (j.get("info") or {}).get("fps") or 30.0
    out = {}
    for name, tr in (j.get("tracks") or {}).items():
        fr = sorted(int(k) for k, b in (tr.get("boxes") or {}).items() if b is not None)
        runs = []
        for f in fr:
            if runs and f == runs[-1][1] + 1:
                runs[-1][1] = f
            else:
                runs.append([f, f])
        out[name] = [[num(a / fps, 5), num(b / fps, 5)] for a, b in runs]
    return out


PATH_DT = 1 / 24     # sampling step (s) of non-parabolic object paths for the 3D viewer


def sampled_path(o: dict, j: dict):
    """[[t, x, y, z], ...] of an object's centre from pipeline.scene_fit._json_pos (ballistic replay or motion pieces),
    over the times its fit covers (to the last results / visibility time, at least 2 s)."""
    from pipeline.scene_fit import _json_pos
    ends = [iv[1] for k in ("visible", "out_of_frame") for iv in (o.get(k) or [])]
    T = max(ends + [2.0])
    n = int(round(T / PATH_DT))
    out = []
    for i in range(n + 1):
        t = i * PATH_DT
        try:
            P = _json_pos(o, t)
        except Exception:   # noqa: BLE001 - a broken entry just falls back to the parabola on the page
            return None
        out.append([num(t, 4)] + [num(float(x), 5) for x in P])
    return out


def align(a):
    """t_q alignment check of one answer (scene json "align_tq"): a number (IoU) or a dict -> {"iou", "dc"} | None."""
    if a is None:
        return None
    if isinstance(a, (int, float)):
        return {"iou": num(a, 3), "dc": None, "px": None, "t": None, "frame": None}
    if isinstance(a, dict):
        # 3D: iou, dcenter_rel (centre offset / box size); 2D: iou, dev_rel, dev_px
        iou = next((a[k] for k in ("iou", "IoU", "box_iou") if a.get(k) is not None), None)
        dc = next((a[k] for k in ("dcenter_rel", "dev_rel") if a.get(k) is not None), None)
        px = next((a[k] for k in ("dev_px", "dcenter_px", "center_px") if a.get(k) is not None), None)
        if iou is None and dc is None and px is None:
            return None
        return {"iou": num(iou, 6), "dc": num(dc, 6), "px": num(px, 3), "t": num(a.get("t"), 4), "frame": a.get("frame")}
    return None


def question_times(q: str, meta: dict | None, scene: dict | None, is3d: bool):
    """t_q of a question for the page: the times it names (pipeline.parse), and the time to jump to — the named time
    (midpoint of a [t1, t2] window), else the middle of the asked object's visible interval, else None."""
    try:
        times = [float(t) for t in parse_question(q).times]
    except Exception:   # noqa: BLE001 - a question the parser cannot read simply gets no time
        times = []
    jump = (sum(times[:2]) / len(times[:2])) if times else None
    if jump is None and meta and (meta.get("align") or {}).get("t") is not None:   # the frame the alignment check used
        jump = meta["align"]["t"]
    if jump is None and meta and scene:
        obj = meta.get("obj")
        items = scene.get("objects") if is3d else scene.get("layers")
        it = next((o for o in items or [] if o.get("name") == obj), None)
        vis = (it or {}).get("visible") or []
        if vis:
            jump = (vis[0][0] + vis[-1][1]) / 2
    return [num(t, 4) for t in times], num(jump, 4)


def scene_fit(split: str, vid: str, *, state=None, source_path=None):
    p = source_path or SCENE_DIRS[split] / f"{vid}.json"
    if state is None:
        if not p.exists():
            return None
        try:
            state = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None
    j = state
    pl = j.get("plane") or {}
    plane = {k: vec(pl.get(k)) for k in ("normal", "u1", "u2")} | {"dist": num(pl.get("dist"), 5)} if pl else None
    objs = []
    for name, o in (j.get("objects") or {}).items():
        objs.append({"name": name, "shape": o.get("shape"), "dims": {k: num(v) for k, v in (o.get("dims") or {}).items()},
                     "rms": num(o.get("rms"), 3), "p0": vec(o.get("p0")), "v0": vec(o.get("v0")), "a": vec(o.get("a")),
                     "yaw": num(o.get("yaw"), 5), "t_on": num(o.get("t_on"), 5),
                     # where each dimension comes from: observed / prior / bound (not seen, anti-collapse floor)
                     "dim_source": o.get("dim_source"), "support": o.get("support"), "hang": o.get("hang"),
                     "length": num(o.get("length")), "mode": o.get("mode"),
                     "out_of_frame": o.get("out_of_frame"), "visibility": o.get("visibility"),
                     "overhead": bool(o.get("overhead")), "events": ((o.get("ballistic") or {}).get("events") or [])[:12],
                     # bounce model / motion pieces are not a single parabola: sample the centre for the viewer
                     "path": sampled_path(o, j) if (o.get("ballistic") or o.get("segments")) else None})
    res = []
    for r in j.get("results") or []:
        qual = r.get("quality") or {}
        res.append({"q": r.get("question"), "obj": r.get("object"), "pred": num(r.get("pred")),
                    "method": r.get("method"), "answer": num(r.get("answer")) if split == "val" else None,
                    "mra": num(r.get("mra"), 3) if split == "val" else None,
                    "conf": num(qual.get("conf"), 3), "quality": {k: (num(v, 3) if isinstance(v, float) else v)
                                                                  for k, v in qual.items() if k != "conf"} or None})
    # per question: the fitted object and the t_q alignment check (quality.align_tq), when the scene json has them
    q_meta = {r.get("question"): {"obj": r.get("object"), "align": align((r.get("quality") or {}).get("align_tq"))}
              for r in j.get("results") or [] if r.get("question")}
    sg = j.get("scene_graph")
    inv = inventory(j.get("inventory") or j.get("objects_inventory"))
    if inv is None and sg:   # no inventory in the json: the scene graph's object nodes carry the same names and mentions
        inv = inventory({"objects": [{"name": n.get("id"), "aliases": list(dict.fromkeys(
            m.get("alias") for m in n.get("mentions") or [] if isinstance(m, dict) and m.get("alias"))),
            "mentions": n.get("mentions") or []} for n in sg.get("nodes") or [] if n.get("type") == "object"]})
    # visible intervals: the scene json's own "visible" when it has one, else the object's track (its name, else an alias)
    tracks = track_intervals(split, vid)
    aliases = {o["name"]: o["aliases"] for o in inv or []}
    for o, (name, raw) in zip(objs, (j.get("objects") or {}).items()):
        vis = raw.get("visible")
        if vis is None:
            key = next((k for k in [name, *aliases.get(name, [])] if k in tracks), None)
            vis = tracks.get(key) if key else None
        o["visible"] = [[num(a, 5), num(b, 5)] for a, b in vis] if vis else None
    return {"file": str(p.relative_to(ROOT)), "f": num(j.get("f"), 5), "plane": plane, "objects": objs, "results": res,
            "main_support": j.get("main_support"), "inventory": inv, "scene_graph": graph(sg), "q_meta": q_meta,
            "count_check": j.get("count_check"),
            "f_source": text(j.get("f_source")) or None, "f_first": j.get("f_first"),
            "focal_votes": [{"source": text(x.get("source")), "value": num(x.get("value"), 4), "weight": num(x.get("weight"), 3)}
                            for x in j.get("focal_votes") or []] or None}


def layer_temporal(L: dict) -> dict:
    """stable extent (rigid_constant layers: one w x h for every frame), occluded / merged frames as indices into the
    layer's frame list, and the temporal_consistency numbers (scene2d fields; empty when absent)."""
    se = L.get("stable_extent_px") or {}
    idx = {f: i for i, f in enumerate(L.get("frames") or [])}
    out = {}
    if se.get("rigid_constant") and se.get("w") and se.get("h"):
        out["stable"] = [num(se["w"], 5), num(se["h"], 5)]
    occ = [idx[f] for f in L.get("occluded") or [] if f in idx]
    mrg = [idx[f] for f in L.get("merged") or [] if f in idx]
    if occ:
        out["occ"] = occ
    if mrg:
        out["merged"] = mrg
    tc = L.get("temporal_consistency")
    if tc:
        out["tc"] = {k: (num(v, 4) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in tc.items()}
    return out


def scene2d_checks(j: dict) -> dict:
    """s2d_v5 consistency checks, compacted: relation checks per track candidate, size-order checks, frame-level
    reference measurements and the critic's applied / rejected issues (each only when present)."""
    out = {}
    rc = []
    for r in j.get("relation_checks") or []:
        rc.append({"from": text(r.get("from")), "candidates": [
            {"cache": text(c.get("cache")), "ok": c.get("ok"),
             "checks": [{"rel": text(x.get("rel")), "to": text(x.get("to")), "ok": x.get("ok"), "why": text(x.get("why"))}
                        for x in c.get("checks") or []]} for c in r.get("candidates") or []]})
    if rc:
        out["relation_checks"] = rc
    sc = [{"smaller": text(x.get("smaller")), "larger": text(x.get("larger")), "ok": x.get("ok"),
           "px": [num(a, 4) for a in x.get("px") or []], "why": text(x.get("why"))} for x in j.get("size_checks") or []]
    if sc:
        out["size_checks"] = sc
    rm = {text(k): {"px": num(v.get("px"), 4), "conf": num(v.get("conf"), 3), "relation": text(v.get("relation")),
                    "frame": v.get("frame"), "p0": vec(v.get("p0")), "p1": vec(v.get("p1")), "spread": num(v.get("spread"), 3)}
          for k, v in (j.get("reference_measurements") or {}).items() if isinstance(v, dict)}
    if rm:
        out["refmeasure"] = rm
    cq = j.get("critique") or {}
    if cq.get("applied") or cq.get("rejected"):
        out["critique"] = {"applied": [json.dumps(a, ensure_ascii=False)[:200] for a in cq.get("applied") or []],
                           "rejected": [json.dumps(a, ensure_ascii=False)[:200] for a in cq.get("rejected") or []],
                           "rerun": text(cq.get("rerun"))}
    return out


def scene2d(split: str, vid: str, *, state=None, source_path=None, asset_tag=None):
    """Compact copy of the 2D layered scene (pipeline/scene2d.py docstring); pixel coordinates of that json's width/height."""
    d = SCENE2D_DIRS.get(split)
    p = source_path or (d / f"{vid}.json" if d else None)
    if state is None:
        if p is None or not p.exists():
            return None
        try:
            state = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None
    j = state
    bg = None
    bgname = (j.get("background") or {}).get("image")
    if bgname and (d / bgname).exists():
        # val keeps data/scene2d/<name>; test (same video ids, higher resolution) goes to data/scene2d/test/, shrunk
        out_dir = SCENE2D_WEB / asset_tag if asset_tag else SCENE2D_WEB if split == "val" else SCENE2D_WEB / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / bgname
        img = cv2.imread(str(d / bgname))
        if img is not None and img.shape[1] > BG_MAX_W:
            img = cv2.resize(img, (BG_MAX_W, round(img.shape[0] * BG_MAX_W / img.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        else:
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

    # size quantities the questions ask of each layer, so the page labels a layer with those sizes only
    size_q = {}
    for r in j.get("results") or []:
        q = r.get("quantity")
        if q in ("length", "width", "height", "diameter") and r.get("layer"):
            size_q.setdefault(r["layer"], [])
            if q not in size_q[r["layer"]]:
                size_q[r["layer"]].append(q)
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
                       "size_q": size_q.get(L.get("name"), []), "orient": vec(L.get("orientation")), "hulls": hulls(L, j.get("fps")),
                       # cross-frame layer correspondence: constant extent of a rigid layer, occluded /
                       # merged frames (as indices into "t"), size statistics
                       **layer_temporal(L),
                       "constraint": {k: (L.get("constraint") or {}).get(k) for k in ("mode", "surface", "motion_plane", "applied", "note")}})
    return {"file": str(p.relative_to(ROOT)), "fps": num(j.get("fps")), "width": j.get("width"), "height": j.get("height"),
            "n_frames": j.get("n_frames"), "bg": bg, "prior": prior, "scale": scale,
            "gravity_dir": vec(j.get("gravity_dir")), "gravity_source": text(j.get("gravity_source")),
            "inventory": inventory(j.get("objects")), "scene_graph": graph(j.get("scene_graph")),
            "rejected": {text(k): text(v) for k, v in (j.get("rejected_layers") or {}).items()}, "layers": layers,
            "count_check": j.get("count_check"),
            "q_meta": {r.get("question"): {"obj": r.get("layer"), "align": align(r.get("align_tq"))}
                       for r in j.get("results") or [] if r.get("question")},
            **scene2d_checks(j)}


LA = ZoneInfo("America/Los_Angeles")


def run_at(split: str, vid: str, csv: Path) -> dict:
    """Last time this video was run, in Los Angeles time: the mtime of its scene json (3D scene fit / 2D layered
    scene); only a video without a scene json falls back to the per-question csv."""
    scenes = [p for p in (d / f"{vid}.json" for d in (SCENE_DIRS.get(split), SCENE2D_DIRS.get(split)) if d) if p.exists()]
    ts = max(p.stat().st_mtime for p in scenes or [csv])
    t = datetime.fromtimestamp(ts, LA)
    return {"run_at": t.strftime("%Y-%m-%d %H:%M %Z"), "run_at_iso": t.isoformat(timespec="seconds")}


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


def video_info(split: str, vid: str, *, allow_test_copy=True) -> dict:
    """Page-relative URL of the original mp4 (media/<split> are git-ignored symlinks to the video dirs of video_path()).

    Browsers only decode H.264 here. All validation mp4s are MPEG-4 Part 2 ('mp4v'); the test set has the same
    video_id at higher resolution, usually in H.264, so the page plays that copy instead and says so.
    "video" is None when no browser-playable copy exists; "orig" is always the original file.
    """
    orig = f"media/{split}/{video_path(split, vid).name}"
    codec = fourcc(video_path(split, vid))
    info = {"orig": orig, "codec": codec, "video": orig if codec in ("avc1", "h264") else None, "video_from": split}
    if allow_test_copy and info["video"] is None and split == "val" and video_path("test", vid).exists():
        if fourcc(video_path("test", vid)) in ("avc1", "h264"):
            info.update(video=f"media/test/{video_path('test', vid).name}", video_from="test")
    return info


def harness_groups(saved: dict) -> list[dict]:
    """Never substitute an aggregate's last scene for a missing group scene."""
    if saved.get("groups"):
        return [{"group": str(g.get("group", i)), "prior": text(g.get("prior")),
                 "depth_info": text(g.get("depth_info")), "state": g.get("scene") or {}}
                for i, g in enumerate(saved["groups"])]
    context = saved.get("question_context") or [{}]
    return [{"group": "0", "prior": text(context[0].get("prior", saved.get("constraint_prior"))),
             "depth_info": text(context[0].get("depth_info", saved.get("depth_info"))), "state": saved}]


def bind_harness_question(row: dict, answers: list[dict], groups: list[dict]) -> tuple[dict | None, dict | None, str | None]:
    """Bind row identity and scene_id exactly, including same text under different priors."""
    candidates = [a for a in answers if a.get("row") == row.get("row") and a.get("question") == row.get("question")
                  and text(a.get("prior")) == text(row.get("prior"))]
    if len(candidates) != 1:
        return None, None, "answer identity missing or duplicated"
    answer = candidates[0]
    scene_id = answer.get("scene_id")
    if not scene_id or text(row.get("scene_id")) != scene_id:
        return answer, None, "scene_id missing or differs between CSV and answer record"
    matches = [g for g in groups if g["state"].get("scene_id") == scene_id]
    if len(matches) != 1:
        return answer, None, "scene_id missing or duplicated among independent group scenes"
    group = matches[0]
    contexts = [c for c in group["state"].get("question_context") or []
                if c.get("row") == row.get("row") and c.get("question") == row.get("question")
                and text(c.get("prior")) == text(row.get("prior"))
                and text(c.get("depth_info")) == text(row.get("depth_info"))]
    if len(contexts) != 1:
        return answer, None, "row/prior/depth identity not unique in the matched scene"
    return answer, group, None


def runtime_overlays(tag: str, vid: str, scene: dict, dim: int, question: str, fps: float, duration: float) -> list[dict]:
    """Render saved state only; no fitting, assembly, model requests or GT inputs."""
    from pipeline.data import Video
    try:
        times = list(parse_question(question).times or [])
    except (TypeError, ValueError):
        times = []
    times = list(dict.fromkeys(float(t) for t in times))[:2] or [0.]
    if dim == 2:
        from pipeline.scene2d import render_overlay
        sources = ("scene2d.py", "scene_state2d.py")
    else:
        from pipeline.scene_state3d import render_overlay
        sources = ("scene_state3d.py",)
    renderer_hash = hashlib.sha256(b"".join((ROOT / "pipeline" / name).read_bytes() for name in sources)).hexdigest()[:16]
    renderer_version = (ROOT / "pipeline/VERSION").read_text().strip()
    out_dir = WEB / "data/runtime" / tag / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs, video = [], None
    try:
        for t in times:
            entry = {"t": t, "scene_id": scene.get("scene_id"), "renderer": f"scene_state{dim}d",
                     "renderer_version": renderer_version, "renderer_hash": renderer_hash}
            if not math.isfinite(t) or t < 0 or (duration and t > duration + 1 / max(fps, 1)):
                entry["unavailable"] = "asked time outside the recorded video"
            else:
                frame = round(t * fps)
                identity = hashlib.sha256(f"{scene['scene_id']}:{renderer_hash}:{t:.9g}".encode()).hexdigest()[:24]
                path = out_dir / f"{identity}_f{frame}.jpg"
                try:
                    if not path.exists():
                        if video is None:
                            video = Video(video_path("val", vid), fps=fps)
                        image = render_overlay(scene, t, frame=video.frame_at(t))
                        if image is None or not cv2.imwrite(str(path), image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 86]):
                            raise ValueError("runtime produced no image")
                    entry["path"] = str(path.relative_to(WEB))
                except Exception as exc:
                    entry["unavailable"] = f"{type(exc).__name__}: {exc}"[:240]
            outputs.append(entry)
    finally:
        if video is not None:
            video.close()
    return outputs


def critique_summary(scene: dict) -> dict:
    keep = ("iter", "diagnosis", "layer", "observations", "tool_calls", "results", "accepted", "why", "execution_status",
            "decision_status", "budget_exhausted", "rejected_calls", "deferred_calls", "failure_stage",
            "model_policy_results", "control_before", "control_after")
    return {"status": {k: v for k, v in (scene.get("agent_status") or {}).items() if k != "self_check"},
            "iterations": [{k: i[k] for k in keep if k in i} for i in (scene.get("critique") or {}).get("iterations") or []]}


def validate_harness_csv(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Require the complete ordered manifest; labels always come from that manifest."""
    frame, manifest = frame.reset_index(drop=True).copy(), manifest.reset_index(drop=True)
    if len(frame) != len(manifest) or "row" not in frame:
        raise ValueError("harness CSV is incomplete or lacks row identities")
    rows = pd.to_numeric(frame["row"], errors="coerce")
    if not np.array_equal(rows.to_numpy(), np.arange(len(manifest))):
        raise ValueError("harness CSV row identities differ from manifest order")
    for key in ("video_id", "question", "prior", "depth_info"):
        actual = frame[key].map(text) if key in frame else pd.Series([""] * len(frame))
        expected = manifest[key].map(text) if key in manifest else pd.Series([""] * len(manifest))
        if not actual.equals(expected):
            raise ValueError(f"harness CSV {key} identities differ from manifest")
    for key in ("answer", "category", "dim", "fps", "video_type", "video_source", "inference_type"):
        if key in manifest:
            frame[key] = manifest[key]
    return frame


def build_harness_val(tag: str, make_thumb: bool = False) -> dict:
    """Display a completed harness CSV and its saved answers/groups without re-running anything."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError(f"invalid harness run tag: {tag!r}")
    csv_path = ROOT / "outputs" / f"val_{tag}_answers.csv"
    if not csv_path.exists():
        raise ValueError(f"completed full harness CSV is unavailable: {csv_path}")
    df = validate_harness_csv(pd.read_csv(csv_path), load_split("val"))
    df["mra"] = [mra_row(p, g) for p, g in zip(df["pred"], df["answer"])]
    ev = evaluate(df)
    SCENE_DIRS["val"] = ROOT / "outputs/scene_fit_eval" / f"h_{tag}"
    SCENE2D_DIRS["val"] = ROOT / "outputs/scene2d_eval" / f"h_{tag}"
    TRACKS_KIND["val"] = f"tracks_{tag}"
    videos = []
    for vid, rows in df.groupby("video_id", sort=True):
        first = rows.iloc[0]
        dim = int(first.dim)
        answer_path = ROOT / "outputs/harness" / tag / "answers" / f"{vid}.json"
        record = json.loads(answer_path.read_text()) if answer_path.exists() else {}
        version = record.get("version") or {}
        if record and version.get("pipeline") != tag.removesuffix("_control"):
            raise ValueError(f"{vid}: answer artifact version differs from requested {tag}")
        scene_path = (SCENE_DIRS if dim == 3 else SCENE2D_DIRS)["val"] / "agent" / f"{vid}.json"
        saved = json.loads(scene_path.read_text()) if scene_path.exists() else {}
        groups = harness_groups(saved) if saved else []
        duration, width, height, thumb = video_meta("val", vid, make_thumb)
        compact = []
        for group in groups:
            state = group["state"]
            has_state = bool(state.get("scene_id"))
            view = (scene_fit("val", vid, state=state, source_path=scene_path) if dim == 3 else
                    scene2d("val", vid, state=state, source_path=scene_path, asset_tag=tag)) if has_state else None
            compact.append({"group": group["group"], "prior": group["prior"], "depth_info": group["depth_info"],
                            "scene_id": state.get("scene_id"), "version": state.get("version") or version,
                            "scene": view if dim == 3 else None, "scene2d": view if dim == 2 else None,
                            "critique": critique_summary(state), "display_projection": "approximate_legacy_viewer"})
        questions = []
        for row in rows.to_dict("records"):
            answer, group, error = bind_harness_question(row, record.get("answers") or [], groups)
            answer = answer or {}
            state = group["state"] if group else {}
            results = [r for r in state.get("results") or [] if r.get("question") == row["question"]
                       and (r.get("row") is None or r["row"] == row["row"])]
            result = results[0] if len(results) == 1 else {}
            quality = result.get("quality") or {}
            alignment = align(quality.get("align_tq") or result.get("align_tq"))
            diagnostic = (answer.get("measurement") or quality.get("measurement")) if group else None
            overlays = runtime_overlays(tag, vid, state, dim, row["question"], float(first.fps), duration) if group else []
            query = {"harness": True, "row": int(row["row"]), "cat": row["category"], "type": row["inference_type"],
                     "q": row["question"], "prior": text(row.get("prior")), "depth": text(row.get("depth_info")),
                     "answer": num(row.get("answer")), "pred": num(row.get("pred")), "mra": num(row["mra"], 6),
                     "src": text(answer.get("source", row.get("source"))), "conf": num(answer.get("conf", row.get("conf")), 6),
                     "scene_conf": num(answer.get("scene_conf", row.get("scene_conf")), 6),
                     "confidence_kind": text(answer.get("confidence_kind", row.get("confidence_kind"))) or None,
                     "status": text(answer.get("status", row.get("status"))), "method": answer.get("method", row.get("method")),
                     "scene_id": answer.get("scene_id") or text(row.get("scene_id")), "group": group["group"] if group else None,
                     "binding_status": "matched" if group else "unavailable", "binding_error": error,
                     "measurement": diagnostic, "measurement_id": result.get("measurement_id"),
                     "measurement_record": (state.get("measurements") or {}).get(result.get("measurement_id")),
                     "provenance": answer.get("provenance"), "quality": quality, "align": alignment,
                     "runtime_overlays": overlays, "version": version,
                     "execution_status": record.get("execution_status", "unavailable")}
            query["tq"], query["tjump"] = question_times(query["q"], {"align": alignment}, None, dim == 3)
            questions.append(query)
        # No aggregate geometry: the browser must choose a complete independent group explicitly.
        videos.append({"id": vid, "harness": True, "source": first.video_source, "video_type": first.video_type,
                       "fps": num(first.fps), "dim": dim, "cats": sorted(set(rows.category)), "duration": duration,
                       "size": [width, height], "thumb": thumb, **video_info("val", vid), "groups": compact,
                       "scene": None, "scene2d": None, "has_scene": dim == 3 and bool(compact),
                       "has_scene2d": dim == 2 and bool(compact), "questions": questions,
                       "execution_status": record.get("execution_status", "unavailable"),
                       "run_at": version.get("time"), "eval": None, "objtab": None, "stree": None})
    print(f"harness {tag}: {len(df)} questions, {len(videos)} videos, macro {ev['average']:.6f}")
    return {"file": str(csv_path.relative_to(ROOT)), "macro": num(ev["average"], 8),
            "per_cat": {c: num(ev[c], 8) for c in CATEGORIES}, "videos": videos}


def replace_validation(existing: dict, validation: dict) -> dict:
    """Keep the existing test block and all other top-level data when no --test-sub is given."""
    if "test" not in existing:
        raise ValueError("no --test-sub given and data/videos.json has no test block to keep")
    return {**existing, "generated": date.today().isoformat(), "val": validation}


TRACKS_KIND = {"val": "tracks", "test": "tracks"}   # track cache kind for the visible intervals (--test-tracks)
TEST_ROUTE: Path | None = None     # --test-route: per-question source of the test predictions
TEST_AUDIT2D: Path | None = None   # --test-audit2d: 2D test audit *_pred.csv for the no-answer list
REASON_ZH = {"needs-track": "缓存里没有这个物体的轨迹", "no-layer": "没有图层", "no-reference-layer": "参照物没有图层",
             "unsupported": "这类题拟合不支持", "object-not-fitted": "物体没拟合上", "reference-not-fitted": "参照物没拟合上",
             "no-fit": "视频没有场景拟合"}


def _reason_zh(method: str) -> str:
    m = str(method or "")
    if m in REASON_ZH:
        return REASON_ZH[m]
    if m.endswith("(no scale)"):
        return "没有尺度（prior 定不了 m/px）"
    if m.endswith("-failed"):
        return m[:-7] + " 量不出来"
    return m or "—"


def test_noanswer(ref: pd.DataFrame, route_csv: Path, audit2d: Path) -> dict:
    """Test questions the scene gave no usable value for: 2D from the latest scene2d test audit (pred empty, its method
    and notes as the reason), 3D from the test scene-fit dir (no result value, or no fit for the video). The submitted
    value comes from the submission; its source (vlm / geo / s2d) from the routing csv when it has a source column."""
    src = {}
    if route_csv is not None and route_csv.exists():
        r = pd.read_csv(route_csv)
        if "source" in r.columns:
            src = {(a, b): c for a, b, c in zip(r.video_id, r.question, r.source)}
    a2d = {}
    if audit2d is not None and audit2d.exists():
        a = pd.read_csv(audit2d)
        for v, q, pr, m, n in zip(a.video_id, a.question, a.pred, a["method"], a.get("notes", pd.Series([""] * len(a)))):
            a2d[(v, q)] = (pr, m, n)
    fits = {}
    d3 = SCENE_DIRS.get("test")
    rows, totals = [], {c: 0 for c in CATEGORIES}
    for i, r in ref.iterrows():
        totals[r.category] = totals.get(r.category, 0) + 1
        key = (r.video_id, r.question)
        if str(r.dim) == "2":
            if key not in a2d:
                continue
            pr, m, n = a2d[key]
            if isinstance(pr, float) and math.isfinite(pr):
                continue
            reason, detail = _reason_zh(m), text(n)[:220]
        else:
            if r.video_id not in fits:
                p = d3 / f"{r.video_id}.json" if d3 else None
                try:
                    fits[r.video_id] = ({x["question"]: (x.get("pred"), x.get("method")) for x in json.load(open(p)).get("results", [])}
                                        if p is not None and p.exists() else None)
                except (OSError, json.JSONDecodeError):
                    fits[r.video_id] = None
            fv = fits[r.video_id]
            if fv is None:
                reason, detail = REASON_ZH["no-fit"], ""
            else:
                pr, m = fv.get(r.question, (None, "unsupported"))
                if pr is not None and not (isinstance(pr, float) and math.isnan(pr)):
                    continue
                reason, detail = _reason_zh(m), text(m)
        rows.append({"vid": r.video_id, "row": int(i) + 1, "cat": r.category, "q": r.question, "pred": num(r.pred),
                     "src": src.get(key), "reason": reason, "detail": detail})
    by_cat = {c: [sum(x["cat"] == c for x in rows), totals.get(c, 0)] for c in CATEGORIES}
    reasons = {}
    for x in rows:
        reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
    print(f"test no-answer: {len(rows)} / {len(ref)} (" + ", ".join(f"{c} {a}/{b}" for c, (a, b) in by_cat.items()) + ")")
    return {"rows": rows, "by_cat": by_cat, "by_reason": sorted(reasons.items(), key=lambda kv: -kv[1]),
            "total": len(ref)}


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
    n_fit = n_2d = 0
    for vid, g in ref.groupby("video_id", sort=True):
        first = g.iloc[0]
        dur, w, h, thumb = video_meta("test", vid, make_thumb)
        is3d = str(first.dim) == "3"
        sc = scene_fit("test", vid) if is3d else None
        s2 = None if is3d else scene2d("test", vid)
        n_fit += sc is not None
        n_2d += s2 is not None
        ref_file = None
        if sc or s2:   # kept out of videos.json (568 videos): written per video and fetched when the card opens
            SCENES_TEST.mkdir(parents=True, exist_ok=True)
            out = SCENES_TEST / f"{vid}.json"
            out.write_text(json.dumps({"scene": sc, "scene2d": s2}, ensure_ascii=False, separators=(",", ":")))
            ref_file = str(out.relative_to(WEB))
        qs = [{"row": int(i) + 1, "cat": r.category, "type": r.inference_type, "q": r.question, "prior": text(r.prior),
               "depth": text(r.depth_info), "pred": num(r.pred)} for i, r in zip(g.index, g.itertuples())]
        for q in qs:
            q["tq"], q["tjump"] = question_times(q["q"], ((sc or s2) or {}).get("q_meta", {}).get(q["q"]), sc or s2, is3d)
        videos.append({"id": vid, "source": first.video_source, "video_type": first.video_type, "fps": num(first.fps),
                       "dim": int(first.dim), "cats": sorted(set(g.category)), "duration": dur, "size": [w, h],
                       "thumb": thumb, **video_info("test", vid), "scene": None, "scene2d": None, "scene_ref": ref_file,
                       "has_scene": sc is not None, "has_scene2d": s2 is not None, **run_at("test", vid, TEST_CSV),
                       "questions": qs})
    n3d = sum(v["dim"] == 3 for v in videos)
    print(f"test: {len(ref)} questions, {len(videos)} videos, scene fit {n_fit}/{n3d} 3D videos, "
          f"2D layered scene {n_2d}/{len(videos) - n3d} 2D videos")
    return {"file": str(TEST_CSV), "videos": videos, "noans": test_noanswer(ref, TEST_ROUTE, TEST_AUDIT2D)}


def main():
    global TEST_CSV, TEST_ROUTE, TEST_AUDIT2D
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-run", default=(ROOT / "pipeline" / "VERSION").read_text().strip(),
                    help="completed harness validation run tag (default: pipeline/VERSION, %(default)s)")
    ap.add_argument("--rethumb", action="store_true", help="regenerate thumbnails that already exist")
    ap.add_argument("--test-sub", type=Path, help="test predictions csv; without it the existing test block is kept")
    ap.add_argument("--test-route", type=Path, help="its per-question csv with a source column")
    ap.add_argument("--test-fit", type=Path, help="test scene-fit dir")
    ap.add_argument("--test-s2d", type=Path, help="test 2D layered-scene dir")
    ap.add_argument("--test-tracks", help="track cache kind for the test visible intervals (default: tracks)")
    ap.add_argument("--test-audit2d", type=Path, help="2D test audit *_pred.csv for the no-answer list")
    args = ap.parse_args()
    existing = json.loads(OUT.read_text()) if not args.test_sub and OUT.exists() else {}
    if not args.test_sub and "test" not in existing:
        ap.error("--test-sub is required when data/videos.json has no test block to keep")
    TEST_CSV, TEST_ROUTE, TEST_AUDIT2D = args.test_sub, args.test_route, args.test_audit2d
    if args.test_fit:
        SCENE_DIRS["test"] = args.test_fit.resolve()
    if args.test_s2d:
        SCENE2D_DIRS["test"] = args.test_s2d.resolve()
    if args.test_tracks:
        TRACKS_KIND["test"] = args.test_tracks
    THUMBS.mkdir(parents=True, exist_ok=True)
    validation = build_harness_val(args.val_run, args.rethumb)
    data = replace_validation(existing, validation) if not args.test_sub else {
        "generated": date.today().isoformat(), "val": validation, "test": build_test(args.rethumb)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    sizes = [p.stat().st_size for p in THUMBS.glob("*.jpg")]
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB); {len(sizes)} thumbs, {sum(sizes) / 1e6:.2f} MB, max {max(sizes, default=0)} B")


if __name__ == "__main__":
    main()
