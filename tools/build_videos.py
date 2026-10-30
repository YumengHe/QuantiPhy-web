"""Build web/data/videos.json (+ first-frame thumbnails) for evaluation.html from one version's validation run.

Reads only that version's outputs; never runs the pipeline. The page lists one row per question; a question is the
unit <video_id>_q<k> (k = its 1-based order among the video's rows of the split table, docs/PER_QUESTION.md).

  python -B web/tools/build_videos.py                      # the current version (pipeline/VERSION)
  python -B web/tools/build_videos.py --val-run <tag>      # a specific version's run

Every version is independent (docs/WORKING_RULES.md): a version is shown only from the artifacts its own run wrote,
read in the format VERSION_FORMAT declares for it. Anything else is an error, never a substitute.

Inputs (relative to the repo root):
  outputs/val_<tag>_answers.csv                       harness CSV (one row per question run, "row" = table row)
  outputs/harness/<tag>/answers/<video>.json           per-video answer records
  outputs/scene_fit_eval/h_<tag>/agent/<unit>.json     saved scenes (per video: one per prior group)
  outputs/scenegraph_<tag>/<unit>.json                 scene graph cache
  outputs/tracks_<tag>/<unit>.json                     track cache (visible intervals per object)
  docs/DEV10.md                                        development question set -> val "dev"
  web/data/runtime/<tag>/overlays.json                 overlays rendered for that version (images beside it)

Overlays: the record web/data/runtime/<tag>/overlays.json lists every overlay of the version. Only the current version
(pipeline/VERSION) may render the ones it lacks, with the current code; building an older version needs every overlay
in its record, with its image on disk.

Every question of the run (every row of its CSV) goes into videos.json; "dev" lists the development question set
(the units of docs/DEV10.md, in that order), each of which must have its row in the CSV.

Layout: {"generated", "val": {"run", "unit", "dev", "videos", "questions"}}. With unit "video"
(the run fits one scene per video and prior group) a video carries its groups' scenes, its scene graph and tracks, and
each question names its video and group. With unit "question" each question carries its own scene / scene graph /
tracks / fit and a video carries none.

The page plays the original videos from media/val (git-ignored symlinks, see video_info).

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
from datetime import date
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pipeline.data import load_split, video_path  # noqa: E402
from pipeline.metrics import mra_row  # noqa: E402
from pipeline.parse import parse_question  # noqa: E402

OUT = WEB / "data" / "videos.json"
THUMBS = WEB / "data" / "thumbs"
THUMB_W = 240
THUMB_MAX_BYTES = 30_000
CURRENT = (ROOT / "pipeline" / "VERSION").read_text().strip()

# The artifact format of each version. A version not listed here cannot be built: add its entry first.
#   motion   key of the physics-law motion record of a scene object ("ballistic" in v1, "physics" from v2 on)
#   paths    object keys whose motion is sampled for the 3D viewer (pipeline.scene_state3d.position); v1 "ballistic"
#            motion is not executable by scene_state3d, so v1 balls are drawn from p0 / v0 / a only
#   unit     processing unit of the run: "video" (one scene per video and prior group) or "question"
VERSION_FORMAT = {
    "v1": {"motion": "ballistic", "paths": ("segments",), "unit": "video"},
    "v2": {"motion": "physics", "paths": ("physics", "segments"), "unit": "video"},
}
MOTION_KEYS = ("ballistic", "physics")
FMT: dict = {}     # VERSION_FORMAT entry of the version being built
DIRS: dict = {}    # artifact dirs of the version being built


def version_format(tag: str) -> dict:
    base = tag.removesuffix("_control")
    if base not in VERSION_FORMAT:
        raise ValueError(f"{tag}: no VERSION_FORMAT entry for {base}; declare its artifact format first")
    return VERSION_FORMAT[base]


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


def read_json(p: Path, what: str):
    if not p.exists():
        raise ValueError(f"{what} missing: {p.relative_to(ROOT)}")
    return json.loads(p.read_text())


def inventory(sg: dict) -> list:
    """Object inventory as shown on the page, from the scene graph's object nodes (the scene json has no separate
    inventory): canonical name, aliases, where it is mentioned."""
    out = []
    for n in sg.get("nodes") or []:
        if n.get("type") != "object":
            continue
        where, aliases = [], []
        for m in n.get("mentions") or []:
            w, a = text(m.get("where")), text(m.get("alias"))
            if w and w not in where:
                where.append(w)
            if a and a not in aliases:
                aliases.append(a)
        out.append({"name": text(n.get("id")), "aliases": aliases, "track": None, "dyn": None, "role": None,
                    "where": where})
    return out


def graph(sg):
    """Scene graph of a scene json without the raw LLM text: support nodes, edges, same-instance checks."""
    edges = [{"from": text(e.get("from")), "to": e.get("to"), "rel": text(e.get("rel")), "conf": num(e.get("conf"), 3),
              "t0": num(e.get("t_start"), 4), "t1": num(e.get("t_end"), 4)} for e in sg.get("edges") or []]
    support = [text(n.get("id")) for n in sg.get("nodes") or [] if n.get("type") == "support"]
    same = [{"a": text(x.get("a")), "b": text(x.get("b")), "same": x.get("same"), "conf": num(x.get("conf"), 3)}
            for x in sg.get("same_instance") or []]
    return {"source": text(sg.get("source")), "support": support, "edges": edges, "same": same}


SUPPORT_RELS = ("on", "in", "held_by", "attached_to", "above", "airborne")   # contact edges are not support


def sgraph(unit: str):
    """The run's scene graph cache outputs/scenegraph_<tag>/<unit>.json as written, for the tree on the page:
    {"nodes": [{"id", "type": object | support | holder}], "edges": [{"from", "to" (null: airborne), "rel", "t0", "t1"}]}
    (t0 / t1 null = the whole video; contact edges dropped). None when the graph has no support relation."""
    g = read_json(DIRS["sgraph"] / f"{unit}.json", "scene graph")
    edges = [{"from": text(e.get("from")), "to": text(e.get("to")) or None, "rel": text(e.get("rel")),
              "t0": num(e.get("t_start"), 4), "t1": num(e.get("t_end"), 4)}
             for e in g.get("edges") or [] if e.get("rel") in SUPPORT_RELS]
    nodes = [{"id": text(n.get("id")), "type": text(n.get("type"))} for n in g.get("nodes") or []]
    return {"nodes": nodes, "edges": edges} if edges else None


def track_intervals(unit: str):
    """{track name: [[t0, t1], ...]} — the frames where the track cache outputs/tracks_<tag>/<unit>.json has a box,
    merged into runs of consecutive frames, in seconds (frame / fps of the cache). None when it has no track."""
    j = read_json(DIRS["tracks"] / f"{unit}.json", "track cache")
    fps = j["info"]["fps"]
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
    return out or None


PATH_DT = 1 / 24     # sampling step (s) of non-parabolic object paths for the 3D viewer


def sampled_path(o: dict):
    """[[t, x, y, z], ...] of an object's centre (pipeline.scene_state3d.position), over the times its fit covers
    (to the last visibility time, at least 2 s)."""
    from pipeline.scene_state3d import position
    ends = [iv[1] for k in ("visible", "out_of_frame") for iv in (o.get(k) or [])]
    T = max(ends + [2.0])
    n = int(round(T / PATH_DT))
    return [[num(i * PATH_DT, 4)] + [num(float(x), 5) for x in position(o, i * PATH_DT)] for i in range(n + 1)]


def motion_events(name: str, o: dict) -> list:
    """An object's motion events ([t, kind] list) from the version's physics-law motion record (FMT["motion"])."""
    other = [k for k in MOTION_KEYS if k != FMT["motion"] and o.get(k)]
    if other:
        raise ValueError(f"object {name!r} has a {other[0]!r} motion record; this version's format is {FMT['motion']!r}")
    return list((o.get(FMT["motion"]) or {}).get("events") or [])


def align(a):
    """t_q alignment check of one answer (scene result quality "align_tq"): {"iou", "dc", "px", "t", "frame"} | None."""
    if a is None:
        return None
    return {"iou": num(a.get("iou"), 6), "dc": num(a.get("dcenter_rel"), 6), "px": num(a.get("dcenter_px"), 3),
            "t": num(a.get("t"), 4), "frame": a.get("frame")}


def question_times(q: str, alignment: dict | None):
    """t_q of a question for the page: the times it names (pipeline.parse), and the time to jump to — the named time
    (midpoint of a [t1, t2] window); for a question that names no time, the frame its alignment check used."""
    times = [float(t) for t in parse_question(q).times]
    jump = (sum(times[:2]) / len(times[:2])) if times else (alignment or {}).get("t")
    return [num(t, 4) for t in times], num(jump, 4)


def fit_row(r: dict) -> dict:
    """One scene result (scene json "results" entry) as shown on the page."""
    qual = r.get("quality") or {}
    return {"q": r.get("question"), "obj": r.get("object"), "pred": num(r.get("pred")),
            "method": r.get("method"), "answer": num(r.get("answer")), "mra": num(r.get("mra"), 3),
            "conf": num(qual.get("conf"), 3), "quality": {k: (num(v, 3) if isinstance(v, float) else v)
                                                          for k, v in qual.items() if k != "conf"} or None}


def scene_view(state: dict, source_path: Path) -> dict:
    """A saved scene as the page draws it, without the per-question results (a shared scene must not show other
    questions' answers)."""
    j = state
    if j.get("f") is None:
        raise ValueError(f"{source_path.relative_to(ROOT)}: scene has no focal length")
    if not j.get("scene_graph"):
        raise ValueError(f"{source_path.relative_to(ROOT)}: scene has no scene graph")
    pl = j.get("plane") or {}
    plane = {k: vec(pl.get(k)) for k in ("normal", "u1", "u2")} | {"dist": num(pl.get("dist"), 5)} if pl else None
    objs = []
    for name, o in (j.get("objects") or {}).items():
        if "visible" not in o:
            raise ValueError(f"{source_path.relative_to(ROOT)}: object {name!r} has no visible intervals")
        vis = o["visible"]
        objs.append({"name": name, "shape": o.get("shape"), "dims": {k: num(v) for k, v in (o.get("dims") or {}).items()},
                     "rms": num(o.get("rms"), 3), "p0": vec(o.get("p0")), "v0": vec(o.get("v0")), "a": vec(o.get("a")),
                     "yaw": num(o.get("yaw"), 5), "t_on": num(o.get("t_on"), 5),
                     # where each dimension comes from: observed / prior / bound (not seen, anti-collapse floor)
                     "dim_source": o.get("dim_source"), "support": o.get("support"), "hang": o.get("hang"),
                     "length": num(o.get("length")), "mode": o.get("mode"),
                     "out_of_frame": o.get("out_of_frame"), "visibility": o.get("visibility"),
                     "overhead": bool(o.get("overhead")), "events": motion_events(name, o)[:12],
                     # motion that is not a single parabola: sample the centre for the viewer
                     "path": sampled_path(o) if any(o.get(k) for k in FMT["paths"]) else None,
                     "visible": [[num(a, 5), num(b, 5)] for a, b in vis] if vis else None})
    sg = j["scene_graph"]
    return {"file": str(source_path.relative_to(ROOT)), "f": num(j["f"], 5), "plane": plane, "objects": objs,
            "main_support": j.get("main_support"), "inventory": inventory(sg), "scene_graph": graph(sg),
            "f_source": text(j.get("f_source")) or None, "f_first": j.get("f_first"),
            "focal_votes": [{"source": text(x.get("source")), "value": num(x.get("value"), 4), "weight": num(x.get("weight"), 3)}
                            for x in j.get("focal_votes") or []] or None}


def video_meta(vid: str, make_thumb: bool):
    """(duration_s, width, height, thumb_rel_path) from the mp4; thumbnail from the first frame."""
    p = video_path("val", vid)
    if not p.exists():
        raise ValueError(f"{vid}: video missing: {p}")
    cap = cv2.VideoCapture(str(p))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    cap.release()
    if not (ok and fps > 0 and n > 0):
        raise ValueError(f"{vid}: video unreadable: {p}")
    out = THUMBS / f"{vid}.jpg"
    if make_thumb or not out.exists():
        th = max(1, round(frame.shape[0] * THUMB_W / frame.shape[1]))
        small = cv2.resize(frame, (THUMB_W, th), interpolation=cv2.INTER_AREA)
        for q in (72, 60, 48, 36, 25):
            ok2, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok2 and len(buf) <= THUMB_MAX_BYTES:
                break
        out.write_bytes(buf.tobytes())
    return num(n / fps, 4), w, h, str(out.relative_to(WEB))


def fourcc(p: Path) -> str:
    cap = cv2.VideoCapture(str(p))
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    return "".join(chr((code >> (8 * k)) & 0xFF) for k in range(4)).strip("\x00").lower()


def video_info(vid: str) -> dict:
    """Page-relative URL of the original mp4 (media/<split> are git-ignored symlinks to the video dirs of video_path()).

    Browsers only decode H.264 here. All validation mp4s are MPEG-4 Part 2 ('mp4v'); the test set has the same
    video_id at higher resolution, usually in H.264, so the page plays that copy instead and says so.
    "video" is None when no browser-playable copy exists; "orig" is always the original file.
    """
    orig = f"media/val/{video_path('val', vid).name}"
    codec = fourcc(video_path("val", vid))
    info = {"orig": orig, "codec": codec, "video": orig if codec in ("avc1", "h264") else None, "video_from": "val"}
    if info["video"] is None and video_path("test", vid).exists() and fourcc(video_path("test", vid)) in ("avc1", "h264"):
        info.update(video=f"media/test/{video_path('test', vid).name}", video_from="test")
    return info


def harness_groups(saved: dict) -> list[dict]:
    """The independent scenes of one video: one per prior group ("groups"), or the video's single scene."""
    if "groups" in saved:
        return [{"group": str(g["group"]), "prior": text(g["prior"]), "depth_info": text(g["depth_info"]),
                 "state": g["scene"]} for g in saved["groups"]]
    context = saved["question_context"][0]
    return [{"group": "0", "prior": text(context["prior"]), "depth_info": text(context["depth_info"]), "state": saved}]


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


OUTSIDE_VIDEO = "asked time outside the recorded video"


class Overlays:
    """The overlay record of one version, web/data/runtime/<tag>/overlays.json: {"<scene_id> <t>": entry}, entry =
    {"t", "scene_id", "renderer", "renderer_version", "renderer_hash", "path" | "unavailable"}."""

    def __init__(self, tag: str):
        self.tag, self.dir = tag, WEB / "data" / "runtime" / tag
        self.file = self.dir / "overlays.json"
        self.render = tag == CURRENT
        if not self.file.exists() and not self.render:
            raise ValueError(f"{tag} is not the current version ({CURRENT}) and has no overlay record {self.file.relative_to(WEB)}")
        self.record = json.loads(self.file.read_text()) if self.file.exists() else {}
        self.added = 0

    def get(self, vid: str, scene: dict, question: str, fps: float, duration: float) -> list[dict]:
        times = list(dict.fromkeys(float(t) for t in parse_question(question).times))[:2] or [0.]
        out = []
        for t in times:
            key = f"{scene['scene_id']} {t:.9g}"
            entry = self.record.get(key)
            if entry is None:
                if not self.render:
                    raise ValueError(f"{vid}: overlay {key} not in {self.file.relative_to(WEB)}; only {CURRENT} renders")
                entry = self.record[key] = self._render(vid, scene, t, fps, duration)
                self.added += 1
            if entry.get("path") and not (WEB / entry["path"]).exists():
                raise ValueError(f"{vid}: overlay image missing: {entry['path']}")
            out.append(entry)
        return out

    def _render(self, vid: str, scene: dict, t: float, fps: float, duration: float) -> dict:
        """Render saved state only; no fitting, assembly, model requests or GT inputs."""
        from pipeline.data import Video
        from pipeline.scene_state3d import render_overlay
        renderer_hash = hashlib.sha256((ROOT / "pipeline" / "scene_state3d.py").read_bytes()).hexdigest()[:16]
        entry = {"t": t, "scene_id": scene["scene_id"], "renderer": "scene_state3d", "renderer_version": CURRENT,
                 "renderer_hash": renderer_hash}
        if not math.isfinite(t) or t < 0 or t > duration + 1 / fps:
            entry["unavailable"] = OUTSIDE_VIDEO
            return entry
        frame = round(t * fps)
        identity = hashlib.sha256(f"{scene['scene_id']}:{renderer_hash}:{t:.9g}".encode()).hexdigest()[:24]
        path = self.dir / vid / f"{identity}_f{frame}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            video = Video(video_path("val", vid), fps=fps)
            try:
                image = render_overlay(scene, t, frame=video.frame_at(t))
            finally:
                video.close()
            if image is None or not cv2.imwrite(str(path), image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 86]):
                raise ValueError("runtime produced no image")
            entry["path"] = str(path.relative_to(WEB))
        except Exception as exc:   # noqa: BLE001 - the renderer's failure is this version's result, shown on the page
            entry["unavailable"] = f"{type(exc).__name__}: {exc}"[:240]
        return entry

    def save(self):
        if self.added:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.file.write_text(json.dumps(self.record, ensure_ascii=False, indent=1, sort_keys=True))


def manifest_units() -> pd.DataFrame:
    """The validation table with each row's unit id (<video_id>_q<k>, k = order of the row within its video)."""
    manifest = load_split("val").reset_index(drop=True)
    manifest["k"] = manifest.groupby("video_id", sort=False).cumcount() + 1
    manifest["unit"] = [unit_id(v, int(k)) for v, k in zip(manifest["video_id"], manifest["k"])]
    return manifest


def harness_rows(frame: pd.DataFrame, manifest: pd.DataFrame, units: list[str]) -> pd.DataFrame:
    """The rows of `units` from a harness CSV, in that order, each bound to its manifest row by row identity and
    checked against it; labels always come from the manifest. A unit without its row in the CSV is an error."""
    if "row" not in frame:
        raise ValueError("harness CSV lacks row identities")
    by_unit = manifest.set_index("unit")
    missing = [u for u in units if u not in by_unit.index]
    if missing:
        raise ValueError(f"units not in the validation table: {missing}")
    rows = pd.to_numeric(frame["row"], errors="raise").astype(int)
    if rows.duplicated().any():
        raise ValueError("harness CSV has duplicated row identities")
    frame = frame.set_axis(rows.to_numpy())
    absent = [u for u in units if int(by_unit.index.get_loc(u)) not in frame.index]
    if absent:
        raise ValueError(f"units without an answer row in the harness CSV: {absent}")
    out = []
    for u in units:
        i = int(by_unit.index.get_loc(u))
        r, m = frame.loc[i].to_dict(), manifest.loc[i]
        for key in ("video_id", "question", "prior", "depth_info"):
            if text(r[key]) != text(m[key]):
                raise ValueError(f"{u}: harness CSV {key} differs from the validation table")
        for key in ("answer", "category", "dim", "fps", "video_type", "video_source", "inference_type", "k"):
            r[key] = m[key]
        r["row"], r["unit"] = i, u
        out.append(r)
    return pd.DataFrame(out)


def unit_id(vid: str, k: int) -> str:
    """The unit of one question: <video_id>_q<k>, k = its 1-based order among the video's rows of the split table."""
    return f"{vid}_q{k}"


def dev_units() -> list[str]:
    """The development question set: the unit ids in the first column of the table in docs/DEV10.md."""
    units = re.findall(r"^\|\s*(\S+_q\d+)\s*\|", (ROOT / "docs" / "DEV10.md").read_text(), re.M)
    if not units or len(set(units)) != len(units):
        raise ValueError(f"docs/DEV10.md: no or duplicated unit ids: {units}")
    return units


def own_result(state: dict, row: dict) -> dict | None:
    """This question's entry of a scene's "results" (exactly one, or none when the scene gave it no value)."""
    results = [r for r in state.get("results") or [] if r.get("question") == row["question"]
               and (r.get("row") is None or r["row"] == row["row"])]
    if len(results) > 1:
        raise ValueError(f"row {row['row']}: {len(results)} scene results for one question")
    return results[0] if results else None


def build_harness_val(tag: str, make_thumb: bool = False) -> dict:
    """Display a completed harness CSV and its saved answers/groups without re-running anything."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError(f"invalid harness run tag: {tag!r}")
    FMT.clear()
    FMT.update(version_format(tag))
    unit_kind = FMT["unit"]
    DIRS.update(scenes=ROOT / "outputs/scene_fit_eval" / f"h_{tag}" / "agent", sgraph=ROOT / "outputs" / f"scenegraph_{tag}",
                tracks=ROOT / "outputs" / f"tracks_{tag}")
    csv_path = ROOT / "outputs" / f"val_{tag}_answers.csv"
    if not csv_path.exists():
        raise ValueError(f"harness CSV missing: {csv_path}")
    frame, manifest = pd.read_csv(csv_path), manifest_units()
    if "row" not in frame:
        raise ValueError("harness CSV lacks row identities")
    ran = set(pd.to_numeric(frame["row"], errors="raise").astype(int))
    dev = dev_units()
    missing = [u for u in dev if u not in set(manifest.loc[sorted(ran & set(manifest.index)), "unit"])]
    if missing:
        raise ValueError(f"docs/DEV10.md units not in {csv_path.name}: {missing}")
    df = harness_rows(frame, manifest, [u for i, u in enumerate(manifest["unit"]) if i in ran])
    df["mra"] = [mra_row(p, g) for p, g in zip(df["pred"], df["answer"])]
    overlays = Overlays(tag)
    videos, questions = [], []
    for vid, rows in df.groupby("video_id", sort=True):
        first = rows.iloc[0]
        record = read_json(ROOT / "outputs/harness" / tag / "answers" / f"{vid}.json", "answer record")
        version = record["version"]
        if version["pipeline"] != tag.removesuffix("_control"):
            raise ValueError(f"{vid}: answer artifact version {version['pipeline']} differs from requested {tag}")
        duration, width, height, thumb = video_meta(vid, make_thumb)
        video = {"id": vid, "source": first.video_source, "video_type": first.video_type,
                 "fps": num(first.fps), "duration": duration, "size": [width, height], "thumb": thumb,
                 **video_info(vid)}
        groups = []
        if unit_kind == "video":
            scene_path = DIRS["scenes"] / f"{vid}.json"
            if scene_path.exists():
                groups = harness_groups(json.loads(scene_path.read_text()))
            elif record["execution_status"] != "failed":
                raise ValueError(f"{vid}: scene missing for a run that did not fail: {scene_path.relative_to(ROOT)}")
            video |= {"groups": [{"group": g["group"], "scene_id": g["state"]["scene_id"],
                                  "scene": scene_view(g["state"], scene_path)} for g in groups],
                      "run_at": version["time"], "sgraph": sgraph(vid), "tracks": track_intervals(vid)}
        else:
            video["run_at"] = version["time"]
        for row in rows.to_dict("records"):
            unit = row["unit"]
            query = {"id": unit, "video": vid, "k": int(row["k"]), "row": int(row["row"]),
                     "cat": row["category"], "type": row["inference_type"],
                     "q": row["question"], "prior": text(row["prior"]), "depth": text(row["depth_info"]),
                     "answer": num(row["answer"]), "pred": num(row["pred"]), "mra": num(row["mra"], 6),
                     "src": text(row["source"]), "conf": num(row["conf"], 6), "status": text(row["status"]),
                     "method": row["method"], "scene_id": text(row["scene_id"])}
            if unit_kind == "video":
                answer, group, error = bind_harness_question(row, record["answers"], groups)
                state = group["state"] if group else {}
                result = own_result(state, row) if group else None
            else:
                scene_path = DIRS["scenes"] / f"{unit}.json"
                state = read_json(scene_path, "question scene")
                group, error = {"group": None, "state": state}, None
                answers = [a for a in record["answers"] if a.get("row") == row["row"]]
                if len(answers) != 1:
                    raise ValueError(f"{unit}: {len(answers)} answer records")
                answer = answers[0]
                result = own_result(state, row)
            quality = (result or {}).get("quality") or {}
            alignment = align(quality.get("align_tq"))
            mid = (result or {}).get("measurement_id")
            if mid is not None and mid not in (state.get("measurements") or {}):
                raise ValueError(f"{unit}: measurement {mid} not in its scene")
            query |= {"group": group["group"] if group else None, "binding_error": error,
                      "measurement": answer["measurement"] if group else None, "measurement_id": mid,
                      "measurement_record": state["measurements"][mid] if mid is not None else None,
                      "provenance": (answer or {}).get("provenance"), "quality": quality, "align": alignment,
                      "fit": fit_row(result) if result else None,
                      "overlays": overlays.get(vid, state, row["question"], float(first.fps), duration) if group else [],
                      "execution_status": record["execution_status"]}
            if unit_kind == "question":
                query |= {"scene": scene_view(state, scene_path), "sgraph": sgraph(unit), "tracks": track_intervals(unit)}
            query["tq"], query["tjump"] = question_times(query["q"], alignment)
            questions.append(query)
        videos.append(video)
    overlays.save()
    print(f"harness {tag}: {len(questions)} questions ({len(dev)} DEV10), {len(videos)} videos; "
          f"{overlays.added} overlays rendered")
    return {"run": tag, "unit": unit_kind, "dev": dev, "videos": videos, "questions": questions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-run", default=CURRENT,
                    help="completed harness validation run tag (default: pipeline/VERSION, %(default)s)")
    ap.add_argument("--rethumb", action="store_true", help="regenerate thumbnails that already exist")
    args = ap.parse_args()
    THUMBS.mkdir(parents=True, exist_ok=True)
    data = {"generated": date.today().isoformat(), "val": build_harness_val(args.val_run, args.rethumb)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    sizes = [p.stat().st_size for p in THUMBS.glob("*.jpg")]
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB); {len(sizes)} thumbs, {sum(sizes) / 1e6:.2f} MB, max {max(sizes, default=0)} B")


if __name__ == "__main__":
    main()
