"""Build web/data/videos.json (+ first-frame thumbnails) for evaluation.html from validation runs.

Reads only the runs' outputs; never runs the pipeline. The page lists one row per question; a question is the
unit <video_id>_q<k> (k = its 1-based order among the video's rows of the split table, docs/PER_QUESTION.md).

  python -B web/tools/build_videos.py --val-runs v1
  python -B web/tools/build_videos.py --val-runs v4=outputs/val_v4_answers_dev10.csv v1
  python -B web/tools/build_videos.py --val-runs v5gpt=outputs/val_v5gpt_answers_dev10.csv v1

Runs are listed in order of preference: each question is shown from the first run whose CSV has its row, and every
value of that question (pred, MRA, source, scene, overlays, tracks, scene graph) comes from that run alone. The page
labels each question with its version. Every version is independent (docs/WORKING_RULES.md): a run is read only from
the artifacts it wrote, in the format VERSION_FORMAT declares for it. A run missing an artifact of a question it has
is an error, never a substitute.

Inputs of a run <tag> (relative to the repo root):
  outputs/val_<tag>_answers.csv (or the CSV given as <tag>=<path>)
                                                       harness CSV (one row per question run, "row" = table row)
  outputs/harness/<tag>/answers/<key>.json             answer records ("version"."pipeline" = <tag>)
  outputs/scene_fit_eval/h_<tag>/agent/<key>.json      saved scenes (unit "video": one per prior group)
  outputs/scenegraph_<tag>/<key>.json                  scene graph cache
  outputs/tracks_<tag>/<key>.json                      track cache (visible intervals per object)
  outputs/objects_<tag>/<key>.json                     object inventory (unit "question": identity checked)
  web/data/runtime/<tag>/overlays.json                 overlays rendered for that version (images beside it)
A run tag is a version, optionally with the model backend ("v5gpt": v5 run with --model-backend gpt) and "_control";
its format is the version's. <key> is the video for unit "video" and the question's unit <video_id>_q<k> for unit "question"; a unit-question file
whose "unit" (or video) is not that question's raises. Answer records of unit "question" have schema "unit-1", with
the question's timing. Also docs/DEV10.md, the development question set -> val "dev"; each of its units must be in some run.

Overlays: the record web/data/runtime/<tag>/overlays.json lists every overlay of the version. Only the current version
(pipeline/VERSION) may render the ones it lacks, with the current code; building an older version needs every overlay
in its record, with its image on disk.

Layout: {"generated", "val": {"runs": [{"version", "unit", "csv"}], "dev", "videos", "questions"}}. Each question
names its "version". A video carries "runs": {version: {"run_at", ...}}; with unit "video" (the run fits one scene
per video and prior group) that entry also holds the groups' scenes, the scene graph and tracks, and each question
names its group. With unit "question" each question carries its own scene / scene graph / tracks / fit.

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
#   decide   how a scene value becomes the answer: "conf" (a numeric confidence: CSV / scene result "conf"),
#            "checks" (named pass/fail checks: answer record "checks" / "failed" / "status", CSV "failed"; no conf) or
#            "agree" (v5, docs/ANSWER_AGREEMENT.md section 5: the named checks, plus every measurement of the asked
#            quantity in the scene result -- "measurements", "chosen", "selection" -- the checks "agree" {chi2, dof,
#            bound, values} and "constraints" {failed_ids}, and the answer's "constraint_residuals" with sigma_stated /
#            sigma_pred / bound / pass / testable; each scene object has a "motion_law")
#   steps    the answer record's timing steps, in pipeline order (unit "question" only)
#   identity the track cache has "meta"."identity" {name: {"candidates", "chosen", "untracked"}} (the track identity
#            judge: static / verify / class_size / presence / border); each scene object has "dim_bounds"
#            {dimension: [lo, hi | null]} for exactly its "bound" dimensions; a prior-scaled scene's "scale" names its
#            source, and a prior-extent scale its "extent" (the frames and depth observations it was measured on)
STEPS_V4 = ("objects", "scenegraph_grid", "track", "scenegraph_pf", "observe", "scene", "agent", "answer")
STEPS_V5 = ("objects", "scenegraph_grid", "observe", "track", "measure", "scenegraph_pf", "ground", "scene", "agent",
            "answer")
VERSION_FORMAT = {
    "v1": {"motion": "ballistic", "paths": ("segments",), "unit": "video", "decide": "conf"},
    "v2": {"motion": "physics", "paths": ("physics", "segments"), "unit": "video", "decide": "conf"},
    "v3": {"motion": "physics", "paths": ("physics", "segments"), "unit": "video", "decide": "conf"},
    "v4": {"motion": "physics", "paths": ("physics", "segments"), "unit": "question", "decide": "checks",
           "steps": STEPS_V4},
    "v5": {"motion": "physics", "paths": ("physics", "segments"), "unit": "question", "decide": "agree",
           "steps": STEPS_V5, "identity": True},
}
CHECK_SOURCES = ("scene", "measurement")   # sources of a resolved answer under "checks"; unresolved: "unresolved"
MOTION_KEYS = ("ballistic", "physics")
FMT: dict = {}     # VERSION_FORMAT entry of the version being built
DIRS: dict = {}    # artifact dirs of the version being built
AT = {"where": ""}  # "<version> <video or unit>" being read, for error messages


TAG_RE = re.compile(r"(v\d+)(gpt)?(_control)?")


def base_version(tag: str) -> str:
    """The version of a run tag: "v5gpt" -> "v5" (a backend suffix), "v4_control" -> "v4"."""
    m = TAG_RE.fullmatch(tag)
    if not m:
        raise ValueError(f"{tag}: not a run tag <version>[gpt][_control]")
    return m.group(1)


def version_format(tag: str) -> dict:
    base = base_version(tag)
    if base not in VERSION_FORMAT:
        raise ValueError(f"{tag}: no VERSION_FORMAT entry for {base}; declare its artifact format first")
    return VERSION_FORMAT[base]


def run_label(tag: str, pipeline: str) -> bool:
    """Is `pipeline` (an answer record's version.pipeline) the label of run `tag`? The harness writes the tag, with
    "-dirty" when the run's working tree had uncommitted changes."""
    t = tag.removesuffix("_control")
    return pipeline in (t, f"{t}-dirty")


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


def req(d: dict, key: str, what: str):
    """d[key], a field every artifact of the version's format has; its absence is a format mismatch, not a default."""
    if key not in d:
        raise ValueError(f"{AT['where']}: {what} has no {key!r} field")
    return d[key]


def read_json(p: Path, what: str):
    if not p.exists():
        raise ValueError(f"{what} missing: {p.relative_to(ROOT)}")
    return json.loads(p.read_text())


SUPPORT_RELS = ("on", "in", "held_by", "attached_to", "above", "airborne")   # contact edges are not support


# the field naming the video in each per-unit artifact
VIDEO_FIELD = {"scene graph": "video", "track cache": "video_id", "object inventory": "video_id",
               "question scene": "video_id", "answer record": "video_id"}


def unit_json(kind: str, key: str, what: str):
    """The artifact DIRS[kind]/<key>.json; for unit "question" its "unit" and video fields must name that question."""
    j = read_json(DIRS[kind] / f"{key}.json", what)
    if FMT["unit"] == "question":
        vid = key.rsplit("_q", 1)[0]
        if req(j, "unit", what) != key or req(j, VIDEO_FIELD[what], what) != vid:
            raise ValueError(f"{AT['where']}: {what} {DIRS[kind].name}/{key}.json is for unit {j.get('unit')!r} "
                             f"/ video {j.get(VIDEO_FIELD[what])!r}, not {key}")
    return j


def sgraph(key: str):
    """The run's scene graph cache outputs/scenegraph_<tag>/<key>.json as written, for the tree on the page:
    {"nodes": [{"id", "type": object | support | holder}], "edges": [{"from", "to" (null: airborne), "rel", "t0", "t1"}]}
    (t0 / t1 null = the whole video; contact edges dropped). None when the graph has no support relation."""
    g = unit_json("sgraph", key, "scene graph")
    edges = [{"from": text(req(e, "from", "edge")), "to": text(req(e, "to", "edge")) or None,   # null "to": airborne
              "rel": text(e["rel"]), "t0": num(req(e, "t_start", "edge"), 4), "t1": num(req(e, "t_end", "edge"), 4)}
             for e in req(g, "edges", "scene graph cache") if req(e, "rel", "edge") in SUPPORT_RELS]
    nodes = [{"id": text(req(n, "id", "scene graph node")), "type": text(req(n, "type", "scene graph node"))}
             for n in req(g, "nodes", "scene graph cache")]
    return {"nodes": nodes, "edges": edges} if edges else None


def track_intervals(key: str):
    """{track name: [[t0, t1], ...]} — the frames where the track cache outputs/tracks_<tag>/<key>.json has a box,
    merged into runs of consecutive frames, in seconds (frame / fps of the cache). None when it has no track."""
    j = unit_json("tracks", key, "track cache")
    fps = j["info"]["fps"]
    out = {}
    for name, tr in req(j, "tracks", "track cache").items():
        fr = sorted(int(k) for k, b in req(tr, "boxes", f"track {name!r}").items() if b is not None)
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
    ends = [iv[1] for k in ("visible", "out_of_frame") for iv in req(o, k, "scene object")]
    T = max(ends + [2.0])
    n = int(round(T / PATH_DT))
    return [[num(i * PATH_DT, 4)] + [num(float(x), 5) for x in position(o, i * PATH_DT)] for i in range(n + 1)]


def motion_events(name: str, o: dict) -> list:
    """An object's motion events ([t, kind] list) from the version's physics-law motion record (FMT["motion"])."""
    other = [k for k in MOTION_KEYS if k != FMT["motion"] and o.get(k)]
    if other:
        raise ValueError(f"object {name!r} has a {other[0]!r} motion record; this version's format is {FMT['motion']!r}")
    record = req(o, FMT["motion"], f"object {name!r}")    # null: no physics-law motion
    return [] if record is None else list(req(record, "events", f"object {name!r} motion record"))


def motion_law(name: str, o: dict):
    """The motion law of a scene object (v5 "motion_law"): {"model", "rotation": {"rel", "target", "pieces",
    "circle_candidate"} | None, "phases": [[t, state, moving]] | None}; None: no physics-law motion record."""
    law = req(o, "motion_law", f"scene object {name!r}")
    if law is None:
        return None
    rot = law.get("rotation")
    if rot is not None:
        rot = {"rel": text(req(rot, "rel", "rotation law")), "target": text(req(rot, "target", "rotation law")) or None,
               "pieces": list(req(rot, "pieces", "rotation law")),
               "circle_candidate": bool(req(rot, "circle_candidate", "rotation law"))}
    phases = [[num(p[0], 4), text(p[1]), bool(p[2])] for p in law["phases"]] if law.get("phases") else None
    return {"model": text(req(law, "model", f"object {name!r} motion law")), "rotation": rot, "phases": phases}


def align(a):
    """t_q alignment check of one answer (scene result quality "align_tq"): {"iou", "dc", "t", "frame"} | None."""
    if a is None:
        return None
    return {"iou": num(req(a, "iou", "align_tq"), 6), "dc": num(req(a, "dcenter_rel", "align_tq"), 6),
            "t": num(req(a, "t", "align_tq"), 4),
            "frame": req(a, "frame", "align_tq")}


def question_times(q: str, alignment: dict | None):
    """t_q of a question for the page: the times it names (pipeline.parse), and the time to jump to — the named time
    (midpoint of a [t1, t2] window); for a question that names no time, the frame its alignment check used."""
    times = [float(t) for t in parse_question(q).times]
    jump = (sum(times[:2]) / len(times[:2])) if times else (alignment or {}).get("t")
    return [num(t, 4) for t in times], num(jump, 4)


def fit_row(r: dict) -> dict:
    """One scene result (scene json "results" entry) as shown on the page."""
    qual = req(r, "quality", "scene result")
    out = {"q": req(r, "question", "scene result"), "obj": req(r, "object", "scene result"),
           "pred": num(req(r, "pred", "scene result")), "method": req(r, "method", "scene result"),
           "quality": {k: (num(v, 3) if isinstance(v, float) else v) for k, v in qual.items() if k != "conf"} or None}
    if FMT["decide"] == "conf":
        out["conf"] = num(req(qual, "conf", "scene result quality"), 3)
    elif "conf" in qual:
        raise ValueError(f"{AT['where']}: scene result quality has a 'conf' field; this version decides by named checks")
    return out


def answer_checks(answer: dict, row: dict) -> dict:
    """The named pass/fail checks of a unit's answer (answer record "answer"), for versions that decide by checks:
    {"checks": [{"name", "pass", "reason"}], "failed": [names], "scene_value" (unresolved only, when kept)}.
    The CSV row's status / source / failed / pred must agree with the record."""
    if "conf" in answer or "conf" in row:
        raise ValueError(f"{AT['where']}: answer has a 'conf' field; this version decides by named checks")
    checks = req(answer, "checks", "answer")
    failed = list(req(answer, "failed", "answer"))
    status = req(answer, "status", "answer")
    out = []
    for name, c in checks.items():
        ok = req(c, "pass", f"check {name!r}")
        if not isinstance(ok, bool):
            raise ValueError(f"{AT['where']}: check {name!r} pass = {ok!r}")
        out.append({"name": name, "pass": ok, "reason": text(req(c, "reason", f"check {name!r}"))}
                   | ({"which": text(c["which"])} if c.get("which") else {}))
    if failed != [c["name"] for c in out if not c["pass"]]:
        raise ValueError(f"{AT['where']}: failed {failed} differs from the checks that did not pass")
    csv_failed = [x for x in text(req(row, "failed", "harness CSV row")).split(";") if x]
    if csv_failed != failed:
        raise ValueError(f"{AT['where']}: CSV failed {csv_failed} differs from the answer record's {failed}")
    if text(row["status"]) != status or text(row["source"]) != text(req(answer, "source", "answer")):
        raise ValueError(f"{AT['where']}: CSV status / source differs from the answer record")
    pred = num(row["pred"])
    if status == "resolved":
        if failed or pred is None or row["source"] not in CHECK_SOURCES:
            raise ValueError(f"{AT['where']}: resolved answer with failed {failed}, pred {pred}, source {row['source']!r}")
    elif status == "unresolved":
        if not failed or pred is not None or row["source"] != "unresolved":
            raise ValueError(f"{AT['where']}: unresolved answer with failed {failed}, pred {pred}, source {row['source']!r}")
    else:
        raise ValueError(f"{AT['where']}: answer status {status!r}")
    res = {"checks": out, "failed": failed}
    if status == "unresolved" and answer.get("scene_value") is not None:   # kept for inspection, not the answer
        res["scene_value"] = num(answer["scene_value"])
    return res


def measurement_row(m: dict) -> dict:
    """One measurement of the asked quantity (scene result "measurements", docs/ANSWER_AGREEMENT.md section 5)."""
    what = f"measurement {m.get('name')!r}"
    adm = req(m, "admissible", what)
    ok = req(adm, "pass", f"{what} admissible")
    if not isinstance(ok, bool):
        raise ValueError(f"{AT['where']}: {what} admissible pass = {ok!r}")
    checks = req(m, "checks", what)
    failed = list(req(m, "failed", what))
    for k in failed:
        if k not in checks or checks[k]["pass"] is not False:
            raise ValueError(f"{AT['where']}: {what} lists {k!r} as failed, its checks do not")
    parts = req(m, "sigma_parts", what) or {}
    terms = {k: num(v, 3) for k, v in parts.items() if k in ("track", "depth", "align", "fit_se", "stated")}
    lst = lambda v: [num(x, 3) for x in v] if isinstance(v, list) else ([num(v, 3)] if v is not None else None)
    dof = parts.get("dof")
    inputs = req(m, "inputs", what) or {}
    return {"name": text(req(m, "name", what)), "method": text(req(m, "method", what)),
            "value": num(req(m, "value", what)), "unit": text(req(m, "unit", what)), "sigma": num(req(m, "sigma", what), 3),
            "terms": terms or None, "scale": lst(parts.get("scale")), "q": lst(parts.get("q")),
            "dof": [int(x) for x in dof] if isinstance(dof, list) else ([int(dof)] if dof is not None else None),
            "shared": list(req(m, "shared", what)), "admissible": ok, "why": text(req(adm, "reason", f"{what} admissible")),
            "failed": [{"name": k, "reason": text(req(checks[k], "reason", f"{what} check {k!r}"))}
                       | ({"which": text(checks[k]["which"])} if checks[k].get("which") else {}) for k in failed],
            "checks": [{"name": k, "pass": bool(req(c, "pass", f"{what} check {k!r}"))} for k, c in checks.items()],
            "window": [num(x, 4) for x in inputs["window"]] if inputs.get("window") else None,
            "piece": [num(max(x, 0.0), 4) if x > -1e8 else None for x in inputs["piece"]] if inputs.get("piece") else None,
            "n": inputs.get("n"), "state": text(inputs.get("state")) or None}


def agreement_view(result: dict, answer: dict, status: str) -> dict:
    """The measurements of the asked quantity (scene result), the chosen one, and the checks "agree" / "constraints"
    of the answer with each constraint residual (answer "measurement"."constraint_residuals")."""
    ms = [measurement_row(m) for m in req(result, "measurements", "scene result")]
    names = [m["name"] for m in ms]
    if len(set(names)) != len(names):
        raise ValueError(f"{AT['where']}: measurement names repeat: {names}")
    chosen = req(result, "chosen", "scene result")
    req(result, "selection", "scene result")
    if chosen is not None:
        m = next((x for x in ms if x["name"] == chosen), None)
        if m is None or not m["admissible"]:
            raise ValueError(f"{AT['where']}: chosen {chosen!r} is not an admissible measurement")
        kept = num(answer.get("pred") if status == "resolved" else answer.get("scene_value"))
        if kept != m["value"]:
            raise ValueError(f"{AT['where']}: the answer's value {kept} is not the chosen measurement's {m['value']}")
    checks = req(answer, "checks", "answer")
    ag, cs = req(checks, "agree", "answer checks"), req(checks, "constraints", "answer checks")
    agree = {"pass": req(ag, "pass", "agree"), "chi2": num(req(ag, "chi2", "agree"), 4), "dof": req(ag, "dof", "agree"),
             "bound": num(req(ag, "bound", "agree"), 4),
             "values": [[k, num(v[0]), num(v[1], 3)] for k, v in req(ag, "values", "agree").items()]}
    res = req(req(answer, "measurement", "answer"), "constraint_residuals", "answer measurement")
    rows = []
    for c in res:
        w = f"constraint {c.get('id')!r}"
        obs, pred = req(c, "observed", w), req(c, "predicted", w)
        rows.append({"id": text(req(c, "id", w)), "kind": text(req(c, "kind", w)), "units": text(req(c, "units", w)),
                     "observed": num(obs), "predicted": num(pred),
                     "diff": num(abs(float(pred) - float(obs)), 3) if pred is not None else None,
                     "sigma_stated": num(req(c, "sigma_stated", w), 3), "sigma_pred": num(req(c, "sigma_pred", w), 3),
                     "bound": num(req(c, "bound", w), 3), "pass": req(c, "pass", w), "testable": req(c, "testable", w)})
    ids = {r["id"] for r in rows}
    failed_ids = list(req(cs, "failed_ids", "constraints"))
    if not set(failed_ids) <= ids:
        raise ValueError(f"{AT['where']}: constraints failed_ids {failed_ids} not among the residuals {sorted(ids)}")
    return {"measurements": ms, "chosen": chosen, "agree": agree, "constraints": rows}


def dim_bounds(name: str, o: dict):
    """{dimension: [lo, hi | None]} of the object's "bound" dimensions (scene_fit rule D2a: an unobserved dimension is
    an interval, never a value); None for versions without the field. Its keys must be exactly the bound dimensions."""
    if not FMT.get("identity"):
        return None
    b = req(o, "dim_bounds", f"scene object {name!r}")
    bound = {k for k, v in req(o, "dim_source", f"scene object {name!r}").items() if v == "bound"}
    if set(b) != bound:
        raise ValueError(f"{AT['where']}: object {name!r} dim_bounds {sorted(b)} differ from its bound dims {sorted(bound)}")
    return {k: [num(lo), None if hi is None else num(hi)] for k, (lo, hi) in b.items()}


def scale_view(state: dict):
    """The scale of a prior-scaled scene: {"s", "source", "object", "prior", "how", "extent"} ("extent" for a
    prior-extent scale: the frames and depth observations the nominal extent was measured on); None without one."""
    if not FMT.get("identity"):
        return None
    sc = req(state, "scale", "scene")
    if sc is None:
        return None
    src = text(req(sc, "source", "scale"))
    ext = sc.get("extent")
    if src == "prior_object_extent" and not ext:
        raise ValueError(f"{AT['where']}: prior-extent scale without its 'extent'")
    return {"s": num(req(sc, "s", "scale")), "source": src, "object": text(sc.get("object")) or None,
            "prior": text(req(sc, "prior", "scale")), "how": text(req(sc, "how", "scale")), "extent": text(ext) or None}


def dims_check(state: dict) -> list[dict]:
    """plausible.dims per object (pipeline.critic, rule D2a): the largest dimension as an interval [lo, hi | None]
    (a bound dimension contributes its dim_bounds interval) against the class range widened by critic.SIZE_MARGIN;
    fails only when the whole interval is outside. Recomputed from the saved scene and checked against the scene's
    own checks.plausible.parts.dims."""
    from pipeline.critic import SIZE_MARGIN, _class_range, _size_interval
    rows = []
    for name, o in req(state, "objects", "scene").items():
        lo, hi, _ = _size_interval(o)
        rng = _class_range(name)
        tested = rng is not None and not (hi is not None and hi <= 0)
        ok = not tested or not (lo > rng[1] * SIZE_MARGIN or (hi is not None and hi < rng[0] / SIZE_MARGIN))
        rows.append({"obj": name, "lo": num(lo), "hi": None if hi is None else num(hi),
                     "range": [num(rng[0] / SIZE_MARGIN), num(rng[1] * SIZE_MARGIN)] if tested else None, "pass": ok})
    part = req(req(req(req(state, "checks", "scene"), "plausible", "scene checks"), "parts", "plausible"), "dims", "plausible parts")
    if req(part, "pass", "plausible.dims") != all(r["pass"] for r in rows):
        raise ValueError(f"{AT['where']}: plausible.dims {part['pass']} differs from the recomputed intervals {rows}")
    return rows


def identity_view(key: str) -> dict:
    """The track identity judge of every object (track cache "meta"."identity"): its candidates in the order they were
    judged, with the outcome and why; the chosen one, or why the object is untracked."""
    j = unit_json("tracks", key, "track cache")
    out = {}
    for name, v in req(req(j, "meta", "track cache"), "identity", "track cache meta").items():
        w = f"identity of {name!r}"
        cands, chosen, untracked = req(v, "candidates", w), req(v, "chosen", w), req(v, "untracked", w)
        if (chosen is None) == (untracked is None):
            raise ValueError(f"{AT['where']}: {w} has chosen {chosen!r} and untracked {untracked!r}")
        if chosen is not None and req(cands[chosen], "outcome", w) != "accepted":
            raise ValueError(f"{AT['where']}: {w} chose candidate {chosen}, outcome {cands[chosen]['outcome']!r}")
        rows = []
        for c in cands:
            pres, mv = c.get("presence"), c.get("moves")
            rows.append({"source": text(req(c, "source", w)), "frame": req(c, "frame", w),
                         "box": [round(float(x)) for x in c["box"]] if req(c, "box", w) else None,
                         "n": req(c, "n_frames", w), "px": num(mv["px"], 3) if mv else None,
                         "moves": bool(mv["moves"]) if mv else None,
                         "votes": list(c["votes"]) if c.get("votes") else None,
                         "size": [{"ref": text(x["ref"]), "ratio": num(x["ratio"], 3),
                                   "bounds": [num(b, 3) if b is not None else None for b in x["bounds"]],
                                   "view": text(x["view"]).split(" (")[0], "pass": bool(x["pass"])}
                                  for x in c.get("class_size") or []],
                         "span": [pres["first"], pres["last"]] if pres and pres.get("pass") else None,
                         "border": num(c["border_frac"], 2) if c.get("border_frac") is not None else None,
                         "outcome": text(req(c, "outcome", w)), "why": text(c.get("why")) or None})
        out[name] = {"candidates": rows, "chosen": chosen, "untracked": text(untracked) or None}
    return out


def scene_view(state: dict, source_path: Path) -> dict:
    """A saved scene as the page draws it, without the per-question results (a shared scene must not show other
    questions' answers)."""
    j = state
    if req(j, "f", "scene") is None:
        raise ValueError(f"{source_path.relative_to(ROOT)}: scene has no focal length")
    pl = req(j, "plane", "scene")
    # u1 / u2 (in-plane axes) are written only for some planes
    plane = {k: vec(req(pl, k, "plane") if k == "normal" else pl.get(k)) for k in ("normal", "u1", "u2")} | \
        {"dist": num(req(pl, "dist", "plane"), 5)}
    objs = []
    for name, o in req(j, "objects", "scene").items():
        what = f"scene object {name!r}"
        vis = req(o, "visible", what)
        objs.append({"name": name, "shape": req(o, "shape", what),
                     "dims": {k: num(v) for k, v in req(o, "dims", what).items()},
                     "rms": num(req(o, "rms", what), 3), "p0": vec(req(o, "p0", what)), "v0": vec(req(o, "v0", what)),
                     "a": vec(req(o, "a", what)), "yaw": num(req(o, "yaw", what), 5), "t_on": num(req(o, "t_on", what), 5),
                     # where each dimension comes from: observed / prior / bound (not seen, anti-collapse floor)
                     "dim_source": req(o, "dim_source", what), "dim_bounds": dim_bounds(name, o),
                     "support": req(o, "support", what),
                     "hang": req(o, "hang", what), "mode": req(o, "mode", what),
                     "out_of_frame": req(o, "out_of_frame", what), "visibility": req(o, "visibility", what),
                     "overhead": bool(req(o, "overhead", what)), "events": motion_events(name, o)[:12],
                     "law": motion_law(name, o) if FMT["decide"] == "agree" else None,
                     # motion that is not a single parabola: sample the centre for the viewer
                     "path": sampled_path(o) if any(req(o, k, what) for k in FMT["paths"]) else None,
                     "visible": [[num(a, 5), num(b, 5)] for a, b in vis] if vis else None})
    return {"file": str(source_path.relative_to(ROOT)), "f": num(j["f"], 5), "plane": plane, "objects": objs,
            "scale": scale_view(j),
            "main_support": req(j, "main_support", "scene"),
            "f_source": text(req(j, "f_source", "scene")) or None, "f_first": req(j, "f_first", "scene"),
            # an empty vote list is legitimate (no focal vote taken): written as null
            "focal_votes": [{"source": text(req(x, "source", "focal vote")), "value": num(req(x, "value", "focal vote"), 4),
                             "weight": num(req(x, "weight", "focal vote"), 3)}
                            for x in req(j, "focal_votes", "scene")] or None}


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
    candidates = [a for a in answers if req(a, "row", "answer record") == row["row"]
                  and req(a, "question", "answer record") == row["question"]
                  and text(req(a, "prior", "answer record")) == text(row["prior"])]
    if len(candidates) != 1:
        return None, None, "answer identity missing or duplicated"
    answer = candidates[0]
    scene_id = req(answer, "scene_id", "answer record")
    if not scene_id or text(row["scene_id"]) != scene_id:
        return answer, None, "scene_id missing or differs between CSV and answer record"
    matches = [g for g in groups if req(g["state"], "scene_id", "scene") == scene_id]
    if len(matches) != 1:
        return answer, None, "scene_id missing or duplicated among independent group scenes"
    group = matches[0]
    contexts = [c for c in req(group["state"], "question_context", "scene")
                if req(c, "row", "question context") == row["row"] and req(c, "question", "question context") == row["question"]
                and text(req(c, "prior", "question context")) == text(row["prior"])
                and text(req(c, "depth_info", "question context")) == text(row["depth_info"])]
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
        self.render = base_version(tag) == CURRENT
        if not self.file.exists() and not self.render:
            raise ValueError(f"{tag} is not the current version ({CURRENT}) and has no overlay record {self.file.relative_to(WEB)}")
        self.record = json.loads(self.file.read_text()) if self.file.exists() else {}
        self.added = 0
        self.used: set[str] = set()

    def get(self, vid: str, scene: dict, question: str, fps: float, duration: float, subdir: str) -> list[dict]:
        """The overlays of one question at its times; rendered images go to <tag>/<subdir>/."""
        times = list(dict.fromkeys(float(t) for t in parse_question(question).times))[:2] or [0.]
        out = []
        for t in times:
            key = f"{scene['scene_id']} {t:.9g}"
            entry = self.record.get(key)
            self.used.add(key)
            if entry is None:
                if not self.render:
                    raise ValueError(f"{vid}: overlay {key} not in {self.file.relative_to(WEB)}; only {CURRENT} renders")
                entry = self.record[key] = self._render(vid, scene, t, fps, duration, subdir)
                self.added += 1
            if entry.get("path") and not (WEB / entry["path"]).exists():
                raise ValueError(f"{vid}: overlay image missing: {entry['path']}")
            out.append(entry)
        return out

    def _render(self, vid: str, scene: dict, t: float, fps: float, duration: float, subdir: str) -> dict:
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
        path = self.dir / subdir / f"{identity}_f{frame}.jpg"
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
        """Write the record. The current version also drops the overlays no question of this build references and
        deletes their images (the build covers every question the version is shown for)."""
        removed = 0
        if self.render:
            for key in sorted(set(self.record) - self.used):
                path = self.record.pop(key).get("path")
                if path and (WEB / path).exists():
                    (WEB / path).unlink()
                removed += 1
            for d in sorted(self.dir.glob("*/"), reverse=True):
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
        if removed:
            print(f"{self.tag}: {removed} unreferenced overlays removed")
        if self.added or removed:
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
        if "k" in r and int(r["k"]) != int(m["k"]):
            raise ValueError(f"{u}: harness CSV k {r['k']} differs from the validation table's {m['k']}")
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
    results = [r for r in req(state, "results", "scene") if req(r, "question", "scene result") == row["question"]
               and req(r, "row", "scene result") == row["row"]]
    if len(results) > 1:
        raise ValueError(f"row {row['row']}: {len(results)} scene results for one question")
    return results[0] if results else None


def run_rows(tag: str, csv_path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    """The harness CSV of one run, its "row" identities checked to be unique rows of the validation table."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError(f"invalid harness run tag: {tag!r}")
    version_format(tag)
    if not csv_path.exists():
        raise ValueError(f"{tag}: harness CSV missing: {csv_path}")
    frame = pd.read_csv(csv_path)
    if "row" not in frame:
        raise ValueError(f"{csv_path.name}: harness CSV lacks row identities")
    rows = pd.to_numeric(frame["row"], errors="raise").astype(int)
    outside = sorted(set(rows) - set(manifest.index))
    if outside:
        raise ValueError(f"{csv_path.name}: rows not in the validation table: {outside}")
    return frame


def timing_view(t: dict, csv_seconds) -> dict:
    """A unit's timing ("timing" of its answer record) for the page: total seconds and one entry per step, in pipeline
    order: {"step", "s", "reused" (false | "partial" | true), "cached", "shared"}. The CSV "seconds" must be the total."""
    total = req(t, "total_seconds", "timing")
    if not (isinstance(csv_seconds, (int, float)) and math.isfinite(csv_seconds) and abs(csv_seconds - total) < 0.05):
        raise ValueError(f"{AT['where']}: CSV seconds {csv_seconds} differs from the answer record's total {total}")
    steps, order = req(t, "steps", "timing"), FMT["steps"]
    if list(steps) != list(order):
        raise ValueError(f"{AT['where']}: timing steps {list(steps)}, expected {list(order)}")
    out = []
    for name in order:
        x = steps[name]
        reused = req(x, "reused", f"timing step {name}")
        if reused not in (False, True, "partial"):
            raise ValueError(f"{AT['where']}: timing step {name} reused = {reused!r}")
        out.append({"step": name, "s": num(req(x, "seconds", f"timing step {name}"), 4), "reused": reused,
                    "cached": bool(req(x, "cached", f"timing step {name}")), "shared": x.get("shared")})
    return {"total": num(total, 5), "steps": out}


def question_record(tag: str, unit: str, row: dict) -> dict:
    """The answer record outputs/harness/<tag>/answers/<unit>.json of a unit-question run (schema "unit-1"), its
    identity checked against the question's row."""
    rec = unit_json("answers", unit, "answer record")
    if req(rec, "schema", "answer record") != "unit-1":
        raise ValueError(f"{unit}: answer record schema {rec['schema']!r}, expected 'unit-1'")
    for key in ("k", "row", "question", "prior", "depth_info"):
        if text(req(rec, key, "answer record")) != text(row[key]):
            raise ValueError(f"{unit}: answer record {key} {rec[key]!r} differs from the table's {row[key]!r}")
    if not run_label(tag, req(rec, "version", "answer record")["pipeline"]):
        raise ValueError(f"{unit}: answer artifact version {rec['version']['pipeline']} differs from requested {tag}")
    return rec


def question_scene(unit: str, row: dict, record: dict) -> dict | None:
    """The unit's own saved scene (agent/<unit>.json): exactly its one row as question context, no groups.
    None only when the run failed before writing it."""
    if not (DIRS["scenes"] / f"{unit}.json").exists() and record["execution_status"] == "failed":
        return None
    state = unit_json("scenes", unit, "question scene")
    if "groups" in state or int(req(state, "k", "question scene")) != int(row["k"]):
        raise ValueError(f"{unit}: question scene has groups or k {state.get('k')}")
    ctx = req(state, "question_context", "question scene")
    if len(ctx) != 1 or req(ctx[0], "row", "question context") != row["row"] or ctx[0].get("unit") != unit \
            or req(ctx[0], "question", "question context") != row["question"]:
        raise ValueError(f"{unit}: question scene context is not exactly this question")
    return state


def build_run(tag: str, frame: pd.DataFrame, manifest: pd.DataFrame, units: list[str], videos: dict,
              make_thumb: bool) -> list[dict]:
    """The questions `units` from one run (all of them must be in its CSV), every value read from that run's own
    artifacts in its own format. Unit "video": the run's per-video artifacts go to videos[vid]["runs"][tag]. Unit
    "question": each question reads only its own unit's files and carries them; nothing per video."""
    FMT.clear()
    FMT.update(version_format(tag))
    unit_kind = FMT["unit"]
    DIRS.clear()
    DIRS.update(scenes=ROOT / "outputs/scene_fit_eval" / f"h_{tag}" / "agent", sgraph=ROOT / "outputs" / f"scenegraph_{tag}",
                tracks=ROOT / "outputs" / f"tracks_{tag}", objects=ROOT / "outputs" / f"objects_{tag}",
                answers=ROOT / "outputs/harness" / tag / "answers")
    df = harness_rows(frame, manifest, units)
    df["mra"] = [mra_row(p, g) for p, g in zip(df["pred"], df["answer"])]
    overlays = Overlays(tag)
    questions = []
    for vid, rows in df.groupby("video_id", sort=True):
        first = rows.iloc[0]
        AT["where"] = f"{tag} {vid}"
        if vid not in videos:
            duration, width, height, thumb = video_meta(vid, make_thumb)
            videos[vid] = {"id": vid, "source": first.video_source, "video_type": first.video_type,
                           "fps": num(first.fps), "duration": duration, "size": [width, height], "thumb": thumb,
                           **video_info(vid), "runs": {}}
        video = videos[vid]
        duration = video["duration"]
        groups = []
        if unit_kind == "video":
            record = read_json(DIRS["answers"] / f"{vid}.json", "answer record")
            version = record["version"]
            if version["pipeline"] != tag.removesuffix("_control"):
                raise ValueError(f"{vid}: answer artifact version {version['pipeline']} differs from requested {tag}")
            run = video["runs"][tag] = {"run_at": version["time"]}
            scene_path = DIRS["scenes"] / f"{vid}.json"
            if scene_path.exists():
                groups = harness_groups(json.loads(scene_path.read_text()))
            elif record["execution_status"] != "failed":
                raise ValueError(f"{vid}: scene missing for a run that did not fail: {scene_path.relative_to(ROOT)}")
            run |= {"groups": [{"group": g["group"], "scene_id": g["state"]["scene_id"],
                                "scene": scene_view(g["state"], scene_path)} for g in groups],
                    "sgraph": sgraph(vid), "tracks": track_intervals(vid)}
        for row in rows.to_dict("records"):
            unit = row["unit"]
            AT["where"] = f"{tag} {unit}"
            query = {"id": unit, "version": tag, "video": vid, "k": int(row["k"]), "row": int(row["row"]),
                     "cat": row["category"], "type": row["inference_type"],
                     "q": row["question"], "prior": text(row["prior"]), "depth": text(row["depth_info"]),
                     "answer": num(row["answer"]), "pred": num(row["pred"]), "mra": num(row["mra"], 6),
                     "src": text(row["source"]), "status": text(row["status"]),
                     "method": row["method"], "scene_id": text(row["scene_id"]), "time": None}
            if FMT["decide"] == "conf":
                query["conf"] = num(req(row, "conf", "harness CSV row"), 6)
            if unit_kind == "video":
                answer, group, error = bind_harness_question(row, record["answers"], groups)
                state = group["state"] if group else {}
                result = own_result(state, row) if group else None
                exec_status = record["execution_status"]
            else:
                rec = question_record(tag, unit, row)
                unit_json("objects", unit, "object inventory")
                state = question_scene(unit, row, rec)
                answer, error = req(rec, "answer", "answer record"), None
                if req(answer, "row", "answer") != row["row"] or text(req(answer, "scene_id", "answer")) != text(row["scene_id"]):
                    raise ValueError(f"{unit}: answer row / scene_id differs from the CSV")
                if state is not None and req(state, "scene_id", "question scene") != row["scene_id"]:
                    raise ValueError(f"{unit}: scene_id of its scene differs from the CSV")
                group = {"group": None, "state": state} if state is not None else None
                result = own_result(state, row) if state is not None else None
                exec_status = rec["execution_status"]
                query["time"] = timing_view(req(rec, "timing", "answer record"), row.get("seconds"))
            if FMT["decide"] in ("checks", "agree"):
                if unit_kind != "question":
                    raise ValueError(f"{tag}: named checks are read from per-question answer records only")
                query |= answer_checks(answer, row)
            if FMT["decide"] == "agree":
                if result is None:
                    raise ValueError(f"{unit}: the scene has no result for this question")
                query |= agreement_view(result, answer, query["status"])
            quality = req(result, "quality", "scene result") if result else {}
            alignment = align(quality.get("align_tq"))   # absent or null when no alignment check ran
            mid = (result or {}).get("measurement_id")
            if mid is not None and mid not in req(state, "measurements", "scene"):
                raise ValueError(f"{unit}: measurement {mid} not in its scene")
            query |= {"group": group["group"] if group else None, "binding_error": error,
                      "measurement": answer["measurement"] if group else None, "measurement_id": mid,
                      "measurement_record": state["measurements"][mid] if mid is not None else None,
                      "provenance": req(answer, "provenance", "answer record") if answer else None, "quality": quality, "align": alignment,
                      "fit": fit_row(result) if result else None,
                      "overlays": overlays.get(vid, state, row["question"], float(first.fps), duration,
                                               vid if unit_kind == "video" else unit) if group else [],
                      "execution_status": exec_status}
            if unit_kind == "question":
                query |= {"scene": scene_view(state, DIRS["scenes"] / f"{unit}.json") if state is not None else None,
                          "sgraph": sgraph(unit), "tracks": track_intervals(unit)}
                if FMT.get("identity"):
                    query |= {"identity": identity_view(unit),
                              "dims_check": dims_check(state) if state is not None else None}
            query["tq"], query["tjump"] = question_times(query["q"], alignment)
            questions.append(query)
    overlays.save()
    print(f"{tag}: {len(questions)} questions, {overlays.added} overlays rendered")
    return questions


def build_harness_val(runs: list[tuple[str, Path]], make_thumb: bool = False) -> dict:
    """Display completed harness runs without re-running anything. `runs` is ordered: each question of the
    validation table is shown from the first run whose CSV has its row, and every value of that question comes from
    that run alone (labelled with its version on the page). Every DEV10 question must be in some run."""
    if len({t for t, _ in runs}) != len(runs):
        raise ValueError(f"a run is listed twice: {[t for t, _ in runs]}")
    manifest, dev = manifest_units(), dev_units()
    claimed: dict[int, str] = {}
    frames = {}
    for tag, csv_path in runs:
        frames[tag] = run_rows(tag, csv_path, manifest)
        for i in pd.to_numeric(frames[tag]["row"], errors="raise").astype(int):
            claimed.setdefault(int(i), tag)
    by_unit = {u: i for i, u in enumerate(manifest["unit"])}
    missing = [u for u in dev if by_unit[u] not in claimed]
    if missing:
        raise ValueError(f"docs/DEV10.md units in none of the runs: {missing}")
    videos, questions = {}, []
    for tag, _ in runs:
        units = [u for i, u in enumerate(manifest["unit"]) if claimed.get(i) == tag]
        if not units:
            raise ValueError(f"{tag}: every question of this run is taken by an earlier run")
        questions += build_run(tag, frames[tag], manifest, units, videos, make_thumb)
    questions.sort(key=lambda q: q["row"])
    print(f"val: {len(questions)} questions ({len(dev)} DEV10), {len(videos)} videos")
    return {"runs": [{"version": t, "unit": version_format(t)["unit"], "decide": version_format(t)["decide"],
                      "csv": str(p.relative_to(ROOT))} for t, p in runs],
            "dev": dev, "videos": [videos[v] for v in sorted(videos)], "questions": questions}


def parse_run(spec: str) -> tuple[str, Path]:
    """<tag> (CSV outputs/val_<tag>_answers.csv) or <tag>=<CSV path relative to the repo root>."""
    tag, _, csv = spec.partition("=")
    return tag, (ROOT / csv if csv else ROOT / "outputs" / f"val_{tag}_answers.csv").resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-runs", nargs="+", default=[CURRENT], metavar="TAG[=CSV]",
                    help="completed harness validation runs, in order of preference: a question is shown from the "
                         "first run that has it (default: pipeline/VERSION, %(default)s)")
    ap.add_argument("--rethumb", action="store_true", help="regenerate thumbnails that already exist")
    args = ap.parse_args()
    THUMBS.mkdir(parents=True, exist_ok=True)
    data = {"generated": date.today().isoformat(),
            "val": build_harness_val([parse_run(s) for s in args.val_runs], args.rethumb)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    sizes = [p.stat().st_size for p in THUMBS.glob("*.jpg")]
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB); {len(sizes)} thumbs, {sum(sizes) / 1e6:.2f} MB, max {max(sizes, default=0)} B")


if __name__ == "__main__":
    main()
