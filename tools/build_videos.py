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
#   gravity  the scene has "gravity" {g, sigma, source, rule, up_source, evidence, checks, failed} (rule B2) and a
#            prior-scaled scene's "scale" its named "checks" / "failed"; a measurement read from a ballistic piece has
#            "inputs"."law" {g, source, up}; each ballistic piece of the gravity evidence has "admissible" {pass, reason}
#            and an inferred gravity its "earth_test" {pass, reason}
#   release  every scene graph edge with "t_release" has its "release_rule" (release.last_contact / free_flight), an
#            airborne one its "release_why"; the graph has "untracked" {name: why}
#   refmeasure the reference measurement record outputs/refmeasure_v<refmeasure.VERSION>_<tag>/<unit>.json (when the
#            unit has one) has "checks" / "failed" per "<object>|<quantity>" entry
#   kept     the answer record has "kept_state" {iter, scenegraph, ...}: the scene graph the answer used (the agent
#            round kept, 0 = the initial fit) and "iterations" (the agent rounds with their tool results); a scene graph
#            edge rewritten by scene_fit carries "was" {rel, to} and "rule"; every constraint residual has "stated"
#            {text, digits, unit, sigma, source}; an identity class_size entry has "dims" {candidate, ref}
#   selftest every scene graph edge has the builder's "test" and a merged one "absorbed" [{rel, to, t_start, t_end, test}];
#            the graph has "unknown" [{from, rel, to, t_start, t_end, test, reason}] and "self_test" {tested, fail,
#            merged, unknown} (rule builder.self_test); each scene object has "segfit" [piece] | null with "point" /
#            "sigma" / "bound" / "rms" / "run" / "pass" (rule segfit.point), a physics record "events_graph" and a
#            motion law "events" / "unknown" (rule events.graph)
#   horizon  the scene has "focal" {votes [{object, span, hop, f, sigma, law, n, rms, used, why}], f_votes, sigma,
#            n_votes, f, source} (rule focal.gravity_hop) and "support_horizon" {pass, reason, phi_plane_deg,
#            phi_horizon_deg, sigma_deg, z, row_plane, row_horizon, horizon {...}, repaired} (rule support.horizon);
#            camera.support_plane has "pitch_source" / "depth_plane"; the answer record has "stop" {reason, round, why,
#            tried} and each agent round "newly_failed" {check: reason}
#   camera   the scene camera has "f_sigma" / "f_sigma_how" (rule C3a, the scene's "f_source" names its source) and
#            "focal"."consistent" [{source, f_obj, sigma_obj, f, sigma, bound, pass, reason, how, class_size?}]; the scene
#            has "support_plane" {pass, failed, reason, tests {name: {pass, reason, ..}}} (pass null, no tests: no
#            support plane; rule D9); a direct
#            measurement with a state span has "window" {t0, t1, state, clipped_by {start, end}, window} (rule
#            window.state), a direct size "support_depth" {applies, ..} and, when it applies, "cross_checks"
#            {"direct.depth_agree"} (rule C12); each agent round past the accept test has "checks_fixed" /
#            "checks_broken" (rule accept.full_checks) and a rebuilding round "rebuilt" {steps, failed, self_test}
#   roll     the scene has "camera_roll" {pass, reason, roll_source, roll_fit_deg?, roll_fit_how?, roll_image_deg?,
#            sigma_deg?, z?, image {status, why, n_lines, n_frames, ..}?, refit {was, now}?} (rule camera.roll); a
#            direct window has "law" / "check" (rule window.driven: law null, check "window.smooth", inputs "driven" /
#            "noise_px" / "noise_how"); an agent round has "repair_candidates" [{for, call, class}] and
#            "repair_candidates_dropped" [{call, why}] (rule repair.window_checks); the unit's extents record
#            outputs/extents2d_<tag>/<unit>.json, when it exists, has "format" 3 with per-frame "extent" /
#            "speed_px" / "overlay" (rule extent.overlay)
#   static   the scene has "extents" {object: {dim: {sigma_mask, sigma_mask_how, at: {label: {value, sigma, frames_used,
#            frames_rejected [{frame, speed_px, extent, why}], how}}}}} with "objects"[name]."extent" the same record or
#            null (rule extent.static), "camera_sources" [{object, frames, t, rule}], every focal.consistent row
#            "independent" (rule focal.independent), "camera_one" {f, sigma, how, votes [{source, object, f, sigma,
#            used, why}]} or null (rule camera.one) and "depth_bindings" {given, bound [{name, node, as, used {by,
#            why}}], unbound [{name, reason}]} (rule C14); a direct speed measurement may have "inputs"."centred"
#            {from, extremum, stop} (rule window.centred); the harness CSV has "unit_set"
#   fusion   the scene's "focal" has "fusion" {pass, reason, f, sigma, n_sources, camera_f, camera_sigma, sources
#            [{object, kind, f, sigma, n_votes, camera, how, verdict (used / outlier / disagree), why, bounded?}]} (rule
#            focal.fusion; a fail adds the measurement check "focal"); a measurement's sigma text is "inputs"."sigma_how"
#            with "inputs"."chain_piece" {piece, events, window_short, events_near_t} (rule fit.sigma_chain) or, when
#            the fit has no chain piece, the text "inputs"."sigma"; "inputs"."sigma_events" {one_sided, event_unstable}
#   fused    a scene with "f_source" "camera.fused" has "focal"."fused" {f, sigma, n_sources, how, was {f, source}}
#            (rule C1a) and no other scene has it; "focal"."agency_unknown" [{object, why}] when present; a motion law
#            may have "agency" "untestable" with "why" (rule agency.unknown); every fit measurement with a piece has
#            "inputs"."chain_piece" with "chain" (bool) and "inputs"."piece" equal to its "piece" (rule C1b); a
#            constraint residual with "pass" null has "untestable" {rule, sigma_pred, source, reason} and
#            "checks"."constraints" has "untestable_ids" / "untestable" (rule G13); a failed unit's answer record has
#            "error" (StageError: objects.llm_invalid / scenegraph.llm_invalid ...) and a timing prefix; a scene's
#            plausible parts may have "untestable" [reason] (critic.agency_unknown)
#   impulse  every physics record has "impulse" [{t, event, agent, source, verdict (kick / agent.unobserved / rejected:
#            .. / chain ends / no_law), why, jump_px_s?, sigma_px_s?, jump_how?}] equal to its motion law's (rule
#            law.impulse), "onset" {phase: {t, sigma, how}} and "incline_law" [{support, t_on, sigma_t_on, accel,
#            sigma_accel, accel_source, accel_geometry, kappa, shape, theta_deg, law}] (rule support.incline); the scene
#            has "support_incline" {support: {method, theta_deg, sigma_deg, length_m, inclined, up {level, angle_deg,
#            sigma_deg, source, why}, why}} and "incline_up"; a measurement may have "kinematic_onset" {pass, t_on,
#            sigma_t_on, v, integral, sigma, bound, reason} (rule kinematic.onset); the scene graph cache has "format" 2:
#            every edge a "source" (agent / per-frame / grid) and no "conf", "conflicts" [{name, from, source, t_start,
#            t_end, edges, why}] (rule scenegraph.overlap); the presence record
#            outputs/objects_<tag>_presence_v4/<unit>.json has {name: {present, rule, votes, zoom_votes?}} and no
#            confidence (rule decisions.no_conf)
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
           "steps": STEPS_V5, "identity": True, "gravity": True, "release": True, "refmeasure": True, "kept": True,
           "pendulum": True, "selftest": True, "horizon": True, "camera": True, "roll": True,
           "static": True, "fusion": True, "fused": True, "impulse": True},
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


def sgraph(key: str, rec: dict | None = None, state: dict | None = None):
    """The run's scene graph cache outputs/scenegraph_<tag>/<key>.json as written (with FMT "kept": the graph the
    answer used, the answer record's kept_state.scenegraph, and "kept" its agent round), for the tree on the page:
    {"nodes": [{"id", "type": object | support | holder}], "edges": [{"from", "to" (null: airborne), "rel", "t0", "t1"}]}
    (t0 / t1 null = the whole video; contact edges dropped); with FMT "release" also "releases" [{"obj", "t", "rule",
    "why"}] (one per held -> airborne release) and "untracked" [[name, why]]; with FMT "kept" "rewritten" [{"obj", "rel",
    "to", "t0", "t1", "was", "rule"}] (edges of the scene's own graph rewritten by scene_fit). None when it has none."""
    if FMT.get("kept"):
        ks = req(rec, "kept_state", "answer record")
        g = read_json(ROOT / text(req(ks, "scenegraph", "kept_state")), "kept scene graph")
        if req(g, "unit", "kept scene graph") != key:
            raise ValueError(f"{AT['where']}: kept scene graph is for unit {g['unit']!r}")
    else:
        g = unit_json("sgraph", key, "scene graph")
    if FMT.get("impulse"):
        if req(g, "format", "scene graph") != 2:
            raise ValueError(f"{AT['where']}: scene graph format {g['format']!r}, expected 2")
        for e in req(g, "edges", "scene graph cache"):
            if "conf" in e or req(e, "source", "edge") not in EDGE_SOURCES:
                raise ValueError(f"{AT['where']}: edge {e.get('from')!r} {e.get('rel')!r} source {e.get('source')!r}"
                                 f"{' with a conf' if 'conf' in e else ''}")
    edges = [{"from": text(req(e, "from", "edge")), "to": text(req(e, "to", "edge")) or None,   # null "to": airborne
              "rel": text(e["rel"]), "t0": num(req(e, "t_start", "edge"), 4), "t1": num(req(e, "t_end", "edge"), 4)}
             | ({"source": text(e["source"])} if FMT.get("impulse") else {})
             | ({"test": test_view(e["test"], f"edge {e.get('from')!r} {e.get('rel')!r}")} if "test" in e else {})
             | ({"absorbed": [span_edge(x, f"edge {e.get('from')!r} {e.get('rel')!r} absorbed") for x in e["absorbed"]]}
                if FMT.get("selftest") and e.get("absorbed") else {})
             for e in req(g, "edges", "scene graph cache") if req(e, "rel", "edge") in SUPPORT_RELS]
    nodes = [{"id": text(req(n, "id", "scene graph node")), "type": text(req(n, "type", "scene graph node"))}
             for n in req(g, "nodes", "scene graph cache")]
    out = {"nodes": nodes, "edges": edges} if edges else {}
    if FMT.get("release"):
        rel = {}     # (object, t_release) -> {"obj", "t", "rule", "why"}: the held_by and the airborne edge of one release
        for e in req(g, "edges", "scene graph cache"):
            if "t_release" not in e:
                continue
            w = f"edge {e.get('from')!r} {e.get('rel')!r}"
            rule = text(req(e, "release_rule", w))
            if rule not in ("release.last_contact", "release.free_flight"):
                raise ValueError(f"{AT['where']}: {w} release_rule {rule!r}")
            t = [num(x, 4) for x in req(e, "t_release", w)]
            r = rel.setdefault((text(e["from"]), tuple(t)), {"obj": text(e["from"]), "t": t, "rule": rule, "why": None})
            if r["rule"] != rule:
                raise ValueError(f"{AT['where']}: {w} release_rule {rule!r} differs from its other edge's {r['rule']!r}")
            if e["rel"] == "airborne":
                r["why"] = text(req(e, "release_why", w))
        unt = req(g, "untracked", "scene graph cache")
        if rel:
            out["releases"] = list(rel.values())
        if unt:
            out["untracked"] = [[text(k), text(v)] for k, v in unt.items()]
    if FMT.get("selftest"):
        st = req(g, "self_test", "scene graph cache")
        out["self_test"] = {k: int(req(st, k, "self_test")) for k in ("tested", "fail", "merged", "unknown")}
        unk = [span_edge(x, "unknown span") | {"obj": text(req(x, "from", "unknown span")),
                                                 "reason": text(req(x, "reason", "unknown span"))}
               for x in req(g, "unknown", "scene graph cache")]
        if unk:
            out["unknown"] = unk
    if FMT.get("impulse"):
        cf = [conflict_view(x) for x in req(g, "conflicts", "scene graph cache")]
        if cf:
            out["conflicts"] = cf
    if FMT.get("kept") and state is not None:
        rw = []
        for e in req(req(state, "scene_graph", "scene"), "edges", "scene graph"):
            if "was" not in e:
                continue
            w = f"scene graph edge {e.get('from')!r} {e.get('rel')!r}"
            was = req(e, "was", w)
            rw.append({"obj": text(req(e, "from", w)), "rel": text(req(e, "rel", w)), "to": text(e.get("to")) or None,
                       "t0": num(req(e, "t_start", w), 4), "t1": num(req(e, "t_end", w), 4),
                       "was": {"rel": text(req(was, "rel", f"{w} was")), "to": text(was.get("to")) or None},
                       "rule": text(req(e, "rule", w))})
        if rw:
            out["rewritten"] = rw
    if out and "nodes" not in out:
        out |= {"nodes": nodes, "edges": []}
    if out and FMT.get("kept"):
        out["kept"] = int(req(rec["kept_state"], "iter", "kept_state"))
    return out or None


TEST_RESULTS = ("pass", "fail", "untestable")
EDGE_SOURCES = ("agent", "per-frame", "grid")   # pipeline.scenegraph.EDGE_SOURCES


def conflict_view(x: dict) -> dict:
    """An overlap of two same-source state edges of one object (rule scenegraph.overlap): {"obj", "source", "t0", "t1"
    (null: open), "edges": [{"rel", "to", "t0", "t1"} x 2], "why"}."""
    w = "scene graph conflict"
    if req(x, "name", w) != "scenegraph.overlap":
        raise ValueError(f"{AT['where']}: {w} name {x['name']!r}")
    es = req(x, "edges", w)
    if len(es) != 2:
        raise ValueError(f"{AT['where']}: {w} has {len(es)} edges")
    return {"obj": text(req(x, "from", w)), "source": text(req(x, "source", w)), "t0": num(req(x, "t_start", w), 4),
            "t1": num(req(x, "t_end", w), 4),
            "edges": [{"rel": text(req(e, "rel", w)), "to": text(e.get("to")) or None, "t0": num(req(e, "t_start", w), 4),
                       "t1": num(req(e, "t_end", w), 4)} for e in es], "why": text(req(x, "why", w))}


def span_edge(x: dict, what: str) -> dict:
    """A tested span of a scene graph (an absorbed or unknown edge): {"rel", "to", "t0", "t1", "test"}."""
    return {"rel": text(req(x, "rel", what)), "to": text(x.get("to")) or None, "t0": num(req(x, "t_start", what), 4),
            "t1": num(req(x, "t_end", what), 4), "test": test_view(req(x, "test", what), what)}


def test_view(t: dict, what: str) -> dict:
    """The geometric test of an agent state edit (rule graph_edit.retest): {"name", "result", "why"}."""
    r = text(req(t, "result", f"{what} test"))
    if r not in TEST_RESULTS:
        raise ValueError(f"{AT['where']}: {what} test result {r!r}")
    return {"name": text(req(t, "name", f"{what} test")), "result": r, "why": text(req(t, "why", f"{what} test"))}


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
    out = {"model": text(req(law, "model", f"object {name!r} motion law")), "rotation": rot, "phases": phases}
    if FMT.get("fused") and law.get("agency") is not None:
        if law["agency"] != "untestable":
            raise ValueError(f"{AT['where']}: object {name!r} motion law agency {law['agency']!r}")
        out["agency"] = {"why": text(req(law, "why", f"object {name!r} motion law"))}
    if FMT.get("selftest") and rot is None:
        out["unknown"] = [[num(a, 4), None if b >= OPEN_END else num(b, 4)]
                          for a, b in req(law, "unknown", f"object {name!r} motion law")] or None
    return out


OPEN_END = 1e8      # the scene json writes an open span end as +-1e9


def segfit_view(name: str, o: dict):
    """The motion-law test of each piece (rule segfit.point): {"model", "t0", "t1" (null: open), "point", "sigma",
    "bound", "rms", "run", "n_test", "pass" (null: untestable)}; None when the object has no segfit record."""
    sf = req(o, "segfit", f"scene object {name!r}")
    if sf is None:
        return None
    out = []
    for p in sf:
        w = f"object {name!r} segfit piece"
        t0, t1 = float(req(p, "t0", w)), float(req(p, "t1", w))
        rms = req(p, "rms", w)
        out.append({"model": text(req(p, "model", w)), "t0": None if t0 <= -OPEN_END else num(t0, 4),
                    "t1": None if t1 >= OPEN_END else num(t1, 4), "point": text(req(p, "point", w)),
                    "sigma": [num(x, 3) for x in req(p, "sigma", w)], "bound": [num(x, 3) for x in req(p, "bound", w)],
                    "rms": None if rms is None else num(rms, 3), "run": int(req(p, "run", w)),
                    "n_test": int(req(p, "n_test", w)), "pass": req(p, "pass", w)})
    return out


def events_view(name: str, o: dict):
    """The boundaries of the physics chain, each a scene-graph event (rule events.graph): [{"phase", "event", "t",
    "window", "precision", "t_fit", "kick"}]; None without a physics record."""
    ph = req(o, "physics", f"scene object {name!r}")
    if ph is None:
        return None
    w = f"object {name!r} physics events_graph"
    return [{"phase": int(req(e, "phase", w)), "event": text(req(e, "event", w)), "t": num(e["t"], 4) if e.get("t") is not None else None,
             "window": [num(x, 4) for x in e["window"]] if e.get("window") else None,
             "precision": text(e.get("precision")) or None, "t_fit": num(req(e, "t_fit", w), 4), "kick": bool(req(e, "kick", w))}
            for e in req(ph, "events_graph", w)] or None


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
        req(qual, "conf", "scene result quality")    # read as the format's field, not shown (decisions.no_conf)
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
    terms = {k: num(v, 3) for k, v in parts.items() if k in ("track", "depth", "align", "focal", "gravity", "fit_se", "stated")}
    lst = lambda v: [num(x, 3) for x in v] if isinstance(v, list) else ([num(v, 3)] if v is not None else None)
    dof = parts.get("dof")
    inputs = req(m, "inputs", what) or {}
    if FMT.get("fused") and inputs.get("piece") is not None:
        cp = req(inputs, "chain_piece", what)
        if [num(x, 4) for x in inputs["piece"]] != [num(x, 4) for x in req(cp, "piece", f"{what} chain_piece")]:
            raise ValueError(f"{AT['where']}: {what} piece {inputs['piece']} differs from its chain_piece {cp['piece']}")
    return {"name": text(req(m, "name", what)), "method": text(req(m, "method", what)),
            "value": num(req(m, "value", what)), "unit": text(req(m, "unit", what)), "sigma": num(req(m, "sigma", what), 3),
            "terms": terms or None, "scale": lst(parts.get("scale")), "q": lst(parts.get("q")),
            "dof": [int(x) for x in dof] if isinstance(dof, list) else ([int(dof)] if dof is not None else None),
            "shared": list(req(m, "shared", what)), "admissible": ok, "why": text(req(adm, "reason", f"{what} admissible")),
            "failed": [{"name": k, "reason": text(req(checks[k], "reason", f"{what} check {k!r}"))}
                       | ({"which": text(checks[k]["which"])} if checks[k].get("which") else {}) for k in failed],
            "checks": [{"name": k, "pass": bool(req(c, "pass", f"{what} check {k!r}")),
                        "reason": text(req(c, "reason", f"{what} check {k!r}"))} for k, c in checks.items()],
            "window": [num(x, 4) for x in inputs["window"]] if inputs.get("window") else None,
            "piece": [num(max(x, 0.0), 4) if x > -1e8 else None for x in inputs["piece"]]
            if inputs.get("piece") and not FMT.get("fused") else None,
            "n": inputs.get("n"), "state": text(inputs.get("state")) or None,
            "law": law_view(inputs["law"], what) if FMT.get("gravity") and inputs.get("law") else None,
            "profile": [profile_view(x, what) for x in inputs["at_bound.profile"]] if inputs.get("at_bound.profile") else None} \
        | (direct_view(m, what) if FMT.get("camera") else {}) | (sigma_view(inputs, what) if FMT.get("fusion") else {}) \
        | (onset_check_view(m, what) if FMT.get("impulse") and "kinematic_onset" in m else {})


def onset_check_view(m: dict, what: str) -> dict:
    """A law-driven onset's speed bound (rule kinematic.onset): {"kin_onset": {pass (null: untestable), t_on,
    sigma_t_on, v, integral, sigma, bound, reason}}."""
    k = req(m, "kinematic_onset", what)
    w = f"{what} kinematic_onset"
    ok = req(k, "pass", w)
    if ok not in (True, False, None):
        raise ValueError(f"{AT['where']}: {w} pass {ok!r}")
    return {"kin_onset": {"pass": ok, "t_on": num(k.get("t_on"), 5), "sigma_t_on": num(k.get("sigma_t_on"), 3),
                          "v": num(k.get("v"), 4), "integral": num(k.get("integral"), 4), "sigma": num(k.get("sigma"), 3),
                          "bound": num(k.get("bound"), 4), "reason": text(req(k, "reason", w))}}


def plain(x):
    """A JSON value as plain JSON (numbers rounded), for records shown as written."""
    if isinstance(x, (list, tuple)):
        return [plain(a) for a in x]
    if isinstance(x, dict):
        return {text(k): plain(v) for k, v in x.items()}
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, (int, float)):
        return num(x, 4)
    return text(x)


def sigma_view(inputs: dict, what: str) -> dict:
    """How the measurement's sigma was found (rule fit.sigma_chain): the text "sigma_how" with the chain piece holding
    t {piece, events [[t, kind]], window_short, events_near_t}; without a chain piece the text "sigma" when the
    measurement records one; the whole-fit events whose finite difference was one-sided or unstable."""
    cp = inputs.get("chain_piece")
    if cp is not None:
        w = f"{what} chain_piece"
        how = text(req(inputs, "sigma_how", what))
        chain = {"piece": [num(x, 4) for x in req(cp, "piece", w)],
                 "events": [[num(t, 4), text(k)] for t, k in req(cp, "events", w)],
                 "window_short": plain(req(cp, "window_short", w)), "events_near_t": plain(req(cp, "events_near_t", w))}
        if FMT.get("fused"):
            ch = req(cp, "chain", w)
            if not isinstance(ch, bool) or (not ch and chain["events"]):
                raise ValueError(f"{AT['where']}: {w} chain = {ch!r} with events {chain['events']}")
            chain["chain"] = ch
    else:
        if inputs.get("sigma_how") is not None:
            raise ValueError(f"{AT['where']}: {what} has sigma_how without chain_piece")
        sg = inputs.get("sigma")
        how = text(sg) if isinstance(sg, str) else None
        chain = None
    se = inputs.get("sigma_events") or {}
    flags = {k: plain(se.get(k) or []) for k in ("one_sided", "event_unstable")}
    return {"sigma_how": how, "chain": chain, "sigma_flags": {k: v for k, v in flags.items() if v} or None}


def direct_view(m: dict, what: str) -> dict:
    """A direct measurement's state span (rule window.state), support-plane depth and its silhouette cross-check
    (rule C12); each None when the measurement does not carry it."""
    w = m.get("window")
    win = None
    if w is not None:
        cb = req(w, "clipped_by", f"{what} window")
        win = {"t0": num(req(w, "t0", f"{what} window"), 4), "t1": num(req(w, "t1", f"{what} window"), 4),
               "state": text(req(w, "state", f"{what} window")) or None,
               "start": text(req(cb, "start", f"{what} window clipped_by")) or None,
               "end": text(req(cb, "end", f"{what} window clipped_by")) or None,
               "used": [num(x, 4) for x in w["window"]] if req(w, "window", f"{what} window") else None}
        if FMT.get("roll"):
            inputs = req(m, "inputs", what) or {}
            win |= {"law": text(req(w, "law", f"{what} window")) or None, "check": text(req(w, "check", f"{what} window"))}
            if win["law"] is None:   # rule window.driven: the state's own jitter is the noise of window.smooth
                if win["check"] != "window.smooth" or "window.smooth" not in req(m, "checks", what):
                    raise ValueError(f"{AT['where']}: {what} driven window without the check window.smooth")
                win["driven"] = {"state": text(req(inputs, "driven", f"{what} inputs")),
                                 "noise": [num(x, 3) for x in req(inputs, "noise_px", f"{what} inputs")],
                                 "how": text(req(inputs, "noise_how", f"{what} inputs"))}
    sd = m.get("support_depth")
    sdv = None
    if sd is not None:
        if req(sd, "applies", f"{what} support_depth"):
            z = float(req(sd, "z_plane", f"{what} support_depth"))
            sdv = {"applies": True, "z": num(z, 3), "n": int(req(sd, "n", f"{what} support_depth")),
                   "sigma_plane": num(req(sd, "sigma_plane", f"{what} support_depth"), 3),
                   "sigma_row": num(z * float(req(sd, "rel_noise", f"{what} support_depth")), 3)}
        else:
            sdv = {"applies": False, "reason": text(req(sd, "reason", f"{what} support_depth"))}
    da = (m.get("cross_checks") or {}).get("direct.depth_agree")
    if sdv and sdv["applies"] and da is None:
        raise ValueError(f"{AT['where']}: {what} has a support-plane depth without direct.depth_agree")
    dav = None if da is None else {
        "pass": req(da, "pass", f"{what} direct.depth_agree"), "z_mask": num(req(da, "z_mask", f"{what} direct.depth_agree"), 3),
        "z_plane": num(req(da, "z_plane", f"{what} direct.depth_agree"), 3),
        "diff": num(abs(float(da["z_mask"]) - float(da["z_plane"])), 3),
        "bound": num(req(da, "bound", f"{what} direct.depth_agree"), 3)}
    ce = (m.get("inputs") or {}).get("centred") if FMT.get("static") else None
    cev = None if ce is None else {
        "from": [num(x, 4) for x in req(ce, "from", f"{what} centred")],
        "extremum": text(req(ce, "extremum", f"{what} centred")), "stop": text(req(ce, "stop", f"{what} centred"))}
    return {"wstate": win, "support_depth": sdv, "depth_agree": dav, "centred": cev}


def profile_view(x: dict, what: str) -> dict:
    """One parameter of rule at_bound.profile: the answer at the physical bound and at the bound + 1 sigma inward."""
    w = f"{what} at_bound.profile"
    return {"param": text(req(x, "param", w)), "side": text(req(x, "side", w)), "bound": num(req(x, "bound", w), 3),
            "sigma": num(req(x, "sigma", w), 3), "at": num(req(x, "profiled_at", w), 3),
            "beyond": bool(req(x, "beyond_range", w)), "values": [num(v) for v in req(x, "values", w)],
            "delta": num(req(x, "delta", w), 3), "sigma_answer": num(req(x, "sigma_answer", w), 3),
            "pass": bool(req(x, "pass", w)), "why": text(req(x, "why", w))}


def law_view(law: dict, what: str) -> dict:
    """The ballistic law a measurement was read under (rule B2): {"g", "source"}."""
    return {"g": num(req(law, "g", f"{what} law")), "source": text(req(law, "source", f"{what} law"))}


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
                     "bound": num(req(c, "bound", w), 3), "pass": req(c, "pass", w), "testable": req(c, "testable", w)}
                    | ({"stated": stated_view(req(c, "stated", w), w)} if FMT.get("kept") else {})
                    | ({"untestable": untestable_view(c, w)} if FMT.get("fused") and c.get("pass") is None else {}))
    ids = {r["id"] for r in rows}
    failed_ids = list(req(cs, "failed_ids", "constraints"))
    if not set(failed_ids) <= ids:
        raise ValueError(f"{AT['where']}: constraints failed_ids {failed_ids} not among the residuals {sorted(ids)}")
    if FMT.get("fused"):
        un_ids = list(req(cs, "untestable_ids", "constraints"))
        if sorted(un_ids) != sorted(r["id"] for r in rows if r["pass"] is None) or \
                len(req(cs, "untestable", "constraints")) != len(un_ids):
            raise ValueError(f"{AT['where']}: constraints untestable_ids {un_ids} differ from the residuals with pass null")
    return {"measurements": ms, "chosen": chosen, "agree": agree, "constraints": rows}


def untestable_view(c: dict, what: str) -> dict:
    """A constraint that tests nothing (rule G13 constraint.untestable): its sigma_pred is at least its stated value;
    {rule, sigma_pred, source (sigma_pred's delta-method text), reason}."""
    if req(c, "testable", what) is not False:
        raise ValueError(f"{AT['where']}: {what} pass null but testable {c['testable']!r}")
    u = req(c, "untestable", what)
    w = f"{what} untestable"
    rule = text(req(u, "rule", w))
    if rule != "constraint.untestable":
        raise ValueError(f"{AT['where']}: {w} rule {rule!r}")
    return {"rule": rule, "sigma_pred": num(req(u, "sigma_pred", w), 4), "source": text(req(u, "source", w)),
            "reason": text(req(u, "reason", w))}


def stated_view(st: dict, what: str) -> dict:
    """A constraint's stated number as written (rule G10): its text (null: no text of the unit states it), digits,
    unit, sigma (SI, half a unit of the last written digit) and source."""
    return {"text": text(req(st, "text", f"{what} stated")) or None, "digits": text(req(st, "digits", f"{what} stated")),
            "unit": text(req(st, "unit", f"{what} stated")), "sigma": num(req(st, "sigma", f"{what} stated"), 3),
            "source": text(req(st, "source", f"{what} stated"))}


def dropped_view(c: dict, what: str) -> dict:
    """A repair candidate the loop dropped before the planner: its call and the rule that rejected it with the test's
    reason ("rule repair.applicable: rejected (rule <name>): <reason>")."""
    why = text(req(c, "why", f"{what} dropped candidate"))
    m = re.match(r"^rule repair\.applicable: rejected \(rule ([\w.]+)\):\s*(.*)$", why, re.S)
    call = req(c, "call", f"{what} dropped candidate")
    return {"name": text(req(call, "name", what)), "params": {k: v for k, v in (call.get("params") or {}).items() if v is not None},
            "rule": m.group(1) if m else None, "why": m.group(2) if m else why}


def rounds_view(rec: dict) -> dict | None:
    """The agent rounds of an answer record ("iterations"): per round its tool calls with their results (a rejected
    edit's result starts "rejected: "), the plan's rejected issues, whether it was accepted and why; "kept" = the
    round whose state the answer used (kept_state.iter, 0 = the initial fit). None when the agent made no round."""
    its = req(rec, "iterations", "answer record")
    kept = int(req(req(rec, "kept_state", "answer record"), "iter", "kept_state"))
    if kept and kept not in [it.get("iter") for it in its]:
        raise ValueError(f"{AT['where']}: kept_state.iter {kept} is not an agent round")
    rows = []
    for it in its:
        w = f"agent round {it.get('iter')!r}"
        calls = [{"name": text(req(r, "name", w)), "params": {k: v for k, v in (r.get("params") or {}).items() if v is not None},
                  "result": text(req(r, "result", w))}
                 | ({"test": test_view(r["test"], w)} if FMT.get("pendulum") and r.get("test") else {})
                 for r in req(it, "results", w)]
        rej = [text(req(x, "why", f"{w} rejected")) for x in (it.get("rejected") or [])]
        acc = req(it, "accepted", w)
        row = {"iter": int(req(it, "iter", w)), "layer": text(it.get("layer")) or None, "calls": calls,
               "rejected": rej, "accepted": bool(acc), "why": text(req(it, "why", w))}
        if FMT.get("horizon"):
            row["newly_failed"] = [[text(k), text(v)] for k, v in (it.get("newly_failed") or {}).items()]
        if FMT.get("camera"):
            # a round stopped before its accept test (loop.no_repair) has no fixed / broken record
            fx, br = it.get("checks_fixed"), it.get("checks_broken")
            if (fx is None) != (br is None):
                raise ValueError(f"{AT['where']}: {w} has only one of checks_fixed / checks_broken")
            row["swap"] = None if fx is None else [[text(x) for x in fx], [text(x) for x in br]]
            rb = it.get("rebuilt")
            row["rebuilt"] = None if rb is None else {
                "steps": [text(x) for x in req(rb, "steps", f"{w} rebuilt")], "failed": text(req(rb, "failed", f"{w} rebuilt")) or None,
                "self_test": {k: v for k, v in (req(rb, "self_test", f"{w} rebuilt") or {}).items()
                              if isinstance(v, (int, float, str, bool))} or None}
        if FMT.get("roll"):
            row["candidates"] = [{"for": text(req(c, "for", f"{w} repair candidate")), "name": text(req(req(c, "call", w), "name", w)),
                                  "params": {k: v for k, v in (c["call"].get("params") or {}).items() if v is not None},
                                  "class": text(c.get("class")) or None} for c in (it.get("repair_candidates") or [])]
            row["dropped"] = [dropped_view(c, w) for c in (it.get("repair_candidates_dropped") or [])]
        rows.append(row)
    out = {"kept": kept, "rounds": rows}
    if FMT.get("horizon"):
        st = req(rec, "stop", "answer record")
        tried = [{"iter": x.get("iter"), "name": text(req(x, "name", "stop.tried")),
                  "params": {k: v for k, v in (x.get("params") or {}).items() if v is not None},
                  "class": text(req(x, "class", "stop.tried")), "outcome": text(x.get("outcome"))}
                 for x in (st.get("tried") or [])]
        out["stop"] = {"reason": text(req(st, "reason", "stop")), "round": st.get("round"),
                       "why": text(st.get("why")) or None, "tried": tried}
    return out if rows else None


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
    out = {"s": num(req(sc, "s", "scale")), "source": src, "object": text(sc.get("object")) or None,
           "prior": text(req(sc, "prior", "scale")), "how": text(req(sc, "how", "scale")), "extent": text(ext) or None}
    if FMT.get("gravity"):
        out["checks"] = named_checks(req(sc, "checks", "scale"), list(req(sc, "failed", "scale")), "scale")
    return out


def named_checks(checks: dict, failed: list, what: str) -> list[dict]:
    """[{"name", "pass", "reason"}] of a record's named checks; `failed` must be exactly the ones that did not pass."""
    out = [{"name": k, "pass": bool(req(c, "pass", f"{what} check {k!r}")), "reason": text(req(c, "reason", f"{what} check {k!r}"))}
           for k, c in checks.items()]
    if failed != [c["name"] for c in out if not c["pass"]]:
        raise ValueError(f"{AT['where']}: {what} failed {failed} differs from its checks that did not pass")
    return out


GRAVITY_SOURCES = ("earth", "inferred", "prior", "unknown")


def gravity_view(state: dict):
    """The scene's gravity (scene_fit.gravity_pass, rule B2): g, sigma, source, rule, up source, every evidence entry
    (a ballistic piece with its free-g fit and residual test, or a physics-chain probe) and the gravity checks."""
    if not FMT.get("gravity"):
        return None
    g = req(state, "gravity", "scene")
    src = text(req(g, "source", "gravity"))
    if src not in GRAVITY_SOURCES:
        raise ValueError(f"{AT['where']}: gravity source {src!r}")
    ev = []
    for e in req(g, "evidence", "gravity"):
        w = f"gravity evidence {e.get('object')!r}"
        kind = text(req(e, "kind", w))
        if kind == "ballistic_piece":
            law, adm = req(e, "law", w), req(e, "admissible", w)
            ev.append({"obj": text(req(e, "object", w)), "kind": kind, "t": [num(x, 4) for x in req(e, "t", w)],
                       "n": req(e, "n", w), "g": num(req(e, "g", w)), "sigma": num(req(e, "sigma", w), 3),
                       "law": {"pass": bool(req(law, "pass", f"{w} law")), "run": req(law, "run", f"{w} law"),
                               "n_test": req(law, "n_test", f"{w} law")},
                       "admissible": {"pass": bool(req(adm, "pass", f"{w} admissible")),
                                      "reason": text(req(adm, "reason", f"{w} admissible"))},
                       "use": bool(req(e, "use", w)), "why": text(e.get("why")) or None})
        elif kind == "physics_probe":
            ev.append({"obj": text(req(e, "object", w)), "kind": kind, "g": num(req(e, "g", w)),
                       "rho": num(req(e, "rho", w), 3), "agree": bool(req(e, "agree", w))})
        else:
            raise ValueError(f"{AT['where']}: {w} kind {kind!r}")
    et = g.get("earth_test")
    if (et is not None) != (src == "inferred"):
        raise ValueError(f"{AT['where']}: gravity source {src!r} with earth_test {et!r}")
    return {"g": num(req(g, "g", "gravity")), "sigma": num(req(g, "sigma", "gravity"), 3), "source": src,
            "earth_test": {"pass": bool(req(et, "pass", "earth_test")), "reason": text(req(et, "reason", "earth_test"))}
            if et is not None else None,
            "rule": text(req(g, "rule", "gravity")), "up_source": text(req(g, "up_source", "gravity")), "evidence": ev,
            "checks": named_checks(req(g, "checks", "gravity"), list(req(g, "failed", "gravity")), "gravity")}


def refmeasure_view(tag: str, unit: str):
    """The unit's reference measurements (pipeline.refmeasure record, keyed "<object>|<quantity>"): object, quantity,
    graduation unit, length in px, its frame and named checks; None when the unit has no record."""
    if not FMT.get("refmeasure"):
        return None
    from pipeline.refmeasure import VERSION as REFM_VERSION
    p = ROOT / "outputs" / f"refmeasure_v{REFM_VERSION}_{tag}" / f"{unit}.json"
    if not p.exists():
        return None
    j = read_json(p, "refmeasure record")
    out = []
    for key, e in j.items():
        w = f"refmeasure {key!r}"
        if req(e, "unit_id", w) != unit:
            raise ValueError(f"{AT['where']}: {w} is for unit {e['unit_id']!r}")
        out.append({"obj": text(req(e, "object", w)), "quantity": text(req(e, "quantity", w)),
                    "method": text(e.get("method")) or None, "unit": text(e.get("unit")) or None,
                    "px": num(e.get("px")), "frame": e.get("frame"), "pass": bool(req(e, "pass", w)),
                    "checks": named_checks(req(e, "checks", w), list(req(e, "failed", w)), w)})
    return out or None


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


def plausible_untestable(state: dict):
    """The plausible parts that could not judge an object (critic.agency_unknown): [[part, [reason]]] or None."""
    parts = req(req(req(state, "checks", "scene"), "plausible", "scene checks"), "parts", "plausible")
    out = [[text(k), [text(x) for x in v["untestable"]]] for k, v in parts.items() if v.get("untestable")]
    return out or None


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
                                  | ({"dims": {k: text(req(req(x, "dims", w), k, w)) for k in ("candidate", "ref")}}
                                     if FMT.get("kept") else {})
                                  for x in c.get("class_size") or []],
                         "span": [pres["first"], pres["last"]] if pres and pres.get("pass") else None,
                         "border": num(c["border_frac"], 2) if c.get("border_frac") is not None else None,
                         "outcome": text(req(c, "outcome", w)), "why": text(c.get("why")) or None})
        out[name] = {"candidates": rows, "chosen": chosen, "untracked": text(untracked) or None}
    return out


def focal_view(state: dict):
    """The scene's focal record (rule focal.gravity_hop): one row per hop of every passive object's airborne edges."""
    fo = req(state, "focal", "scene")
    rows = []
    for x in req(fo, "votes", "focal"):
        w = "focal vote"
        law = req(x, "law", w)
        rows.append({"object": text(req(x, "object", w)), "span": [num(a, 4) for a in req(x, "span", w)],
                     "hop": [num(a, 4) for a in req(x, "hop", w)], "f": num(req(x, "f", w), 4),
                     "sigma": num(req(x, "sigma", w), 3), "law": text(req(law, "result", w)),
                     "n": x.get("n"), "rms": num(x.get("rms"), 3), "used": bool(req(x, "used", w)),
                     "why": text(req(x, "why", w))})
    return {"votes": rows, "f_votes": num(fo.get("f_votes"), 4), "sigma": num(fo.get("sigma"), 3),
            "n_votes": fo.get("n_votes"), "f": num(req(fo, "f", "focal"), 4), "source": text(req(fo, "source", "focal"))}


def horizon_view(state: dict):
    """The support plane's pitch check (rule support.horizon) and the support plane's pitch source; None when the
    scene has no support plane record and no horizon record."""
    sh = req(state, "support_horizon", "scene")
    sp = (req(state, "camera", "scene") or {}).get("support_plane")
    plane = None
    if sp:
        dp = sp.get("depth_plane")
        plane = {"tilt": num(sp.get("tilt_from_up_deg"), 3), "pitch_source": text(sp.get("pitch_source")) or None,
                 "depth_tilt": num(dp.get("tilt_from_up_deg"), 3) if dp else None}
    if sh is None:
        return {"plane": plane, "check": None} if plane else None
    hz = req(sh, "horizon", "support_horizon")
    far = hz.get("far")
    check = {"pass": sh.get("pass"), "reason": text(req(sh, "reason", "support_horizon")), "repaired": bool(sh.get("repaired")),
             "phi_plane": num(sh.get("phi_plane_deg"), 3), "phi_horizon": num(sh.get("phi_horizon_deg"), 3),
             "sigma": num(sh.get("sigma_deg"), 2), "z": num(sh.get("z"), 2),
             "row_plane": num(sh.get("row_plane"), 4), "row_horizon": num(sh.get("row_horizon"), 4),
             "status": text(req(hz, "status", "horizon")), "row": num(hz.get("row"), 4), "sigma_row": num(hz.get("sigma_row"), 2),
             "spread": num(hz.get("spread_px"), 3), "n_found": hz.get("n_found"), "n_frames": hz.get("n_frames"),
             "far": {"band": num(far.get("band"), 3), "support_p95": num(far.get("support_p95"), 3)} if far else None}
    return {"plane": plane, "check": check}


def camera_roll_view(state: dict) -> dict:
    """The scene's roll check (rule camera.roll): the roll the scene used against the roll of the image's
    near-vertical segments, and the pose refit when they disagree."""
    cr = req(state, "camera_roll", "scene")
    ok = req(cr, "pass", "camera_roll")
    out = {"pass": ok, "reason": text(req(cr, "reason", "camera_roll")), "source": text(req(cr, "roll_source", "camera_roll")) or None,
           "fit": num(cr.get("roll_fit_deg"), 6), "how": text(cr.get("roll_fit_how")) or None, "image": None, "refit": None}
    im = cr.get("image")
    if im:
        out["image"] = {"status": text(req(im, "status", "camera_roll image")), "why": text(req(im, "why", "camera_roll image")),
                        "n_lines": im.get("n_lines"), "n_frames": im.get("n_frames")}
    if ok is not None:
        out |= {"image_roll": num(req(cr, "roll_image_deg", "camera_roll"), 6), "sigma": num(req(cr, "sigma_deg", "camera_roll"), 4),
                "z": num(req(cr, "z", "camera_roll"), 3)}
    rf = cr.get("refit")
    if rf is not None:
        if ok is not False:
            raise ValueError(f"{AT['where']}: camera_roll refit without a failed check")
        out["refit"] = {k: [num(req(rf[s], k, f"camera_roll refit {s}"), 6) for s in ("was", "now")]
                        for k in ("roll_deg", "pitch_deg", "dist", "rms")}
    return out


def extents_view(tag: str, unit: str, state: dict | None):
    """The unit's mask extents: per tracked object the frames and overlay-ring frames of pipeline.extents2d format 3
    (rule extent.overlay), and per object, dimension and anchor the blur-free extent the scene used (scene json
    "extents", rule extent.static): value +- sigma, frames used, frames rejected with why. None without either."""
    p = ROOT / "outputs" / f"extents2d_{tag}" / f"{unit}.json"
    rows = []
    if p.exists():
        j = read_json(p, "extents2d record")
        if req(j, "format", "extents2d record") != 3 or req(j, "unit", "extents2d record") != unit:
            raise ValueError(f"{AT['where']}: extents2d record format {j.get('format')} / unit {j.get('unit')!r}")
        kinds = req(j, "tracks", "extents2d record")
        for kind, objs in kinds.items():
            for name, frames in objs.items():
                w = f"extents2d {kind} {name!r}"
                fs = sorted((int(k), v) for k, v in frames.items())
                if not fs:
                    continue
                rings = [float(req(v, "ring_px", w)) for k, v in fs if req(v, "overlay", w)]
                rows.append({"obj": name, "kind": kind if len(kinds) > 1 else None, "n": len(fs), "frames": [fs[0][0], fs[-1][0]],
                             "overlay": len(rings), "ring": num(float(np.median(rings)), 3) if rings else None, "at": []})
    if FMT.get("static") and state is not None:
        ex = req(state, "extents", "scene")
        for name, o in req(state, "objects", "scene").items():
            oe = req(o, "extent", f"scene object {name!r}")
            if (oe or {}) != (ex.get(name) or {}):
                raise ValueError(f"{AT['where']}: object {name!r} extent differs from the scene's extents record")
        for name, dims in ex.items():
            row = next((r for r in rows if r["obj"] == name), None)
            if row is None:
                row = {"obj": name, "kind": None, "n": None, "frames": None, "overlay": None, "ring": None, "at": []}
                rows.append(row)
            for dim, d in dims.items():
                w = f"extents {name!r} {dim}"
                sm = num(req(d, "sigma_mask", w), 3)
                for label, x in req(d, "at", w).items():
                    row["at"].append({"dim": text(dim), "label": text(label), "value": num(req(x, "value", w), 4),
                                      "sigma": num(req(x, "sigma", w), 3), "sigma_mask": sm,
                                      "sigma_mask_how": text(req(d, "sigma_mask_how", w)),
                                      "used": [int(f) for f in req(x, "frames_used", w)],
                                      "rejected": [{"frame": int(req(r, "frame", w)), "speed": num(req(r, "speed_px", w), 3),
                                                    "extent": num(req(r, "extent", w), 3), "why": text(req(r, "why", w))}
                                                   for r in req(x, "frames_rejected", w)],
                                      "how": text(req(x, "how", w))})
    return {"rows": rows} if rows else None


def depth_bindings_view(state: dict):
    """The scene's depth statements (rule C14): how many were given, each bound one with its node, role and what the
    scene used it for (by null: not used, with why), each unbound one with its reason; None when none was given."""
    db = req(state, "depth_bindings", "scene")
    given = req(db, "given", "depth_bindings")
    bound = [{"name": text(req(b, "name", "depth binding")), "node": text(req(b, "node", "depth binding")),
              "as": text(req(b, "as", "depth binding")), "end": text(b.get("end")) or None,
              "by": text(req(req(b, "used", "depth binding"), "by", "depth binding used")) or None,
              "why": text(req(b["used"], "why", "depth binding used"))} for b in req(db, "bound", "depth_bindings")]
    unbound = [{"name": text(req(b, "name", "unbound depth statement")), "reason": text(req(b, "reason", "unbound depth statement"))}
               for b in req(db, "unbound", "depth_bindings")]
    if not given:
        if bound or unbound:
            raise ValueError(f"{AT['where']}: depth_bindings bound / unbound without a given statement")
        return None
    return {"given": len(given), "names": len({text(g.get("name")) for g in given}), "bound": bound, "unbound": unbound}


def camera_one_view(state: dict):
    """The scene camera fused from stage A and the pendulums' joint f (rule camera.one); None when no pendulum votes."""
    co = req(state, "camera_one", "scene")
    if co is None:
        return None
    votes = [{"source": text(req(v, "source", "camera_one vote")), "object": text(req(v, "object", "camera_one vote")),
              "f": num(req(v, "f", "camera_one vote"), 4), "sigma": num(req(v, "sigma", "camera_one vote"), 3),
              "used": bool(req(v, "used", "camera_one vote")), "why": text(req(v, "why", "camera_one vote"))}
             for v in req(co, "votes", "camera_one")]
    if text(req(state, "f_source", "scene")) != "camera.one":
        raise ValueError(f"{AT['where']}: camera_one record with f_source {state.get('f_source')!r}")
    return {"f": num(req(co, "f", "camera_one"), 4), "sigma": num(req(co, "sigma", "camera_one"), 3),
            "how": text(req(co, "how", "camera_one")), "votes": votes}


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
                     "visible": [[num(a, 5), num(b, 5)] for a, b in vis] if vis else None,
                     "pendulum": pendulum_view(req(o, "pendulum", what), what) if FMT.get("pendulum") else None}
                    | ({"segfit": segfit_view(name, o), "events_graph": events_view(name, o)} if FMT.get("selftest") else {})
                    | (impulse_view(name, o) if FMT.get("impulse") else {}))
    return {"file": str(source_path.relative_to(ROOT)), "f": num(j["f"], 5), "plane": plane, "objects": objs,
            "scale": scale_view(j), "gravity": gravity_view(j),
            "main_support": req(j, "main_support", "scene"),
            "f_source": text(req(j, "f_source", "scene")) or None, "f_first": req(j, "f_first", "scene"),
            "f_first_clamp": clamp_view(req(j, "f_first_clamp", "scene")) if FMT.get("pendulum") else None,
            # an empty vote list is legitimate (no focal vote taken): written as null
            "focal_votes": [{"source": text(req(x, "source", "focal vote")), "value": num(req(x, "value", "focal vote"), 4),
                             "weight": num(req(x, "weight", "focal vote"), 3)}
                            for x in req(j, "focal_votes", "scene")] or None} | \
        ({"focal": focal_view(j), "horizon": horizon_view(j)} if FMT.get("horizon") else {}) | \
        ({"camera": camera_view(j), "support_plane": support_plane_view(j)} if FMT.get("camera") else {}) | \
        ({"camera_roll": camera_roll_view(j)} if FMT.get("roll") else {}) | \
        ({"camera_one": camera_one_view(j)} if FMT.get("static") else {}) | \
        ({"fusion": fusion_view(j)} if FMT.get("fusion") else {}) | \
        ({"incline": incline_view(j)} if FMT.get("impulse") else {})


IMPULSE_VERDICTS = ("kick", "agent.unobserved", "chain ends", "no_law")   # or "rejected: <why>"


def impulse_view(name: str, o: dict) -> dict:
    """The physics record's impulse decisions (rule law.impulse), its refined onsets (kinematic.onset) and incline laws
    (support.incline): {"impulse": [{t, event, agent, source, verdict, why, jump, sigma, jump_how}] | None,
    "onset": [{phase, t, sigma, how}] | None, "incline_law": [{..}] | None}; all None without a physics record."""
    ph = req(o, "physics", f"scene object {name!r}")
    if ph is None:
        return {"impulse": None, "onset": None, "incline_law": None}
    w = f"object {name!r} physics"
    imp = req(ph, "impulse", w)
    if (o.get("motion_law") or {}).get("impulse", imp) != imp:
        raise ValueError(f"{AT['where']}: {w} impulse differs from its motion law's")
    rows = []
    for x in imp:
        v = text(req(x, "verdict", f"{w} impulse"))
        if v not in IMPULSE_VERDICTS and not v.startswith("rejected: "):
            raise ValueError(f"{AT['where']}: {w} impulse verdict {v!r}")
        rows.append({"t": num(req(x, "t", f"{w} impulse"), 4), "event": text(req(x, "event", f"{w} impulse")),
                     "agent": text(req(x, "agent", f"{w} impulse")) or None,
                     "source": text(req(x, "source", f"{w} impulse")) or None, "verdict": v,
                     "why": text(req(x, "why", f"{w} impulse")), "jump": num(x.get("jump_px_s"), 3),
                     "sigma": num(x.get("sigma_px_s"), 3), "jump_how": text(x.get("jump_how")) or None})
    onset = [{"phase": int(k), "t": num(req(v, "t", f"{w} onset"), 5), "sigma": num(req(v, "sigma", f"{w} onset"), 3),
              "how": text(req(v, "how", f"{w} onset"))} for k, v in req(ph, "onset", w).items()]
    il = []
    for x in req(ph, "incline_law", w):
        wl = f"{w} incline_law"
        il.append({"phase": int(req(x, "phase", wl)), "support": text(req(x, "support", wl)),
                   "t_on": num(req(x, "t_on", wl), 5), "sigma_t_on": num(req(x, "sigma_t_on", wl), 3),
                   "accel": num(req(x, "accel", wl), 4), "sigma_accel": num(req(x, "sigma_accel", wl), 3),
                   "accel_source": text(req(x, "accel_source", wl)), "accel_geometry": num(req(x, "accel_geometry", wl), 4),
                   "kappa": num(req(x, "kappa", wl), 4), "shape": text(req(x, "shape", wl)),
                   "theta": num(req(x, "theta_deg", wl), 4), "law": text(req(x, "law", wl))})
    return {"impulse": rows or None, "onset": onset or None, "incline_law": il or None}


def incline_view(state: dict):
    """The measured slope of every inclined support (rule support.incline): [{support, method, theta, sigma, sigma_f,
    length, inclined, up {level, angle, sigma, source, why}, why}] | None."""
    out = []
    for name, x in req(state, "support_incline", "scene").items():
        w = f"support_incline {name!r}"
        up = req(x, "up", w)
        out.append({"support": text(name), "method": text(req(x, "method", w)), "theta": num(req(x, "theta_deg", w), 4),
                    "sigma": num(req(x, "sigma_deg", w), 3), "sigma_f": num(x.get("sigma_f_deg"), 3),
                    "sigma_f_how": text(x.get("sigma_f_how")) or None, "length": num(x.get("length_m"), 4),
                    "inclined": bool(req(x, "inclined", w)),
                    "up": {"level": bool(req(up, "level", f"{w} up")), "angle": num(req(up, "angle_deg", f"{w} up"), 3),
                           "sigma": num(req(up, "sigma_deg", f"{w} up"), 3), "source": text(req(up, "source", f"{w} up")),
                           "why": text(req(up, "why", f"{w} up"))},
                    "why": text(req(x, "why", w))})
    return out or None


def presence_view(tag: str, unit: str):
    """The unit's presence verdicts (outputs/objects_<tag>_presence_v4/<unit>.json, rule decisions.no_conf): [[name,
    present (visible / partial / absent / null), rule, votes, zoom_votes | null]]."""
    j = read_json(ROOT / "outputs" / f"objects_{tag}_presence_v4" / f"{unit}.json", "presence record")
    out = []
    for name, v in j.items():
        w = f"presence of {name!r}"
        if any("conf" in k for k in v):
            raise ValueError(f"{AT['where']}: {w} has a confidence field; format 4 decides by votes")
        rule = text(req(v, "rule", w))
        if not rule.startswith("presence."):
            raise ValueError(f"{AT['where']}: {w} rule {rule!r}")
        out.append([text(name), req(v, "present", w), rule, [text(x) for x in req(v, "votes", w)],
                    [text(x) for x in v["zoom_votes"]] if v.get("zoom_votes") else None])
    return out or None


FUSION_VERDICTS = ("used", "outlier", "disagree")


def fusion_view(state: dict) -> dict:
    """The scene's focal votes fused by independent source (rule focal.fusion): one row per source (object x kind)
    with its fused f +- sigma, votes, whether it is one of camera.f's own sources, verdict and why; the fused f +- sigma,
    the number of sources, pass (None: untestable) and reason, and camera.f the fits keep."""
    fu = req(req(state, "focal", "scene"), "fusion", "focal")
    w = "focal.fusion"
    rows = []
    for x in req(fu, "sources", w):
        v = text(req(x, "verdict", f"{w} source"))
        if v not in FUSION_VERDICTS:
            raise ValueError(f"{AT['where']}: {w} verdict {v!r}")
        b = x.get("bounded")
        rows.append({"object": text(req(x, "object", f"{w} source")), "kind": text(req(x, "kind", f"{w} source")),
                     "f": num(req(x, "f", f"{w} source"), 4), "sigma": num(req(x, "sigma", f"{w} source"), 3),
                     "n": int(req(x, "n_votes", f"{w} source")), "camera": bool(req(x, "camera", f"{w} source")),
                     "how": text(req(x, "how", f"{w} source")), "verdict": v, "why": text(req(x, "why", f"{w} source")) or None,
                     "bounded": [num(a, 4) for a in b] if b else None})
    n = int(req(fu, "n_sources", w))
    if n != len(rows):
        raise ValueError(f"{AT['where']}: {w} n_sources {n} with {len(rows)} sources")
    ok = req(fu, "pass", w)
    if ok not in (True, False, None):
        raise ValueError(f"{AT['where']}: {w} pass = {ok!r}")
    return {"sources": rows, "n": n, "f": num(fu.get("f"), 4), "sigma": num(fu.get("sigma"), 3),
            "camera_f": num(req(fu, "camera_f", w), 4), "camera_sigma": num(req(fu, "camera_sigma", w), 3),
            "pass": ok, "reason": text(req(fu, "reason", w))}


def camera_view(state: dict) -> dict:
    """The scene's one camera (rule C3a): f +- sigma, how sigma was found, its source, and each focal.consistent
    cross-check of a focal length one prior or one pendulum implies."""
    cam = req(state, "camera", "scene")
    rows = []
    for x in req(req(state, "focal", "scene"), "consistent", "focal"):
        w = "focal.consistent"
        fo, f = req(x, "f_obj", w), req(x, "f", w)
        cs = x.get("class_size")
        ind = req(x, "independent", w) if FMT.get("static") else None
        if FMT.get("static") and not isinstance(ind, bool):
            raise ValueError(f"{AT['where']}: focal.consistent independent = {ind!r}")
        if ind is False and x.get("pass") is not None:
            raise ValueError(f"{AT['where']}: a vote that is not independent has a verdict")
        xe = x.get("extent") if FMT.get("static") else None
        ext = None if xe is None else {"value": num(req(xe, "value", w), 4), "sigma": num(req(xe, "sigma", w), 3),
                                       "used": [int(f) for f in req(xe, "frames_used", w)],
                                       "rejected": len(req(xe, "frames_rejected", w))}
        rows.append({"independent": ind, "extent": ext} | {"source": text(req(x, "source", w)), "f_obj": num(fo, 4), "sigma_obj": num(req(x, "sigma_obj", w), 3),
                     "f": num(f, 4), "sigma": num(req(x, "sigma", w), 3), "bound": num(x.get("bound"), 3),
                     "diff": num(abs(float(fo) - float(f)), 3) if fo is not None and f is not None else None,
                     "pass": req(x, "pass", w), "reason": text(req(x, "reason", w)), "how": text(req(x, "how", w)),
                     "class_size": {"pass": bool(req(cs, "pass", w)), "reason": text(req(cs, "reason", w))} if cs else None})
    out = {"f": num(req(cam, "f", "camera"), 4), "sigma": num(req(cam, "f_sigma", "camera"), 3),
           "how": text(req(cam, "f_sigma_how", "camera")), "source": text(req(state, "f_source", "scene")) or None,
           "consistent": rows}
    if FMT.get("fused"):
        out |= {"fused": fused_view(state, out), "agency_unknown": agency_unknown_view(state)}
    return out


def fused_view(state: dict, cam: dict):
    """camera.f as the fused focal estimate (rule C1a camera.fused): {f, sigma, n, how, was {f, source}}; None when
    camera.f keeps its own source."""
    fo = req(state, "focal", "scene")
    if cam["source"] != "camera.fused":
        if fo.get("fused") is not None:
            raise ValueError(f"{AT['where']}: focal.fused with f_source {cam['source']!r}")
        return None
    fz, w = req(fo, "fused", "focal"), "focal.fused"
    was = req(fz, "was", w)
    out = {"f": num(req(fz, "f", w), 4), "sigma": num(req(fz, "sigma", w), 3), "n": int(req(fz, "n_sources", w)),
           "how": text(req(fz, "how", w)),
           "was": {"f": num(req(was, "f", f"{w} was"), 4), "source": text(req(was, "source", f"{w} was"))}}
    if out["f"] != cam["f"]:
        raise ValueError(f"{AT['where']}: focal.fused f {out['f']} is not camera.f {cam['f']}")
    return out


def agency_unknown_view(state: dict):
    """Objects whose agency is unknown cast no gravity focal vote (rule agency.unknown): [{object, why}] or None."""
    rows = req(state, "focal", "scene").get("agency_unknown") or []
    return [{"object": text(req(x, "object", "focal.agency_unknown")), "why": text(req(x, "why", "focal.agency_unknown"))}
            for x in rows] or None


def support_plane_view(state: dict):
    """The named check support.plane (rule D9) with its sub-tests in the scene's order; pass None with the reason and
    no sub-test when the scene has no support plane."""
    sp = req(state, "support_plane", "scene")
    if req(sp, "pass", "support_plane") is None:
        if req(sp, "tests", "support_plane"):
            raise ValueError(f"{AT['where']}: support_plane without a result has sub-tests")
        return {"pass": None, "reason": text(req(sp, "reason", "support_plane")), "tests": []}
    tests = [{"name": k, "pass": req(t, "pass", f"support.plane {k}"), "reason": text(req(t, "reason", f"support.plane {k}"))}
             for k, t in req(sp, "tests", "support_plane").items()]
    failed = list(req(sp, "failed", "support_plane"))
    if failed != [t["name"] for t in tests if t["pass"] is False] or bool(req(sp, "pass", "support_plane")) == bool(failed):
        raise ValueError(f"{AT['where']}: support_plane pass / failed {failed} differ from its sub-tests")
    return {"pass": bool(sp["pass"]), "reason": None, "tests": tests}


def pendulum_view(p: dict | None, what: str):
    """A pendulum object's record (rules scale.converged / pendulum.period): fitted L, the swing period and its sigma,
    and the focal length solved jointly with the swing under the stated size (f_free). None for a non-pendulum."""
    if p is None:
        return None
    w = f"{what} pendulum"
    ff = req(p, "f_free", w)
    out = {"L": num(req(p, "L", w), 4), "period": num(req(p, "period_s", w), 4), "period_sigma": num(req(p, "period_sigma_s", w), 3),
           "periods": req(p, "periods", w), "period_from": text(req(p, "period_from", w)), "f": num(req(p, "f", w), 4),
           "f_free": None}
    if ff is not None:
        wf = f"{w} f_free"
        ab = req(ff, "at_bound", wf)
        out["f_free"] = {"f": num(req(ff, "f", wf), 4), "f_start": num(req(ff, "f_start", wf), 4),
                         "bounds": [num(x, 4) for x in req(ff, "bounds", wf)], "at_bound": text(ab) if ab is not None else None,
                         "D": num(req(ff, "D", wf), 4), "D_prior": num(req(ff, "D_prior", wf), 4),
                         "rel": num(req(ff, "rel", wf), 3), "converged": bool(req(ff, "converged", wf))}
    return out


def clamp_view(c: dict | None):
    """The first focal estimate clamped into the person-height band (scene f_first_clamp), reported; None if unclamped."""
    if c is None:
        return None
    return {"from": num(req(c, "from", "f_first_clamp"), 4), "to": num(req(c, "to", "f_first_clamp"), 4),
            "band": [num(x, 4) for x in req(c, "band", "f_first_clamp")]}


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


def failed_unit(rec: dict, row: dict) -> dict:
    """A unit whose chain stopped (answer record "execution_status" failed): its "error" as the harness wrote it (an
    unresolved inventory or scene graph names its reason, objects.llm_invalid / scenegraph.llm_invalid), the steps it
    ran; no answer, no scene."""
    err = text(req(rec, "error", "answer record"))
    if not err or req(rec, "answer", "answer record") is not None:
        raise ValueError(f"{AT['where']}: failed unit with error {err!r} and an answer")
    if text(row["status"]) != "failed" or text(row["source"]) != "execution_failed" or num(row["pred"]) is not None:
        raise ValueError(f"{AT['where']}: failed unit with CSV status {row['status']!r} / source {row['source']!r}")
    return {"error": err, "execution_status": "failed", "time": timing_view(req(rec, "timing", "answer record"),
                                                                           row.get("seconds"), prefix=True),
            "checks": [], "failed": [], "scene": None, "sgraph": None, "tracks": None, "overlays": [], "fit": None,
            "rounds": None, "identity": None, "refmeasure": None, "extents": None, "depth_bindings": None,
            "measurement": None, "provenance": None, "quality": {}, "align": None, "group": None, "binding_error": None}


def timing_view(t: dict, csv_seconds, prefix: bool = False) -> dict:
    """A unit's timing ("timing" of its answer record) for the page: total seconds and one entry per step, in pipeline
    order: {"step", "s", "reused" (false | "partial" | true), "cached", "shared"}. The CSV "seconds" must be the total.
    prefix: a failed unit, whose steps are the first steps of the order."""
    total = req(t, "total_seconds", "timing")
    if not (isinstance(csv_seconds, (int, float)) and math.isfinite(csv_seconds) and abs(csv_seconds - total) < 0.05):
        raise ValueError(f"{AT['where']}: CSV seconds {csv_seconds} differs from the answer record's total {total}")
    steps, order = req(t, "steps", "timing"), FMT["steps"]
    if prefix:
        order = order[:len(steps)]
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
                req(row, "conf", "harness CSV row")    # read as the format's field, not shown (decisions.no_conf)
            if unit_kind == "video":
                answer, group, error = bind_harness_question(row, record["answers"], groups)
                state = group["state"] if group else {}
                result = own_result(state, row) if group else None
                exec_status = record["execution_status"]
            else:
                rec = question_record(tag, unit, row)
                if FMT.get("fused") and req(rec, "execution_status", "answer record") == "failed":
                    query |= failed_unit(rec, row)
                    query["tq"], query["tjump"] = question_times(query["q"], None)
                    questions.append(query)
                    continue
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
                query["refmeasure"] = refmeasure_view(tag, unit)
                if FMT.get("roll"):
                    query["extents"] = extents_view(tag, unit, state)
                if FMT.get("static"):
                    query["depth_bindings"] = depth_bindings_view(state) if state is not None else None
                query |= {"scene": scene_view(state, DIRS["scenes"] / f"{unit}.json") if state is not None else None,
                          "sgraph": sgraph(unit, rec, state), "tracks": track_intervals(unit)}
                if FMT.get("kept"):
                    query["rounds"] = rounds_view(rec)
                if FMT.get("identity"):
                    query |= {"identity": identity_view(unit),
                              "dims_check": dims_check(state) if state is not None else None}
                if FMT.get("fused"):
                    query["plausible_untestable"] = plausible_untestable(state) if state is not None else None
                if FMT.get("impulse"):
                    query["presence"] = presence_view(tag, unit)
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
                      "csv": str(p.relative_to(ROOT)), "unit_set": unit_set(t, frames[t], p)} for t, p in runs],
            "dev": dev, "videos": [videos[v] for v in sorted(videos)], "questions": questions}


def unit_set(tag: str, frame: pd.DataFrame, csv_path: Path):
    """The unit set a run's CSV names (the harness names its CSV by the set: "_dev10"; the full split has no suffix);
    None for formats without it. One value per CSV, and the file name must carry it."""
    if not version_format(tag).get("static"):
        return None
    vals = sorted({text(x) for x in req(frame, "unit_set", f"{csv_path.name}")})
    if len(vals) != 1 or not vals[0] or \
            not csv_path.stem.endswith("_answers" if vals[0] == "full" else f"_answers_{vals[0]}"):
        raise ValueError(f"{csv_path.name}: unit_set {vals} does not name this CSV")
    return vals[0]


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
    used = {Path(v["thumb"]).name for v in data["val"]["videos"]}
    stale = [p for p in THUMBS.glob("*.jpg") if p.name not in used]
    for p in stale:
        p.unlink()
    if stale:
        print(f"{len(stale)} unreferenced thumbs removed")
    sizes = [p.stat().st_size for p in THUMBS.glob("*.jpg")]
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB); {len(sizes)} thumbs, {sum(sizes) / 1e6:.2f} MB, max {max(sizes, default=0)} B")


if __name__ == "__main__":
    main()
