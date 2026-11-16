"""Build web/data/videos.json (+ first-frame thumbnails) for evaluation.html from one run on the validation set.

Reads only the run's outputs; never runs the pipeline. The page lists one row per question; a question is the
unit <video_id>_q<k> (k = its 1-based order among the video's rows of the split table, docs/PER_QUESTION.md).

  python -B web/tools/build_videos.py --run main

--run TAG names the run to read; a run tag is only a name. The harness stops after tracking (objects, scene graph
grid, track, focal); the scene program does the rest (masks, build, fit, answer). Inputs of run <tag> (relative to the
repo root), one per question unit:
  outputs/sceneprog/<tag>/<unit>/                      answer.json (answer_scene, answer_final), fit.json, plan.json,
                                                       the chain's overlay frames, plan grid, shapes, scene.json
  outputs/harness/<tag>/answers/<unit>.json            harness record (its timing per step)
  web/media/sceneprog/<tag>                            git-ignored symlink to outputs/sceneprog/<tag> (plan grid, shapes)
Also docs/DEV20.md, the development question set -> val "dev". A missing product is an error, never a substitute.

Writes data/videos.json ({"generated", "val": {"dev", "videos", "questions", "sceneprog": {"build": <tag>}}}),
data/comparison_ours.json (write_comparison_ours), the redrawn overlays under data/runtime/<tag>/ with their record
overlays.json (run_overlays), and two tables of pipeline.html: the class sizes from pipeline/class_sizes.py
(write_class_table) and the harness time per step (write_run_time_table).

The page plays the original videos from media/val (git-ignored symlinks, see video_info).
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
from pipeline.parse import parse_question  # noqa: E402

OUT = WEB / "data" / "videos.json"
THUMBS = WEB / "data" / "thumbs"
THUMB_W = 240
THUMB_MAX_BYTES = 30_000
AT = {"where": ""}  # "<run> <unit>" being read, for error messages


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


def req(d: dict, key: str, what: str):
    """d[key], a field every artifact of its format has; its absence is a format mismatch, not a default."""
    if key not in d:
        raise ValueError(f"{AT['where']}: {what} has no {key!r} field")
    return d[key]


def read_json(p: Path, what: str):
    if not p.exists():
        raise ValueError(f"{what} missing: {p.relative_to(ROOT)}")
    return json.loads(p.read_text())


def question_times(q: str, alignment: dict | None):
    """t_q of a question for the page: the times it names (pipeline.parse), and the time to jump to — the named time
    (midpoint of a [t1, t2] window); for a question that names no time, the frame its alignment check used."""
    times = [float(t) for t in parse_question(q).times]
    jump = (sum(times[:2]) / len(times[:2])) if times else (alignment or {}).get("t")
    return [num(t, 4) for t in times], num(jump, 4)


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


def manifest_units() -> pd.DataFrame:
    """The validation table with each row's unit id (<video_id>_q<k>, k = order of the row within its video)."""
    manifest = load_split("val").reset_index(drop=True)
    manifest["k"] = manifest.groupby("video_id", sort=False).cumcount() + 1
    manifest["unit"] = [unit_id(v, int(k)) for v, k in zip(manifest["video_id"], manifest["k"])]
    return manifest


def unit_id(vid: str, k: int) -> str:
    """The unit of one question: <video_id>_q<k>, k = its 1-based order among the video's rows of the split table."""
    return f"{vid}_q{k}"


def dev_units() -> list[str]:
    """The development question set: the unit ids in the first column of the table in docs/DEV20.md."""
    units = re.findall(r"^\|\s*(\S+_q\d+)\s*\|", (ROOT / "docs" / "DEV20.md").read_text(), re.M)
    if not units or len(set(units)) != len(units):
        raise ValueError(f"docs/DEV20.md: no or duplicated unit ids: {units}")
    return units


SP_MEDIA = "media/sceneprog"          # git-ignored symlinks media/sceneprog/<tag> -> outputs/sceneprog/<tag>
SP_RAW: dict[str, tuple] = {}         # unit -> (answer_final, answer_scene or None) unrounded, for comparison_ours


def sp_media(tag: str, unit: str, name: str) -> str:
    """The page path of one scene-program product; the file must exist behind the media symlink."""
    p = WEB / SP_MEDIA / tag / unit / name
    if not p.is_file():
        raise ValueError(f"{unit}: {SP_MEDIA}/{tag}/{unit}/{name} missing (symlink {SP_MEDIA}/{tag} -> outputs/sceneprog/{tag})")
    return f"{SP_MEDIA}/{tag}/{unit}/{name}"


def sp_mra(pred, truth) -> float:
    """MRA of one answer: the baselines' rule (tools/build_baseline.py mra), so our row and theirs are scored alike."""
    from build_baseline import mra
    return mra(None if pred is None else float(pred), float(truth))


# ---------- the run (--run TAG): harness until track, then masks -> build -> fit -> answer ----------
# The harness stops after tracking (pipeline.harness --until track: objects, scene graph grid, track, focal), so a
# question has no harness scene; everything shown comes from outputs/sceneprog/<TAG>/<unit>/ (answer.json with both
# answers, fit.json, plan.json, the plan grid and the shapes) and the harness record outputs/harness/<TAG>/answers/<unit>.json
# (its timing). The fitted program's overlays are redrawn at the frames the chain drew (tools/sceneprog_overlay.py,
# pipeline.palette) into data/runtime/<TAG>/<unit>/, recorded in data/runtime/<TAG>/overlays.json with the hash of
# the drawing code; an image whose record matches the current code and program is kept.
RUN_STEPS = ("objects", "scenegraph_grid", "track", "focal")
RUN_STEP_LABELS = {"objects": "1 Objects", "scenegraph_grid": "3 Scene graph", "track": "4 Track", "focal": "5 Focal"}


def drawing_hash() -> str:
    files = [WEB / "tools" / "sceneprog_overlay.py", WEB / "tools" / "overlay_style.py", ROOT / "pipeline" / "palette.py",
             ROOT / "pipeline" / "sceneprog" / "render.py", WEB / "site.css"]
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()[:16]


def run_overlays(tag: str, units: list[str], workers: int) -> dict[str, list[str]]:
    """{unit: [page paths of its redrawn overlays]}; draws only what the record lacks (or drew with other code or from
    another program), with `workers` processes; drops images no unit references."""
    from concurrent.futures import ProcessPoolExecutor
    out_root = WEB / "data" / "runtime" / tag
    rec_file = out_root / "overlays.json"
    record = json.loads(rec_file.read_text()) if rec_file.exists() else {}
    h = drawing_hash()
    todo, result = [], {}
    for u in units:
        d = ROOT / "outputs" / "sceneprog" / tag / u
        frames = sorted(p.name for p in d.glob("overlay_f*.jpg"))
        if not frames:
            raise ValueError(f"{u}: no overlay_f*.jpg in {d.relative_to(ROOT)}")
        scene = hashlib.sha256((d / "scene.json").read_bytes()).hexdigest()[:16]
        paths = [f"data/runtime/{tag}/{u}/{n}" for n in frames]
        e = record.get(u)
        if not (e and e["drawing"] == h and e["scene"] == scene and e["paths"] == paths
                and all((WEB / p).is_file() for p in paths)):
            todo.append(u)
            record[u] = {"drawing": h, "scene": scene, "paths": paths}
        result[u] = paths
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for u, outs in zip(todo, ex.map(_draw_unit, [(tag, u) for u in todo])):
                if [f"data/runtime/{tag}/{u}/{p.name}" for p in outs] != result[u]:
                    raise ValueError(f"{u}: redrawn frames differ from the chain's")
    for k in set(record) - set(units):
        record.pop(k)
    keep = {p for ps in result.values() for p in ps}
    stale = [p for p in out_root.rglob("*.jpg") if str(p.relative_to(WEB)) not in keep]
    for p in stale:
        p.unlink()
    for d in sorted(out_root.glob("*/"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    out_root.mkdir(parents=True, exist_ok=True)
    rec_file.write_text(json.dumps(record, indent=1, sort_keys=True))
    print(f"{tag}: {len(todo)} units redrawn ({sum(len(result[u]) for u in todo)} overlays), {len(stale)} stale images removed")
    return result


def _draw_unit(arg):
    tag, unit = arg
    sys.path.insert(0, str(WEB / "tools"))
    from sceneprog_overlay import draw_unit
    return draw_unit(ROOT / "outputs" / "sceneprog" / tag / unit, WEB / "data" / "runtime" / tag / unit)


def run_answer(tag: str, unit: str, truth) -> dict:
    """One question's products of the run: both answers (answer.json: answer_scene, the strict read-out, undetermined
    = no value; answer_final, the agent's, the reported one) with their MRA, the fit's status, each chain step's time
    (build: plan.json timing; fit: fit.json time_s; the answer record has no timing) and the plan
    grid and shapes of the build."""
    AT["where"] = f"{tag} {unit}"
    bd = ROOT / "outputs" / "sceneprog" / tag / unit
    a = read_json(bd / "answer.json", "scene-program answer")
    if req(a, "schema", "scene-program answer") != "sceneprog-answer/1" or a["unit"] != unit:
        raise ValueError(f"{unit}: answer record schema / unit {a.get('schema')!r} {a.get('unit')!r}")
    sc, fi = req(a, "answer_scene", "answer record"), req(a, "answer_final", "answer record")
    det = bool(req(sc, "determined", "answer_scene"))
    if det and sc.get("value") is None:
        raise ValueError(f"{unit}: a determined read-out without a value")
    fit, plan = read_json(bd / "fit.json", "fit"), read_json(bd / "plan.json", "plan")
    scene_value = sc["value"] if det else None
    SP_RAW[unit] = (fi.get("value"), scene_value)
    return {"scene": num(scene_value), "sigma": num(sc.get("sigma"), 3) if det else None, "determined": det,
            "reason": text(sc.get("reason")) or None, "scene_mra": sp_mra(scene_value, truth),
            "final": num(fi.get("value")), "final_mra": sp_mra(fi.get("value"), truth),
            "unit": text(fi.get("unit") or sc.get("unit")), "rationale": text(fi.get("rationale")) or None,
            "fit_status": text(req(fit, "status", "fit")),
            "time": {"build": num(req(req(plan, "timing", "plan"), "total_s", "plan timing"), 4),
                     "fit": num(req(fit, "time_s", "fit"), 4)},
            "renders": {"plan_grid": sp_media(tag, unit, "plan_grid.jpg"), "shapes": sp_media(tag, unit, "shapes.png")}}


def build_program_val(tag: str, make_thumb: bool, workers: int) -> dict:
    """Every question of the validation table from run `tag` (see RUN_STEPS above). Every unit must have its
    products; a missing one is an error."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError(f"invalid run tag: {tag!r}")
    manifest, dev = manifest_units(), dev_units()
    overlays = run_overlays(tag, list(manifest["unit"]), workers)
    videos, questions = {}, []
    for i, row in manifest.iterrows():
        unit, vid = row["unit"], row["video_id"]
        AT["where"] = f"{tag} {unit}"
        if vid not in videos:
            duration, width, height, thumb = video_meta(vid, make_thumb)
            videos[vid] = {"id": vid, "source": row["video_source"], "video_type": row["video_type"],
                           "fps": num(row["fps"]), "duration": duration, "size": [width, height], "thumb": thumb,
                           **video_info(vid)}
        rec = read_json(ROOT / "outputs/harness" / tag / "answers" / f"{unit}.json", "harness record")
        for key in ("unit", "row", "question", "prior", "depth_info"):
            want = unit if key == "unit" else i if key == "row" else row[key]
            if text(req(rec, key, "harness record")) != text(want):
                raise ValueError(f"{unit}: harness record {key} differs from the validation table")
        t = req(rec, "timing", "harness record")
        steps = req(t, "steps", "timing")
        if list(steps) != list(RUN_STEPS):
            raise ValueError(f"{unit}: harness timing steps {list(steps)}, expected {list(RUN_STEPS)}")
        tq, tjump = question_times(row["question"], None)
        sp = run_answer(tag, unit, row["answer"])
        sp["renders"]["overlays"] = overlays[unit]
        questions.append({"id": unit, "video": vid, "k": int(row["k"]), "row": int(i), "cat": row["category"],
                          "type": row["inference_type"], "q": row["question"], "prior": text(row["prior"]),
                          "depth": text(row["depth_info"]), "answer": num(row["answer"]), "tq": tq, "tjump": tjump,
                          "time": {"total": num(req(t, "total_seconds", "timing"), 5),
                                   "steps": [{"step": k, "s": num(req(steps[k], "seconds", f"timing step {k}"), 4),
                                              "reused": steps[k].get("reused", False),
                                              "cached": bool(steps[k].get("cached", False)),
                                              "shared": steps[k].get("shared")} for k in RUN_STEPS]},
                          "sp": sp})
    print(f"val: {len(questions)} questions ({len(dev)} DEV20), {len(videos)} videos, "
          f"{sum(q['sp']['determined'] for q in questions)} determined read-outs")
    return {"dev": dev, "videos": [videos[v] for v in sorted(videos)], "questions": questions}


def write_run_time_table(questions: list[dict]) -> int:
    """The time-per-step table of pipeline.html (between TIME_TABLE_MARKS) from the harness records' timing of the
    run: median / min / max seconds of each harness step."""
    page = PIPELINE_HTML.read_text()
    a, b = TIME_TABLE_MARKS
    if page.count(a) != 1 or page.count(b) != 1 or page.index(a) > page.index(b):
        raise ValueError(f"{PIPELINE_HTML.name}: step-time table marks missing or out of order")
    cells = lambda v: "".join(f'<td class="n">{x:.1f}</td>' for x in (float(np.median(v)), min(v), max(v)))
    rows = [f'  <tr><td data-i18n>{RUN_STEP_LABELS[k]}</td><td><code>{k}</code></td>'
            f'{cells([q["time"]["steps"][i]["s"] for q in questions])}</tr>' for i, k in enumerate(RUN_STEPS)]
    table = ('\n<div class="scroll-x"><table class="tt">\n'
             '  <tr><th data-i18n>Step</th><th data-i18n>Timing key</th><th class="n" data-i18n>Median (s)</th>'
             '<th class="n" data-i18n>Min</th><th class="n" data-i18n>Max</th></tr>\n'
             + "\n".join(rows) + '\n</table></div>\n')
    new = page[:page.index(a) + len(a)] + table + page[page.index(b):]
    if new != page:
        PIPELINE_HTML.write_text(new)
    return len(questions)


OURS_JSON = WEB / "data" / "comparison_ours.json"


def write_comparison_ours(questions: list[dict], sp: dict) -> int:
    """data/comparison_ours.json, our row of the method comparison (index.html, baseline.html): every question of the
    baselines' set (data/baseline.json) with its category and both MRAs of the scene-program run against that set's
    ground truth (the baselines' rule, sp_mra); the pages compute the scores with the baselines' own rule. Each baseline question must be one of ours."""
    base = read_json(WEB / "data" / "baseline.json", "baseline data")
    byid = {q["id"]: q for q in questions}
    out = {}
    for b in req(base, "questions", "baseline data"):
        q = byid.get(b["id"])
        if q is None or "sp" not in q or q["cat"] != b["cat"]:
            raise ValueError(f"baseline question {b['id']} not among the scene-program questions (or another category)")
        final, scene = SP_RAW[b["id"]]
        out[b["id"]] = {"cat": q["cat"], "final": q["sp"]["final"], "final_mra": sp_mra(final, b["answer"]),
                        "scene_mra": sp_mra(scene, b["answer"])}
    OURS_JSON.write_text(json.dumps({"label": "Ours", "run": [sp["build"]], "questions": out},
                                    ensure_ascii=False, separators=(",", ":")))
    return len(out)


PIPELINE_HTML = WEB / "pipeline.html"
CLASS_TABLE_MARKS = ("<!-- class_sizes: generated by tools/build_videos.py from pipeline/class_sizes.py -->",
                     "<!-- /class_sizes -->")


def write_class_table() -> int:
    """Regenerate the class-size table of pipeline.html (scene program section, rule C15) from pipeline.class_sizes itself, between
    CLASS_TABLE_MARKS: one row per class with what it is, every dim's range lo-hi (m) and sigma, the primary dim, role,
    observe and source, under the module's TABLE_VERSION. Nothing is copied by hand; the marks must be there."""
    from html import escape
    from pipeline.class_sizes import CLASSES, PART_DIMS, size_reference
    page = PIPELINE_HTML.read_text()
    a, b = CLASS_TABLE_MARKS
    if page.count(a) != 1 or page.count(b) != 1 or page.index(a) > page.index(b):
        raise ValueError(f"{PIPELINE_HTML.name}: class-size table marks missing or out of order")
    g = lambda x: f"{x:g}"
    rows = []
    for key, c in CLASSES.items():
        ref = size_reference(key)
        dims = "".join(f'<span class="dim">{escape(d["dim"].replace("_", " "))} {g(d["lo"])}–{g(d["hi"])}'
                       f' <span class="muted">σ {g(round(d["sigma"], 6))}</span></span>' for d in ref["dims"])
        rows.append(f'  <tr><td><code>{escape(key)}</code></td><td>{escape(c["what"])}</td><td class="dims">{dims}</td>'
                    f'<td>{escape(c["primary"].replace("_", " "))}{" (part)" if c["primary"] in PART_DIMS else ""}</td>'
                    f'<td>{escape(c["role"])}</td><td>{escape(c["observe"].replace("_", " "))}</td>'
                    f'<td class="small">{escape(c["source"])}</td></tr>')
    # data-i18n: the UI text is translated in i18n.js (the class descriptions and sources are data and stay English)
    table = (f'\n<details class="classtable"><summary><code>pipeline/class_sizes.py</code> · '
             f'<span data-i18n="pl.classes" data-i18n-args=\'{{"n": {len(CLASSES)}}}\'>{len(CLASSES)} classes</span>'
             f'</summary>\n<div class="scroll-x"><table class="small classsizes">\n'
             '  <tr><th data-i18n>Class</th><th data-i18n>What</th><th data-i18n>Dims (m)</th><th data-i18n>Primary</th>'
             '<th data-i18n>Role</th><th data-i18n>Observe</th><th data-i18n>Source</th></tr>\n'
             + "\n".join(rows) + '\n</table></div>\n</details>\n')
    new = page[:page.index(a) + len(a)] + table + page[page.index(b):]
    if new != page:
        PIPELINE_HTML.write_text(new)
    return len(CLASSES)


TIME_TABLE_MARKS = ("<!-- step_times: generated by tools/build_videos.py from the answer records of the current run -->",
                    "<!-- /step_times -->")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", metavar="TAG", required=True,
                    help="the run to show: every question from outputs/sceneprog/<TAG>/<unit>/ and "
                         "outputs/harness/<TAG>/answers/")
    ap.add_argument("--rethumb", action="store_true", help="regenerate thumbnails that already exist")
    ap.add_argument("--workers", type=int, default=3, help="processes redrawing overlays")
    args = ap.parse_args()
    THUMBS.mkdir(parents=True, exist_ok=True)
    data = {"generated": date.today().isoformat(), "val": build_program_val(args.run, args.rethumb, args.workers)}
    data["val"]["sceneprog"] = {"build": args.run}
    print(f"comparison_ours.json: {write_comparison_ours(data['val']['questions'], data['val']['sceneprog'])} baseline questions")
    finish(data)
    print(f"pipeline.html: step-time table, {write_run_time_table(data['val']['questions'])} questions")


def finish(data: dict):
    """Write videos.json, drop unreferenced thumbs, regenerate the class-size table."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    used = {Path(v["thumb"]).name for v in data["val"]["videos"]}
    stale = [p for p in THUMBS.glob("*.jpg") if p.name not in used]
    for p in stale:
        p.unlink()
    if stale:
        print(f"{len(stale)} unreferenced thumbs removed")
    print(f"pipeline.html: class-size table, {write_class_table()} classes")
    sizes = [p.stat().st_size for p in THUMBS.glob("*.jpg")]
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB); {len(sizes)} thumbs, {sum(sizes) / 1e6:.2f} MB, max {max(sizes, default=0)} B")


if __name__ == "__main__":
    main()
