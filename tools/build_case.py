"""One question walked through from the video to the answer: the data and pictures of case.html.

Reads one unit's real products of one run and writes web/data/case/<unit>/ (case.json, the video, the pictures):

  outputs/harness/<tag>/answers/<unit>.json         harness timing per step, paths of its products
  outputs/objects_<tag>/, task_plan_<tag>/, scenegraph_grid_<tag>/, tracks_<tag>/, trackcheck/, trackcoverage/,
  outputs/depth_focal/<video>.json                   object list, detection choice, track checks, focal start
  outputs/logs/harness_<tag>/<unit>.log              the task plan's call count and seconds
  outputs/sceneprog/<tag>/<unit>/                    plan.json, structure.json, shape_check.json, fit.json,
                                                     scene.json / scene.scad, answer.json, plan grid, shapes
  outputs/sceneprog_masks_<tag>/<unit>/              the observed masks
  outputs/driver/<driver>/state.json, logs/<unit>.<step>.1.log   wall time per chain step, every solve of the fit

Pictures that no product holds are drawn from recorded values only: the masks on their frames, the detector's box on
the choice frame, and the program at the fit's recorded start values (fit.json params[].init, the camera start
chosen, the focal start, the plan's segment boundaries) next to the program at its fitted values, both with the
pipeline's own renderer (pipeline.sceneprog.render). No fitting, no model calls.

  python -B web/tools/build_case.py --unit captured_0021x_q1 --tag main --driver main_val
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
sys.path.insert(0, str(ROOT))

from pipeline import palette  # noqa: E402
from pipeline.data import Video  # noqa: E402
from pipeline.sceneprog.masks import ObservedMasks  # noqa: E402
from pipeline.sceneprog.metric import frame_record  # noqa: E402
from pipeline.sceneprog.poses import camera  # noqa: E402
from pipeline.sceneprog.program import load  # noqa: E402
from pipeline.sceneprog.render import render  # noqa: E402
from pipeline.sceneprog.shapes import compile_scene  # noqa: E402

OUT = ROOT / "outputs"
PALETTE = ["#5B8DB8", "#E09A6E", "#92B573", "#C58BC0", "#E5B455", "#6FB5AE", "#C0564F", "#9B93C9", "#DE8C9D",
           "#C2BB66", "#7FA6CF", "#B9BEC4"]
TEXT = (40, 35, 31)  # BGR of the site's --text #1f2328
SCALE = 3            # crops are enlarged this many times (nearest for masks, cubic for the frame)


def jload(p: Path):
    return json.loads(Path(p).read_text())


def bgr(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def contour(image, mask, colour, width):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, cs, -1, colour, width, cv2.LINE_AA)


def observed_outline(image, mask):
    """The site's mark of an observed mask: dark with a white core."""
    contour(image, mask, TEXT, 3)
    contour(image, mask, (255, 255, 255), 1)


def up(a, interp):
    return cv2.resize(a, (a.shape[1] * SCALE, a.shape[0] * SCALE), interpolation=interp)


class Crop:
    """The part of the frame the asked body moves through, with a margin; every crop of the page uses it."""

    def __init__(self, boxes: list, shape: tuple, margin: int = 40):
        x0 = min(b[0] for b in boxes) - margin
        y0 = min(b[1] for b in boxes) - margin
        x1 = max(b[2] for b in boxes) + margin
        y1 = max(b[3] for b in boxes) + margin
        h, w = shape
        self.x0, self.y0 = max(0, int(x0)), max(0, int(y0))
        self.x1, self.y1 = min(w, int(x1)), min(h, int(y1))

    def img(self, a):
        return a[self.y0:self.y1, self.x0:self.x1]

    def box(self):
        return [self.x0, self.y0, self.x1, self.y1]


def log_fit(path: Path) -> dict:
    """Every solve the fit log lists, grouped by the stage it belongs to."""
    screen, refine, joint, shifts = [], [], [], []
    cand = None
    for line in path.read_text().splitlines():
        m = re.search(r"start pitch (-?\d+) yaw (-?\d+) deg: ([\d.e+]+) -> ([\d.e+]+) \(([\d.]+) s\)", line)
        if m:
            screen.append({"pitch_deg": int(m[1]), "yaw_deg": int(m[2]), "before": float(m[3]), "after": float(m[4]),
                           "s": float(m[5])})
            continue
        m = re.search(r"refine start (\d+) round (\d+) \[status \d+\]: (\d+) vars, (\d+) frames, objective ([\d.e+]+) -> "
                      r"([\d.e+]+) in (\d+) evaluations, ([\d.]+) s", line)
        if m:
            refine.append({"start": int(m[1]), "round": int(m[2]), "vars": int(m[3]), "frames": int(m[4]),
                           "before": float(m[5]), "after": float(m[6]), "nfev": int(m[7]), "s": float(m[8])})
            continue
        m = re.search(r"camera candidate start (\d+) \((\d+) of (\d+)\)", line)
        if m:
            cand = int(m[1])
            continue
        m = re.search(r"joint (round|descent step) (\d+) \[status \d+\]: (\d+) vars, (\d+) frames, objective ([\d.e+]+) -> "
                      r"([\d.e+]+) in (\d+) evaluations, ([\d.]+) s", line)
        if m:
            joint.append({"candidate": cand, "kind": m[1], "n": int(m[2]), "vars": int(m[3]), "frames": int(m[4]),
                          "before": float(m[5]), "after": float(m[6]), "nfev": int(m[7]), "s": float(m[8])})
            continue
        m = re.search(r"(\w+): boundary (\S+) moved ([+-]\d+) frames \((\d+) -> (\d+)\)", line)
        if m and cand is None:
            shifts.append({"object": m[1], "frames": int(m[3]), "from": int(m[4]), "to": int(m[5])})
    m = re.search(r"done in (\d+) s; mean of object median IoU ([\d.]+)", path.read_text())
    return {"screen": screen, "refine": refine, "joint": joint, "shifts": shifts,
            "done_s": int(m[1]) if m else None, "iou_median": float(m[2]) if m else None}


def ground_truth(split: str, official_id) -> str | None:
    if split != "val":
        return None
    path = ROOT / "data/QuantiPhy-validation/validation_dataset.csv"
    with path.open(encoding="utf-8-sig") as f:
        rd = csv.reader(f)   # the header repeats the empty name, so columns are read by position
        col = next(rd).index("ground_truth_posterior")
        for row in rd:
            if row and row[0].strip() == str(official_id):
                return row[col].strip()
    return None


def program_at_start(program: dict, fit: dict, structure: dict) -> dict:
    """The program at the fit's recorded start values: every free parameter's init, the focal start, the plan's
    segment boundaries. A parameter without a recorded init keeps its fitted value (the heading of a sphere, the
    untracked table) and is listed."""
    p = copy.deepcopy(program)
    kept = []
    for k, rec in fit["params"].items():
        init = rec.get("init")
        if k == "cam.f":
            p["params"][k]["value"] = fit["f_start"]["value"]
            continue
        if init is None or (isinstance(init, list) and all(v is None for v in init)):
            kept.append(k)
            continue
        if isinstance(init, list):
            cur = list(p["params"][k]["value"])
            p["params"][k]["value"] = [i if i is not None else c for i, c in zip(init, cur)]
        else:
            p["params"][k]["value"] = init
    segs = {o["id"]: o["segments"] for o in structure["objects"]}
    for o in p["objects"]:
        for s, s0 in zip(o.get("segments", []), segs.get(o["id"], [])):
            s["t0"], s["t1"] = s0["t0"], s0["t1"]
    return p, kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--driver", required=True, help="outputs/driver/<name> with the chain's state.json and logs")
    a = ap.parse_args()
    unit, tag = a.unit, a.tag
    sp = OUT / "sceneprog" / tag / unit
    dst = WEB / "data" / "case" / unit
    dst.mkdir(parents=True, exist_ok=True)

    harness = jload(OUT / "harness" / tag / "answers" / f"{unit}.json")
    paths = {k: ROOT / v for k, v in harness["paths"].items()}
    objects = jload(paths["inventory"])
    sgrid = jload(paths["scenegraph_grid"])
    tracks = jload(paths["tracks"])
    tcheck = jload(paths["trackcheck"])
    tcover = jload(paths["trackcoverage"])
    focal = jload(paths["depth_focal"])
    plan = jload(sp / "plan.json")
    structure = jload(sp / "structure.json")
    shape = jload(sp / "shape_check.json")
    fit = jload(sp / "fit.json")
    answer = jload(sp / "answer.json")
    state = jload(OUT / "driver" / a.driver / "state.json")["units"][unit]
    logs = OUT / "driver" / a.driver / "logs"
    fitlog = log_fit(logs / f"{unit}.fit.1.log")
    hlog = paths["log"].read_text()
    m = re.search(rf"task_plan {re.escape(unit)}: status (\w+), calls (\d+), ([\d.]+) s", hlog)
    task_plan = {"status": m[1], "calls": int(m[2]), "s": float(m[3])} if m else None
    m = re.search(rf"{re.escape(unit)}: (\d+) masks .* in (\d+) s", (logs / f"{unit}.masks.1.log").read_text())
    masks_log = {"n": int(m[1]), "s": int(m[2])} if m else None
    m = re.search(r"requests (\d+) \(new (\d+), memo (\d+)\); (\d+) s", (logs / f"{unit}.answer.1.log").read_text())
    answer_log = {"requests": int(m[1]), "s": int(m[4])} if m else None

    # colours: one per object of the plan, in plan order with the asked object first
    ids = [o["id"] for o in plan["plan"]["objects"]]
    asked = plan["query"]["object"]
    ids.sort(key=lambda i: i != asked)
    colours = {oid: PALETTE[k] for k, oid in enumerate(ids)}

    # the video
    vpath = Path(structure["video"]["path"])
    # the clip re-encoded for browsers (the source may be MPEG-4 part 2, which browsers do not play): H.264 and VP9
    for out, codec in (("video.mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart"]),
                       ("video.webm", ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-crf", "32", "-b:v", "0"])):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(vpath), "-an", *codec, str(dst / out)], check=True)
    fps = float(structure["video"]["fps"])
    video = Video(vpath, fps=fps)
    frame = lambda i: cv2.cvtColor(video.frame(i), cv2.COLOR_RGB2BGR)  # noqa: E731

    track_names = list(tracks["tracks"])
    tmain = track_names[0] if track_names else None
    body_of_track = {o["track"]: o["id"] for o in plan["plan"]["objects"] if o.get("track")}
    shape_hw = (structure["video"]["height"], structure["video"]["width"])
    all_boxes = [b for n in track_names for b in tracks["tracks"][n]["boxes"].values()]
    crop = Crop(all_boxes, shape_hw)

    # 3. the detector's boxes on the choice frame
    detect = {}
    for name in track_names:
        ch = tracks["meta"]["choose"][name]
        f = ch["frame"]
        img = frame(f).copy()
        col = bgr(colours[body_of_track[name]])
        for k, b in enumerate(ch["boxes"]):
            chosen = k in ch.get("choices", [ch.get("choice")])
            x0, y0, x1, y1 = (int(round(v)) for v in b)
            cv2.rectangle(img, (x0, y0), (x1, y1), palette.darker(col) if chosen else (150, 150, 150), 2)
        out = f"detect_{body_of_track[name]}.jpg"
        cv2.imwrite(str(dst / out), up(crop.img(img), cv2.INTER_CUBIC), [cv2.IMWRITE_JPEG_QUALITY, 88])
        detect[name] = {"frame": f, "image": out, "n_candidates": ch["n_candidates"],
                        "boxes": [[round(v, 1) for v in b] for b in ch["boxes"]], "choice": ch["choice"],
                        "reason": ch["reason"], "rejected": ch["rejected"], "scan": ch["scan"],
                        "choice_frame": ch["choice_frame"], "motion": ch["motion"], "binding": ch["binding"],
                        "crop": crop.box()}

    # 4. masks over time
    obs = ObservedMasks(tag, unit, structure["split"])
    strip = []
    for name in track_names:
        frames = obs.frames(name)
        pick = sorted(set(frames[:: max(1, len(frames) // 8)] + [frames[-1]]))
        for f in pick:
            img = up(crop.img(frame(f)), cv2.INTER_CUBIC)
            msk = up(crop.img(obs.get(name, f)).astype(np.uint8), cv2.INTER_NEAREST) > 0
            observed_outline(img, msk)
            out = f"mask_{body_of_track[name]}_f{f:04d}.jpg"
            cv2.imwrite(str(dst / out), img, [cv2.IMWRITE_JPEG_QUALITY, 86])
            strip.append({"object": body_of_track[name], "frame": f, "t": round(f / fps, 3), "image": out,
                          "area_px": int(obs.get(name, f).sum())})

    # 9-10. the program at its start values and at its fitted values, on the same frames
    program, scad = load(sp)
    meshes = compile_scene(program, scad)
    start_prog, start_kept = program_at_start(program, fit, structure)
    pair_frames = sorted({int(p.stem.removeprefix("overlay_f")) for p in sp.glob("overlay_f*.jpg")})
    pairs = []
    for f in pair_frames:
        row = {"frame": f, "t": round(f / fps, 3)}
        for which, prog in (("start", start_prog), ("fitted", program)):
            cam = camera(prog)
            r = render(prog, meshes, f / fps, cam)
            img = frame(f).copy()
            img = up(crop.img(img), cv2.INTER_CUBIC)
            ious = {}
            for oid, name in ((o["id"], o["track"]) for o in prog["objects"] if o.get("track")):
                o_m = obs.get(name, f)
                rr = r.get(oid)
                if o_m is not None:
                    others = [obs.get(n, f) for n in obs.names() if n != name]
                    ious[oid] = round(frame_record(rr, o_m, [x for x in others if x is not None])["iou"], 3)
                    observed_outline(img, up(crop.img(o_m).astype(np.uint8), cv2.INTER_NEAREST) > 0)
                if rr is not None:
                    col = palette.darker(bgr(colours[oid]))
                    contour(img, up(crop.img(rr["silhouette"]).astype(np.uint8), cv2.INTER_NEAREST) > 0, col, 1)
                    contour(img, up(crop.img(rr["visible"]).astype(np.uint8), cv2.INTER_NEAREST) > 0, col, 3)
            out = f"program_{which}_f{f:04d}.jpg"
            cv2.imwrite(str(dst / out), img, [cv2.IMWRITE_JPEG_QUALITY, 86])
            row[which] = {"image": out, "iou": ious}
        pairs.append(row)
    video.close()

    # pictures the products hold
    copied = []
    for name in ["plan_grid.jpg"] + sorted(p.name for p in sp.glob("shape_*.png")):
        shutil.copyfile(sp / name, dst / name)
        copied.append(name)

    case = {
        "unit": unit,
        "split": structure["split"],
        "video": {"files": ["video.webm", "video.mp4"], **{k: structure["video"][k] for k in ("fps", "n_frames", "width", "height")}},
        "question": plan["question"],
        "prior": plan["prior"],
        "depth_info": plan["depth_info"],
        "statements": plan["statements"],
        "ground_truth": ground_truth(structure["split"], harness.get("official_id")),
        "colours": colours,
        "names": {o["id"]: o["name"] for o in plan["plan"]["objects"]},
        "harness_timing": harness["timing"],
        "task_plan": task_plan,
        "objects": {"list": objects["objects"], "rendered": objects.get("rendered"),
                    "reference_rejected": objects.get("reference_rejected", [])},
        "scenegraph": {"edges": sgrid["edges"], "frames_t": sgrid.get("frames_t")},
        "detect": detect,
        "tracks": {n: {"span": [min(map(int, tracks["tracks"][n]["boxes"])), max(map(int, tracks["tracks"][n]["boxes"]))],
                       "n_frames": len(tracks["tracks"][n]["boxes"]), "ends": tracks["tracks"][n]["ends"],
                       "overlap_same_class": tracks["tracks"][n].get("overlap_same_class"),
                       "identity": tracks["meta"]["identity"][n]["candidates"][tracks["meta"]["identity"][n]["chosen"]],
                       "check": tcheck["checks"][n], "coverage": tcover["bodies"].get(n),
                       "body": body_of_track[n]} for n in track_names},
        "masks": {"strip": strip, "log": masks_log, "crop": crop.box()},
        "mask_noise": fit["mask_noise"],
        "focal": {k: focal[k] for k in ("f", "sigma", "mad_rel", "per_frame", "frames", "rule")},
        "focal_seconds": focal["provenance"]["seconds"],
        "plan": {k: plan[k] for k in ("plan", "initial", "review", "repairs", "final_issues", "final_errors", "timing",
                                      "n_calls", "authoring", "shape_check", "structure_check", "query", "depth",
                                      "grid_frames")},
        "plan_calls": [{"stage": c.get("stage"), "attempt": c.get("attempt"), "prompt_chars": c.get("prompt_chars")}
                       for c in plan["calls"]],
        "shape": shape,
        "structure": {"params": structure["params"], "segments": {o["id"]: o["segments"] for o in structure["objects"]}},
        "fit": {k: fit[k] for k in ("camera_starts", "camera_start_chosen_pitch_deg", "camera_candidates", "focal",
                                     "f_start", "joint", "constraints", "identifiability", "params", "query",
                                     "gauges", "reference_depth", "untracked_bodies", "relation_unobserved",
                                     "time_s", "camera_bounds")},
        "fit_objects": {oid: {k: v for k, v in o.items() if k != "starts"} for oid, o in fit["objects"].items()},
        "fit_log": fitlog,
        "start_kept": start_kept,
        "pairs": pairs,
        "answer": answer,
        "driver": {k: {kk: v.get(kk) for kk in ("seconds", "start", "end")} for k, v in state.items()
                   if isinstance(v, dict) and "seconds" in v},
        "answer_log": answer_log,
        "copied": copied,
    }
    (dst / "case.json").write_text(json.dumps(case, indent=1, default=float))
    print(f"{unit}: {len(strip)} mask crops, {len(pairs)} start / fitted pairs, {len(copied)} copied -> {dst}")


if __name__ == "__main__":
    main()
