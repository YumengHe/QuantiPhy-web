"""Build web/data/wild.json (+ images under web/data/wild/) for wild.html: the scene-program run on real videos outside
the benchmark, which have no ground truth.

  python -B web/tools/build_wild.py --build v6wild --loop v6wildloop

Reads only the run's outputs; never runs the pipeline.

Inputs (relative to the repo root):
  data/QuantiPhy-wild/wild_dataset.parquet              questions and priors (unit <video_id>_q<k>, k = the question's
                                                        1-based order among the video's rows)
  data/QuantiPhy-wild/videos/<video>.mp4                frames of a video none of whose questions was answered
  outputs/sceneprog_answers_wild_<LOOP>.csv             one row per question: status answered / failed(step:kind) /
                                                        stopped, both answers (the authoritative table of the run)
  outputs/sceneprog_wild/<LOOP>/<unit>/answer.json      both answers with the scene answer's sigma and units
  outputs/sceneprog_wild/<LOOP>/<unit>/metric.json      alignment of the final program (unit_summary.iou_median_mean)
  outputs/sceneprog_wild/<LOOP>/<unit>/overlay_f*.jpg   the final program rendered over frames
  outputs/sceneprog_wild/<BUILD>/<unit>/plan_grid.jpg, shapes.png

An answered question missing any of these is an error. Videos are sorted by the best mean IoU among their answered
questions; videos with none come last, in table order.

Writes data/wild/<unit>/{overlay_f<frame>,plan_grid,shapes}.jpg (downscaled), data/wild/<video>/frame_f<frame>.jpg
(raw frames, only for videos without an answered question) and data/wild.json:
{"generated", "groups": [{"key", "videos", "priors"}], "counts": {"videos", "questions", "answered", "failed",
"stopped", "determined"}, "videos": [{"video", "group", "fps", "frames", "questions": [{"unit", "q", "prior",
"status", "step", "final": {"value", "unit"}, "scene": {"value", "sigma", "unit", "determined", "reason"}, "iou",
"overlays", "plan_grid", "shapes"}]}]}. Images of an earlier build that nothing references are removed.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import re
import shutil
from pathlib import Path

import cv2
import pandas as pd

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
DATASET = ROOT / "data" / "QuantiPhy-wild"
OUT_JSON = WEB / "data" / "wild.json"
OUT_IMG = WEB / "data" / "wild"
WIDTH = {"overlay": 1024, "plan_grid": 1280, "shapes": 1024, "frame": 1024}
QUALITY = 82
# group of a video, from the subject of its prior
GROUPS = [("ball", re.compile(r"\b(soccer ball|basketball)\b")), ("quadruped", re.compile(r"\brobot dog\b")),
          ("lab", re.compile(r"\bceiling light panel\b"))]


def group_of(prior: str) -> str:
    for key, rx in GROUPS:
        if rx.search(prior):
            return key
    raise ValueError(f"prior {prior!r} matches no group")


def shrink(src: Path, dst: Path, width: int) -> str:
    im = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(src)
    return write(im, dst, width)


def write(im, dst: Path, width: int) -> str:
    h, w = im.shape[:2]
    if w > width:
        im = cv2.resize(im, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, QUALITY]):
        raise OSError(f"cannot write {dst}")
    return dst.relative_to(WEB).as_posix()


def num(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


def raw_frames(video: str, n: int = 5) -> tuple[list[str], int]:
    cap = cv2.VideoCapture(str(DATASET / "videos" / f"{video}.mp4"))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"{video}: no frames")
    out = []
    for f in sorted({round(i * (total - 1) / (n - 1)) for i in range(n)}):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if not ok:
            raise ValueError(f"{video}: cannot read frame {f}")
        out.append(write(im, OUT_IMG / video / f"frame_f{f:04d}.jpg", WIDTH["frame"]))
    cap.release()
    return out, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", required=True, help="scene-program build tag (outputs/sceneprog_wild/<BUILD>)")
    ap.add_argument("--loop", required=True, help="loop tag (outputs/sceneprog_wild/<LOOP>, the answers CSV)")
    args = ap.parse_args()
    build_dir = ROOT / "outputs" / "sceneprog_wild" / args.build
    loop_dir = ROOT / "outputs" / "sceneprog_wild" / args.loop
    answers = ROOT / "outputs" / f"sceneprog_answers_wild_{args.loop}.csv"

    table = pd.read_parquet(DATASET / "wild_dataset.parquet")
    units: dict[str, dict] = {}
    order: list[str] = []
    for video, rows in table.groupby("video_id", sort=False):
        order.append(video)
        for k, (_, r) in enumerate(rows.iterrows(), 1):
            units[f"{video}_q{k}"] = {"video": video, "q": r["question"], "prior": r["prior"], "fps": float(r["fps"])}
    with open(answers, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if sorted(r["unit"] for r in rows) != sorted(units):
        raise ValueError(f"{answers}: units differ from the dataset's")

    if OUT_IMG.exists():
        shutil.rmtree(OUT_IMG)
    by_video: dict[str, list] = {v: [] for v in order}
    for r in rows:
        u, meta = r["unit"], units[r["unit"]]
        if r["question"] != meta["q"]:
            raise ValueError(f"{u}: question differs from the dataset's")
        m = re.fullmatch(r"(answered|stopped|failed)(?:\((\w+):(\w*)\))?", r["status"])
        if not m:
            raise ValueError(f"{u}: status {r['status']!r}")
        q = {"unit": u, "q": meta["q"], "prior": meta["prior"], "status": m.group(1),
             "step": f"{m.group(2)}: {m.group(3)}" if m.group(2) else None}
        if q["status"] == "answered":
            d = loop_dir / u
            a = json.loads((d / "answer.json").read_text())
            if a["unit"] != u:
                raise ValueError(f"{d}/answer.json: unit {a['unit']}")
            s, f = a["answer_scene"], a["answer_final"]
            q["final"] = {"value": num(f["value"]), "unit": f["unit"]}
            q["scene"] = {"value": num(s["value"]), "sigma": num(s["sigma"]), "unit": s["unit"],
                          "determined": bool(s["determined"]), "reason": s.get("reason") or r["scene_reason"] or None}
            if str(q["scene"]["determined"]) != r["determined"]:
                raise ValueError(f"{u}: determined differs between {answers.name} and answer.json")
            met = json.loads((d / "metric.json").read_text())
            if met["unit"] != u:
                raise ValueError(f"{d}/metric.json: unit {met['unit']}")
            q["iou"] = met["unit_summary"]["iou_median_mean"]
            overlays = sorted(d.glob("overlay_f*.jpg"))
            if not overlays:
                raise FileNotFoundError(f"{d}: no overlays")
            q["overlays"] = [shrink(p, OUT_IMG / u / p.name, WIDTH["overlay"]) for p in overlays]
            q["plan_grid"] = shrink(build_dir / u / "plan_grid.jpg", OUT_IMG / u / "plan_grid.jpg", WIDTH["plan_grid"])
            q["shapes"] = shrink(build_dir / u / "shapes.png", OUT_IMG / u / "shapes.jpg", WIDTH["shapes"])
        by_video[meta["video"]].append(q)

    videos = []
    for v in order:
        qs = sorted(by_video[v], key=lambda q: int(q["unit"].rsplit("_q", 1)[1]))
        groups = {group_of(q["prior"]) for q in qs}
        if len(groups) != 1:
            raise ValueError(f"{v}: questions in groups {groups}")
        entry = {"video": v, "group": groups.pop(), "fps": units[qs[0]["unit"]]["fps"], "questions": qs}
        if not any(q["status"] == "answered" for q in qs):
            entry["frames"], _ = raw_frames(v)
        videos.append(entry)
    best = lambda e: max((q["iou"] for q in e["questions"] if "iou" in q), default=None)  # noqa: E731
    videos.sort(key=lambda e: (best(e) is None, -(best(e) or 0.0)))

    groups = []
    for key, _ in GROUPS:
        vs = [e for e in videos if e["group"] == key]
        priors = sorted({q["prior"] for e in vs for q in e["questions"]})
        groups.append({"key": key, "videos": len(vs), "priors": priors})
    qs = [q for e in videos for q in e["questions"]]
    counts = {"videos": len(videos), "questions": len(qs),
              **{s: sum(q["status"] == s for q in qs) for s in ("answered", "failed", "stopped")},
              "determined": sum(q.get("scene", {}).get("determined", False) for q in qs)}
    out = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "groups": groups, "counts": counts,
           "videos": videos}
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"{OUT_JSON.relative_to(ROOT)}: {counts}")


if __name__ == "__main__":
    main()
