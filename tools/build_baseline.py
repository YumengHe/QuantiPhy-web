"""Build data/baseline.json for baseline.html from baseline prediction CSVs.

Reads only the baselines' prediction CSVs (qid, video_id, category, question, truth, pred, raw: the format of
physmind_quantiphy/run_caw.py) and data/videos.json (question ids, videos, thumbnails). Never runs a model.

  python -B tools/build_baseline.py \
      --run physmind=/nas/yumenghe/QuantiPhy-data/physmind_result_new/quantiphy/scores/physmind_preds.csv \
      --run spatialcode=/nas/yumenghe/QuantiPhy-data/spatial_code_result/quantiphy/answers/preds.csv

The methods given with --run are (re)built from their CSVs; every other method already in data/baseline.json keeps
its predictions there (the Code-as-World, GPT-5.6-luna and Qwen3.5-27B CSVs are not on this machine). Methods are
listed in the order of METHODS.

A CSV row is matched to the page's question id <video_id>_q<k> by video, ground truth (the page's table rounds some
values, so within 0.5 %) and the closest question text; every row must match exactly one question. Questions of the
page without a row in the CSVs (captured_0027s_q4 is not in the HF validation release) are left out. GT and MRA are
the CSV's (the HF validation table), MRA as on the Evaluation page.

Layout: {"generated", "methods": [{"key", "label", ...}], "videos": [...], "questions": [{"id", "video", "cat",
"q", "prior", "depth", "answer", "preds": {key: {"pred", "mra", "raw"}}}]}.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import difflib
import json
import math
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
TH = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

# What each run is.
METHODS = {
    "caw_4b": {
        "label": "Code-as-World-VL-4B",
        "source": "https://github.com/MirroS-Lab/Code-as-World",
        "setup": "released evaluator code_as_world.evaluation @1353bf0, 16 frames, temperature 0.01, seed 1, "
                 "no thinking; gpu_memory_utilization 0.85 (released 0.2 is sized for 80 GB); 1× L40S, 2026-11-03",
    },
    "caw_9b": {
        "label": "Code-as-World-VL-9B",
        "source": "https://github.com/MirroS-Lab/Code-as-World",
        "setup": "released evaluator code_as_world.evaluation @1353bf0, 16 frames, temperature 0.01, seed 1, "
                 "no thinking; gpu_memory_utilization 0.85 (released 0.5 is sized for 80 GB); 1× L40S, 2026-11-03",
    },
    "physmind": {
        "label": "PhysMind",
        "source": "https://github.com/ccyydd/PhysMind",
        "setup": "PhysMind @8c106b7 stages 1-3 (SAM 3 tracks from VLM concept prompts naming the question objects, PhysMind's own vlm-concept-prompts route; MoGe-2, Metric Video Depth Anything, SAM 3D Objects, FoundationPose, GeoCalib), quantity measured on the posed metric meshes (no prior used, as in PhysMind); adapter: one SAM 3 session per prompt, untagged tracks matched to question objects by Gemini; 90 of 158 measured, the other 68 (no posed track 46, SAM 3 found nothing 15, no poses 7) fall back to the VLM's direct answer on the whole video, as PhysMind's own pipeline does (FAIL-001); VLM gemini-3-flash-preview via OpenRouter; 1× L40S, 2026-11-04",
    },
    "spatialcode": {
        "label": "Thinking with Spatial Code",
        "source": "https://github.com/Beckschen/spatialcode",
        "setup": "spatialcode @cf7fffc (arXiv 2603.05591): object boxes by Gemini 3 Flash, each object boxed on the "
                 "earliest of 8 shown frames where it is visible; Spatial Encoder with the released weights "
                 "(qicq1c/SpatialCode), 32 frames spread over each video; spatial code = one 3D box per object plus "
                 "its per-frame box centres with timestamps (metres, first camera frame); reasoner "
                 "Qwen/Qwen3-4B-Instruct-2507 on SGLang, temperature 0, max_tokens 8192; prompt = spatial code, "
                 "prior, depth info, question; 1× L40S, 2026-11-11",
    },
    "gpt56_luna": {
        "label": "GPT-5.6-luna",
        "source": "https://openrouter.ai/openai/gpt-5.6-luna",
        "setup": "pure model inference, no tools: 16 frames sampled uniformly over the video (long side 768 px, each "
                 "labelled with its timestamp) + one prompt (expert persona, prior, the dataset's depth info for 3D "
                 "questions, question, 'directly tell me the number'); default reasoning; openai/gpt-5.6-luna via "
                 "OpenRouter (provider OpenAI), physmind_quantiphy/run_gpt.py; 2026-11-06",
    },
    "qwen35_27b": {
        "label": "Qwen3.5-27B",
        "source": "https://huggingface.co/Qwen/Qwen3.5-27B",
        "setup": "pure model inference, no tools, same frames and prompt as GPT-5.6-luna (16 frames, long side 768 px, "
                 "timestamps; expert persona, prior, depth info for 3D questions, question, 'directly tell me the "
                 "number'); thinking on; qwen/qwen3.5-27b via OpenRouter pinned to Alibaba (the model's own host, no "
                 "fallback), physmind_quantiphy/run_gpt.py; 2026-11-07",
    },
}


def mra(pred: float | None, truth: float) -> float:
    if pred is None or not math.isfinite(pred) or truth == 0:
        return 0.0
    rel = abs(abs(pred) - abs(truth)) / abs(truth)
    return sum(rel < 1 - th for th in TH) / len(TH)


def match_ids(rows: list[dict], questions: list[dict]) -> dict[str, str]:
    """qid -> page question id: same video, GT within 0.5 %, then the most similar question text."""
    out, taken = {}, {}
    for r in rows:
        truth = float(r["truth"])
        cands = [q for q in questions if q["video"] == r["video_id"]
                 and abs(q["answer"] - truth) <= 0.005 * max(abs(truth), 1e-9)]
        if not cands:
            raise SystemExit(f"row {r['qid']} ({r['video_id']}: {r['question']}) has no question on the page")
        best = max(cands, key=lambda q: difflib.SequenceMatcher(None, q["q"].strip(), r["question"].strip()).ratio())
        if best["id"] in taken:
            raise SystemExit(f"rows {taken[best['id']]} and {r['qid']} both match {best['id']}")
        taken[best["id"]] = r["qid"]
        out[r["qid"]] = best["id"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="key=path/to/preds.csv (key in METHODS)")
    args = ap.parse_args()

    val = json.loads((WEB / "data" / "videos.json").read_text())["val"]
    page_q = {q["id"]: q for q in val["questions"]}
    out = WEB / "data" / "baseline.json"
    given = [s.split("=", 1)[0] for s in args.run]
    kept: dict[str, dict] = {}  # methods not given: as in the current data/baseline.json
    questions: dict[str, dict] = {}
    if out.exists():
        old = json.loads(out.read_text())
        kept = {m["key"]: m for m in old["methods"] if m["key"] not in given}
        for q in old["questions"]:
            q["preds"] = {k: v for k, v in q["preds"].items() if k in kept}
            if q["preds"]:
                questions[q["id"]] = q
    for spec in args.run:
        key, path = spec.split("=", 1)
        if key not in METHODS:
            raise SystemExit(f"unknown run {key}: add it to METHODS")
        rows = list(csv.DictReader(Path(path).expanduser().open(encoding="utf-8")))
        for qid, qid_page in match_ids(rows, val["questions"]).items():
            r = next(x for x in rows if x["qid"] == qid)
            pq = page_q[qid_page]
            truth = float(r["truth"])
            entry = questions.setdefault(qid_page, {
                "id": qid_page, "video": pq["video"], "k": pq["k"], "cat": pq["cat"], "q": pq["q"].strip(),
                "prior": pq["prior"], "depth": pq["depth"], "answer": truth, "preds": {}})
            if entry["answer"] != truth:
                raise SystemExit(f"{qid_page}: GT {truth} in {key} differs from {entry['answer']}")
            pred = float(r["pred"]) if r["pred"] not in ("", "None") else None
            entry["preds"][key] = {"pred": pred, "mra": mra(pred, truth), "raw": r["raw"]}

    keys = [k for k in METHODS if k in given or k in kept]
    missing = [q["id"] for q in questions.values() if set(q["preds"]) != set(keys)]
    if missing:
        raise SystemExit(f"questions not in every run: {missing[:5]}")
    used = {q["video"] for q in questions.values()}
    order = {v["id"]: i for i, v in enumerate(val["videos"])}
    data = {
        "generated": datetime.date.today().isoformat(),
        "methods": [{"key": k, **METHODS[k]} if k in given else kept[k] for k in keys],
        "videos": [{k: v.get(k) for k in ("id", "source", "thumb", "orig", "video", "video_from", "codec")}
                   for v in val["videos"] if v["id"] in used],
        "questions": sorted(questions.values(), key=lambda q: (order[q["video"]], q["k"])),
        "left_out": sorted(set(page_q) - set(questions)),
    }
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    print(f"data/baseline.json: {len(questions)} questions, {len(used)} videos, left out {data['left_out']}")


if __name__ == "__main__":
    main()
