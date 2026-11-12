"""Home > Qualitative Comparison: on each frame where a baseline drew its reconstruction, the final scene program of
one of the frame's questions, PhysMind's posed meshes and Thinking with Spatial Code's 3D boxes.

  python -B web/tools/build_qualitative.py --loop v6valloop

Frames: those of data/physmind/index.json (PhysMind's images, tools/render_physmind.py; not touched here) and the
frames Thinking with Spatial Code drew its encoder's boxes on (drawn_frame of SPATIALCODE/index.json, the encoder frame
nearest to the PhysMind frame; its images are copied to data/spatialcode/<video>_f<drawn_frame>.jpg). A frame drawn by
only one of the two leaves the other's image out.

For every frame, the first of its questions that has a final scene program (outputs/sceneprog/<loop>/<unit>/scene.json;
a frame without questions: the video's first question with one) is drawn at that frame exactly as
pipeline.sceneprog.evaluate draws its overlays: the program rendered from its camera (visible outline 2 px, occluded
silhouette 1 px, object colour), the observed masks of the tracks it is bound to (white), each object's IoU on that
frame, the unit and frame in the corner. Reads saved programs only (no fitting, no model calls). A frame none of whose
questions has a program raises.

Writes data/qualitative/<loop>/<unit>_f<frame>.jpg and data/qualitative/index.json:
{"generated", "ours" (loop tag), "videos": [{"video", "physmind": {"question_objects", "posed_question_objects",
"failed_stage"}, "frames": [{"frame", "t", "questions", "ours", "ours_unit", "physmind", "physmind_missing",
"spatialcode", "spatialcode_missing"}]}]}; images of an earlier build that no frame references are removed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from pipeline.data import Video  # noqa: E402
from pipeline.sceneprog.masks import ObservedMasks  # noqa: E402
from pipeline.sceneprog.metric import frame_record  # noqa: E402
from pipeline.sceneprog.poses import camera  # noqa: E402
from pipeline.sceneprog.program import load  # noqa: E402
from pipeline.sceneprog.render import overlay, render  # noqa: E402
from pipeline.sceneprog.shapes import compile_scene  # noqa: E402

PHYSMIND = WEB / "data" / "physmind" / "index.json"
SPATIALCODE = Path("/nas/yumenghe/QuantiPhy-data/spatial_code_result/quantiphy/reconstructions")
INDEX = WEB / "data" / "qualitative" / "index.json"


def draw(directory: Path, frame: int, out: Path) -> None:
    """One overlay of the program in `directory` at `frame` (pipeline.sceneprog.evaluate._write_overlays)."""
    program, scad = load(directory)
    observed = ObservedMasks(program["observations"]["tracks_tag"], program["unit"], program["split"])
    meshes = compile_scene(program, scad)
    fps = float(program["video"]["fps"])
    cam = camera(program)
    if (cam["height"], cam["width"]) != observed.shape:
        raise ValueError(f"{directory}: camera image {cam['height']}x{cam['width']} differs from the masks' {observed.shape}")
    modelled = {o["id"]: o["track"] for o in program["objects"] if o["kind"] == "body" and o.get("track") is not None}
    r = render(program, meshes, frame / fps, cam)
    labels, obs = {}, {}
    for oid, name in modelled.items():
        o = observed.get(name, frame)
        obs[oid] = o
        if o is None:
            labels[oid] = f"{oid} untracked"
            continue
        others = [observed.get(n, frame) for n in observed.names() if n != name]
        rec = frame_record(r.get(oid), o, [x for x in others if x is not None])
        labels[oid] = f"{oid} IoU {rec['iou']:.2f}"
    video = Video(Path(program["video"]["path"]), fps=fps)
    try:
        image = overlay(video.frame(frame), r, obs, labels, f"{program['unit']}  frame {frame}  t={frame / fps:.2f}s")
    finally:
        video.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86]):
        raise ValueError(f"{out}: not written")


def frames(pm: dict, sc: dict) -> list[dict]:
    """One record per video and frame drawn by PhysMind or Thinking with Spatial Code; copies the latter's images."""
    sc_videos = {v["video"]: v for v in sc["videos"]}
    if set(sc_videos) != {v["video"] for v in pm["videos"]}:
        raise ValueError("PhysMind and Thinking with Spatial Code drew different videos")
    sc_dir = WEB / "data" / "spatialcode"
    sc_dir.mkdir(parents=True, exist_ok=True)
    copied = set()
    videos = []
    for v in pm["videos"]:
        rows: dict[int, dict] = {}
        blank = {"physmind": None, "physmind_missing": [], "spatialcode": None, "spatialcode_missing": []}
        for f in v["frames"]:
            rows[int(f["frame"])] = {"frame": int(f["frame"]), "t": f["t"], "questions": f["questions"], **blank,
                                     "physmind": f["physmind"], "physmind_missing": f["missing"] if f["physmind"] else []}
        for f in sc_videos[v["video"]]["frames"]:
            fi = int(f["drawn_frame"])
            row = rows.setdefault(fi, {"frame": fi, "t": f["drawn_t"], "questions": f["questions"], **blank})
            src = SPATIALCODE / Path(f["spatialcode"]).name
            dst = sc_dir / f"{v['video']}_f{fi}.jpg"
            shutil.copyfile(src, dst)
            copied.add(dst)
            row["spatialcode"], row["spatialcode_missing"] = str(dst.relative_to(WEB)), f["missing"]
        videos.append({"video": v["video"],
                       "physmind": {k: v[k] for k in ("question_objects", "posed_question_objects", "failed_stage")},
                       "frames": [rows[k] for k in sorted(rows)]})
    for p in sc_dir.glob("*.jpg"):
        if p not in copied:
            p.unlink()
    return videos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loop", required=True, help="scene-program loop tag (outputs/sceneprog/<loop>/<unit>)")
    args = ap.parse_args()
    videos = frames(json.loads(PHYSMIND.read_text()), json.loads((SPATIALCODE / "index.json").read_text()))
    out_dir = WEB / "data" / "qualitative" / args.loop
    written, drawn = set(), 0
    for v in videos:
        for f in v["frames"]:
            # a frame without a question of ours (PhysMind's frame with most posed objects) takes the video's questions
            qs = f["questions"] or sorted((p.parent.name for p in (ROOT / "outputs" / "sceneprog" / args.loop).glob(f"{v['video']}_q*/scene.json")
                                           if re.fullmatch(rf"{re.escape(v['video'])}_q\d+", p.parent.name)),
                                          key=lambda u: int(u.rsplit("_q", 1)[1]))
            units = [u for u in qs if (ROOT / "outputs" / "sceneprog" / args.loop / u / "scene.json").is_file()]
            if not units:
                raise ValueError(f"{v['video']} frame {f['frame']}: none of {f['questions']} has a program in {args.loop}")
            unit, frame = units[0], int(f["frame"])
            out = out_dir / f"{unit}_f{frame}.jpg"
            if not out.exists():
                draw(ROOT / "outputs" / "sceneprog" / args.loop / unit, frame, out)
                drawn += 1
            f["ours"], f["ours_unit"] = str(out.relative_to(WEB)), unit
            written.add(out)
    stale = [p for p in out_dir.glob("*.jpg") if p not in written]
    for p in stale:
        p.unlink()
    INDEX.write_text(json.dumps({"generated": datetime.date.today().isoformat(), "ours": args.loop, "videos": videos},
                                ensure_ascii=False, indent=1))
    n = sum(len(v["frames"]) for v in videos)
    print(f"{n} frames ({len(written)} of ours, {drawn} drawn now, {len(stale)} stale removed); "
          f"PhysMind {sum(bool(f['physmind']) for v in videos for f in v['frames'])}, "
          f"Spatial Code {sum(bool(f['spatialcode']) for v in videos for f in v['frames'])}; {INDEX.relative_to(WEB)} written")


if __name__ == "__main__":
    main()
