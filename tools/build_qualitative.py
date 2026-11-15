"""Home > Qualitative Comparison: on each frame where a baseline drew its reconstruction, the final scene program of
one of the frame's questions, PhysMind's posed meshes and Thinking with Spatial Code's 3D boxes.

  python -B web/tools/build_qualitative.py --tag v6val

Frames: those of data/physmind/index.json and the frames Thinking with Spatial Code drew its encoder's boxes on
(drawn_frame of data/spatialcode/index.json, the encoder frame nearest to the PhysMind frame). Both runs' images are
redrawn in the site's style by tools/restyle_baselines.py and not touched here. A frame drawn by only one of the two
leaves the other's image out.

For every frame, the first of its questions that has a fitted scene program (outputs/sceneprog/<tag>/<unit>/scene.json;
a frame without questions: the video's first question with one) is drawn at that frame exactly as
pipeline.sceneprog.evaluate draws its overlays, in the site's style (tools/overlay_style.py): the program rendered from
its camera (visible outline 2 px, occluded silhouette 1 px, in the object's outline shade), the observed masks of the
tracks it is bound to (white on dark), each object's IoU on that frame on a tag tinted with its colour, the unit and
frame on a white tag. Reads saved programs only (no fitting, no model calls). A frame none of whose questions has a
program raises.

Colours, one per object and per frame (overlay_style.object_colours, the rule of the paper's figures): the program's
bodies posed at the frame in program order with the asked object (query.object) first, then the names PhysMind and
Thinking with Spatial Code gave that frame's objects; a name that matches a known one takes its colour. The map (every
name's key -> colour) is written to the frame's "colours" and tools/render_physmind.py and tools/restyle_baselines.py draw with it, so run
this tool first.

Writes data/qualitative/<tag>/<unit>_f<frame>.jpg and data/qualitative/index.json:
{"generated", "ours" (scene-program tag), "videos": [{"video", "physmind": {"question_objects", "posed_question_objects",
"failed_stage"}, "frames": [{"frame", "t", "questions", "ours", "ours_unit", "physmind", "physmind_missing",
"spatialcode", "spatialcode_missing", "colours" (name key -> colour)}]}]}; images of an earlier build that no frame references are removed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(WEB / "tools"))
import overlay_style as S  # noqa: E402

from pipeline import palette  # noqa: E402
from pipeline.data import Video  # noqa: E402
from pipeline.sceneprog.masks import ObservedMasks  # noqa: E402
from pipeline.sceneprog.metric import frame_record  # noqa: E402
from pipeline.sceneprog.poses import camera, poses_at  # noqa: E402
from pipeline.sceneprog.program import load  # noqa: E402
from pipeline.sceneprog.render import render  # noqa: E402
from pipeline.sceneprog.shapes import compile_scene  # noqa: E402

PHYSMIND = WEB / "data" / "physmind" / "index.json"
SPATIALCODE = WEB / "data" / "spatialcode" / "index.json"
INDEX = WEB / "data" / "qualitative" / "index.json"


def draw(directory: Path, frame: int, out: Path, colours: dict[str, str]) -> None:
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
        image = overlay(cv2.cvtColor(video.frame(frame), cv2.COLOR_RGB2BGR), r, obs, labels, colours,
                        f"{program['unit']}  frame {frame}  t={frame / fps:.2f}s")
    finally:
        video.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), image, [cv2.IMWRITE_JPEG_QUALITY, 86]):
        raise ValueError(f"{out}: not written")


def overlay(bgr: np.ndarray, rendered: dict, observed: dict, labels: dict, colours: dict[str, str],
            header: str) -> np.ndarray:
    """pipeline.sceneprog.render.overlay in the site's style: per object (in the order of `colours`) the observed mask
    outline (white on dark, 1 px), the rendered occluded silhouette (1 px) and visible outline (2 px) in the object's
    outline shade; the labels as tags down the left edge under the header."""
    image = np.ascontiguousarray(bgr).copy()
    tags = [(header, 4, None, None)]
    order = list(colours)
    for oid in sorted(set(labels), key=lambda o: order.index(S.name_key(o))):
        colour = colours[S.name_key(oid)]
        obs = observed.get(oid)
        if obs is not None and obs.any():
            cs, _ = cv2.findContours(obs.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image, cs, -1, S.bgr(S.TEXT), 3, cv2.LINE_AA)
            cv2.drawContours(image, cs, -1, (255, 255, 255), 1, cv2.LINE_AA)
        r = rendered.get(oid)
        if r is not None:
            cs, _ = cv2.findContours(r["silhouette"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image, cs, -1, palette.darker(S.bgr(colour)), 1, cv2.LINE_AA)
            cs, _ = cv2.findContours(r["visible"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image, cs, -1, palette.darker(S.bgr(colour)), 2, cv2.LINE_AA)
        tags.append((labels[oid], 4, None, colour))
    return S.tags(image, tags)


def comparison_colours(program: dict, frame: int, theirs: list[str]) -> dict[str, str]:
    """name key -> colour for one frame, as the paper's figures colour a row (paper/figures/src/overlays.py): the
    program's bodies posed at the frame in program order with the asked object first, then the baselines' names;
    bodies not posed at the frame come last (they have a tag but no outline)."""
    posed = poses_at(program, frame / float(program["video"]["fps"]))
    bodies = [o["id"] for o in program["objects"] if o["kind"] == "body"]
    present = [b for b in bodies if posed.get(b) is not None]
    asked = (program.get("query") or {}).get("object")
    present.sort(key=lambda b: b != asked)
    return S.object_colours(present, theirs, [b for b in bodies if b not in present])


def frames(pm: dict, sc: dict) -> list[dict]:
    """One record per video and frame drawn by PhysMind or Thinking with Spatial Code."""
    sc_videos = {v["video"]: v for v in sc["videos"]}
    if set(sc_videos) != {v["video"] for v in pm["videos"]}:
        raise ValueError("PhysMind and Thinking with Spatial Code drew different videos")
    videos = []
    for v in pm["videos"]:
        rows: dict[int, dict] = {}
        blank = {"physmind": None, "physmind_missing": [], "spatialcode": None, "spatialcode_missing": []}
        for f in v["frames"]:
            rows[int(f["frame"])] = {"frame": int(f["frame"]), "t": f["t"], "questions": f["questions"], **blank,
                                     "physmind": f["physmind"], "physmind_missing": f["missing"] if f["physmind"] else [],
                                     "names": [d["label"] for d in f["drawn"]] + f["missing"]}
        for f in sc_videos[v["video"]]["frames"]:
            fi = int(f["drawn_frame"])
            row = rows.setdefault(fi, {"frame": fi, "t": f["drawn_t"], "questions": f["questions"], **blank})
            row["spatialcode"], row["spatialcode_missing"] = f["spatialcode"], f["missing"]
            row["names"] = row.get("names", []) + [d["label"] for d in f["drawn"]] + f["missing"]
        videos.append({"video": v["video"],
                       "physmind": {k: v[k] for k in ("question_objects", "posed_question_objects", "failed_stage")},
                       "frames": [rows[k] for k in sorted(rows)]})
    return videos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="scene-program tag of the fitted programs (outputs/sceneprog/<tag>/<unit>)")
    args = ap.parse_args()
    videos = frames(json.loads(PHYSMIND.read_text()), json.loads(SPATIALCODE.read_text()))
    out_dir = WEB / "data" / "qualitative" / args.tag
    written, drawn = set(), 0
    for v in videos:
        for f in v["frames"]:
            # a frame without a question of ours (PhysMind's frame with most posed objects) takes the video's questions
            qs = f["questions"] or sorted((p.parent.name for p in (ROOT / "outputs" / "sceneprog" / args.tag).glob(f"{v['video']}_q*/scene.json")
                                           if re.fullmatch(rf"{re.escape(v['video'])}_q\d+", p.parent.name)),
                                          key=lambda u: int(u.rsplit("_q", 1)[1]))
            units = [u for u in qs if (ROOT / "outputs" / "sceneprog" / args.tag / u / "scene.json").is_file()]
            if not units:
                raise ValueError(f"{v['video']} frame {f['frame']}: none of {f['questions']} has a program in {args.tag}")
            unit, frame = units[0], int(f["frame"])
            directory = ROOT / "outputs" / "sceneprog" / args.tag / unit
            f["colours"] = comparison_colours(json.loads((directory / "scene.json").read_text()), frame, f.pop("names", []))
            out = out_dir / f"{unit}_f{frame}.jpg"
            draw(directory, frame, out, f["colours"])
            drawn += 1
            f["ours"], f["ours_unit"] = str(out.relative_to(WEB)), unit
            written.add(out)
    stale = [p for p in out_dir.glob("*.jpg") if p not in written]
    for p in stale:
        p.unlink()
    INDEX.write_text(json.dumps({"generated": datetime.date.today().isoformat(), "ours": args.tag, "videos": videos},
                                ensure_ascii=False, indent=1))
    n = sum(len(v["frames"]) for v in videos)
    print(f"{n} frames ({len(written)} of ours, {drawn} drawn, {len(stale)} stale removed); "
          f"PhysMind {sum(bool(f['physmind']) for v in videos for f in v['frames'])}, "
          f"Spatial Code {sum(bool(f['spatialcode']) for v in videos for f in v['frames'])}; {INDEX.relative_to(WEB)} written")


if __name__ == "__main__":
    main()
