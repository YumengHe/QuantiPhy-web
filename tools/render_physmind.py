"""Render PhysMind's 3D reconstruction back onto the video for Home > Qualitative Comparison, from its meshes.

The frames are the ones of the run's own reconstructions/index.json; draws PhysMind's posed meshes of the question
objects on each frame in the site's style (tools/overlay_style.py): light fill and outline of the projected mesh, its
box and the object's name on a tinted tag, t on a white tag. Our side of each frame is written by
tools/build_qualitative.py, not by this tool.

Needs the run's full artifacts (meshes, poses, frames: about 2 GB of the 7 GB under run/runs/qp_physmind_v2/artifacts);
the copy on this machine holds two scenes only, and the site's images are redrawn from the run's published images by
tools/restyle_baselines.py instead (same file names, same style).

  python -B tools/render_physmind.py   # --run defaults to RUN below

PhysMind run: physmind_quantiphy/run_physmind.py, variant B, with all artifacts kept. Its geometry is used as is,
nothing is re-fitted: each object's FoundationPose mesh (mesh_path of foundationpose_poses.json, metres) is moved into
the camera frame by that frame's pose_4x4 and projected with the intrinsics FoundationPose used (MoGe-2 K in the saved
video-depth frames' pixel coordinates, which are the video's). Objects occlude each other (faces drawn far to near).
Only objects matched to a question object are drawn (the matching of run_physmind.load_objects: the question
concept carried by the SAM 3 track, else the VLM match; several tracks for one concept: the longest pose track),
labelled with the question object's name. A video without one of our overlays gets the frame with most posed objects.

Writes data/physmind/<video>_f<frame>.jpg and data/physmind/index.json:
{"generated", "run", "videos": [{"video", "question_objects", "posed_question_objects", "failed_stage",
"frames": [{"frame", "t", "physmind", "questions", "drawn", "missing"}]}]}.
Needs numpy, opencv, trimesh (for this tool only, not for the site).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
RUN = Path("/nas/yumenghe/QuantiPhy-data/physmind_result_new/quantiphy/run/runs/qp_physmind_v2")
FRAMES = RUN.parents[2] / "reconstructions" / "index.json"  # the frames the run drew

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overlay_style as S  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def run_frames() -> dict[str, dict[int, dict]]:
    """video -> frame -> {"t", "questions"} of the run's reconstructions/index.json."""
    return {v["video"]: {int(f["frame"]): {"t": f["t"], "questions": f["questions"]} for f in v["frames"]}
            for v in read_json(FRAMES).get("videos", [])}


def question_objects(world: Path, plan_q: dict, match: dict) -> dict[str, tuple[str, dict]]:
    """concept prompt -> (object_id, foundationpose object), as run_physmind.load_objects picks it."""
    concepts = {c["concept_id"]: c["prompt"] for c in plan_q.get("concepts", [])}
    tracks = read_json(world / "object-segmentation-and-event-detection/sam3_video_tracks/sam3_video_tracks.json")
    track_concept = {tid: (recs[0].get("concept_id") if recs[0].get("concept_id") in concepts else None)
                     for tid, recs in tracks.get("tracks_by_object", {}).items() if recs}
    plan = read_json(world / "object-identification-and-planning/object_plan/object_plan.json")
    object_track = {o["object_id"]: o.get("source_track_id") for o in plan.get("target_objects", [])}
    vlm: set[str] = set()
    for c, oids in (match or {}).items():  # the VLM match fills concepts no prompted track carries
        if c in set(track_concept.values()) - {None} and c not in vlm:
            continue
        for oid in oids:
            tid = object_track.get(oid)
            if tid and track_concept.get(tid) is None:
                track_concept[tid] = c
                vlm.add(c)
    best: dict[str, tuple[str, dict]] = {}
    poses = read_json(world / "pose-estimation-and-tracking/foundationpose/foundationpose_poses.json")
    for obj in poses.get("objects", []):
        c = track_concept.get(object_track.get(obj["object_id"]) or obj["object_id"])
        if c is None or not obj.get("poses") or not obj.get("mesh_path"):
            continue
        if c not in best or len(obj["poses"]) > len(best[c][1]["poses"]):
            best[c] = (obj["object_id"], obj)
    return {concepts[c]: v for c, v in best.items()}


def draw(frame: np.ndarray, K: np.ndarray, items: list[tuple[str, np.ndarray, np.ndarray, str]], t: float) -> np.ndarray:
    """items: (label, camera-frame vertices, faces, colour). Rasterises object ids far to near, then per object:
    light fill, outline, box, name tag."""
    h, w = frame.shape[:2]
    ids = np.zeros((h, w), np.int32)  # 0 = background, i+1 = items[i]
    polys = []
    for i, (_, verts, faces, _) in enumerate(items):
        tri = verts[faces]
        tri = tri[(tri[:, :, 2] > 0.05).all(axis=1)]
        if not len(tri):
            continue
        uv = tri @ K.T
        uv = uv[:, :, :2] / uv[:, :, 2:3]
        for d, pts in zip(tri[:, :, 2].mean(axis=1), uv):
            polys.append((d, pts, i + 1))
    polys.sort(key=lambda p: -p[0])  # far first, near faces overwrite
    for _, pts, i in polys:
        if np.abs(pts).max() < 1e5:
            cv2.fillConvexPoly(ids, np.round(pts * 4).astype(np.int32), i, shift=2)
    out = frame.copy()
    labels = [S.time_tag(t)]
    for i, (label, _, _, col) in enumerate(items):
        m = (ids == i + 1).astype(np.uint8)
        if not m.any():
            continue
        S.fill_and_outline(out, m, col, box=True)
        x, y, _, _ = cv2.boundingRect(m)
        labels.append((label, x, y - 2, col))
    return S.tags(out, labels)


def one_video(run: Path, measure: dict, wanted: dict[int, dict], out_dir: Path) -> dict:
    vid = measure["video_id"]
    world = run / "artifacts" / f"scene_{measure['scene_index']}" / "world-modeling"
    plan_q = read_json(run / "plans" / f"{vid}.json")
    rec = {"video": vid, "question_objects": len(plan_q.get("concepts", [])),
           "posed_question_objects": len(measure.get("posed_concepts", {})),
           "failed_stage": measure.get("reconstruction", {}).get("failed_stage"), "frames": []}
    frame_dir = world / "metric-mesh-reconstruction/video_metric_depth/video_metric_frames"
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    objs = question_objects(world, plan_q, measure.get("match")) if frames else {}
    fps = read_json(world / "pose-estimation-and-tracking/pose_frames/pose_frames.json").get("video_metadata", {}).get("fps")
    wanted = dict(sorted(wanted.items()))
    if not wanted and frames:  # a video the run drew no frame of: the frame with most posed question objects
        count = np.zeros(len(frames), int)
        for _, o in objs.values():
            for p in o["poses"]:
                if p["frame_index"] < len(frames):
                    count[p["frame_index"]] += 1
        fi = int(np.argmax(count))
        wanted = {fi: {"t": fi / fps if fps else 0.0, "questions": []}}
    poses = read_json(world / "pose-estimation-and-tracking/foundationpose/foundationpose_poses.json")
    K = np.asarray(poses["fixed_intrinsics"], float) if poses.get("fixed_intrinsics") else None
    meshes = {}
    colour = {label: S.BASELINE[i % len(S.BASELINE)] for i, label in enumerate(sorted(objs))}
    for fi, o in wanted.items():
        fr = {"frame": fi, "t": o["t"], "physmind": None, "questions": o["questions"], "drawn": [], "missing": []}
        if frames and fi < len(frames):
            img = cv2.imread(str(frames[fi]))
            items = []
            for label, (oid, obj) in sorted(objs.items()):
                pose = next((p for p in obj["poses"] if p["frame_index"] == fi), None)
                if pose is None:
                    fr["missing"].append(label)
                    continue
                if oid not in meshes:
                    m = trimesh.load(run.parent.parent / obj["mesh_path"], force="mesh")
                    meshes[oid] = (np.asarray(m.vertices, float), np.asarray(m.faces))
                v, f = meshes[oid]
                T = np.asarray(pose.get("pose_4x4") or pose["raw_pose_4x4"], float)
                items.append((label, v @ T[:3, :3].T + T[:3, 3], f, colour[label]))
                fr["drawn"].append({"label": label, "color": colour[label]})
            name = f"{vid}_f{fi}.jpg"
            cv2.imwrite(str(out_dir / name), draw(img, K, items, o["t"]) if K is not None else img,
                        [cv2.IMWRITE_JPEG_QUALITY, 86])
            fr["physmind"] = f"data/physmind/{name}"
        rec["frames"].append(fr)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=RUN, type=Path, help="runs/qp_physmind_v2 of a downloaded PhysMind run")
    ap.add_argument("--out", default=ROOT / "data" / "physmind", type=Path)
    args = ap.parse_args()
    run = args.run.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.jpg"):  # outputs of this tool only
        old.unlink()
    chosen = run_frames()
    videos = []
    for f in sorted((run / "measure").glob("*.json")):
        m = json.loads(f.read_text())
        rec = one_video(run, m, chosen.get(m["video_id"], {}), args.out)
        videos.append(rec)
        print(rec["video"], [(fr["frame"], len(fr["drawn"]), len(fr["missing"])) for fr in rec["frames"]], rec["failed_stage"] or "")
    index = {"generated": datetime.date.today().isoformat(),
             "run": "PhysMind variant B (qp_physmind_v2, all artifacts)", "videos": videos}
    (args.out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
