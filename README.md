# Project website

Site for "Reconstructing measurable physical scenes from video". Static pages, no build step.

| File | Content |
|---|---|
| `index.html` | Home: project scope, keywords, pipeline, results, comparison, browse |
| `pipeline.html` | Flow diagram (a single chain), every step with input, output and the named checks it feeds, answer sources |
| `versions.html` | Version history with the 10-question scores |
| `evaluation.html` + `videos.js` | Evaluation: per-question results on the validation set with video, 3D scene, overlays, scene graph, the scene camera (the fused focal estimate and what it replaced) with its focal cross-checks and its focal votes fused by source, the piece each fit measurement's σ was found over, the support plane's checks and the camera roll, the per-object mask extents, every measurement of the asked quantity, agree / constraints with the untestable ones apart, named checks, each object's presence votes and rule, scene-graph edge sources and conflicts, every impulse decision of a physics chain, a measured slope with its rolling law and refined onset, the kinematic onset bound, each object's fit as a whole (`fit.object`: s² vs its bound, χ² per row kind) with its inadmissible constraints and excluded focal votes apart, the scale object's contacts as depth equations and the plane's rescaling, the main support's own plane with its `level` test and refit, scene-graph events (a catch ending a flight) as ticks on the track timeline, presence reasons and unknown zoom votes, the reference objects' size reads (frames read and left out, rms, dims) and their static poses in the 3D view, class votes with their role, a stopped unit's error, a partial unit's failed planning round and time per step, a motion prior's read per motion piece on the clip's time axis, a class prior's instances with their reads, an object in a container against the container's depth, each on / in edge's mask containment, each object's choice frame with its rule and probe-frame scan, where its track starts and the record's binding to it, the unobserved phases on the track timeline, a kick's frame-pair speeds ± σ and its ramp end, a pendulum's period read from both swing series, the frame indices beside every window and piece, the agent's no-repeat repairs, edits kept off unobserved phases and the constraint contract replayed per round, the mask components each track dropped and its leak frames, an anchored class vote's reading per stated depth, an overhead structure's seen-from-below test |
| `site.css` | Shared styles (color tokens, type, top bar) |
| `data/` | Data for the evaluation page (`videos.json`, thumbnails; `runtime/<version>/` holds that version's overlays and their record `overlays.json`), built by `tools/build_videos.py` |
| `media/val`, `media/test` | Symlinks to the original video folders (not in git) |

## Serve

```bash
./serve.sh start    # port 9002 (override with PORT=...), Range support so videos can seek
./serve.sh status
./serve.sh stop
```

## Rebuild the scene data

From the repository root:

```bash
python -B web/tools/build_videos.py                    # validation run of the current version (pipeline/VERSION)
python -B web/tools/build_videos.py --val-runs <tag>[=<csv>] ...   # runs in order of preference
python -B web/tools/build_videos.py --val-runs v5gpt=outputs/val_v5gpt_answers_dev20.csv v1
```

The script reads only that version's own artifacts. It never runs the pipeline and never falls back to another version or field: artifacts are read in the format `VERSION_FORMAT` declares for the version (motion record fields, processing unit), and anything else is an error. Only the current version may render missing overlays; building an older version fails if an overlay is missing. Register a new version in `VERSION_FORMAT` first.

The validation results table in `index.html` is updated by hand from `docs/VERSIONS.md`. The build also regenerates two tables of `pipeline.html` between their marks: the class-size table (from `pipeline/class_sizes.py`) and the time per step (from the first run's answer records).
