# Project website

Site for "Reconstructing measurable physical scenes from video". Static pages, no build step.

| File | Content |
|---|---|
| `index.html` + `comparison.js` | Home: project scope, keywords, pipeline, quantitative comparison (the Baseline page's method summary, computed from `data/baseline.json`, plus our row from `data/comparison_ours.json`), qualitative comparison (`data/qualitative/index.json` from `tools/build_qualitative.py`: PhysMind's posed meshes of the question objects, `data/physmind/`, from `tools/render_physmind.py`; Thinking with Spatial Code's encoder boxes, `data/spatialcode/`, copied from `/nas/yumenghe/QuantiPhy-data/spatial_code_result/quantiphy/reconstructions/`; on each of their frames the final scene program of one of the frame's questions, `data/qualitative/<loop tag>/`) |
| `pipeline.html` | Flow diagram (a single chain), every step with input, output and the named checks it feeds, answer sources |
| `evaluation.html` + `videos.js` | Evaluation: the scene-program run on the validation set — both answers (`answer_scene`, `answer_final`) over all questions and the ones shared with the baselines, per category, by source of `answer_final`, loop stops — and per question its two answers, the loop, the final program's overlays, each round's views, the plan grid and shapes (`media/sceneprog/<tag>/`), then its harness results with video, 3D scene, overlays, scene graph, the scene camera (the fused focal estimate and what it replaced) with its focal cross-checks and its focal votes fused by source, the piece each fit measurement's σ was found over, the support plane's checks and the camera roll, the per-object mask extents, every measurement of the asked quantity, agree / constraints with the untestable ones apart, named checks, each object's presence votes and rule, scene-graph edge sources and conflicts, every impulse decision of a physics chain, a measured slope with its rolling law and refined onset, the kinematic onset bound, each object's fit as a whole (`fit.object`: s² vs its bound, χ² per row kind) with its inadmissible constraints and excluded focal votes apart, the scale object's contacts as depth equations and the plane's rescaling, the main support's own plane with its `level` test and refit, scene-graph events (a catch ending a flight) as ticks on the track timeline, presence reasons and unknown zoom votes, the reference objects' size reads (frames read and left out, rms, dims) and their static poses in the 3D view, class votes with their role, a stopped unit's error, a partial unit's failed planning round and time per step, a motion prior's read per motion piece on the clip's time axis, a class prior's instances with their reads, an object in a container against the container's depth, each on / in edge's mask containment, each object's choice frame with its rule and probe-frame scan, where its track starts and the record's binding to it, the unobserved phases on the track timeline, a kick's frame-pair speeds ± σ and its ramp end, a pendulum's period read from both swing series, the frame indices beside every window and piece, the agent's no-repeat repairs, edits kept off unobserved phases and the constraint contract replayed per round, the mask components each track dropped and its leak frames, an anchored class vote's reading per stated depth, an overhead structure's seen-from-below test |
| `wild.html` + `wild.js` | In the Wild: the pipeline on real videos outside the benchmark (`data/QuantiPhy-wild`, no ground truth, no scores), one block per video, best aligned first (mean IoU of the final program, `metric.json`): the prior, both questions with `answer_final` and `answer_scene` ± σ (or the run status), the final program's overlays, plan grid and shapes. `data/wild.json` and the downscaled images `data/wild/` from `tools/build_wild.py` |
| `site.css` | Shared styles (color tokens, type, top bar) |
| `data/` | Data for the evaluation page (`videos.json`, thumbnails; `runtime/<version>/` holds that version's overlays and their record `overlays.json`), built by `tools/build_videos.py` |
| `media/val`, `media/test` | Symlinks to the original video folders (not in git) |
| `media/sceneprog/<tag>` | Symlinks to `outputs/sceneprog/<tag>` (not in git): `ln -sfn /nas/yumenghe/QuantiPhy-data/outputs/sceneprog/<tag> media/sceneprog/<tag>` |

## Serve

```bash
./serve.sh start    # port 9002 (override with PORT=...), Range support so videos can seek
./serve.sh status
./serve.sh stop
```

## Rebuild the scene data

From the repository root:

```bash
python -B web/tools/build_videos.py --val-runs v6gpt --sceneprog v6val v6valloop   # harness run, scene-program build and loop tags
python -B web/tools/build_qualitative.py --loop v6valloop                         # the qualitative comparison: our overlays on the baselines' frames
python -B web/tools/build_wild.py --build v6wild --loop v6wildloop                # In the Wild: outputs/sceneprog_wild/<tag>, outputs/sceneprog_answers_wild_<loop>.csv
python -B web/tools/build_baseline.py --run physmind=/nas/yumenghe/QuantiPhy-data/physmind_result_new/quantiphy/scores/physmind_preds.csv \
    --run spatialcode=/nas/yumenghe/QuantiPhy-data/spatial_code_result/quantiphy/answers/preds.csv   # baselines given here are rebuilt, the others kept
```

The script reads only that version's own artifacts. It never runs the pipeline and never falls back to another version or field: artifacts are read in the format `VERSION_FORMAT` declares for the version (motion record fields, processing unit), and anything else is an error. Only the current version may render missing overlays; building an older version fails if an overlay is missing. Register a new version in `VERSION_FORMAT` first.

Our row in the method comparison (Home and Baseline) is `data/comparison_ours.json`, written by `build_videos.py`: every question of `data/baseline.json` with its category, `answer_final`, its source and both MRAs; the pages score it with the baselines' rule (mean of the four category means). A version with the scene program needs `--sceneprog BUILD LOOP` and reads `outputs/sceneprog/<BUILD>`, `outputs/sceneprog/<LOOP>` and `outputs/sceneprog_answers_<LOOP>.json`. The build also regenerates two tables of `pipeline.html` between their marks: the class-size table (from `pipeline/class_sizes.py`) and the harness time per step (from the first run's answer records themselves, not its CSV).
