# Project website

Site for "Reconstructing measurable physical scenes from video". Static pages, no build step.

| File | Content |
|---|---|
| `index.html` + `comparison.js` | Home: project scope, keywords, pipeline, quantitative comparison (the Baseline page's method summary, computed from `data/baseline.json`, plus our row from `data/comparison_ours.json`), qualitative comparison (`data/qualitative/index.json` from `tools/build_qualitative.py`: PhysMind's posed meshes of the question objects, `data/physmind/`, from `tools/render_physmind.py`; Thinking with Spatial Code's encoder boxes, `data/spatialcode/`, copied from `/nas/yumenghe/QuantiPhy-data/spatial_code_result/quantiphy/reconstructions/`; on each of their frames the fitted scene program of one of the frame's questions, `data/qualitative/<tag>/`) |
| `pipeline.html` | The chain (perception → scene program → Physics-guided Fitting → answers) with each step's input, processing and output, the class-size table, time per step, and the harness's five steps (objects, task plan, scene graph, track, focal) |
| `evaluation.html` + `videos.js` | Evaluation: the run on the validation set — both answers (`answer_scene`, `answer_final`) over all questions and the ones shared with the baselines, per category — and per question its prior, its two answers with their MRA, the fitted program's overlays, the plan grid and shapes (`media/sceneprog/<tag>/`), the video and the time per step |
| `wild.html` + `wild.js` | In the Wild: the pipeline on real videos outside the benchmark (`data/QuantiPhy-wild`, no ground truth, no scores), one block per video, best aligned first (mean IoU of the fitted program, `metric.json`): the prior, both questions with `answer_final` and `answer_scene` ± σ (or the run status), the fitted program's overlays, plan grid and shapes. `data/wild.json` and the downscaled images `data/wild/` from `tools/build_wild.py` |
| `site.css` | Shared styles (color tokens, type, top bar) |
| `data/` | Data for the evaluation page (`videos.json`, thumbnails; `runtime/<tag>/` holds the run's redrawn overlays and their record `overlays.json`), built by `tools/build_videos.py` |
| `media/val`, `media/test` | Symlinks to the original video folders (not in git) |
| `media/sceneprog/<tag>` | Symlinks to `outputs/sceneprog/<tag>` (not in git): `ln -sfn /data/yumenghe/QuantiPhy/outputs/sceneprog/<tag> media/sceneprog/<tag>` |

## Serve

```bash
./serve.sh start    # port 9002 (override with PORT=...), Range support so videos can seek
./serve.sh status
./serve.sh stop
```

## Rebuild the scene data

From the repository root:

```bash
python -B web/tools/build_videos.py --run main                                    # the run: answers, read-outs, plans, overlays redrawn
python -B web/tools/build_qualitative.py --tag main                               # the qualitative comparison: our overlays on the baselines' frames
python -B web/tools/render_physmind.py --masks <scratch dir>                      # PhysMind's side in the comparison's colours
python -B web/tools/restyle_baselines.py                                          # Thinking with Spatial Code's side in the same colours
python -B web/tools/build_wild.py --build v6wild --status outputs/sceneprog_answers_wild_v6wildloop.csv \
    --answers /nas/yumenghe/QuantiPhy-data/scratch/qp-code-3d-chain/answers_wild_fix.json --newer main   # In the Wild
python -B web/tools/build_baseline.py --run physmind=/nas/yumenghe/QuantiPhy-data/physmind_result_new/quantiphy/scores/physmind_preds.csv \
    --run spatialcode=/nas/yumenghe/QuantiPhy-data/spatial_code_result/quantiphy/answers/preds.csv   # baselines given here are rebuilt, the others kept
```

`--run TAG` reads a run of the current chain (harness until track, then masks, build, fit, answer): every question's `outputs/sceneprog/<TAG>/<unit>/` (`answer.json` with `answer_scene` and `answer_final`, `fit.json`, `plan.json`, plan grid, shapes) and its harness record `outputs/harness/<TAG>/answers/<unit>.json` (timing). The fitted program's overlays are redrawn at the frames the chain drew, in the site's style with a legend of the line styles (`tools/sceneprog_overlay.py`, colours from `pipeline/palette.py`), into `data/runtime/<TAG>/<unit>/`; `data/runtime/<TAG>/overlays.json` records the drawing code's hash per unit, so only changed units are redrawn. Needs the symlink `media/sceneprog/<TAG>` -> `outputs/sceneprog/<TAG>` (plan grids and shapes). The wild page and the qualitative comparison redraw their overlays with the same module.

It reads only that run's own products and never runs the pipeline; a missing product is an error, never a substitute. A run tag is only a name: outputs keep the name they were run under.

Our row in the method comparison (Home and Baseline) is `data/comparison_ours.json`, written by `build_videos.py`: every question of `data/baseline.json` with its category, `answer_final` and both MRAs, each against that file's ground truth with the baselines' MRA (`tools/build_baseline.py` `mra`); the pages score it with the baselines' rule (mean of the four category means). The build also regenerates two tables of `pipeline.html` between their marks: the class-size table (from `pipeline/class_sizes.py`) and the harness time per step (from the run's harness records).
