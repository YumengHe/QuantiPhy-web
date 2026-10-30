# Project website

Site for "Reconstructing measurable physical scenes from video". Static pages, no build step.

| File | Content |
|---|---|
| `index.html` | Home: project scope, pipeline overview, datasets, validation results, keywords, links |
| `pipeline.html` | Flow diagram (a single chain), answer sources |
| `evaluation.html` + `videos.js` | Evaluation: per-question results on the validation set with video, 3D scene, overlays, scene graph, answer and source |
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
python -B web/tools/build_videos.py --val-run <tag>    # a specific version
```

The script reads only that version's own artifacts. It never runs the pipeline and never falls back to another version or field: artifacts are read in the format `VERSION_FORMAT` declares for the version (motion record fields, processing unit), and anything else is an error. Only the current version may render missing overlays; building an older version fails if an overlay is missing. Register a new version in `VERSION_FORMAT` first.

The validation results table in `index.html` is updated by hand from `docs/VERSIONS.md`.
