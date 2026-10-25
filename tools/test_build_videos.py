"""Offline viewer binding tests; no model requests or real test-set reads."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

spec = importlib.util.spec_from_file_location("build_videos", Path(__file__).with_name("build_videos.py"))
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def fixture():
    rows, answers, parts = [], [], []
    for i, prior in enumerate(("diameter = 1 m", "diameter = 2 m")):
        context = {"row": i, "question": "What is the width of the ball in meters?", "prior": prior,
                   "depth_info": "", "category": "S2"}
        row = {**context, "video_id": "synthetic", "video_source": "synthetic", "video_type": "2D",
               "fps": 10., "dim": 2, "inference_type": "static", "answer": 1., "pred": 1.,
               "source": "scene", "conf": .7, "status": "resolved", "scene_id": f"scene-{i}"}
        answer = {**context, "scene_id": f"scene-{i}", "source": "measurement" if i else "scene",
                  "conf": .6 + .1 * i, "status": "resolved", "measurement": {"reasons": [f"group-{i}"]}}
        state = {"scene_id": f"scene-{i}", "question_context": [context], "results": [
            {"row": i, "question": context["question"], "quality": {"align_tq": {"iou": .3 + .1 * i}}}],
            "layers": [], "version": {"pipeline": "v77"}}
        rows.append(row)
        answers.append(answer)
        parts.append({"group": i, "prior": prior, "depth_info": "", "scene": state})
    return rows, answers, {"groups": parts, "scene_id": "last-group-must-not-be-used"}


class HarnessIdentityTests(unittest.TestCase):
    def test_manifest_order_and_identity_are_required_and_labels_are_authoritative(self):
        rows, _, _ = fixture()
        manifest = pd.DataFrame(rows)
        altered = manifest.copy()
        altered["answer"] = 999.
        altered["category"] = "D3"
        checked = builder.validate_harness_csv(altered, manifest)
        self.assertEqual(checked.answer.tolist(), manifest.answer.tolist())
        self.assertEqual(checked.category.tolist(), manifest.category.tolist())
        for bad in (manifest.iloc[::-1], manifest.iloc[:1], manifest.assign(prior="foreign")):
            with self.assertRaises(ValueError):
                builder.validate_harness_csv(bad, manifest)

    def test_same_question_different_prior_and_row_binds_own_scene(self):
        rows, answers, saved = fixture()
        groups = builder.harness_groups(saved)
        for i, row in enumerate(rows):
            answer, group, error = builder.bind_harness_question(row, answers, groups)
            self.assertIsNone(error)
            self.assertEqual(group["state"]["scene_id"], f"scene-{i}")
            self.assertEqual(answer["measurement"]["reasons"], [f"group-{i}"])

    def test_missing_or_duplicate_scene_id_is_unavailable(self):
        rows, answers, saved = fixture()
        groups = builder.harness_groups(saved)
        for mode in ("missing", "duplicate", "wrong-depth"):
            with self.subTest(mode=mode):
                current = copy.deepcopy(groups)
                if mode == "missing":
                    current[0]["state"].pop("scene_id")
                elif mode == "duplicate":
                    current[1]["state"]["scene_id"] = current[0]["state"]["scene_id"]
                else:
                    current[0]["state"]["question_context"][0]["depth_info"] = "different depth"
                _, group, error = builder.bind_harness_question(rows[0], answers, current)
                self.assertIsNone(group)
                self.assertTrue(error)

    def test_duplicate_answer_identity_and_mismatched_csv_id_are_unavailable(self):
        rows, answers, saved = fixture()
        groups = builder.harness_groups(saved)
        self.assertIsNone(builder.bind_harness_question(rows[0], answers + [answers[0]], groups)[1])
        rows[0]["scene_id"] = "other"
        self.assertIsNone(builder.bind_harness_question(rows[0], answers, groups)[1])

    def test_harness_build_needs_no_legacy_geo_vlm_and_renders_matching_state(self):
        rows, answers, saved = fixture()
        rendered = []
        def render(tag, vid, state, dim, question, fps, duration):
            rendered.append(state["scene_id"])
            return [{"scene_id": state["scene_id"], "path": f"data/runtime/{tag}/{state['scene_id']}.jpg", "t": 0.}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "outputs"
            (out / "harness/v77/answers").mkdir(parents=True)
            (out / "scene2d_eval/h_v77/agent").mkdir(parents=True)
            pd.DataFrame(rows).to_csv(out / "val_v77_answers.csv", index=False)
            (out / "harness/v77/answers/synthetic.json").write_text(json.dumps({
                "version": {"pipeline": "v77", "commit": "saved-commit"}, "answers": answers, "execution_status": "complete"}))
            (out / "scene2d_eval/h_v77/agent/synthetic.json").write_text(json.dumps(saved))
            with patch.object(builder, "ROOT", root), patch.object(builder, "load_split", return_value=pd.DataFrame(rows)), \
                    patch.object(builder, "video_meta", return_value=(2., 854, 480, None)), \
                    patch.object(builder, "video_info", return_value={}), \
                    patch.object(builder, "scene2d", side_effect=lambda *a, **k: {"scene_id": k["state"]["scene_id"]}), \
                    patch.object(builder, "runtime_overlays", side_effect=render), \
                    patch.dict(builder.SCENE2D_DIRS, {}), patch.dict(builder.SCENE_DIRS, {}), \
                    patch.dict(builder.TRACKS_KIND, {}):
                output = builder.build_harness_val("v77")
        self.assertEqual(rendered, ["scene-0", "scene-1"])
        self.assertIsNone(output["videos"][0]["scene2d"])
        for i, q in enumerate(output["videos"][0]["questions"]):
            self.assertEqual(q["scene_id"], f"scene-{i}")
            self.assertEqual(q["group"], str(i))
            self.assertEqual(q["runtime_overlays"][0]["scene_id"], q["scene_id"])
            self.assertEqual(q["measurement"]["reasons"], [f"group-{i}"])
            self.assertEqual(q["version"]["commit"], "saved-commit")

    def test_val_only_keeps_test_value_without_test_or_index_work(self):
        existing = {"generated": "old", "test": {"videos": [{"id": "do-not-touch", "questions": [{"pred": 17.}]}]},
                    "val": {"old": True}, "extra": {"keep": True}}
        replacement = {"harness_tag": "v77", "videos": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "videos.json"
            path.write_text(json.dumps(existing))
            with patch.object(builder, "OUT", path), patch.object(builder, "THUMBS", Path(tmp) / "thumbs"), \
                    patch.object(builder, "build_harness_val", return_value=replacement), \
                    patch.object(builder, "build_test", side_effect=AssertionError("test build must not run")), \
                    patch.object(builder, "write_versions", side_effect=AssertionError("index must not change")), \
                    patch("sys.argv", ["build_videos.py", "--val-harness-tag", "v77", "--val-only"]):
                builder.main()
            output = json.loads(path.read_text())
        self.assertEqual(output["test"], existing["test"])
        self.assertEqual(output["extra"], existing["extra"])
        self.assertEqual(output["val"], replacement)


if __name__ == "__main__":
    unittest.main()
