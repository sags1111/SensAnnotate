"""Run with python -m unittest discover -s test -p 'test_manual_annotations.py'."""

import ast
import copy
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotation_colors import color_for, color_hex
from manual_drawing import make_manual_region, polygon_area, classify_manual_region


def triangle(tool: str = "polygon") -> dict:
    return {"tool": tool, "sensitivity": "sensitive",
            "points": [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]]}


class ManualAnnotationTests(unittest.TestCase):
    def test_four_distinct_color_meanings(self):
        self.assertEqual(color_hex(color_for("sensitive", "unknown")), "#ff8000")
        self.assertEqual(color_hex(color_for("non_sensitive", " UNKNOWN ")), "#2ec4a6")
        colors = {color_for(s, c) for s in ("sensitive", "non_sensitive")
                  for c in ("unknown", "person")}
        self.assertEqual(len(colors), 4)

    def test_polygon_preserved_at_original_resolution(self):
        region = make_manual_region(triangle(), 2000, 1000)
        self.assertEqual(region["polygons"], [[200, 100, 1800, 100, 200, 900]])
        self.assertEqual(region["bbox_xyxy"], [200, 100, 1800, 900])
        self.assertEqual(polygon_area(region["polygons"][0]), 640000)
        self.assertIsNone(region["confidence"])

    def test_freehand_and_non_sensitive_metadata(self):
        payload = triangle("freehand")
        payload["sensitivity"] = "non_sensitive"
        region = make_manual_region(payload, 100, 100)
        self.assertEqual(region["sensitivity"], "non_sensitive")
        self.assertEqual(region["segmentation_method"], "Manual freehand region")

    def test_rectangle_on_image_edges(self):
        payload = triangle("rectangle")
        payload["points"] = [[0, 0], [1, 0], [1, 1], [0, 1]]
        region = make_manual_region(payload, 100, 50)
        self.assertEqual(region["bbox_xyxy"], [0, 0, 100, 50])
        self.assertEqual(polygon_area(region["polygons"][0]), 5000)

    def test_closed_outline_and_reversed_winding(self):
        payload = triangle()
        payload["points"].append(payload["points"][0])
        forward = make_manual_region(payload, 100, 100)
        payload["points"].reverse()
        reverse = make_manual_region(payload, 100, 100)
        self.assertEqual(polygon_area(forward["polygons"][0]),
                         polygon_area(reverse["polygons"][0]))

    def test_concave_outline_is_not_replaced_with_rectangle(self):
        payload = triangle()
        payload["points"] = [[0, 0], [1, 0], [.5, .5], [1, 1], [0, 1]]
        region = make_manual_region(payload, 100, 100)
        self.assertEqual(polygon_area(region["polygons"][0]), 7500)
        self.assertEqual(len(region["polygons"][0]), 10)

    def test_bad_coordinates_are_rejected(self):
        for coordinate in (float("nan"), float("inf"), -0.01, 1.01, "0.5", True):
            with self.subTest(coordinate=coordinate):
                payload = triangle()
                payload["points"][1][0] = coordinate
                with self.assertRaises(ValueError):
                    make_manual_region(payload, 100, 100)

    def test_malformed_or_empty_input(self):
        for payload in (None, {}, {**triangle(), "points": []},
                        {**triangle(), "points": [[0, 0]] * 1001},
                        {**triangle(), "tool": []},
                        {**triangle(), "sensitivity": "maybe"},
                        {**triangle(), "points": [[0], [1, 0], [0, 1]]}):
            with self.subTest(payload=str(payload)[:100]):
                with self.assertRaises(ValueError):
                    make_manual_region(payload, 100, 100)

    def test_zero_area_and_self_crossing_are_rejected(self):
        for points in ([[0, 0], [.5, .5], [1, 1]],
                       [[0, 0], [1, 1], [1, 0], [0, 1]],
                       [[0, 0], [1, 0], [1, 1], [.5, 0], [0, 1]]):
            with self.subTest(points=points):
                with self.assertRaises(ValueError):
                    make_manual_region({**triangle(), "points": points}, 100, 100)

    def test_invalid_dimensions(self):
        for width, height in [(0, 100), (100, -1)]:
            with self.assertRaises(ValueError):
                make_manual_region(triangle(), width, height)

    def test_rectangle_metadata_requires_rectangular_geometry(self):
        with self.assertRaises(ValueError):
            make_manual_region(triangle("rectangle"), 100, 100)

    def test_auto_uses_crop_and_preserves_clip_provenance(self):
        class Frame:
            shape = (100, 200, 3)
            def __getitem__(self, key):
                return key
        region = make_manual_region({**triangle(), "sensitivity": "auto"}, 200, 100)
        self.assertIsNone(region["sensitivity"])
        calls = []
        def classify(crop):
            calls.append(crop)
            return "non_sensitive", "non-sensitive-category match", .12
        classify_manual_region(region, Frame(), classify)
        self.assertEqual(calls, [(slice(10, 90), slice(20, 180))])
        self.assertEqual(region["sensitivity"], "non_sensitive")
        self.assertEqual(region["source"], "manual")
        self.assertEqual(region["sensitivity_source"], "clip")
        self.assertEqual(region["clip_suggested_sensitivity"], "non_sensitive")
        self.assertEqual(region["confidence"], .12)
        self.assertTrue(region["needs_review"])
        self.assertEqual(region["class_name"], "unknown")

    def test_invalid_clip_result_does_not_set_a_default_label(self):
        class Frame:
            shape = (100, 100, 3)
            def __getitem__(self, key):
                return key
        for result in [("maybe", "", .5), ("sensitive", "", float("nan"))]:
            region = make_manual_region({**triangle(), "sensitivity": "auto"}, 100, 100)
            with self.assertRaises(ValueError):
                classify_manual_region(region, Frame(), lambda _: result)
            self.assertIsNone(region["sensitivity"])

    def test_coco_export_uses_outline_area(self):
        # Category mapping is unrelated to geometry; avoid loading GPU config
        # in this isolated exporter unit test.
        path = Path(__file__).resolve().parents[1] / "coco_export.py"
        spec = importlib.util.spec_from_file_location("tested_coco_export", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"config": SimpleNamespace(classify_supercategory=lambda _: "object")}):
            spec.loader.exec_module(module)
        region = make_manual_region(triangle(), 100, 100)
        result = module.build_coco({"video.mp4": {
            "type": "video", "width": 100, "height": 100, "fps": 25,
            "frames": {7: [region]},
        }})
        annotation = result["annotations"][0]
        self.assertEqual(annotation["area"], 3200)
        self.assertEqual(annotation["bbox"], [10, 10, 80, 80])
        self.assertEqual(annotation["segmentation"], region["polygons"])
        self.assertEqual(result["images"][0]["timestamp_seconds"], .28)
        region.update(confidence=.12, sensitivity="non_sensitive", sensitivity_source="clip",
                      clip_suggested_sensitivity="non_sensitive", needs_review=True)
        exported = module.build_coco({"image.jpg": {
            "type": "image", "width": 100, "height": 100, "frames": {0: [region]},
        }})["annotations"][0]["attributes"]
        self.assertEqual(exported["source"], "manual")
        self.assertEqual(exported["confidence_type"], "clip_sensitive_softmax_score")
        self.assertEqual(exported["sensitivity_source"], "clip")
        self.assertEqual(exported["clip_suggested_sensitivity"], "non_sensitive")
        self.assertTrue(exported["needs_review"])

    def test_storage_is_exact_frame_and_invalidates_derived_data(self):
        # Execute the actual storage helpers without launching the app/models.
        path = Path(__file__).resolve().parents[1] / "app.py"
        nodes = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef)
                 and node.name in {"add_manual_region", "mark_annotations_changed"}]
        class State(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__
        for kind in ("image", "video"):
            state = State(files={"file": {"type": kind, "width": 100, "height": 100}},
                          annotations={"file": {}}, annotation_revisions={},
                          video_tally_cache={"file": {}}, prepared_exports={"old": True},
                          preview_path_file="old.mp4")
            import logging
            ns = {"st": SimpleNamespace(session_state=state, spinner=lambda _: nullcontext()),
                  "make_manual_region": make_manual_region, "classify_manual_region": classify_manual_region,
                  "logging": logging}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
            region = ns["add_manual_region"]("file", 7, triangle())
            storage = state.annotations["file"]["frames" if kind == "image" else "manual_frames"]
            self.assertEqual(list(storage), [7])
            self.assertIs(storage[7][0], region)
            self.assertIsNone(state.prepared_exports)
            self.assertNotIn("file", state.video_tally_cache)
            self.assertNotIn("preview_path_file", state)
            before = copy.deepcopy(state.annotations)
            with self.assertRaises(ValueError):
                ns["add_manual_region"]("file", 8, {})
            self.assertEqual(state.annotations, before)
            ns["add_manual_region"]("file", 8, {**triangle(), "sensitivity": "non_sensitive"})
            class Frame:
                shape = (100, 100, 3)
                def __getitem__(self, key):
                    return key
            calls = []
            def classify(crop):
                calls.append(crop)
                return "non_sensitive", "non-sensitive-category match", .1
            ns["get_frame_array"] = lambda filename, frame: Frame()
            ns["current_pipeline"] = lambda: SimpleNamespace(classify_with_clip=classify)
            auto = ns["add_manual_region"]("file", 9, {**triangle(), "sensitivity": "auto"})
            self.assertEqual(len(calls), 1)
            self.assertEqual(auto["sensitivity"], "non_sensitive")
            self.assertEqual(storage[9][0]["sensitivity_source"], "clip")
            before = copy.deepcopy(state.annotations)
            def fail(crop):
                raise RuntimeError("Model unavailable")
            ns["current_pipeline"] = lambda: SimpleNamespace(classify_with_clip=fail)
            with self.assertLogs(level="ERROR"), self.assertRaisesRegex(ValueError, "not saved"):
                ns["add_manual_region"]("file", 10, {**triangle(), "sensitivity": "auto"})
            self.assertEqual(state.annotations, before)


if __name__ == "__main__":
    unittest.main()
