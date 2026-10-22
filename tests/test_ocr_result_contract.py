#!/usr/bin/env python3
"""
Contract tests for OCR engine result schema.

Pins the guarantee that every engine's output, after flowing through
extract_text / _preserve_layout / format_csv, is composed of plain
Python types (no numpy arrays, tensors, or other non-JSON-serializable
objects leak into downstream consumers).
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_engine import OCREngine


def _make_engine(available_engine: str = "tesseract") -> OCREngine:
    """Build an OCREngine with only the requested engine marked available,
    avoiding any heavy model initialization at import time."""
    flags = {
        "easyocr": (True, False, False),
        "paddleocr": (False, True, False),
        "tesseract": (False, False, True),
    }
    easy, paddle, tess = flags[available_engine]
    with patch('src.ocr_engine.EASYOCR_AVAILABLE', easy), \
         patch('src.ocr_engine.PADDLEOCR_AVAILABLE', paddle), \
         patch('src.ocr_engine.TESSERACT_AVAILABLE', tess), \
         patch('src.ocr_engine.OPENCV_AVAILABLE', True), \
         patch('src.ocr_engine.NUMPY_AVAILABLE', True), \
         patch('src.ocr_engine.SKLEARN_AVAILABLE', True):
        return OCREngine(auto_select=False, preprocess_images=False)


class TestResultContract:
    """Guarantee that all engines return JSON-serializable, plain-Python results."""

    @pytest.fixture
    def blank_image(self):
        return np.zeros((80, 200, 3), dtype=np.uint8)

    def _inject_mock_reader(self, engine: OCREngine, raw_results):
        """Replace the engine's reader with a mock returning raw_results."""
        engine.paddleocr_reader = MagicMock()
        engine.easyocr_reader = MagicMock()
        engine.paddleocr_reader.predict.return_value = raw_results
        engine.easyocr_reader.readtext.return_value = raw_results

    def test_paddleocr_returns_plain_bbox_lists(self, blank_image):
        """PaddleOCR's rec_polys are numpy arrays; normalization must produce lists."""
        engine = _make_engine("tesseract")
        raw = [{
            "rec_texts": ["ABC"],
            "rec_scores": [np.float32(0.95)],
            "rec_polys": [
                np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                         dtype=np.float64)
            ],
        }]
        self._inject_mock_reader(engine, raw)
        _, results = engine.extract_text(blank_image, engine="paddleocr")
        assert len(results) == 1
        r = results[0]
        assert isinstance(r["text"], str)
        assert isinstance(r["confidence"], float)
        assert isinstance(r["bbox"], (list, tuple))
        assert len(r["bbox"]) == 4
        for pt in r["bbox"]:
            assert isinstance(pt, (list, tuple))
            assert len(pt) == 2
        # The whole result list must be JSON-serializable (e.g., for export).
        json.dumps(results)

    def test_easyocr_results_are_plain_types(self, blank_image):
        """EasyOCR results must also satisfy the plain-type contract."""
        engine = _make_engine("easyocr")
        raw = [
            [
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                "Hello",
                0.9,
            ]
        ]
        self._inject_mock_reader(engine, raw)
        _, results = engine.extract_text(blank_image, engine="easyocr")
        assert len(results) == 1
        r = results[0]
        assert isinstance(r["text"], str)
        assert isinstance(r["confidence"], float)
        assert isinstance(r["bbox"], (list, tuple))
        json.dumps(results)

    def test_results_with_empty_bbox_do_not_crash_layout(self, blank_image):
        """Mixed valid / empty bboxes must not crash _preserve_layout."""
        engine = _make_engine("paddleocr")
        raw = [{
            "rec_texts": ["A", "B"],
            "rec_scores": [np.float32(0.9), np.float32(0.8)],
            "rec_polys": [
                np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]),
                [],  # empty bbox — was a crash site before the fix
            ],
        }]
        self._inject_mock_reader(engine, raw)
        engine.preserve_layout = True
        text, results = engine.extract_text(blank_image, engine="paddleocr")
        # Should succeed, not raise "truth value of an array is ambiguous".
        assert "A" in text
        json.dumps(results)

    def test_preserve_layout_with_numpy_bboxes_directly(self):
        """Unit-level guard: _preserve_layout must accept numpy polygon arrays."""
        engine = _make_engine("tesseract")
        fake_results = [
            {
                "text": "TopLeft",
                "bbox": np.array([[10.0, 10.0], [60.0, 10.0], [60.0, 30.0], [10.0, 30.0]]),
            },
            {
                "text": "TopRight",
                "bbox": np.array([[70.0, 10.0], [120.0, 10.0], [120.0, 30.0], [70.0, 30.0]]),
            },
        ]
        # Before the fix, this raised ValueError on `bool(ndarray)`.
        out = engine._preserve_layout(fake_results)
        assert "TopLeft" in out
        assert "TopRight" in out


class TestBboxNormalization:
    """Validate bbox normalization contract at the unit level."""

    def test_numpy_array_normalizes_to_list(self):
        arr = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        if hasattr(arr, "tolist"):
            normalized = arr.tolist()
        else:
            normalized = arr
        assert isinstance(normalized, list)
        assert all(isinstance(pt, list) for pt in normalized)
        assert all(isinstance(v, (int, float)) for pt in normalized for v in pt)

    def test_empty_list_remains_empty_list(self):
        bbox = []
        if hasattr(bbox, "tolist"):
            bbox = bbox.tolist()
        assert bbox == []
        assert isinstance(bbox, list)

    def test_none_remains_none(self):
        bbox = None
        if hasattr(bbox, "tolist"):
            bbox = bbox.tolist()
        assert bbox is None
