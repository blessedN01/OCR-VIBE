"""Shared OCR result types and normalization helpers."""

from typing import Any, Dict, List


def normalize_bbox(value: Any) -> list:
    """Coerce an engine-specific bbox representation to a plain nested list.

    Accepts None, plain list/tuple of points, or any object with a ``tolist``
    method (e.g. numpy arrays, tensors). Returns a new list so callers can
    mutate freely without affecting the original engine result.
    """
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)
