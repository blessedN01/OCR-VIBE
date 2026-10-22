"""
OCR Application Source Package

This package contains the core OCR functionality and application interfaces.
"""

from .exceptions import OCREngineFallbackError
from .ocr_engine import OCREngine, extract_text_from_image, get_available_engines

__version__ = "2.0.0"
__all__ = [
    "OCREngine",
    "OCREngineFallbackError",
    "extract_text_from_image",
    "get_available_engines",
]