"""
OCR Application Source Package

This package contains the core OCR functionality and application interfaces.
"""

from .ocr_engine import OCREngine, extract_text_from_image, get_available_engines
from .exceptions import OCREngineFallbackError

__version__ = "1.2.0"
__all__ = [
    "OCREngine",
    "extract_text_from_image",
    "get_available_engines",
    "OCREngineFallbackError",
]