"""
OCR Application Source Package

This package contains the core OCR functionality and application interfaces.
"""

from .app_info import APP_NAME, APP_VERSION
from .exceptions import OCREngineFallbackError
from .ocr_engine import OCREngine, extract_text_from_image, get_available_engines

__version__ = APP_VERSION
__app_name__ = APP_NAME
__all__ = [
    "OCREngine",
    "OCREngineFallbackError",
    "extract_text_from_image",
    "get_available_engines",
]