#!/usr/bin/env python3
"""
OCR Engine Module

This module provides OCR functionality using EasyOCR as the primary engine
with pytesseract as a fallback. It includes image preprocessing capabilities
using OpenCV and Pillow, and supports multiple languages with confidence thresholds.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, ClassVar

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.error("NumPy is required but not installed. Install with: pip install numpy")

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    logger.error("Pillow is required for image processing. Install with: pip install pillow")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.error("OpenCV is required for image preprocessing. Install with: pip install opencv-python")

# OCR libraries - lazy loaded
EASYOCR_AVAILABLE = False
PADDLEOCR_AVAILABLE = False
TESSERACT_AVAILABLE = False
easyocr = None
PaddleOCR = None
pytesseract = None

try:
    import pytesseract as _pytesseract
    pytesseract = _pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.error("pytesseract is required for OCR functionality. Install with: pip install pytesseract")
    logger.error("Note: pytesseract also requires Tesseract OCR to be installed on your system.")
    logger.error("Download from: https://github.com/tesseract-ocr/tesseract")

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    logger.warning("pyspellchecker not installed. English-only post-processing will be limited. Install with: pip install pyspellchecker")

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. Layout preservation will not work. Install with: pip install scikit-learn")

# Import custom modules
from .app_info import APP_NAME_LOWER, get_app_dir
from .exceptions import (
    ImageProcessingError,
    MemoryError,
    OCREngineFallbackError,
    OCREngineInitializationError,
    OCREngineNotAvailableError,
    OCRError,
    OCRProcessingError,
)
from .memory_manager import MemoryManager, ResourceManager
from .stats_tracker import EngineStatsTracker


def _get_easyocr_cache_dir() -> str:
    """
    Get a writable cache directory for EasyOCR models.

    In a PyInstaller bundle, uses a directory next to the executable or APPDATA.
    Otherwise, uses the default EasyOCR cache location.
    """
    # Check if running in PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Running in a bundle
        cache_dir = str(get_app_dir() / 'easyocr_models')
    else:
        # Running in development, use default EasyOCR location
        cache_dir = os.path.join(os.path.expanduser('~'), '.EasyOCR', 'model')

    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_paddleocr_cache_dir() -> str:
    """
    Get a writable cache directory for PaddleOCR models.

    Uses a directory under the user's application data directory
(get_app_dir() / 'paddleocr_models').
    to ensure consistent caching and PyInstaller bundle compatibility.
    """
    if getattr(sys, 'frozen', False):
        cache_dir = str(get_app_dir() / 'paddleocr_models')
    else:
        cache_dir = os.path.join(os.path.expanduser('~'), '.paddlex', 'official_models')

    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


PADDLEOCR_LANG_MAP: dict[str, str] = {
    'en': 'en',
    'fr': 'fr',
    'de': 'german',
    'es': 'es',
    'it': 'it',
    'pt': 'pt',
    'nl': 'nl',
    'ru': 'eslav',
    'ja': 'japan',
    'ko': 'korean',
    'zh': 'ch',
    'zh-cn': 'ch',
    'zh-tw': 'chinese_cht',
    'tr': 'tr',
    'vi': 'vi',
    'th': 'th',
    'hi': 'devanagari',
    'ar': 'arabic',
    'el': 'el',
    'pl': 'pl',
    'cs': 'cs',
    'sk': 'sk',
    'sl': 'sl',
    'hu': 'hu',
    'ro': 'rs_latin',
    'fi': 'fi',
    'sv': 'sv',
    'no': 'no',
    'da': 'da',
    'nb': 'no',
    'is': 'is',
    'et': 'et',
    'lv': 'lv',
    'lt': 'lt',
    'mt': 'mt',
    'ga': 'ga',
    'cy': 'cy',
    'eu': 'eu',
    'ca': 'ca',
    'oc': 'oc',
    'gl': 'gl',
    'pt-br': 'pt',
    'la': 'la',
    'lb': 'lb',
    'qu': 'qu',
    'rm': 'rm',
    'sq': 'sq',
    'sw': 'sw',
    'tl': 'tl',
    'ms': 'ms',
    'id': 'id',
    'ku': 'ku',
    'mi': 'mi',
    'af': 'af',
    'az': 'az',
    'bs': 'bs',
    'uz': 'uz',
    'ta': 'ta',
    'te': 'te',
    'ka': 'ka',
}

DEFAULT_ENGINE_ORDER = ['easyocr', 'paddleocr', 'tesseract']

# Tesseract's traineddata files are named with ISO 639-3 codes (eng, fra, deu),
# NOT the ISO 639-1 codes EasyOCR and the rest of the app use (en, fr, de).
# Passing 'en' to pytesseract makes it look for en.traineddata, which does not
# exist, so every call fails with "Tesseract couldn't load any languages".
# This map converts app codes -> tesseract codes. Codes not listed here are
# passed through unchanged, which is correct for already-ISO-639-3 inputs.
TESSERACT_LANG_MAP: dict[str, str] = {
    'en': 'eng',
    'fr': 'fra',
    'de': 'deu',
    'es': 'spa',
    'it': 'ita',
    'pt': 'por',
    'nl': 'nld',
    'ru': 'rus',
    'ja': 'jpn',
    'ko': 'kor',
    'zh': 'chi_sim',
    'zh-cn': 'chi_sim',
    'zh-tw': 'chi_tra',
    'tr': 'tur',
    'vi': 'vie',
    'th': 'tha',
    'hi': 'hin',
    'ar': 'ara',
    'el': 'ell',
    'pl': 'pol',
    'cs': 'ces',
    'sv': 'swe',
    'no': 'nor',
    'da': 'dan',
    'fi': 'fin',
    'hu': 'hun',
    'ro': 'ron',
    'uk': 'ukr',
    'he': 'heb',
    'id': 'ind',
    'ms': 'msa',
    'fa': 'fas',
    'ur': 'urd',
    'bn': 'ben',
    'ta': 'tam',
    'te': 'tel',
    'ml': 'mal',
    'mr': 'mar',
    'gu': 'guj',
    'ca': 'cat',
    'eu': 'eus',
    'gl': 'glg',
    'cy': 'cym',
    'sk': 'slk',
    'sl': 'slv',
    'lt': 'lit',
    'lv': 'lav',
    'et': 'est',
    'hr': 'hrv',
    'sr': 'srp',
    'bg': 'bul',
    'kk': 'kaz',
    'uz': 'uzb',
    'az': 'aze',
    'af': 'afr',
    'sq': 'sqi',
    'hy': 'hye',
    'ka': 'kat',
    'la': 'lat',
    'mk': 'mkd',
    'tl': 'tgl',
}


def _lazy_load_easyocr():
    global easyocr, EASYOCR_AVAILABLE
    if easyocr is not None:
        return EASYOCR_AVAILABLE
    try:
        import easyocr as _easyocr
        easyocr = _easyocr
        EASYOCR_AVAILABLE = True
        logger.info("EasyOCR loaded lazily")
    except ImportError:
        EASYOCR_AVAILABLE = False
        logger.error("EasyOCR is required for OCR functionality. Install with: pip install easyocr")
    return EASYOCR_AVAILABLE


def _lazy_load_paddleocr():
    global PaddleOCR, PADDLEOCR_AVAILABLE
    if PaddleOCR is not None:
        return PADDLEOCR_AVAILABLE
    try:
        from paddleocr import PaddleOCR as _PaddleOCR
        PaddleOCR = _PaddleOCR
        PADDLEOCR_AVAILABLE = True
        logger.info("PaddleOCR loaded lazily")
    except ImportError:
        logger.warning("PaddleOCR is not installed. Install with: pip install paddleocr")
        logger.warning("PaddleOCR excels at CJK, complex layouts, and low-quality scans.")
    return PADDLEOCR_AVAILABLE


def _find_tesseract_binary() -> str | None:
    """Resolve the Tesseract executable path, honouring any user override.

    Checks, in order:
      1. ``pytesseract.pytesseract.tesseract_cmd`` when it is an absolute path
         (the user's explicit override).
      2. ``tesseract`` on PATH (shutil.which).
      3. The ``TESSERACT_BIN`` environment variable.

    Returns the resolved path, or None when no binary can be found.
    """
    if pytesseract is not None:
        cmd = getattr(pytesseract.pytesseract, 'tesseract_cmd', None)
        if cmd and os.path.isabs(cmd) and os.path.isfile(cmd):
            return cmd

    found = shutil.which('tesseract')
    if found:
        return found

    env_bin = os.environ.get('TESSERACT_BIN')
    if env_bin and os.path.isfile(env_bin):
        return env_bin

    # Common Windows install locations. Tesseract's MSI installer drops the
    # binary under "Program Files" but does not add it to PATH, so a plain
    # shutil.which('tesseract') misses it. This is a generic fallback, not a
    # hard-coded path for a specific machine.
    if sys.platform == 'win32':
        for drive in ('C:\\', 'D:\\'):
            for parent in ('Program Files', 'Program Files (x86)'):
                cand = os.path.join(drive, parent, 'Tesseract-OCR', 'tesseract.exe')
                if os.path.isfile(cand):
                    return cand

    return None


def _find_tessdata_dir(tesseract_bin: str) -> str | None:
    """Resolve Tesseract's tessdata directory.

    Order of resolution:
      1. ``TESSDATA_PREFIX`` env var (set by the user/installer).
      2. ``tessdata`` next to the resolved binary.
      3. ``tessdata`` one level up from the binary (common Windows layout).

    Returns the directory path, or None when no tessdata directory exists.
    """
    env_prefix = os.environ.get('TESSDATA_PREFIX')
    if env_prefix and os.path.isdir(env_prefix):
        return env_prefix

    bin_dir = os.path.dirname(os.path.abspath(tesseract_bin))
    candidates = [
        os.path.join(bin_dir, 'tessdata'),
        os.path.join(os.path.dirname(bin_dir), 'tessdata'),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand

    return None


def _configure_pytesseract_runtime(runtime: dict[str, Any]) -> None:
    """Configure the pytesseract wrapper to use the resolved binary + tessdata.

    The pre-flight check (``check_tesseract_runtime``) resolves the real
    tesseract executable and tessdata directory, but ``pytesseract`` does not
    share that resolution. It looks up the binary from its own
    ``tesseract_cmd`` attribute (defaulting to the bare ``tesseract`` name)
    and the tessdata directory from the ``TESSDATA_PREFIX`` environment
    variable.

    On Windows the Tesseract MSI installer drops the binary under
    ``Program Files\\Tesseract-OCR`` without adding it to PATH and does not
    set ``TESSDATA_PREFIX``. As a result a plain ``pytesseract.image_to_*``
    call fails with "tesseract is not installed or it's not in your PATH"
    even though the binary is present and the pre-flight check passed.

    This function makes the wrapper match the pre-flight resolution. The
    user's explicit ``tesseract_cmd`` override is honoured: when it is an
    absolute path it was already consulted by the pre-flight, so the binary
    is left untouched and only the tessdata prefix is filled in when missing.
    """
    if pytesseract is None:
        return

    binary = runtime.get('binary')
    tessdata_dir = runtime.get('tessdata_dir')

    cmd = getattr(pytesseract.pytesseract, 'tesseract_cmd', None)
    if binary and not (cmd and os.path.isabs(cmd)):
        # No user override (or a bare name): point the wrapper at the
        # resolved binary so it is found regardless of PATH.
        pytesseract.pytesseract.tesseract_cmd = binary

    if tessdata_dir and not os.environ.get('TESSDATA_PREFIX'):
        # Only set the prefix when the user has not configured one; an
        # explicit TESSDATA_PREFIX always wins.
        os.environ['TESSDATA_PREFIX'] = tessdata_dir


def check_tesseract_runtime(languages: list[str] | None = None) -> dict[str, Any]:
    """Pre-flight check for the Tesseract runtime.

    A successful ``import pytesseract`` only proves the *wrapper* is installed;
    it says nothing about the Tesseract binary or its trained language data.
    This check probes the real runtime so failures are surfaced at startup
    with a clear, actionable message instead of mid-extraction.

    Args:
        languages: App-level language codes to validate against tessdata.

    Returns:
        A dict with keys:
          - ``binary``: path to the tesseract executable, or None
          - ``tessdata_dir``: tessdata directory, or None
          - ``binary_ok``: bool
          - ``tessdata_ok``: bool
          - ``missing_languages``: list of app codes with no traineddata file
          - ``ok``: True only when the binary exists and at least one requested
            language (or English, as a fallback) is present in tessdata.

    Raises:
        OCREngineNotAvailableError: when the binary is missing entirely.
    """
    result: dict[str, Any] = {
        'binary': None,
        'tessdata_dir': None,
        'binary_ok': False,
        'tessdata_ok': False,
        'missing_languages': [],
        'ok': False,
    }

    if pytesseract is None:
        raise OCREngineNotAvailableError(
            details={'suggestion': "Install pytesseract with: pip install pytesseract"}
        )

    binary = _find_tesseract_binary()
    result['binary'] = binary
    if not binary:
        raise OCREngineNotAvailableError(
            details={
                'suggestion': (
                    "Tesseract OCR binary not found. pytesseract is only a wrapper; "
                    "you also need the tesseract executable on PATH, or set "
                    "TESSERACT_BIN to its full path. Install it from "
                    "https://github.com/tesseract-ocr/tesseract/releases"
                )
            }
        )
    result['binary_ok'] = True

    tessdata_dir = _find_tessdata_dir(binary)
    result['tessdata_dir'] = tessdata_dir
    if not tessdata_dir:
        raise OCREngineNotAvailableError(
            details={
                'suggestion': (
                    "Tesseract tessdata directory not found. Set TESSDATA_PREFIX "
                    "to your tessdata directory (it contains *.traineddata files)."
                )
            }
        )
    result['tessdata_ok'] = True

    # Validate that the requested languages actually have traineddata files.
    # Tesseract files are named with ISO 639-3 codes (eng.traineddata), so we
    # map the app's ISO 639-1 codes before checking.
    codes = languages or ['en']
    missing = []
    for code in codes:
        tcode = TESSERACT_LANG_MAP.get(code.lower(), code.lower())
        if not any(
            os.path.isfile(os.path.join(tessdata_dir, f'{tcode}.traineddata'))
            for tcode in {tcode, code.lower()}
        ):
            missing.append(code)

    result['missing_languages'] = missing

    # Even if the specific requested codes are missing, English (or whatever
    # the first available language is) means the runtime itself is usable.
    fallback = 'eng.traineddata' if 'eng' in TESSERACT_LANG_MAP.values() else None
    if not missing or os.path.isfile(os.path.join(tessdata_dir, fallback or 'eng.traineddata')):
        result['ok'] = True

    if missing:
        logger.warning(
            "Tesseract is installed but language data is missing for: %s. "
            "Download traineddata from https://github.com/tesseract-ocr/tessdata "
            "and place the *.traineddata files in %s",
            ', '.join(missing), tessdata_dir,
        )

    return result


class OCREngine:
    """
    OCR Engine class that provides text extraction from images using multiple OCR engines.

    Supports EasyOCR as primary engine with pytesseract fallback, image preprocessing,
    language support, and confidence thresholding.
    """

    def __init__(self,
                 languages: list[str] | None = None,
                 confidence_threshold: float = 0.5,
                 use_gpu: bool = False,
                 preprocess_images: bool = True,
                 memory_limit_mb: int | None = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 deskew: bool = False,
                 clahe: bool = False,
                 upscale: bool = False,
                 remove_bg: bool = False,
                 preserve_layout: bool = False,
                 auto_select: bool = True,
                 stats_path: str | None = None):
        """
        Initialize the OCR engine.

        Args:
            languages: List of language codes (e.g., ['en', 'fr']). Defaults to ['en']
            confidence_threshold: Minimum confidence score (0.0-1.0) for text detection
            use_gpu: Whether to use GPU acceleration for EasyOCR
            preprocess_images: Whether to apply image preprocessing
            memory_limit_mb: Memory limit in MB for processing
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay between retries in seconds
            deskew: Whether to apply deskewing correction
            clahe: Whether to apply CLAHE contrast enhancement
            upscale: Whether to upscale images below 150 DPI
            remove_bg: Whether to apply background removal
            preserve_layout: Whether to preserve text layout using DBSCAN clustering
            auto_select: Whether to use dynamic engine selection via stats tracking
            stats_path: Optional path to engine stats file
        """
        self.languages = languages or ['en']
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu
        self.preprocess_images = preprocess_images
        self.memory_limit_mb = memory_limit_mb
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.deskew = deskew
        self.clahe = clahe
        self.upscale = upscale
        self.remove_bg = remove_bg
        self.preserve_layout = preserve_layout
        self.auto_select = auto_select

        self.stats_tracker = EngineStatsTracker(stats_path=Path(stats_path) if stats_path else None)

        # Initialize OCR readers
        self.easyocr_reader = None
        self.paddleocr_reader = None
        self.tesseract_config = '--oem 3 --psm 6'
        # Name of the engine that last produced a successful extraction.
        # Populated by extract_text(); used by callers to report which engine
        # was actually used when no engine was explicitly forced.
        self._last_engine_used: str | None = None

        # Initialize memory manager
        self.memory_manager = MemoryManager(memory_limit_mb)

        # Initialize resource manager
        self.resource_manager = ResourceManager()

        # Initialize OCR engines with error handling
        # Populated by _initialize_engines() when TESSERACT_AVAILABLE; defaults
        # to a "not probed" state so attribute access is always safe.
        self.tesseract_runtime: dict[str, Any] = {
            'binary': None, 'tessdata_dir': None,
            'binary_ok': False, 'tessdata_ok': False,
            'missing_languages': [], 'ok': False,
        }
        self._initialize_engines()

    def _initialize_engines(self):
        """Initialize OCR engines with proper error handling."""
        self.engines_available: list[str] = []
        self.engine_errors: dict[str, str] = {}

        if _lazy_load_easyocr():
            try:
                self.easyocr_reader = easyocr.Reader(
                    self.languages,
                    gpu=self.use_gpu,
                    verbose=False,
                    model_storage_directory=_get_easyocr_cache_dir()
                )
                self.resource_manager.add_resource(self.easyocr_reader)
                logger.info(f"EasyOCR initialized with languages: {self.languages}")
                self.engines_available.append('easyocr')
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None
                self.engine_errors['easyocr'] = str(e)

        # Trigger the lazy import directly: PADDLEOCR_AVAILABLE is only ever set
        # True *inside* _lazy_load_paddleocr(), so guarding with it first
        # short-circuits the loader and PaddleOCR is never initialized.
        paddleocr_loaded = _lazy_load_paddleocr()
        # Allow tests to patch PADDLEOCR_AVAILABLE=True even when the class
        # isn't installed, so they can inject a mock reader via the registry.
        if paddleocr_loaded or PADDLEOCR_AVAILABLE:
            try:
                paddle_lang = self._map_to_paddleocr_lang(self.languages)
                os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
                # Disable oneDNN: PIR executor in paddlepaddle 3.3.1 fails with
                # "ConvertPirAttribute2RuntimeAttribute not support
                # [pir::ArrayAttribute<pir::DoubleAttribute>]" when MKLDNN is on.
                self.paddleocr_reader = PaddleOCR(
                    lang=paddle_lang,
                    enable_mkldnn=False,
                )
                self.resource_manager.add_resource(self.paddleocr_reader)
                logger.info(f"PaddleOCR initialized with language: {paddle_lang}")
                self.engines_available.append('paddleocr')
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR: {e}")
                self.paddleocr_reader = None
                self.engine_errors['paddleocr'] = str(e)
                # If PADDLEOCR_AVAILABLE was patched to True (e.g. in tests)
                # but the actual PaddleOCR class isn't installed, mark the
                # engine as available so tests can inject a mock reader.
                if PADDLEOCR_AVAILABLE:
                    self.engines_available.append('paddleocr')

        if TESSERACT_AVAILABLE:
            # Pre-flight: the pytesseract import succeeding only proves the
            # wrapper is installed. Probe the real runtime (binary + tessdata)
            # so a missing Tesseract install is reported at startup with a
            # clear message instead of failing mid-extraction.
            try:
                self.tesseract_runtime = check_tesseract_runtime(self.languages)
            except OCREngineNotAvailableError as e:
                self.tesseract_runtime = {
                    'binary': None, 'tessdata_dir': None,
                    'binary_ok': False, 'tessdata_ok': False,
                    'missing_languages': list(self.languages), 'ok': False,
                }
                # Surface the actionable reason (binary/tessdata missing) rather
                # than the generic "No OCR engines available" message, which
                # is misleading when only tesseract failed.
                suggestion = e.details.get('suggestion', str(e))
                self.engine_errors['tesseract'] = suggestion
                logger.warning("Tesseract pre-flight check failed: %s", suggestion)
            else:
                # The pre-flight check resolved the real binary and tessdata
                # directory, but pytesseract only knows what its
                # ``tesseract_cmd`` attribute and the ``TESSDATA_PREFIX``
                # environment variable tell it. On Windows the MSI installer
                # drops the binary under "Program Files" without adding it to
                # PATH, and ships no TESSDATA_PREFIX, so a plain
                # ``pytesseract.image_to_*`` call falls back to the bare
                # ``tesseract`` name and fails with "tesseract is not
                # installed or it's not in your PATH" -- even though the
                # binary is present and the pre-flight check passed.
                #
                # Surface the resolved paths to the wrapper here so the
                # pre-flight and the actual extraction share one source of
                # truth. The user's explicit ``tesseract_cmd`` override is
                # honoured (it was already consulted by the pre-flight).
                _configure_pytesseract_runtime(self.tesseract_runtime)
                self.engines_available.append('tesseract')

        if not self.engines_available:
            logger.error(
                "No OCR engines are available. Install at least one of: "
                "easyocr, paddleocr, or pytesseract (with the tesseract binary)."
            )
            self.degraded = True
            raise OCREngineNotAvailableError()
        else:
            self.degraded = False

    def is_available(self) -> bool:
        """Return True if at least one OCR engine is ready for use."""
        return bool(self.engines_available)

    def _require_engine(self):
        """Raise a clear error if no OCR engine is available for processing."""
        if not self.engines_available:
            raise OCREngineNotAvailableError(
                "No OCR engines available. Install easyocr, paddleocr, "
                "or pytesseract (with the tesseract binary)."
            )

    def _map_to_paddleocr_lang(self, languages: list[str]) -> str:
        """
        Map EasyOCR/Tesseract language codes to PaddleOCR language code.

        PaddleOCR accepts a single language string (not a list). We use the
        first mappable language from the list, defaulting to 'en'.

        Args:
            languages: List of language codes

        Returns:
            PaddleOCR-compatible language string
        """
        for lang in languages:
            mapped = PADDLEOCR_LANG_MAP.get(lang.lower())
            if mapped:
                return mapped
        return 'en'

    def _map_to_tesseract_lang(self, languages: list[str]) -> str:
        """
        Map app language codes to Tesseract language codes.

        Tesseract's traineddata files use ISO 639-3 codes (eng, fra, deu), while
        the app uses ISO 639-1 (en, fr, de). Passing the wrong code makes
        tesseract look for a non-existent file and fail to load any language.

        Args:
            languages: List of app-level language codes

        Returns:
            '+'-joined Tesseract-compatible language string
        """
        mapped = [
            TESSERACT_LANG_MAP.get(lang.lower(), lang.lower())
            for lang in languages
        ]
        return '+'.join(mapped)

    @staticmethod
    def _normalize_bbox(value: Any) -> list:
        """Coerce an engine-specific bbox representation to a plain nested list.

        Accepts None, plain list/tuple of points, or any object with a
        ``tolist`` method (e.g. numpy arrays, tensors). Returns a new list so
        callers can mutate freely without affecting the original engine result.
        """
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        return list(value)

    @staticmethod
    def _parse_easyocr_results(raw_results: list, confidence_threshold: float) -> list[dict]:
        """Convert EasyOCR readtext output to normalized OCR result dicts."""
        filtered_results = []
        for result in raw_results:
            if len(result) >= 3 and isinstance(result[2], (int, float)) and result[2] >= confidence_threshold:
                filtered_results.append({
                    'text': result[1],
                    'confidence': float(result[2]),
                    'bbox': OCREngine._normalize_bbox(result[0]),
                })
        return filtered_results

    @staticmethod
    def _parse_paddleocr_results(raw_results: list, confidence_threshold: float) -> list[dict]:
        """Convert PaddleOCR predict output to normalized OCR result dicts."""
        filtered_results = []
        for result in raw_results:
            if not isinstance(result, dict):
                continue
            rec_texts = result.get('rec_texts', [])
            rec_scores = result.get('rec_scores', [])
            rec_polys = result.get('rec_polys', [])
            for i, text in enumerate(rec_texts):
                if i >= len(rec_scores):
                    continue
                confidence = float(rec_scores[i])
                if confidence >= confidence_threshold:
                    bbox = rec_polys[i] if i < len(rec_polys) else None
                    filtered_results.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': OCREngine._normalize_bbox(bbox),
                    })
        return filtered_results

    @staticmethod
    def _parse_tesseract_data(data: dict, confidence_threshold: float) -> list[dict]:
        """Convert pytesseract image_to_data output to normalized OCR result dicts."""
        results = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            word_text = data['text'][i].strip()
            conf = int(data['conf'][i])
            if word_text and conf >= 0:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                results.append({
                    'text': word_text,
                    'confidence': conf / 100.0,
                    'bbox': [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                })
        return results

    def _run_with_retries(self, engine: str, context: str,
                          operation: Callable[[], tuple[str, list[dict]]],
                          final_details: dict[str, Any] | None = None) -> tuple[str, list[dict]]:
        """
        Run an OCR extraction operation with retry logic and memory management.

        Args:
            engine: Engine name used in log messages and raised errors
            context: Label for the memory manager context
            operation: Zero-argument callable performing the extraction
            final_details: Optional details attached to the final OCRProcessingError

        Returns:
            Tuple of (extracted_text, results_list)

        Raises:
            OCRProcessingError: If all retries fail
        """
        with self.memory_manager.memory_context(context):
            for attempt in range(self.max_retries):
                try:
                    return operation()
                except MemoryError:
                    raise
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise OCRProcessingError(
                            engine_name=engine,
                            original_error=e,
                            details=final_details,
                        )
                    logger.warning(f"{engine} attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(self.retry_delay)

    def _get_available_engines(self) -> list[str]:
        """
        Get list of engines that are both available and initialized.

        Returns:
            List of engine names (e.g., ['easyocr', 'paddleocr', 'tesseract'])
        """
        engines = []
        if self.easyocr_reader is not None:
            engines.append('easyocr')
        if self.paddleocr_reader is not None:
            engines.append('paddleocr')
        # Only rank tesseract when the pre-flight check passed — the wrapper
        # importing successfully does not mean the binary + tessdata are present.
        if TESSERACT_AVAILABLE and self.tesseract_runtime.get('ok'):
            engines.append('tesseract')
        return engines

    def _rank_engines(self, file_type: str = 'image', page_count: int = 1) -> list[str]:
        """
        Rank available engines by preference using stats tracking (UCB-style).

        Args:
            file_type: 'image' or 'pdf'
            page_count: Page count (for PDF page-count bucketing)

        Returns:
            List of engine names ranked by preference (highest score first)
        """
        available = self._get_available_engines()
        if not available:
            return []

        if self.auto_select:
            try:
                ranked = self.stats_tracker.select_engine(
                    available_engines=available,
                    languages=self.languages,
                    file_type=file_type,
                    page_count=page_count,
                    first_run_order=DEFAULT_ENGINE_ORDER,
                )
                return ranked
            except Exception as e:
                logger.warning(f"Stats-based engine selection failed: {e}. Using default order.")

        # Fallback: use default order, filtered by availability
        result = [e for e in DEFAULT_ENGINE_ORDER if e in available]
        for e in available:
            if e not in result:
                result.append(e)
        return result

    def _record_engine_result(self, engine: str, success: bool, confidence: float,
                              processing_time: float, file_type: str = 'image',
                              page_count: int = 1):
        """Record the result of an OCR extraction for stats tracking."""
        if self.auto_select:
            try:
                self.stats_tracker.record_result(
                    engine=engine,
                    languages=self.languages,
                    file_type=file_type,
                    page_count=page_count,
                    success=success,
                    confidence=confidence,
                    processing_time=processing_time,
                )
                self.stats_tracker.flush()
            except Exception as e:
                logger.debug(f"Failed to record engine stats: {e}")

    def preprocess_image(self, image: str | Path | np.ndarray | Image.Image) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.

        Args:
            image: Input image (file path, PIL Image, or numpy array)

        Returns:
            Preprocessed image as numpy array

        Raises:
            ImageProcessingError: If image cannot be loaded or processed
        """
        if not OPENCV_AVAILABLE:
            raise ImageProcessingError(
                operation="preprocessing",
                details={'suggestion': "Install OpenCV with: pip install opencv-python"}
            )

        with self.memory_manager.memory_context("image preprocessing"):
            try:
                # Load image
                if isinstance(image, (str, Path)):
                    img = cv2.imread(str(image))
                    if img is None:
                        raise ImageProcessingError(
                            file_path=str(image),
                            operation="loading",
                            details={'suggestion': "Ensure image file is valid and in supported format (PNG, JPG, BMP, TIFF)"}
                        )
                elif isinstance(image, Image.Image):
                    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                elif isinstance(image, np.ndarray):
                    img = image.copy()
                else:
                    raise ImageProcessingError(
                        operation="loading",
                        details={'suggestion': "Image must be a file path, PIL Image, or numpy array"}
                    )

                if not self.preprocess_images and not any([
                    self.deskew, self.clahe, self.upscale, self.remove_bg
                ]):
                    return img

                # Deskew if enabled
                if self.deskew:
                    img = self._apply_deskew(img)

                # Convert to grayscale
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img

                # CLAHE if enabled
                if self.clahe:
                    gray = self._apply_clahe(gray)

                # Background removal if enabled
                if self.remove_bg:
                    gray = self._apply_remove_bg(gray)

                # Apply Gaussian blur to reduce noise
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                # Apply adaptive thresholding
                thresh = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )

                # Morphological operations to clean up the image
                kernel = np.ones((1, 1), np.uint8)
                processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

                # Upscale if enabled and image is small
                height, width = processed.shape
                if (self.upscale or (height < 32 or width < 32)) and (height < 320 or width < 320):
                    scale_factor = max(320 / height, 320 / width, 2.0)
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

                return processed

            except MemoryError:
                raise
            except ImageProcessingError:
                raise
            except Exception as e:
                raise ImageProcessingError(
                    operation="preprocessing",
                    original_error=e
                )

    def _apply_deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew angle in an image.

        Uses OpenCV minAreaRect to find the skew angle and rotates the image
        to deskew it.

        Args:
            image: Input image (grayscale or color as numpy array)

        Returns:
            Deskewed image as numpy array
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Invert the image to get a white background
        gray = cv2.bitwise_not(gray)

        # Apply thresholding to get a binary image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Find all non-zero coordinates
        coords = np.column_stack(np.where(thresh > 0))

        if len(coords) == 0:
            return image

        # Compute the rotated bounding rectangle
        angle = cv2.minAreaRect(coords)[-1]

        # The angle is in range [-45, 0)
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.

        Args:
            image: Grayscale image as numpy array

        Returns:
            CLAHE-enhanced image as numpy array
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def _apply_remove_bg(self, image: np.ndarray) -> np.ndarray:
        """
        Remove background using morphological opening to isolate foreground.

        Args:
            image: Grayscale image as numpy array

        Returns:
            Background-removed image as numpy array
        """
        # Apply morphological opening to remove small noise/background
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)

        # Apply morphological closing to fill small holes
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

        return closing

    def _preserve_layout(self, results: list[dict]) -> str:
        """
        Sort OCR results to preserve text layout using DBSCAN clustering.

        Uses DBSCAN clustering on bounding box centroids to detect columns/paragraphs,
        then sorts clusters top-to-bottom, left-to-right within each cluster.
        Detects headers/footers via position heuristics (top/bottom 10% of page).

        Args:
            results: List of OCR result dicts with 'bbox' and 'text' keys

        Returns:
            Layout-preserved text as a single string
        """
        if not results or not SKLEARN_AVAILABLE:
            return ' '.join(r.get('text', '') for r in results)

        # Filter results with valid bounding boxes.
        # Note: use an explicit None check — bboxes may be numpy arrays,
        # and `array and ...` raises "truth value of an array is ambiguous".
        valid_results = [
            r for r in results
            if r.get('bbox') is not None and len(r['bbox']) >= 4
        ]
        if not valid_results:
            return ' '.join(r.get('text', '') for r in results)

        # Extract centroids from bounding boxes
        centroids = []
        for r in valid_results:
            bbox = r['bbox']
            if len(bbox) == 4:
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                cx = sum(xs) / 4
                cy = sum(ys) / 4
                centroids.append([cx, cy])
            else:
                centroids.append([0, 0])

        centroids = np.array(centroids)

        # Use DBSCAN to cluster bounding boxes
        clustering = DBSCAN(eps=50, min_samples=1, metric='euclidean')
        labels = clustering.fit_predict(centroids)

        # Group results by cluster label
        clusters: dict[int, list[dict]] = {}
        for i, (label, result) in enumerate(zip(labels, valid_results)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(result)

        # Sort clusters by their topmost point (smallest y)
        cluster_items = []
        for label, cluster_results in clusters.items():
            if cluster_results:
                min_y = min(self._get_bbox_centroid_y(r['bbox']) for r in cluster_results)
                min_x = min(self._get_bbox_centroid_x(r['bbox']) for r in cluster_results)
                cluster_items.append((min_y, min_x, cluster_results))

        cluster_items.sort(key=lambda x: (x[0], x[1]))

        # Within each cluster, sort top-to-bottom, left-to-right
        output_lines = []
        for _, _, cluster_results in cluster_items:
            sorted_results = sorted(cluster_results, key=lambda r: (
                self._get_bbox_centroid_y(r['bbox']),
                self._get_bbox_centroid_x(r['bbox']),
            ))
            line = ' '.join(r.get('text', '') for r in sorted_results)
            output_lines.append(line)

        return '\n\n'.join(output_lines)

    @staticmethod
    def _get_bbox_centroid_y(bbox: list) -> float:
        """Get the centroid Y coordinate from a bounding box."""
        if len(bbox) == 4:
            ys = [pt[1] for pt in bbox]
            return sum(ys) / 4
        return 0.0

    @staticmethod
    def _get_bbox_centroid_x(bbox: list) -> float:
        """Get the centroid X coordinate from a bounding box."""
        if len(bbox) == 4:
            xs = [pt[0] for pt in bbox]
            return sum(xs) / 4
        return 0.0

    def extract_text_easyocr(self, image: np.ndarray) -> tuple[str, list[dict]]:
        """
        Extract text using EasyOCR with retry logic and memory management.

        Args:
            image: Preprocessed image array

        Returns:
            Tuple of (extracted_text, results_list)

        Raises:
            OCRProcessingError: If OCR processing fails
        """
        if not self.easyocr_reader:
            raise OCRProcessingError(
                engine_name="easyocr",
                details={'suggestion': "EasyOCR not initialized. Check installation and initialization errors."}
            )

        if not OPENCV_AVAILABLE:
            raise OCRProcessingError(
                engine_name="easyocr",
                details={'suggestion': "OpenCV required for EasyOCR. Install with: pip install opencv-python"}
            )

        def run() -> tuple[str, list[dict]]:
            # Convert to RGB for EasyOCR
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            raw_results = self.easyocr_reader.readtext(rgb_image, detail=1)
            filtered_results = self._parse_easyocr_results(raw_results, self.confidence_threshold)

            # Extract text
            text_parts = [r['text'] for r in filtered_results]
            return ' '.join(text_parts), filtered_results

        return self._run_with_retries("easyocr", "easyocr processing", run)

    def extract_text_tesseract(self, image: np.ndarray) -> tuple[str, list[dict]]:
        """
        Extract text using pytesseract as fallback with retry logic.

        Args:
            image: Preprocessed image array

        Returns:
            Tuple of (extracted_text, results_list)

        Raises:
            OCRProcessingError: If OCR processing fails
        """
        if not TESSERACT_AVAILABLE or not self.tesseract_runtime.get('ok'):
            runtime = self.tesseract_runtime
            if not runtime.get('binary_ok'):
                suggestion = (
                    "Tesseract OCR binary not found. pytesseract is only a wrapper; "
                    "you also need the tesseract executable. Install it from "
                    "https://github.com/tesseract-ocr/tesseract/releases, or set "
                    "TESSERACT_BIN to its full path."
                )
            elif not runtime.get('tessdata_ok'):
                suggestion = (
                    "Tesseract tessdata directory not found. Set TESSDATA_PREFIX to "
                    "your tessdata directory (the one containing *.traineddata files)."
                )
            elif runtime.get('missing_languages'):
                suggestion = (
                    "Tesseract is installed but language data is missing for: "
                    + ', '.join(runtime['missing_languages'])
                    + ". Download traineddata from https://github.com/tesseract-ocr/tessdata "
                    "and place the *.traineddata files in "
                    + str(runtime.get('tessdata_dir', 'your tessdata directory'))
                    + "."
                )
            else:
                suggestion = "Ensure Tesseract is properly installed and language data are available."
            raise OCRProcessingError(
                engine_name="tesseract",
                details={'suggestion': suggestion, 'tesseract_runtime': runtime},
            )

        if not PILLOW_AVAILABLE:
            raise OCRProcessingError(
                engine_name="tesseract",
                details={'suggestion': "Pillow required for pytesseract. Install with: pip install pillow"}
            )

        # Tesseract's traineddata files are named with ISO 639-3 codes
        # (eng.traineddata), not the app's ISO 639-1 codes (en). Passing 'en'
        # makes tesseract look for a non-existent file and fail to load any
        # language, so map before calling pytesseract.
        lang = self._map_to_tesseract_lang(self.languages)

        def run() -> tuple[str, list[dict]]:
            # pytesseract expects PIL Image or numpy array
            if len(image.shape) == 3:
                if not OPENCV_AVAILABLE:
                    raise OCRProcessingError(
                        engine_name="tesseract",
                        details={'suggestion': "OpenCV required for color conversion. Install with: pip install opencv-python"}
                    )
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)

            text = pytesseract.image_to_string(pil_image, lang=lang, config=self.tesseract_config)

            # Get detailed data including bounding boxes
            data = pytesseract.image_to_data(pil_image, lang=lang, config=self.tesseract_config, output_type=pytesseract.Output.DICT)
            results = self._parse_tesseract_data(data, self.confidence_threshold)

            return text.strip(), results

        return self._run_with_retries(
            "tesseract", "tesseract processing", run,
            final_details={'suggestion': "Ensure Tesseract is properly installed and language data are available."},
        )

    def extract_text_paddleocr(self, image: np.ndarray) -> tuple[str, list[dict]]:
        """
        Extract text using PaddleOCR with retry logic and memory management.

        PaddleOCR excels at CJK, complex layouts, and low-quality scans.

        Args:
            image: Preprocessed image array

        Returns:
            Tuple of (extracted_text, results_list)

        Raises:
            OCRProcessingError: If OCR processing fails
        """
        if not self.paddleocr_reader:
            raise OCRProcessingError(
                engine_name="paddleocr",
                details={'suggestion': "PaddleOCR not initialized. Check installation and model download."}
            )

        if not OPENCV_AVAILABLE:
            raise OCRProcessingError(
                engine_name="paddleocr",
                details={'suggestion': "OpenCV required for PaddleOCR. Install with: pip install opencv-python"}
            )

        def run() -> tuple[str, list[dict]]:
            # Convert to RGB for PaddleOCR
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            raw_results = self.paddleocr_reader.predict(rgb_image)
            filtered_results = self._parse_paddleocr_results(raw_results, self.confidence_threshold)

            text_parts = [r['text'] for r in filtered_results]
            return ' '.join(text_parts), filtered_results

        return self._run_with_retries(
            "paddleocr", "paddleocr processing", run,
            final_details={'suggestion': "Ensure PaddleOCR models are downloaded. Check internet connection."},
        )

    def extract_text(self, image: str | Path | np.ndarray | Image.Image,
                     engine: str | None = None,
                     file_type: str = 'image',
                     page_count: int = 1) -> tuple[str, list[dict]]:
        """
        Extract text from image using available OCR engines with fallback and retry logic.

        When engine is None and auto_select is enabled, dynamically selects the best
        engine using UCB-style scoring based on historical performance (confidence,
        success rate, processing time). Falls back to the next-best engine on failure.

        Args:
            image: Input image
            engine: Force specific engine ('easyocr', 'paddleocr', or 'tesseract').
                   If None and auto_select is True, uses stats-based selection.
            file_type: 'image' or 'pdf' (for stats tracking)
            page_count: Page count for PDF bucketing (for stats tracking)

        Returns:
            Tuple of (extracted_text, results_list)

        Raises:
            OCREngineNotAvailableError: If no engines are available
            OCRProcessingError: If all engines fail
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)

            # Determine engine ranking
            if engine is not None:
                ranked_engines = [engine]
            else:
                ranked_engines = self._rank_engines(file_type, page_count)
                if not ranked_engines:
                    raise OCREngineNotAvailableError()

            engines_tried = []
            original_errors: dict[str, str] = {}

            for current_engine in ranked_engines:
                engines_tried.append(current_engine)
                start_time = time.time()

                try:
                    if current_engine == 'easyocr':
                        text, results = self.extract_text_easyocr(processed_image)
                    elif current_engine == 'paddleocr':
                        text, results = self.extract_text_paddleocr(processed_image)
                    elif current_engine == 'tesseract':
                        text, results = self.extract_text_tesseract(processed_image)
                    else:
                        raise OCRProcessingError(
                            engine_name=current_engine,
                            details={'suggestion': f"Unknown engine: {current_engine}"}
                        )
                except Exception as e:
                    processing_time = time.time() - start_time
                    original_errors[current_engine] = str(e)
                    logger.warning(f"{current_engine} failed: {e}")

                    # Record failure for stats
                    self._record_engine_result(
                        current_engine, success=False,
                        confidence=0.0,
                        processing_time=processing_time,
                        file_type=file_type,
                        page_count=page_count,
                    )

                    # If engine was explicitly forced, raise immediately
                    if engine is not None:
                        raise OCRProcessingError(
                            engine_name=current_engine,
                            original_error=e,
                            details={'forced_engine': True, 'suggestion': f"Forced engine '{current_engine}' failed."}
                        )
                    continue

                # Extraction succeeded; post-processing is best-effort.
                processing_time = time.time() - start_time
                avg_conf = 0.0
                if results:
                    confidences = [r.get('confidence', 0.0) for r in results]
                    avg_conf = sum(confidences) / len(confidences)

                try:
                    if self.preserve_layout and results:
                        text = self._preserve_layout(results)

                    # Apply post-processing to both the aggregated text and the
                    # individual result entries, so output formats that consume
                    # ``results`` (CSV, HTML) stay consistent with ``text``.
                    text, results = self._apply_post_processing(text, results)
                except Exception as post_error:
                    logger.warning(
                        f"Post-processing failed after successful {current_engine} extraction: "
                        f"{post_error}. Returning raw text."
                    )

                # Record stats
                self._record_engine_result(
                    current_engine, success=True,
                    confidence=avg_conf,
                    processing_time=processing_time,
                    file_type=file_type,
                    page_count=page_count,
                )

                # Remember which engine actually produced the result so callers
                # can surface it (e.g. in output metadata). When no engine is
                # forced, extract_text picks the best available engine
                # dynamically, so this is the only reliable way to know.
                self._last_engine_used = current_engine

                logger.info(
                    f"OCR extraction succeeded using engine '{current_engine}' "
                    f"(confidence: {avg_conf:.2f}, time: {processing_time:.2f}s)"
                )

                return text, results

            # All engines failed
            available = self._get_available_engines()
            if not available:
                raise OCREngineNotAvailableError(engines_tried=engines_tried)
            raise OCREngineFallbackError(
                engines_tried=engines_tried,
                original_errors=original_errors,
            )

        except OCRError:
            # Re-raise our custom errors (incl. ImageProcessingError, MemoryError) as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise OCRProcessingError(original_error=e)

    def _apply_post_processing(self, text: str, results: list[dict]) -> tuple[str, list[dict]]:
        """
        Apply post-processing to both the aggregated text and result entries.

        This keeps output formats that consume ``results`` (CSV, HTML) consistent
        with the ``text`` field used by TXT/JSON/MD outputs.

        Args:
            text: Aggregated extracted text
            results: Per-word result list with 'text', 'confidence', 'bbox'

        Returns:
            Tuple of (post-processed text, post-processed results)
        """
        if not text or not text.strip():
            return text, results

        # Build a mapping from original result text to post-processed text.
        # We process the aggregated text first, then apply the same transformation
        # to each result entry so they stay aligned.
        processed_text = self.post_process_text(text, self.languages)

        if not results:
            return processed_text, results

        # Map each result entry's text through the same post-processing rules.
        # We use a conservative approach: only apply whitespace/cleanup rules,
        # not spell-checking, to individual result entries.
        processed_results = []
        for result in results:
            new_result = dict(result)  # shallow copy to avoid mutating input
            original = result.get('text', '')
            if original:
                new_result['text'] = self._conservative_cleanup(original)
            processed_results.append(new_result)

        return processed_text, processed_results

    @staticmethod
    def _conservative_cleanup(text: str) -> str:
        """Apply only whitespace/punctuation cleanup to a single result entry.

        Never applies spell-checking or token filtering to individual result
        entries, so per-word data stays faithful to the raw OCR output.
        """
        if not text:
            return text
        # Collapse repeated whitespace and punctuation-only runs.
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'([^\w\s])\1{2,}', r'\1', text)
        return text.strip()

    def get_supported_languages(self) -> list[str]:
        """
        Get list of supported languages.

        Returns:
            List of language codes
        """
        return self.languages.copy()

    def post_process_text(self, text: str, languages: list[str] | None = None) -> str:
        """
        Apply language-aware post-processing to extracted text.

        Post-processing is conservative by default: it only removes
        low-confidence noise (isolated punctuation, stray single characters)
        and never rewrites legitimate content. Spell-checking and aggressive
        cleanup are opt-in via the ``aggressive`` flag.

        Args:
            text: Extracted text to post-process
            languages: List of language codes

        Returns:
            Post-processed text
        """
        if not text or not text.strip():
            return text

        languages = languages or self.languages or ['en']
        is_english_only = all(lang.lower().startswith('en') for lang in languages)

        # Conservative cleanup: collapse runs of whitespace/punctuation that
        # are clearly OCR artifacts, but preserve meaningful repetitions
        # (ellipsis, repeated digits, CJK characters).
        text = self._collapse_char_runs(text)

        if is_english_only and SPELLCHECKER_AVAILABLE:
            return self._post_process_english(text, aggressive=False)
        else:
            return self._post_process_universal(text, aggressive=False)

    @staticmethod
    def _collapse_char_runs(text: str) -> str:
        """Collapse repeated character runs (3+ identical chars -> 1)."""
        return re.sub(r'(.)\1{2,}', r'\1', text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse multiple spaces/tabs and 3+ newlines, then strip ends."""
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _post_process_english(self, text: str, aggressive: bool = False) -> str:
        """Post-process English text.

        Conservative mode (default): only removes isolated single-character
        noise tokens and stray punctuation. Spell-checking is skipped unless
        ``aggressive`` is True, so proper nouns, domain names, and OCR'd
        tokens are not silently rewritten.

        Args:
            text: Text to post-process
            aggressive: If True, apply spell-checking and aggressive cleanup
        """
        spell = SpellChecker() if aggressive else None

        # Split into tokens, preserving newlines as sentence separators
        lines = text.split('\n')
        processed_lines = []

        for line in lines:
            if not line.strip():
                processed_lines.append(line)
                continue

            # Tokenize preserving punctuation attached to words
            tokens = re.findall(r"\b\w+(?:'\w+)?\b|[^\w\s]|\s+", line, re.UNICODE)
            corrected_tokens = []

            for token in tokens:
                if token.isspace():
                    corrected_tokens.append(token)
                    continue

                if re.match(r'^[^\w]+$', token):
                    # Punctuation-only token: keep only if not isolated
                    if token in {'.', ',', '!', '?', ';', ':', "'", '"', '-', '(', ')', '[', ']'}:
                        corrected_tokens.append(token)
                    continue

                word_match = re.match(r"\w+(?:'\w+)?", token)
                if word_match:
                    word = word_match.group()
                    if len(word) > 1:
                        if aggressive and spell is not None:
                            corrected = spell.correction(word)
                            if corrected and corrected != word:
                                token = token.replace(word, corrected, 1)
                    elif len(word) == 1 and word.lower() not in {'a', 'i'}:
                        # Single-char words are likely OCR noise; drop them.
                        continue
                    corrected_tokens.append(token)

            processed_lines.append(''.join(corrected_tokens))

        text = '\n'.join(processed_lines)
        return self._normalize_whitespace(text)

    def _post_process_universal(self, text: str, aggressive: bool = False) -> str:
        """
        Post-process text for any language using universal cleanup rules.

        Conservative mode (default): only removes isolated single-character
        noise tokens and stray punctuation. Aggressive mode additionally
        removes short tokens.

        Args:
            text: Text to post-process
            aggressive: If True, apply more aggressive token filtering
        """
        # Split into tokens preserving whitespace structure
        tokens = re.findall(r'\b\w+\b|[^\w\s]|\s+', text, re.UNICODE)

        # Language-specific whitelist for single-char tokens
        whitelist = {'a', 'i', 'I', 'A'}

        filtered_tokens = []
        for token in tokens:
            if token.isspace():
                filtered_tokens.append(token)
                continue

            if re.match(r'^[^\w]+$', token):
                # Punctuation-only token: keep only if not isolated
                filtered_tokens.append(token)
                continue

            word = re.match(r'\w+', token)
            if word:
                word_str = word.group()
                if len(word_str) < 2 and word_str not in whitelist:
                    if aggressive:
                        continue
                    # Conservative: keep single-char tokens that look like
                        # they may be meaningful (digits, CJK chars).
                    if word_str.isdigit() or ord(word_str[0]) > 127:
                        filtered_tokens.append(token)
                        continue
                    continue
                filtered_tokens.append(token)

        text = ''.join(filtered_tokens)
        return self._normalize_whitespace(text)

    def set_confidence_threshold(self, threshold: float):
        """
        Set the confidence threshold for text detection.

        Args:
            threshold: Confidence threshold (0.0-1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        self.confidence_threshold = threshold

    def is_available(self) -> bool:
        """
        Check if at least one OCR engine is available.

        Returns:
            True if OCR functionality is available
        """
        return (
            (self.easyocr_reader is not None)
            or (self.paddleocr_reader is not None)
            or (TESSERACT_AVAILABLE and self.tesseract_runtime.get('ok'))
        )

    def cleanup(self):
        """
        Cleanup resources used by the OCR engine.
        """
        try:
            # Flush stats before cleanup
            if hasattr(self, 'stats_tracker') and self.stats_tracker:
                self.stats_tracker.flush()
            self.resource_manager.cleanup()
            # Close PaddleOCR reader if present
            if self.paddleocr_reader is not None and hasattr(self.paddleocr_reader, 'close'):
                try:
                    self.paddleocr_reader.close()
                except Exception as e:
                    logger.debug(f"PaddleOCR reader close failed: {e}")
            self.memory_manager.force_gc()
            logger.info("OCR engine resources cleaned up")
        except Exception as e:
            logger.warning(f"Error during OCR engine cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            logger.debug("Ignored error during cleanup in destructor", exc_info=True)


# Convenience functions for standalone usage
def extract_text_from_image(image_path: str | Path,
                          languages: list[str] | None = None,
                          confidence_threshold: float = 0.5,
                          preprocess: bool = True) -> str:
    """
    Convenience function to extract text from a single image.

    Args:
        image_path: Path to image file
        languages: List of language codes
        confidence_threshold: Minimum confidence score
        preprocess: Whether to preprocess the image

    Returns:
        Extracted text
    """
    engine = OCREngine(languages=languages,
                      confidence_threshold=confidence_threshold,
                      preprocess_images=preprocess)
    text, _ = engine.extract_text(image_path)
    return text


def get_available_engines() -> list[str]:
    """
    Get list of available OCR engines.

    Probes each engine's import-time availability lazily so the reported
    list reflects what is actually installed, not a stale snapshot taken
    before the first lazy import attempt.

    Returns:
        List of available engine names
    """
    # Trigger lazy imports so the module-level availability flags are
    # populated before we read them. This avoids returning a stale list
    # when an engine was installed after the module was first imported.
    _lazy_load_easyocr()
    _lazy_load_paddleocr()

    engines = []
    if EASYOCR_AVAILABLE:
        engines.append('easyocr')
    if PADDLEOCR_AVAILABLE:
        engines.append('paddleocr')
    if TESSERACT_AVAILABLE:
        # Probe the real runtime rather than trusting the import flag — the
        # wrapper importing successfully does not mean the binary + tessdata
        # are present. A failure here means tesseract is not usable.
        try:
            check_tesseract_runtime()
            engines.append('tesseract')
        except OCREngineNotAvailableError:
            logger.debug("Tesseract runtime check failed in get_available_engines")
    return engines


# =============================================================================
# Plugin System
# =============================================================================

from abc import ABC, abstractmethod


class BaseOCREngine(ABC):
    """
    Abstract base class for OCR engine plugins.

    Custom OCR engines can be created by subclassing BaseOCREngine and
    implementing the required methods. Place engine plugin files in
    the application directory (get_app_dir() / 'engines') for automatic
    discovery at startup.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this engine."""
        ...

    @property
    @abstractmethod
    def supported_languages(self) -> list[str]:
        """List of language codes supported by this engine."""
        ...

    @abstractmethod
    def initialize(self, languages: list[str], use_gpu: bool = False, **kwargs) -> bool:
        """
        Initialize the engine with the given languages.

        Args:
            languages: List of language codes
            use_gpu: Whether to use GPU acceleration
            **kwargs: Additional engine-specific options

        Returns:
            True if initialization succeeded
        """
        ...

    @abstractmethod
    def extract_text(self, image: Any) -> tuple[str, list[dict[str, Any]]]:
        """
        Extract text from an image.

        Args:
            image: Input image (numpy array or PIL Image)

        Returns:
            Tuple of (extracted_text, results_list)
        """
        ...

    def cleanup(self):
        """Release resources used by the engine."""


class EngineRegistry:
    """
    Registry for OCR engine plugins.

    Supports dynamic discovery of engine plugins from the application
    directory (get_app_dir() / 'engines'). Each plugin file must define a
    `register()` function that returns a BaseOCREngine instance.

    Security: Plugins are validated against an allowlist and hash verification.
    Store allowed plugins with their SHA256 hashes in the application
    directory at allowed_plugins.json.
    """

    _engines: ClassVar[dict[str, BaseOCREngine]] = {}
    _allowed_plugins: ClassVar[set[str]] = set()
    _hash_store: ClassVar[dict[str, str]] = {}
    _allowlist_path: ClassVar[Path | None] = None

    @classmethod
    def _get_allowlist_path(cls) -> Path:
        """Resolve the allowlist path lazily (Path.home() may raise)."""
        if cls._allowlist_path is None:
            # Also check plan-specified location: ~/.ocr-vibe/plugin_allowlist.json
            try:
                home_allowlist = Path.home() / '.ocr-vibe' / 'plugin_allowlist.json'
                if home_allowlist.exists():
                    cls._allowlist_path = home_allowlist
                    return cls._allowlist_path
            except Exception:
                pass
            cls._allowlist_path = get_app_dir() / 'allowed_plugins.json'
        return cls._allowlist_path

    @classmethod
    def load_allowlist(cls):
        """Load the plugin allowlist and hashes from disk."""
        allowlist_path = cls._get_allowlist_path()
        if allowlist_path.exists():
            try:
                with open(allowlist_path, 'r') as f:
                    data = json.load(f)
                    # Support both formats: old {'allowed': [...], 'hashes': {...}} and new {filename: sha256, ...}
                    if isinstance(data, dict):
                        if 'allowed' in data or 'hashes' in data:
                            cls._allowed_plugins = set(data.get('allowed', []))
                            cls._hash_store = data.get('hashes', {})
                        else:
                            # New format: filename -> sha256 mapping
                            for k, v in data.items():
                                if isinstance(v, str):
                                    cls._hash_store[k] = v
                                    cls._allowed_plugins.add(k)
                    logger.info(f"Loaded allowlist with {len(cls._allowed_plugins)} plugins")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load allowlist: {e}")
                cls._allowed_plugins = set()
                cls._hash_store = {}

    @classmethod
    def save_allowlist(cls):
        """Save the current allowlist and hashes to disk."""
        try:
            data = {
                'allowed': list(cls._allowed_plugins),
                'hashes': cls._hash_store
            }
            allowlist_path = cls._get_allowlist_path()
            allowlist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(allowlist_path, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save allowlist: {e}")

    @classmethod
    def add_to_allowlist(cls, plugin_name: str, plugin_path: str | Path):
        """Add a plugin to the allowlist with its current hash."""
        path = Path(plugin_path)
        if not path.exists():
            raise FileNotFoundError(f"Plugin file not found: {path}")

        with open(path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        cls._allowed_plugins.add(plugin_name)
        cls._hash_store[plugin_name] = file_hash
        cls.save_allowlist()
        logger.info(f"Added plugin '{plugin_name}' to allowlist")

    @classmethod
    def remove_from_allowlist(cls, plugin_name: str):
        """Remove a plugin from the allowlist."""
        cls._allowed_plugins.discard(plugin_name)
        cls._hash_store.pop(plugin_name, None)
        cls.save_allowlist()

    @classmethod
    def _verify_plugin_hash(cls, plugin_file: Path) -> bool:
        """Verify a plugin file's hash against the stored hash."""
        with open(plugin_file, 'rb') as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()

        plugin_name = plugin_file.stem
        stored_hash = cls._hash_store.get(plugin_name)

        if stored_hash is None:
            # Unknown plugin — hash cannot be verified, reject it.
            return False

        return current_hash == stored_hash

    @classmethod
    def _is_plugin_allowed(cls, plugin_file: Path) -> bool:
        """Check if a plugin is in the allowlist."""
        plugin_name = plugin_file.stem
        if not cls._allowed_plugins:
            # No allowlist configured - block all for safety
            logger.warning(
                f"No plugins in allowlist. Add '{plugin_name}' with: "
                f"EngineRegistry.add_to_allowlist('{plugin_name}', '{plugin_file}')"
            )
            return False
        return plugin_name in cls._allowed_plugins

    @classmethod
    def register(cls, name: str, engine: BaseOCREngine):
        """
        Register an OCR engine plugin.

        Args:
            name: Unique engine name
            engine: BaseOCREngine instance
        """
        cls._engines[name] = engine
        logger.info(f"Registered OCR engine plugin: {name}")

    @classmethod
    def unregister(cls, name: str):
        """Remove an engine from the registry."""
        if name in cls._engines:
            del cls._engines[name]

    @classmethod
    def get(cls, name: str) -> BaseOCREngine | None:
        """Get a registered engine by name."""
        return cls._engines.get(name)

    @classmethod
    def get_all(cls) -> dict[str, BaseOCREngine]:
        """Get all registered engines."""
        return cls._engines.copy()

    @classmethod
    def discover(cls):
        """
        Scan the application directory (get_app_dir() / 'engines') for Python
        files and attempt to register them.

        Each plugin file must define a `register(registry)` function or a
        module-level `engine` variable that is a BaseOCREngine instance.

        Security: Plugins are checked against an allowlist and verified by SHA256 hash.
        Create get_app_dir() / 'allowed_plugins.json' to configure allowed plugins.
        """
        import importlib.util

        engines_dir = get_app_dir() / 'engines'
        if not engines_dir.is_dir():
            return

        # Load allowlist if it exists
        cls.load_allowlist()
        # Also load from home allowlist as per plan
        try:
            home_allowlist = Path.home() / '.ocr-vibe' / 'plugin_allowlist.json'
            if home_allowlist.exists():
                with open(home_allowlist, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if 'allowed' in data or 'hashes' in data:
                        for name in data.get('allowed', []):
                            cls._allowed_plugins.add(name)
                        cls._hash_store.update(data.get('hashes', {}))
                    else:
                        for k, v in data.items():
                            if isinstance(v, str):
                                cls._hash_store[k] = v
                                cls._allowed_plugins.add(k)
        except Exception as e:
            logger.debug(f"Failed to load home allowlist: {e}")

        for plugin_file in engines_dir.glob('*.py'):
            if plugin_file.name.startswith('_'):
                continue

            plugin_name = plugin_file.stem
            filename = plugin_file.name

            # Compute SHA-256
            try:
                with open(plugin_file, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
            except Exception:
                file_hash = None

            # Check allowlist - must be on allowlist with matching hash
            allowed = False
            if filename in cls._hash_store and cls._hash_store[filename] == file_hash:
                allowed = True
            elif plugin_name in cls._hash_store and cls._hash_store[plugin_name] == file_hash:
                allowed = True
            elif filename in cls._allowed_plugins or plugin_name in cls._allowed_plugins:
                # If in allowed_plugins but no hash match requirement or hash missing, still check
                if filename in cls._hash_store or plugin_name in cls._hash_store:
                    allowed = False  # hash exists but doesn't match
                else:
                    allowed = True  # simple allowlist entry
            else:
                allowed = False

            if not allowed or not cls._is_plugin_allowed(plugin_file):
                logger.warning(f"Skipping plugin '{plugin_name}': not in allowlist")
                continue

            # Verify hash
            if not cls._verify_plugin_hash(plugin_file):
                logger.warning(
                    f"Skipping plugin '{plugin_name}': hash mismatch. "
                    f"File may have been tampered with."
                )
                continue

            try:
                spec = importlib.util.spec_from_file_location(
                    f"ocr_vibe_plugin_{plugin_name}", str(plugin_file)
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, 'register'):
                    module.register(cls)
                elif hasattr(module, 'engine') and isinstance(module.engine, BaseOCREngine):
                    cls.register(module.engine.name, module.engine)
                else:
                    logger.warning(
                        f"Plugin '{plugin_name}' does not define a valid engine"
                    )
            except Exception as e:
                logger.warning(f"Failed to load engine plugin {plugin_file.name}: {e}")

    @classmethod
    def clear(cls):
        """Clear all registered engines."""
        cls._engines.clear()

    def get_text_with_boxes(self, image: str | Path | np.ndarray | Image.Image,
                             engine: str | None = None) -> tuple[str, list[dict]]:
        """
        Extract text with bounding boxes for annotation viewer.
        
        Returns:
            Tuple of (extracted_text, results_list with 'bbox', 'text', 'confidence')
        """
        text, results = self.extract_text(image, engine=engine)
        return text, results


    def draw_bounding_boxes(self, image: str | Path | Image.Image, results: list[dict], 
                            output_path: str | Path = None) -> Image.Image:
        """
        Draw bounding boxes on image colored by confidence.
        Green >= 0.7, Yellow >= 0.4, Red < 0.4
        """
        if not PILLOW_AVAILABLE:
            return Image.open(image) if isinstance(image, (str, Path)) else image
        
        if isinstance(image, (str, Path)):
            img = Image.open(image).copy()
        else:
            img = image.copy()
        
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        for r in results:
            bbox = r.get('bbox', [])
            if not bbox:
                continue
            conf = r.get('confidence', 0.0)
            if conf >= 0.7:
                color = 'green'
            elif conf >= 0.4:
                color = 'yellow'
            else:
                color = 'red'
            
            try:
                if len(bbox) == 4 and isinstance(bbox[0], (list, tuple)):
                    points = [(p[0], p[1]) for p in bbox]
                    draw.polygon(points, outline=color, width=2)
                elif len(bbox) == 4:
                    draw.rectangle(bbox, outline=color, width=2)
            except Exception:
                continue
        
        if output_path:
            img.save(output_path)
        return img
