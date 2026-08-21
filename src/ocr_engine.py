#!/usr/bin/env python3
"""
OCR Engine Module

This module provides OCR functionality using EasyOCR as the primary engine
with pytesseract as a fallback. It includes image preprocessing capabilities
using OpenCV and Pillow, and supports multiple languages with confidence thresholds.
"""

import logging
import sys
import os
import time
from typing import List, Optional, Tuple, Union, Dict, Any
from pathlib import Path
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.error("NumPy is required but not installed. Install with: pip install numpy")

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    logging.error("Pillow is required for image processing. Install with: pip install pillow")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logging.error("OpenCV is required for image preprocessing. Install with: pip install opencv-python")

try:
    import re
    RE_AVAILABLE = True
except ImportError:
    RE_AVAILABLE = False

# OCR libraries
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.error("EasyOCR is required for OCR functionality. Install with: pip install easyocr")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.error("pytesseract is required for OCR functionality. Install with: pip install pytesseract")
    logging.error("Note: pytesseract also requires Tesseract OCR to be installed on your system.")
    logging.error("Download from: https://github.com/tesseract-ocr/tesseract")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR is not installed. Install with: pip install paddleocr")
    logging.warning("PaddleOCR excels at CJK, complex layouts, and low-quality scans.")

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    logging.warning("pyspellchecker not installed. English-only post-processing will be limited. Install with: pip install pyspellchecker")

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not installed. Layout preservation will not work. Install with: pip install scikit-learn")

# Import custom modules
from .exceptions import (
    OCRError, OCREngineNotAvailableError, OCREngineInitializationError,
    OCRProcessingError, ImageProcessingError, MemoryError, OCREngineFallbackError
)
from .memory_manager import MemoryManager, ResourceManager, memory_limited
from .stats_tracker import EngineStatsTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def _get_easyocr_cache_dir() -> str:
    """
    Get a writable cache directory for EasyOCR models.

    In a PyInstaller bundle, uses a directory next to the executable or APPDATA.
    Otherwise, uses the default EasyOCR cache location.
    """
    # Check if running in PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Running in a bundle
        if sys.platform == 'win32':
            # Use APPDATA on Windows
            cache_dir = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), 'OCR-VIBE', 'easyocr_models')
        else:
            # Use ~/.ocr-vibe on Unix
            cache_dir = os.path.join(os.path.expanduser('~'), '.ocr-vibe', 'easyocr_models')
    else:
        # Running in development, use default EasyOCR location
        cache_dir = os.path.join(os.path.expanduser('~'), '.EasyOCR', 'model')

    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_paddleocr_cache_dir() -> str:
    """
    Get a writable cache directory for PaddleOCR models.

    Uses a directory under the user's home config directory (~/.ocr-vibe/paddleocr_models)
    to ensure consistent caching and PyInstaller bundle compatibility.
    """
    if getattr(sys, 'frozen', False):
        if sys.platform == 'win32':
            cache_dir = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), 'OCR-VIBE', 'paddleocr_models')
        else:
            cache_dir = os.path.join(os.path.expanduser('~'), '.ocr-vibe', 'paddleocr_models')
    else:
        cache_dir = os.path.join(os.path.expanduser('~'), '.paddlex', 'official_models')

    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


PADDLEOCR_LANG_MAP: Dict[str, str] = {
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


class OCREngine:
    """
    OCR Engine class that provides text extraction from images using multiple OCR engines.

    Supports EasyOCR as primary engine with pytesseract fallback, image preprocessing,
    language support, and confidence thresholding.
    """

    def __init__(self,
                 languages: Optional[List[str]] = None,
                 confidence_threshold: float = 0.5,
                 use_gpu: bool = False,
                 preprocess_images: bool = True,
                 memory_limit_mb: Optional[int] = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 deskew: bool = False,
                 clahe: bool = False,
                 upscale: bool = False,
                 remove_bg: bool = False,
                 preserve_layout: bool = False,
                 auto_select: bool = True,
                 stats_path: Optional[str] = None):
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

        # Initialize memory manager
        self.memory_manager = MemoryManager(memory_limit_mb)

        # Initialize resource manager
        self.resource_manager = ResourceManager()

        # Initialize OCR engines with error handling
        self._initialize_engines()

    def _initialize_engines(self):
        """Initialize OCR engines with proper error handling."""
        engines_initialized = 0

        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(
                    self.languages,
                    gpu=self.use_gpu,
                    verbose=False,
                    model_storage_directory=_get_easyocr_cache_dir()
                )
                self.resource_manager.add_resource(self.easyocr_reader)
                logger.info(f"EasyOCR initialized with languages: {self.languages}")
                engines_initialized += 1
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None
                if not TESSERACT_AVAILABLE and not PADDLEOCR_AVAILABLE:
                    raise OCREngineInitializationError("easyocr", e)

        if PADDLEOCR_AVAILABLE:
            try:
                paddle_lang = self._map_to_paddleocr_lang(self.languages)
                os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
                self.paddleocr_reader = PaddleOCR(lang=paddle_lang)
                self.resource_manager.add_resource(self.paddleocr_reader)
                logger.info(f"PaddleOCR initialized with language: {paddle_lang}")
                engines_initialized += 1
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR: {e}")
                self.paddleocr_reader = None

        if not EASYOCR_AVAILABLE and not TESSERACT_AVAILABLE and not PADDLEOCR_AVAILABLE:
            raise OCREngineNotAvailableError()

    def _map_to_paddleocr_lang(self, languages: List[str]) -> str:
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

    def _get_available_engines(self) -> List[str]:
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
        if TESSERACT_AVAILABLE:
            engines.append('tesseract')
        return engines

    def _rank_engines(self, file_type: str = 'image', page_count: int = 1) -> List[str]:
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

    def preprocess_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
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
                if self.upscale or (height < 32 or width < 32):
                    if height < 320 or width < 320:
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

    def _preserve_layout(self, results: List[dict]) -> str:
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

        # Filter results with valid bounding boxes
        valid_results = [r for r in results if r.get('bbox') and len(r.get('bbox', [])) >= 4]
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
        clusters: Dict[int, List[dict]] = {}
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

    def extract_text_easyocr(self, image: np.ndarray) -> Tuple[str, List[dict]]:
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

        with self.memory_manager.memory_context("easyocr processing"):
            for attempt in range(self.max_retries):
                try:
                    # Convert to RGB for EasyOCR
                    if len(image.shape) == 3:
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                    results = self.easyocr_reader.readtext(rgb_image, detail=1)

                    # Filter by confidence and format results
                    filtered_results = []
                    for result in results:
                        if len(result) >= 3 and isinstance(result[2], (int, float)) and result[2] >= self.confidence_threshold:
                            filtered_results.append({
                                'text': result[1],
                                'confidence': float(result[2]),
                                'bbox': result[0] if len(result) > 0 else []
                            })

                    # Extract text
                    text_parts = [r['text'] for r in filtered_results]
                    full_text = ' '.join(text_parts)

                    return full_text, filtered_results

                except MemoryError:
                    raise
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise OCRProcessingError(
                            engine_name="easyocr",
                            original_error=e
                        )
                    logger.warning(f"EasyOCR attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(self.retry_delay)

    def extract_text_tesseract(self, image: np.ndarray) -> Tuple[str, List[dict]]:
        """
        Extract text using pytesseract as fallback with retry logic.

        Args:
            image: Preprocessed image array

        Returns:
            Tuple of (extracted_text, results_list)

        Raises:
            OCRProcessingError: If OCR processing fails
        """
        if not TESSERACT_AVAILABLE:
            raise OCRProcessingError(
                engine_name="tesseract",
                details={'suggestion': "Install pytesseract and Tesseract OCR. See installation instructions."}
            )

        if not PILLOW_AVAILABLE:
            raise OCRProcessingError(
                engine_name="tesseract",
                details={'suggestion': "Pillow required for pytesseract. Install with: pip install pillow"}
            )

        with self.memory_manager.memory_context("tesseract processing"):
            for attempt in range(self.max_retries):
                try:
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

                    # Configure language
                    lang = '+'.join(self.languages) if len(self.languages) > 1 else self.languages[0]

                    text = pytesseract.image_to_string(pil_image, lang=lang, config=self.tesseract_config)

                    # pytesseract doesn't provide detailed confidence per word easily
                    # We'll return a simplified result format
                    results = [{'text': text.strip(), 'confidence': 1.0}] if text.strip() else []
                    return text.strip(), results

                except pytesseract.TesseractError as e:
                    if attempt == self.max_retries - 1:
                        raise OCRProcessingError(
                            engine_name="tesseract",
                            original_error=e,
                            details={'suggestion': "Ensure Tesseract is properly installed and language data is available."}
                        )
                    logger.warning(f"Tesseract attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(self.retry_delay)
                except MemoryError:
                    raise
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise OCRProcessingError(
                            engine_name="tesseract",
                            original_error=e
                        )
                    logger.warning(f"Tesseract attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(self.retry_delay)

    def extract_text_paddleocr(self, image: np.ndarray) -> Tuple[str, List[dict]]:
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

        with self.memory_manager.memory_context("paddleocr processing"):
            for attempt in range(self.max_retries):
                try:
                    # Convert to RGB for PaddleOCR
                    if len(image.shape) == 3:
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                    # PaddleOCR 3.x: predict returns a list of OCRResult objects
                    results = self.paddleocr_reader.predict(rgb_image)

                    filtered_results = []
                    for result in results:
                        if not isinstance(result, dict):
                            continue

                        rec_texts = result.get('rec_texts', [])
                        rec_scores = result.get('rec_scores', [])
                        rec_polys = result.get('rec_polys', [])

                        for i, text in enumerate(rec_texts):
                            if i >= len(rec_scores):
                                continue
                            confidence = float(rec_scores[i])
                            if confidence >= self.confidence_threshold:
                                bbox = rec_polys[i] if i < len(rec_polys) else []
                                filtered_results.append({
                                    'text': text,
                                    'confidence': confidence,
                                    'bbox': bbox,
                                })

                    text_parts = [r['text'] for r in filtered_results]
                    full_text = ' '.join(text_parts)

                    return full_text, filtered_results

                except MemoryError:
                    raise
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise OCRProcessingError(
                            engine_name="paddleocr",
                            original_error=e,
                            details={'suggestion': "Ensure PaddleOCR models are downloaded. Check internet connection."}
                        )
                    logger.warning(f"PaddleOCR attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(self.retry_delay)

    def extract_text(self, image: Union[str, Path, np.ndarray, Image.Image],
                     engine: Optional[str] = None,
                     file_type: str = 'image',
                     page_count: int = 1) -> Tuple[str, List[dict]]:
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
            elif self.auto_select:
                ranked_engines = self._rank_engines(file_type, page_count)
                if not ranked_engines:
                    raise OCREngineNotAvailableError()
            else:
                ranked_engines = self._rank_engines(file_type, page_count)
                if not ranked_engines:
                    raise OCREngineNotAvailableError()

            engines_tried = []
            original_errors: Dict[str, str] = {}

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

                    processing_time = time.time() - start_time

                    # Compute average confidence for stats
                    avg_conf = 0.0
                    if results:
                        confidences = [r.get('confidence', 0.0) for r in results]
                        avg_conf = sum(confidences) / len(confidences)

                    # Apply layout preservation if enabled
                    if self.preserve_layout and results:
                        text = self._preserve_layout(results)

                    # Record stats
                    self._record_engine_result(
                        current_engine, success=True,
                        confidence=avg_conf,
                        processing_time=processing_time,
                        file_type=file_type,
                        page_count=page_count,
                    )

                    return text, results

                except (OCRProcessingError, MemoryError) as e:
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

                except Exception as e:
                    processing_time = time.time() - start_time
                    original_errors[current_engine] = str(e)
                    logger.warning(f"{current_engine} failed with unexpected error: {e}")

                    # Record failure for stats
                    self._record_engine_result(
                        current_engine, success=False,
                        confidence=0.0,
                        processing_time=processing_time,
                        file_type=file_type,
                        page_count=page_count,
                    )

                    if engine is not None:
                        raise OCRProcessingError(
                            engine_name=current_engine,
                            original_error=e,
                            details={'forced_engine': True, 'suggestion': f"Forced engine '{current_engine}' failed."}
                        )
                    continue

            # All engines failed
            available = self._get_available_engines()
            if not available:
                raise OCREngineNotAvailableError(engines_tried=engines_tried)
            raise OCREngineFallbackError(
                engines_tried=engines_tried,
                original_errors=original_errors,
            )

        except (ImageProcessingError, MemoryError):
            # Re-raise image processing and memory errors as-is
            raise
        except OCRError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise OCRProcessingError(original_error=e)

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.

        Returns:
            List of language codes
        """
        return self.languages.copy()

    def post_process_text(self, text: str, languages: Optional[List[str]] = None) -> str:
        """
        Apply language-aware post-processing to extracted text.

        For English only: uses pyspellchecker for spell correction and filters
        single-character noise tokens.

        For all languages: applies universal cleanup including repeated character
        run removal, whitespace normalization, punctuation stripping, and
        short token filtering.

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

        if is_english_only and SPELLCHECKER_AVAILABLE:
            return self._post_process_english(text)
        else:
            return self._post_process_universal(text)

    def _post_process_english(self, text: str) -> str:
        """Post-process English text using spellchecker."""
        spell = SpellChecker()

        # Remove repeated character runs (3+ identical chars -> 1)
        text = re.sub(r'(.)\1{2,}', r'\1', text)

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
                    # Punctuation-only token
                    if token in {'.', ',', '!', '?', ';', ':', "'", '"', '-', '(', ')', '[', ']'}:
                        corrected_tokens.append(token)
                    continue

                word_match = re.match(r"\w+(?:'\w+)?", token)
                if word_match:
                    word = word_match.group()
                    if len(word) > 1:
                        corrected = spell.correction(word)
                        if corrected and corrected != word:
                            token = token.replace(word, corrected, 1)
                    elif len(word) == 1 and word.lower() not in {'a', 'i'}:
                        continue
                    corrected_tokens.append(token)

            processed_lines.append(''.join(corrected_tokens))

        text = '\n'.join(processed_lines)

        # Collapse multiple spaces/tabs/newlines
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _post_process_universal(self, text: str) -> str:
        """
        Post-process text for any language using universal cleanup rules.

        - Remove repeated character runs (3+ identical chars -> 1)
        - Collapse multiple spaces/tabs/newlines to single spaces
        - Strip isolated punctuation-only tokens
        - Remove tokens shorter than 2 chars unless whitelisted
        """
        # Remove repeated character runs (3+ identical chars -> 1)
        text = re.sub(r'(.)\1{2,}', r'\1', text)

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
                    continue
                filtered_tokens.append(token)

        text = ''.join(filtered_tokens)

        # Collapse multiple spaces/tabs/newlines to single spaces
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

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
        return (self.easyocr_reader is not None) or (self.paddleocr_reader is not None) or TESSERACT_AVAILABLE

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
                except Exception:
                    pass
            self.memory_manager.force_gc()
            logger.info("OCR engine resources cleaned up")
        except Exception as e:
            logger.warning(f"Error during OCR engine cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup in destructor


# Convenience functions for standalone usage
def extract_text_from_image(image_path: Union[str, Path],
                          languages: Optional[List[str]] = None,
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


def get_available_engines() -> List[str]:
    """
    Get list of available OCR engines.

    Returns:
        List of available engine names
    """
    engines = []
    if EASYOCR_AVAILABLE:
        engines.append('easyocr')
    if PADDLEOCR_AVAILABLE:
        engines.append('paddleocr')
    if TESSERACT_AVAILABLE:
        engines.append('tesseract')
    return engines