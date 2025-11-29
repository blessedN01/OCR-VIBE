#!/usr/bin/env python3
"""
OCR Engine Module

This module provides OCR functionality using EasyOCR as the primary engine
with pytesseract as a fallback. It includes image preprocessing capabilities
using OpenCV and Pillow, and supports multiple languages with confidence thresholds.
"""

import logging
import time
from typing import List, Optional, Tuple, Union
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

# Import custom modules
from .exceptions import (
    OCRError, OCREngineNotAvailableError, OCREngineInitializationError,
    OCRProcessingError, ImageProcessingError, MemoryError
)
from .memory_manager import MemoryManager, ResourceManager, memory_limited

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                 retry_delay: float = 1.0):
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
        """
        self.languages = languages or ['en']
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu
        self.preprocess_images = preprocess_images
        self.memory_limit_mb = memory_limit_mb
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize OCR readers
        self.easyocr_reader = None
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
                    verbose=False
                )
                self.resource_manager.add_resource(self.easyocr_reader)
                logger.info(f"EasyOCR initialized with languages: {self.languages}")
                engines_initialized += 1
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None
                raise OCREngineInitializationError("easyocr", e)

        if not EASYOCR_AVAILABLE and not TESSERACT_AVAILABLE:
            raise OCREngineNotAvailableError()

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

                if not self.preprocess_images:
                    return img

                # Convert to grayscale
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img

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

                # Resize if image is too small
                height, width = processed.shape
                if height < 32 or width < 32:
                    scale_factor = max(32 / height, 32 / width)
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

                return processed

            except MemoryError:
                raise
            except Exception as e:
                raise ImageProcessingError(
                    operation="preprocessing",
                    original_error=e
                )

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

    def extract_text(self, image: Union[str, Path, np.ndarray, Image.Image],
                    engine: Optional[str] = None) -> Tuple[str, List[dict]]:
        """
        Extract text from image using available OCR engines with fallback and retry logic.

        Args:
            image: Input image
            engine: Force specific engine ('easyocr' or 'tesseract')

        Returns:
            Tuple of (extracted_text, results_list)

        Raises:
            OCREngineNotAvailableError: If no engines are available
            OCRProcessingError: If all engines fail
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)

            # Try EasyOCR first (unless specified otherwise)
            if engine == 'easyocr' or (engine is None and self.easyocr_reader):
                try:
                    return self.extract_text_easyocr(processed_image)
                except (OCRProcessingError, MemoryError) as e:
                    logger.warning(f"EasyOCR failed: {e}")
                    if engine == 'easyocr':
                        raise OCRProcessingError(
                            engine_name="easyocr",
                            original_error=e,
                            details={'forced_engine': True}
                        )
                except Exception as e:
                    logger.warning(f"EasyOCR failed with unexpected error: {e}")
                    if engine == 'easyocr':
                        raise OCRProcessingError(
                            engine_name="easyocr",
                            original_error=e,
                            details={'forced_engine': True}
                        )

            # Fallback to pytesseract
            if engine == 'tesseract' or (engine is None and TESSERACT_AVAILABLE):
                try:
                    return self.extract_text_tesseract(processed_image)
                except (OCRProcessingError, MemoryError) as e:
                    logger.warning(f"pytesseract failed: {e}")
                    if engine == 'tesseract':
                        raise OCRProcessingError(
                            engine_name="tesseract",
                            original_error=e,
                            details={'forced_engine': True}
                        )
                except Exception as e:
                    logger.warning(f"pytesseract failed with unexpected error: {e}")
                    if engine == 'tesseract':
                        raise OCRProcessingError(
                            engine_name="tesseract",
                            original_error=e,
                            details={'forced_engine': True}
                        )

            # No engines available or all failed
            raise OCREngineNotAvailableError()

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
        return (self.easyocr_reader is not None) or TESSERACT_AVAILABLE

    def cleanup(self):
        """
        Cleanup resources used by the OCR engine.
        """
        try:
            self.resource_manager.cleanup()
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
    if TESSERACT_AVAILABLE:
        engines.append('tesseract')
    return engines