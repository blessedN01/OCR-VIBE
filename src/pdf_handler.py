#!/usr/bin/env python3
"""
PDF Handler Module

This module provides PDF processing functionality, including text extraction from
text-based PDFs and image extraction from image-based PDFs. It integrates with
the OCR engine for processing scanned documents.

Libraries used:
- pdfplumber: For text extraction from text-based PDFs
- PyPDF2: For basic PDF operations and page counting
- PyMuPDF (fitz): For image extraction and advanced PDF handling
- pdf2image: For converting PDF pages to images for OCR
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.error("pdfplumber is required for PDF text extraction. Install with: pip install pdfplumber")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.error("PyPDF2 is required for PDF operations. Install with: pip install PyPDF2")

try:
    import pymupdf as fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.error("PyMuPDF is required for advanced PDF handling. Install with: pip install PyMuPDF")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.error("pdf2image is required for PDF to image conversion. Install with: pip install pdf2image")
    logger.error("Note: pdf2image also requires poppler-utils to be installed on your system.")

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    logger.error("Pillow is required for image processing. Install with: pip install pillow")

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logger.warning("camelot-py not installed. Table extraction will be unavailable. Install with: pip install camelot-py[cv]")

# Import custom modules
from .exceptions import MemoryError, PDFProcessingError
from .memory_manager import MemoryManager, ResourceManager
from .ocr_engine import OCREngine


class PDFHandler:
    """
    PDF Handler class for processing PDF documents.

    Supports text extraction from text-based PDFs and image extraction
    from image-based PDFs, with OCR integration for scanned documents.
    """

    def __init__(self, ocr_engine: OCREngine | None = None, memory_limit_mb: int | None = None):
        """
        Initialize the PDF handler.

        Args:
            ocr_engine: OCR engine instance for processing image-based PDFs
            memory_limit_mb: Memory limit in MB for processing
        """
        self.ocr_engine = ocr_engine or OCREngine(memory_limit_mb=memory_limit_mb)
        self.memory_manager = MemoryManager(memory_limit_mb)
        self.resource_manager = ResourceManager()

        # Cache for PDF type detection to avoid repeated analysis
        self._type_cache: dict[str, str] = {}
        self._cache_expiry: dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes TTL for cache

    def _get_cache_key(self, pdf_path: str | Path) -> str:
        """Generate cache key for PDF path."""
        return str(Path(pdf_path).resolve())

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self._cache_expiry:
            return False
        return time.time() < self._cache_expiry[cache_key]

    def _get_cached_type(self, pdf_path: str | Path) -> str | None:
        """Get cached PDF type if available and valid."""
        cache_key = self._get_cache_key(pdf_path)
        if self._is_cache_valid(cache_key):
            return self._type_cache.get(cache_key)
        return None

    def _set_cached_type(self, pdf_path: str | Path, pdf_type: str):
        """Cache PDF type with expiry."""
        cache_key = self._get_cache_key(pdf_path)
        self._type_cache[cache_key] = pdf_type
        self._cache_expiry[cache_key] = time.time() + self._cache_ttl

    def detect_pdf_type(self, pdf_path: str | Path) -> str:
        """
        Detect whether PDF is text-based or image-based with caching and error handling.

        Args:
            pdf_path: Path to PDF file

        Returns:
            'text' if text-based, 'image' if image-based

        Raises:
            PDFProcessingError: If PDF type detection fails
        """
        # Check cache first
        cached_type = self._get_cached_type(pdf_path)
        if cached_type:
            logger.debug(f"Using cached PDF type for {pdf_path}: {cached_type}")
            return cached_type

        if not PDFPLUMBER_AVAILABLE:
            raise PDFProcessingError(
                file_path=str(pdf_path),
                operation="type detection",
                details={'suggestion': "Install pdfplumber with: pip install pdfplumber"}
            )

        with self.memory_manager.memory_context("PDF type detection"):
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    # Check first page
                    if len(pdf.pages) == 0:
                        pdf_type = 'empty'
                    else:
                        page = pdf.pages[0]
                        text = page.extract_text()

                        # If significant text is found, consider it text-based
                        if text and len(text.strip()) > 50:
                            pdf_type = 'text'
                        elif page.images:
                            pdf_type = 'image'
                        else:
                            # If no text and no images, might be image-based
                            pdf_type = 'image'

                    # Cache the result
                    self._set_cached_type(pdf_path, pdf_type)
                    return pdf_type

            except FileNotFoundError as e:
                raise PDFProcessingError(
                    file_path=str(pdf_path),
                    operation="type detection",
                    original_error=e,
                    details={'suggestion': "Ensure PDF file exists and path is correct"}
                )
            except PermissionError as e:
                raise PDFProcessingError(
                    file_path=str(pdf_path),
                    operation="type detection",
                    original_error=e,
                    details={'suggestion': "Check file permissions"}
                )
            except MemoryError:
                raise
            except Exception as e:
                logger.warning(f"Error detecting PDF type for {pdf_path}: {e}. Defaulting to image-based.")
                # Default to image-based if detection fails and cache it
                pdf_type = 'image'
                self._set_cached_type(pdf_path, pdf_type)
                return pdf_type

    def extract_text_from_text_pdf(self, pdf_path: str | Path) -> str:
        """
        Extract text from a text-based PDF with error handling.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text

        Raises:
            PDFProcessingError: If text extraction fails
        """
        if not PDFPLUMBER_AVAILABLE:
            raise PDFProcessingError(
                file_path=str(pdf_path),
                operation="text extraction",
                details={'suggestion': "Install pdfplumber with: pip install pdfplumber"}
            )

        with self.memory_manager.memory_context("PDF text extraction"):
            full_text = []

            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        try:
                            text = page.extract_text()
                            if text:
                                full_text.append(text)
                        except Exception as e:
                            logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                            continue

                return '\n\n'.join(full_text)

            except FileNotFoundError as e:
                raise PDFProcessingError(
                    file_path=str(pdf_path),
                    operation="text extraction",
                    original_error=e,
                    details={'suggestion': "Ensure PDF file exists and path is correct"}
                )
            except PermissionError as e:
                raise PDFProcessingError(
                    file_path=str(pdf_path),
                    operation="text extraction",
                    original_error=e,
                    details={'suggestion': "Check file permissions"}
                )
            except MemoryError:
                raise
            except Exception as e:
                raise PDFProcessingError(
                    file_path=str(pdf_path),
                    operation="text extraction",
                    original_error=e,
                    details={'suggestion': "PDF may be corrupted, encrypted, or in an unsupported format"}
                )

    def extract_images_from_pdf(self, pdf_path: str | Path) -> list[Image.Image]:
        """
        Extract images from PDF pages using pdf2image with error handling.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of PIL Images, one per page

        Raises:
            PDFProcessingError: If image extraction fails
        """
        if not PDF2IMAGE_AVAILABLE:
            raise PDFProcessingError(
                file_path=str(pdf_path),
                operation="image extraction",
                details={'suggestion': "Install pdf2image with: pip install pdf2image and ensure poppler-utils is installed"}
            )

        with self.memory_manager.memory_context("PDF image extraction"):
            try:
                # Convert PDF pages to images
                images = convert_from_path(pdf_path)

                # Register images for cleanup
                for img in images:
                    self.resource_manager.add_resource(img)

                return images

            except FileNotFoundError as e:
                raise PDFProcessingError(
                    file_path=str(pdf_path),
                    operation="image extraction",
                    original_error=e,
                    details={'suggestion': "Ensure PDF file exists and path is correct"}
                )
            except PermissionError as e:
                raise PDFProcessingError(
                    file_path=str(pdf_path),
                    operation="image extraction",
                    original_error=e,
                    details={'suggestion': "Check file permissions"}
                )
            except MemoryError:
                raise
            except Exception as e:
                raise PDFProcessingError(
                    file_path=str(pdf_path),
                    operation="image extraction",
                    original_error=e,
                    details={'suggestion': "Ensure pdf2image and poppler-utils are properly installed"}
                )

    def process_image_pdf_with_ocr(self, pdf_path: str | Path, engine: str | None = None,
                                    pages: list[int] | None = None) -> str:
        """
        Process image-based PDF using OCR with memory management and error handling.

        Args:
            pdf_path: Path to PDF file
            engine: Specific OCR engine to use
            pages: Optional list of 1-based page numbers to process

        Returns:
            Extracted text from all pages

        Raises:
            PDFProcessingError: If OCR processing fails
        """
        try:
            images = self.extract_images_from_pdf(pdf_path)
            full_text = []
            page_set = set(pages) if pages else None

            for i, image in enumerate(images):
                if page_set is not None and (i + 1) not in page_set:
                    continue
                try:
                    logger.info(f"Processing page {i + 1}/{len(images)}")
                    with self.memory_manager.memory_context(f"OCR page {i + 1}"):
                        text, _ = self.ocr_engine.extract_text(image, engine)
                        if text:
                            full_text.append(text)
                except MemoryError:
                    logger.warning(f"Memory limit exceeded processing page {i + 1}, skipping...")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing page {i + 1}: {e}")
                    continue

            return '\n\n'.join(full_text)

        except (PDFProcessingError, MemoryError):
            raise
        except Exception as e:
            raise PDFProcessingError(
                file_path=str(pdf_path),
                operation="OCR processing",
                original_error=e
            )

    def extract_text(self, pdf_path: str | Path, engine: str | None = None,
                     pages: list[int] | None = None) -> str:
        """
        Extract text from PDF, automatically detecting type and using appropriate method.

        Args:
            pdf_path: Path to PDF file
            engine: Specific OCR engine to use
            pages: Optional list of 1-based page numbers to extract

        Returns:
            Extracted text

        Raises:
            PDFProcessingError: If extraction fails
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise PDFProcessingError(
                file_path=str(pdf_path),
                operation="text extraction",
                details={'suggestion': "Ensure PDF file exists and path is correct"}
            )

        try:
            pdf_type = self.detect_pdf_type(pdf_path)
            logger.info(f"Detected PDF type: {pdf_type}")

            if pdf_type == 'text':
                if not PDFPLUMBER_AVAILABLE:
                    raise PDFProcessingError(
                        file_path=str(pdf_path),
                        operation="text extraction",
                        details={'suggestion': "Install pdfplumber with: pip install pdfplumber"}
                    )

                page_texts = []
                with pdfplumber.open(pdf_path) as pdf:
                    page_set = set(pages) if pages else None
                    for page_num, page in enumerate(pdf.pages):
                        if page_set is not None and (page_num + 1) not in page_set:
                            continue
                        try:
                            text = page.extract_text()
                            if text:
                                page_texts.append(text)
                        except Exception as e:
                            logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                            continue

                return '\n\n'.join(page_texts)

            elif pdf_type == 'image':
                return self.process_image_pdf_with_ocr(pdf_path, engine, pages)
            else:
                raise PDFProcessingError(
                    file_path=str(pdf_path),
                    operation="type detection",
                    details={'pdf_type': pdf_type, 'suggestion': "PDF type not supported for text extraction"}
                )

        except (PDFProcessingError, MemoryError):
            raise
        except Exception as e:
            raise PDFProcessingError(
                file_path=str(pdf_path),
                operation="text extraction",
                original_error=e
            )

    def get_page_count(self, pdf_path: str | Path) -> int:
        """
        Get the number of pages in a PDF with fallback libraries.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of pages

        Raises:
            PDFProcessingError: If page counting fails
        """
        with self.memory_manager.memory_context("PDF page counting"):
            if PYPDF2_AVAILABLE:
                try:
                    with open(pdf_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        return len(pdf_reader.pages)
                except Exception as e:
                    logger.warning(f"Error getting page count with PyPDF2: {e}")

            if PYMUPDF_AVAILABLE:
                try:
                    doc = fitz.open(pdf_path)
                    page_count = doc.page_count
                    doc.close()
                    return page_count
                except Exception as e:
                    logger.warning(f"Error getting page count with PyMuPDF: {e}")

            raise PDFProcessingError(
                file_path=str(pdf_path),
                operation="page counting",
                details={'suggestion': "Install at least one PDF library: pip install PyPDF2 or pip install PyMuPDF"}
            )

    def extract_text_by_page(self, pdf_path: str | Path, engine: str | None = None,
                             pages: list[int] | None = None) -> list[str]:
        """
        Extract text from each page separately with error handling.

        Args:
            pdf_path: Path to PDF file
            engine: Specific OCR engine to use
            pages: Optional list of 1-based page numbers to extract. If None, extracts all pages.

        Returns:
            List of text strings, one per page (only for requested pages if pages is specified)

        Raises:
            PDFProcessingError: If extraction fails
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise PDFProcessingError(
                file_path=str(pdf_path),
                operation="page extraction",
                details={'suggestion': "Ensure PDF file exists and path is correct"}
            )

        page_set = set(pages) if pages else None

        with self.memory_manager.memory_context("PDF page-by-page extraction"):
            try:
                page_texts = []

                if PDFPLUMBER_AVAILABLE:
                    try:
                        with pdfplumber.open(pdf_path) as pdf:
                            for page_num, page in enumerate(pdf.pages):
                                if page_set is not None and (page_num + 1) not in page_set:
                                    page_texts.append("")
                                    continue
                                try:
                                    text = page.extract_text() or ""
                                    page_texts.append(text)
                                except Exception as e:
                                    logger.warning(f"Error extracting text from page: {e}")
                                    page_texts.append("")
                        
                        if any(page_texts):
                            return page_texts
                    except Exception as e:
                        logger.warning(f"pdfplumber extraction failed: {e}")

                if PDF2IMAGE_AVAILABLE:
                    images = self.extract_images_from_pdf(pdf_path)
                    for i, image in enumerate(images):
                        if page_set is not None and (i + 1) not in page_set:
                            page_texts.append("")
                            continue
                        try:
                            with self.memory_manager.memory_context(f"OCR page {i + 1}"):
                                text, _ = self.ocr_engine.extract_text(image, engine)
                                page_texts.append(text)
                        except MemoryError:
                            logger.warning(f"Memory limit exceeded for page {i + 1}, using empty text")
                            page_texts.append("")
                        except Exception as e:
                            logger.warning(f"Error processing page {i + 1}: {e}")
                            page_texts.append("")

                return page_texts

            except (PDFProcessingError, MemoryError):
                raise
            except Exception as e:
                raise PDFProcessingError(
                    file_path=str(pdf_path),
                    operation="page extraction",
                    original_error=e
                )

    def extract_tables(self, pdf_path: str | Path,
                       pages: str | None = None,
                       flavor: str = 'stream') -> list[Any]:
        """
        Extract tables from a PDF using camelot-py.

        Uses the 'stream' flavor by default (OpenCV-based), which avoids
        the need for Ghostscript or Java system dependencies.

        Args:
            pdf_path: Path to PDF file
            pages: Page range string (e.g., '1-3,5') or None for all pages
            flavor: 'stream' (default, for PDFMiner-based extraction) or
                    'lattice' (for lines-based extraction)

        Returns:
            List of camelot Table objects

        Raises:
            PDFProcessingError: If table extraction fails or camelot is not installed
        """
        if not CAMELOT_AVAILABLE:
            raise PDFProcessingError(
                file_path=str(pdf_path),
                operation="table extraction",
                details={'suggestion': "Install camelot-py with: pip install camelot-py[cv]"}
            )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise PDFProcessingError(
                file_path=str(pdf_path),
                operation="table extraction",
                details={'suggestion': "Ensure PDF file exists and path is correct"}
            )

        with self.memory_manager.memory_context("PDF table extraction"):
            try:
                tables = camelot.read_pdf(
                    str(pdf_path),
                    pages=pages if pages else 'all',
                    flavor=flavor,
                )
                logger.info(f"Extracted {len(tables)} tables from {pdf_path.name}")
                return [tables[i] for i in range(len(tables))]
            except FileNotFoundError as e:
                raise PDFProcessingError(
                    file_path=str(pdf_path),
                    operation="table extraction",
                    original_error=e,
                    details={'suggestion': "Ensure PDF file exists and path is correct"}
                )
            except Exception as e:
                raise PDFProcessingError(
                    file_path=str(pdf_path),
                    operation="table extraction",
                    original_error=e,
                    details={'suggestion': "PDF may not contain extractable tables or is in an unsupported format"}
                )

    def generate_searchable_pdf(self, pdf_path: str | Path,
                                ocr_engine: OCREngine,
                                pages: list[int] | None = None,
                                output_path: str | Path | None = None) -> str:
        """
        Generate a searchable PDF with an embedded OCR text layer using PyMuPDF.

        The text layer is invisible but searchable. The original page images
        are preserved as the visual background.

        Args:
            pdf_path: Path to the input PDF
            ocr_engine: OCREngine instance to use for OCR
            pages: Optional list of 1-based page numbers to process
            output_path: Output path for the searchable PDF (defaults to
                         '<original_name>_searchable.pdf')

        Returns:
            Path to the generated searchable PDF

        Raises:
            PDFProcessingError: If searchable PDF generation fails
        """
        if not PYMUPDF_AVAILABLE:
            raise PDFProcessingError(
                file_path=str(pdf_path),
                operation="searchable PDF generation",
                details={'suggestion': "Install PyMuPDF with: pip install PyMuPDF"}
            )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise PDFProcessingError(
                file_path=str(pdf_path),
                operation="searchable PDF generation",
                details={'suggestion': "Ensure PDF file exists and path is correct"}
            )

        if output_path is None:
            output_path = str(pdf_path.with_name(pdf_path.stem + '_searchable.pdf'))
        else:
            output_path = str(output_path)

        with self.memory_manager.memory_context("searchable PDF generation"):
            try:
                doc = fitz.open(str(pdf_path))
                self.resource_manager.add_resource(doc, lambda d: d.close())

                page_set = set(pages) if pages else None

                for page_num in range(len(doc)):
                    if page_set is not None and (page_num + 1) not in page_set:
                        continue

                    page = doc[page_num]

                    # Get the page pixmap (image)
                    pix = page.get_pixmap(dpi=150)
                    if pix is None:
                        continue

                    # Convert pixmap to PIL Image for OCR
                    if PILLOW_AVAILABLE:
                        mode = "RGBA" if pix.alpha else "RGB"
                        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                        if mode == "RGBA":
                            img = img.convert("RGB")
                    else:
                        import numpy as _np
                        img = _np.frombuffer(pix.samples, dtype=_np.uint8).reshape(
                            pix.height, pix.width, pix.n
                        )
                        img = Image.fromarray(img, mode='RGB' if pix.n == 3 else 'RGBA').convert('RGB')

                    # Run OCR on the page image
                    text, _ = ocr_engine.extract_text(img, file_type='pdf', page_count=1)

                    if text and text.strip():
                        page.insert_textbox(
                            page.rect,
                            text,
                            fontname="helv",
                            fontsize=10,
                            color=(1, 1, 1),
                            opacity=0,
                        )

                    pix = None

                doc.save(output_path, garbage=4, deflate=True, clean_text=True)
                doc.close()

                logger.info(f"Searchable PDF saved to {output_path}")
                return output_path

            except fitz.FitzError as e:
                raise PDFProcessingError(
                    file_path=str(pdf_path),
                    operation="searchable PDF generation",
                    original_error=e,
                    details={'suggestion': "Ensure PyMuPDF is properly installed and the PDF is not corrupted."}
                )
            except Exception as e:
                raise PDFProcessingError(
                    file_path=str(pdf_path),
                    operation="searchable PDF generation",
                    original_error=e,
                    details={'suggestion': "Check system resources and PDF integrity."}
                )


# Convenience functions
def extract_text_from_pdf(pdf_path: str | Path,
                         ocr_languages: list[str] | None = None) -> str:
    """
    Convenience function to extract text from a PDF.

    Args:
        pdf_path: Path to PDF file
        ocr_languages: Languages for OCR (if needed)

    Returns:
        Extracted text
    """
    ocr_engine = OCREngine(languages=ocr_languages) if ocr_languages else OCREngine()
    handler = PDFHandler(ocr_engine=ocr_engine)
    return handler.extract_text(pdf_path)


def get_pdf_info(pdf_path: str | Path) -> dict:
    """
    Get basic information about a PDF with error handling.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dictionary with PDF information

    Raises:
        PDFProcessingError: If info retrieval fails
    """
    try:
        handler = PDFHandler()
        return {
            'page_count': handler.get_page_count(pdf_path),
            'type': handler.detect_pdf_type(pdf_path),
            'path': str(pdf_path)
        }
    except Exception as e:
        raise PDFProcessingError(
            file_path=str(pdf_path),
            operation="info retrieval",
            original_error=e
        )