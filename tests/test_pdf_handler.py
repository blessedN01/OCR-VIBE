#!/usr/bin/env python3
"""
Unit tests for PDF Handler Module.

Tests PDF type detection, page counting, and text extraction with mocks.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_handler import PDFHandler
from src.exceptions import PDFProcessingError

# Mark tests that require optional dependencies
pytestmark = pytest.mark.skipif(
    not all([
        __import__('importlib').util.find_spec('pdfplumber'),
        __import__('importlib').util.find_spec('PyPDF2'),
        __import__('importlib').util.find_spec('fitz'),
    ]),
    reason="Optional PDF dependencies not installed"
)


@pytest.fixture
def mock_ocr_engine():
    """Create a mock OCR engine for testing."""
    mock_engine = MagicMock()
    mock_engine.extract_text.return_value = ("test text", [])
    return mock_engine


class TestPDFHandler:
    """Tests for the PDFHandler class."""

    def test_detect_pdf_type_text_based(self, mock_ocr_engine):
        """Test detecting text-based PDF."""
        with patch('src.pdf_handler.PDFPLUMBER_AVAILABLE', True), \
             patch('src.pdf_handler.PDF2IMAGE_AVAILABLE', True), \
             patch('src.pdf_handler.PYMUPDF_AVAILABLE', True), \
             patch('src.pdf_handler.PYPDF2_AVAILABLE', True):
            handler = PDFHandler(ocr_engine=mock_ocr_engine)

            mock_page = MagicMock()
            mock_page.extract_text.return_value = "This is a long text that exceeds fifty characters for testing purposes."
            mock_page.images = []

            mock_pdf = MagicMock()
            mock_pdf.pages = [mock_page]

            with patch('pdfplumber.open', return_value=MagicMock(__enter__=MagicMock(return_value=mock_pdf), __exit__=MagicMock(return_value=False))):
                result = handler.detect_pdf_type(Path("test.pdf"))
                assert result == 'text'

    def test_detect_pdf_type_image_based(self, mock_ocr_engine):
        """Test detecting image-based PDF."""
        with patch('src.pdf_handler.PDFPLUMBER_AVAILABLE', True), \
             patch('src.pdf_handler.PDF2IMAGE_AVAILABLE', True), \
             patch('src.pdf_handler.PYMUPDF_AVAILABLE', True), \
             patch('src.pdf_handler.PYPDF2_AVAILABLE', True):
            handler = PDFHandler(ocr_engine=mock_ocr_engine)

            mock_page = MagicMock()
            mock_page.extract_text.return_value = ""
            mock_page.images = [MagicMock()]

            mock_pdf = MagicMock()
            mock_pdf.pages = [mock_page]

            with patch('pdfplumber.open', return_value=MagicMock(__enter__=MagicMock(return_value=mock_pdf), __exit__=MagicMock(return_value=False))):
                result = handler.detect_pdf_type(Path("test.pdf"))
                assert result == 'image'

    def test_detect_pdf_type_no_pdfplumber(self, mock_ocr_engine):
        """Test detecting PDF type when pdfplumber is not available."""
        with patch('src.pdf_handler.PDFPLUMBER_AVAILABLE', False):
            handler = PDFHandler(ocr_engine=mock_ocr_engine)
            with pytest.raises(PDFProcessingError):
                handler.detect_pdf_type(Path("test.pdf"))

    def test_get_page_count_with_pypdf2(self, mock_ocr_engine):
        """Test getting page count using PyPDF2."""
        with patch('src.pdf_handler.PDFPLUMBER_AVAILABLE', True), \
             patch('src.pdf_handler.PDF2IMAGE_AVAILABLE', True), \
             patch('src.pdf_handler.PYMUPDF_AVAILABLE', True), \
             patch('src.pdf_handler.PYPDF2_AVAILABLE', True):
            handler = PDFHandler(ocr_engine=mock_ocr_engine)

            mock_reader = MagicMock()
            mock_reader.pages = [MagicMock(), MagicMock(), MagicMock()]

            with patch('builtins.open', mock_open()):
                with patch('PyPDF2.PdfReader', return_value=mock_reader):
                    count = handler.get_page_count(Path("test.pdf"))
                    assert count == 3

    def test_get_page_count_fallback_to_pymupdf(self, mock_ocr_engine):
        """Test page count fallback to PyMuPDF when PyPDF2 fails."""
        with patch('src.pdf_handler.PDFPLUMBER_AVAILABLE', True), \
             patch('src.pdf_handler.PDF2IMAGE_AVAILABLE', True), \
             patch('src.pdf_handler.PYMUPDF_AVAILABLE', True), \
             patch('src.pdf_handler.PYPDF2_AVAILABLE', True):
            handler = PDFHandler(ocr_engine=mock_ocr_engine)

            # Make PyPDF2 fail
            with patch('builtins.open', side_effect=Exception("PyPDF2 error")):
                mock_doc = MagicMock()
                mock_doc.page_count = 5
                with patch('fitz.open', return_value=mock_doc):
                    count = handler.get_page_count(Path("test.pdf"))
                    assert count == 5

    def test_extract_text_file_not_found(self, mock_ocr_engine):
        """Test text extraction with non-existent file."""
        with patch('src.pdf_handler.PDFPLUMBER_AVAILABLE', True), \
             patch('src.pdf_handler.PDF2IMAGE_AVAILABLE', True), \
             patch('src.pdf_handler.PYMUPDF_AVAILABLE', True), \
             patch('src.pdf_handler.PYPDF2_AVAILABLE', True):
            handler = PDFHandler(ocr_engine=mock_ocr_engine)
            with pytest.raises(PDFProcessingError):
                handler.extract_text(Path("nonexistent.pdf"))

    def test_cache_mechanism(self, mock_ocr_engine):
        """Test PDF type caching."""
        with patch('src.pdf_handler.PDFPLUMBER_AVAILABLE', True), \
             patch('src.pdf_handler.PDF2IMAGE_AVAILABLE', True), \
             patch('src.pdf_handler.PYMUPDF_AVAILABLE', True), \
             patch('src.pdf_handler.PYPDF2_AVAILABLE', True):
            handler = PDFHandler(ocr_engine=mock_ocr_engine)

            mock_page = MagicMock()
            mock_page.extract_text.return_value = "This is a long text that exceeds fifty characters for testing purposes."
            mock_page.images = []

            mock_pdf = MagicMock()
            mock_pdf.pages = [mock_page]

            with patch('pdfplumber.open', return_value=MagicMock(__enter__=MagicMock(return_value=mock_pdf), __exit__=MagicMock(return_value=False))):
                # First call
                result1 = handler.detect_pdf_type(Path("test.pdf"))
                # Second call should use cache
                result2 = handler.detect_pdf_type(Path("test.pdf"))
                assert result1 == result2 == 'text'


class TestPDFHandlerNoDeps:
    """Tests for PDFHandler that don't require optional dependencies."""

    def test_handler_creation_with_mock_engine(self, mock_ocr_engine):
        """Test creating PDFHandler with a mock OCR engine."""
        handler = PDFHandler(ocr_engine=mock_ocr_engine)
        assert handler.ocr_engine == mock_ocr_engine

    def test_handler_default_creation(self):
        """Test creating PDFHandler with default OCR engine (mocked)."""
        with patch('src.pdf_handler.OCREngine', return_value=MagicMock()):
            handler = PDFHandler()
            assert handler.ocr_engine is not None
