#!/usr/bin/env python3
"""
Unit tests for OCR Engine Module.

Tests preprocessing, engine fallback, confidence filtering, and plugin system.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_engine import OCREngine, BaseOCREngine, EngineRegistry, get_available_engines
from src.exceptions import (
    OCRError, OCREngineNotAvailableError, OCRProcessingError, ImageProcessingError
)


class TestOCREngine:
    """Tests for the OCREngine class."""

    def test_default_initialization(self):
        """Test OCREngine initializes with default values."""
        with patch('src.ocr_engine.EASYOCR_AVAILABLE', False), \
             patch('src.ocr_engine.TESSERACT_AVAILABLE', True), \
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False):
            engine = OCREngine()
            assert engine.languages == ['en']
            assert engine.confidence_threshold == 0.5
            assert engine.use_gpu is False
            assert engine.preprocess_images is True

    def test_custom_initialization(self):
        """Test OCREngine initializes with custom values."""
        with patch('src.ocr_engine.EASYOCR_AVAILABLE', False), \
             patch('src.ocr_engine.TESSERACT_AVAILABLE', True), \
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False):
            engine = OCREngine(
                languages=['en', 'fr'],
                confidence_threshold=0.7,
                use_gpu=True,
                preprocess_images=False,
                deskew=True,
                clahe=True,
                upscale=True,
                remove_bg=True,
                preserve_layout=True,
            )
            assert engine.languages == ['en', 'fr']
            assert engine.confidence_threshold == 0.7
            assert engine.use_gpu is True
            assert engine.preprocess_images is False
            assert engine.deskew is True
            assert engine.clahe is True
            assert engine.upscale is True
            assert engine.remove_bg is True
            assert engine.preserve_layout is True

    def test_confidence_threshold_validation(self):
        """Test confidence threshold setter validates range."""
        with patch('src.ocr_engine.EASYOCR_AVAILABLE', False), \
             patch('src.ocr_engine.TESSERACT_AVAILABLE', True), \
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False):
            engine = OCREngine()
            engine.set_confidence_threshold(0.8)
            assert engine.confidence_threshold == 0.8

            with pytest.raises(ValueError):
                engine.set_confidence_threshold(1.5)

            with pytest.raises(ValueError):
                engine.set_confidence_threshold(-0.1)

    def test_get_supported_languages(self):
        """Test get_supported_languages returns a copy."""
        with patch('src.ocr_engine.EASYOCR_AVAILABLE', False), \
             patch('src.ocr_engine.TESSERACT_AVAILABLE', True), \
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False):
            engine = OCREngine(languages=['en', 'fr'])
            langs = engine.get_supported_languages()
            assert langs == ['en', 'fr']
            # Verify it's a copy
            langs.append('de')
            assert engine.languages == ['en', 'fr']

    def test_is_available_with_no_engines(self):
        """Test is_available returns False when no engines available."""
        with patch('src.ocr_engine.EASYOCR_AVAILABLE', False), \
             patch('src.ocr_engine.TESSERACT_AVAILABLE', False), \
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False):
            with pytest.raises(OCREngineNotAvailableError):
                OCREngine()

    def test_map_to_paddleocr_lang(self):
        """Test language code mapping to PaddleOCR format."""
        with patch('src.ocr_engine.EASYOCR_AVAILABLE', False), \
             patch('src.ocr_engine.TESSERACT_AVAILABLE', True), \
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False):
            engine = OCREngine(languages=['en'])
            assert engine._map_to_paddleocr_lang(['en']) == 'en'
            assert engine._map_to_paddleocr_lang(['fr']) == 'fr'
            assert engine._map_to_paddleocr_lang(['de']) == 'german'
            assert engine._map_to_paddleocr_lang(['ja']) == 'japan'
            assert engine._map_to_paddleocr_lang(['zh']) == 'ch'
            assert engine._map_to_paddleocr_lang(['xx']) == 'en'  # Unknown defaults to en


class TestPostProcessing:
    """Tests for post-processing functionality."""

    def test_post_process_universal_removes_repeated_chars(self):
        """Test universal post-processing removes repeated character runs."""
        with patch('src.ocr_engine.EASYOCR_AVAILABLE', False), \
             patch('src.ocr_engine.TESSERACT_AVAILABLE', True), \
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False), \
             patch('src.ocr_engine.SPELLCHECKER_AVAILABLE', False):
            engine = OCREngine(languages=['fr'])  # Non-English triggers universal
            text = "Hellooo worlddd"
            result = engine.post_process_text(text, ['fr'])
            assert "Hellooo" not in result
            assert "worlddd" not in result

    def test_post_process_universal_collapses_whitespace(self):
        """Test universal post-processing collapses whitespace."""
        with patch('src.ocr_engine.EASYOCR_AVAILABLE', False), \
             patch('src.ocr_engine.TESSERACT_AVAILABLE', True), \
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False), \
             patch('src.ocr_engine.SPELLCHECKER_AVAILABLE', False):
            engine = OCREngine(languages=['fr'])
            text = "Hello    world   test"
            result = engine.post_process_text(text, ['fr'])
            assert "    " not in result

    def test_post_process_empty_text(self):
        """Test post-processing handles empty text."""
        with patch('src.ocr_engine.EASYOCR_AVAILABLE', False), \
             patch('src.ocr_engine.TESSERACT_AVAILABLE', True), \
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False):
            engine = OCREngine()
            assert engine.post_process_text("", ['en']) == ""
            assert engine.post_process_text("   ", ['en']) == "   "


class TestBaseOCREngine:
    """Tests for BaseOCREngine abstract class."""

    def test_base_engine_abstract_methods(self):
        """Test BaseOCREngine raises NotImplementedError for abstract methods."""
        with pytest.raises(TypeError):
            BaseOCREngine()  # Can't instantiate abstract class


class TestEngineRegistry:
    """Tests for EngineRegistry plugin system."""

    def test_register_and_get_engine(self):
        """Test registering and retrieving an engine."""
        EngineRegistry.clear()

        mock_engine = MagicMock(spec=BaseOCREngine)
        mock_engine.name = 'test_engine'
        mock_engine.supported_languages = ['en']

        EngineRegistry.register('test_engine', mock_engine)
        retrieved = EngineRegistry.get('test_engine')
        assert retrieved == mock_engine

    def test_unregister_engine(self):
        """Test unregistering an engine."""
        EngineRegistry.clear()

        mock_engine = MagicMock(spec=BaseOCREngine)
        mock_engine.name = 'test_engine'

        EngineRegistry.register('test_engine', mock_engine)
        EngineRegistry.unregister('test_engine')
        assert EngineRegistry.get('test_engine') is None

    def test_get_all_engines(self):
        """Test getting all registered engines."""
        EngineRegistry.clear()

        engine1 = MagicMock(spec=BaseOCREngine)
        engine1.name = 'engine1'
        engine2 = MagicMock(spec=BaseOCREngine)
        engine2.name = 'engine2'

        EngineRegistry.register('engine1', engine1)
        EngineRegistry.register('engine2', engine2)

        all_engines = EngineRegistry.get_all()
        assert len(all_engines) == 2
        assert 'engine1' in all_engines
        assert 'engine2' in all_engines

        EngineRegistry.clear()

    def test_clear_registry(self):
        """Test clearing all engines from registry."""
        mock_engine = MagicMock(spec=BaseOCREngine)
        mock_engine.name = 'test_engine'
        EngineRegistry.register('test_engine', mock_engine)
        EngineRegistry.clear()
        assert len(EngineRegistry.get_all()) == 0

    def test_discover_no_directory(self):
        """Test discover handles missing directory gracefully."""
        EngineRegistry.clear()
        EngineRegistry.discover()  # Should not raise
        assert len(EngineRegistry.get_all()) == 0


class TestGetAvailableEngines:
    """Tests for get_available_engines function."""

    def test_returns_list_of_available_engines(self):
        """Test get_available_engines returns correct list."""
        with patch('src.ocr_engine.EASYOCR_AVAILABLE', True), \
             patch('src.ocr_engine.TESSERACT_AVAILABLE', True), \
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False):
            engines = get_available_engines()
            assert 'easyocr' in engines
            assert 'tesseract' in engines
            assert 'paddleocr' not in engines
