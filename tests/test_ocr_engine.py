#!/usr/bin/env python3
"""
Unit tests for OCR Engine Module.

Tests preprocessing, engine fallback, confidence filtering, and plugin system.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exceptions import (
    OCREngineNotAvailableError,
)
from src.ocr_engine import BaseOCREngine, EngineRegistry, OCREngine, get_available_engines

# Successful tesseract runtime probe result. Tests that patch
# TESSERACT_AVAILABLE=True also mock this so the real binary + tessdata
# are not required in the test environment.
_MOCK_TESSERACT_RUNTIME = {
    'binary': 'mock',
    'tessdata_dir': 'mock',
    'binary_ok': True,
    'tessdata_ok': True,
    'missing_languages': [],
    'ok': True,
}


class TestOCREngine:
    """Tests for the OCREngine class."""

    def test_default_initialization(self):
        """Test OCREngine initializes with default values."""
        with patch('src.ocr_engine.EASYOCR_AVAILABLE', False), \
             patch('src.ocr_engine.TESSERACT_AVAILABLE', True), \
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False), \
             patch('src.ocr_engine.check_tesseract_runtime', return_value=_MOCK_TESSERACT_RUNTIME):
            engine = OCREngine()
            assert engine.languages == ['en']
            assert engine.confidence_threshold == 0.5
            assert engine.use_gpu is False
            assert engine.preprocess_images is True

    def test_custom_initialization(self):
        """Test OCREngine initializes with custom values."""
        with patch('src.ocr_engine.EASYOCR_AVAILABLE', False), \
             patch('src.ocr_engine.TESSERACT_AVAILABLE', True), \
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False), \
             patch('src.ocr_engine.check_tesseract_runtime', return_value=_MOCK_TESSERACT_RUNTIME):
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
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False), \
             patch('src.ocr_engine.check_tesseract_runtime', return_value=_MOCK_TESSERACT_RUNTIME):
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
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False), \
             patch('src.ocr_engine.check_tesseract_runtime', return_value=_MOCK_TESSERACT_RUNTIME):
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
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False), \
             pytest.raises(OCREngineNotAvailableError):
            OCREngine()

    def test_map_to_paddleocr_lang(self):
        """Test language code mapping to PaddleOCR format."""
        with patch('src.ocr_engine.EASYOCR_AVAILABLE', False), \
             patch('src.ocr_engine.TESSERACT_AVAILABLE', True), \
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False), \
             patch('src.ocr_engine.check_tesseract_runtime', return_value=_MOCK_TESSERACT_RUNTIME):
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
             patch('src.ocr_engine.SPELLCHECKER_AVAILABLE', False), \
             patch('src.ocr_engine.check_tesseract_runtime', return_value=_MOCK_TESSERACT_RUNTIME):
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
             patch('src.ocr_engine.SPELLCHECKER_AVAILABLE', False), \
             patch('src.ocr_engine.check_tesseract_runtime', return_value=_MOCK_TESSERACT_RUNTIME):
            engine = OCREngine(languages=['fr'])
            text = "Hello    world   test"
            result = engine.post_process_text(text, ['fr'])
            assert "    " not in result

    def test_post_process_empty_text(self):
        """Test post-processing handles empty text."""
        with patch('src.ocr_engine.EASYOCR_AVAILABLE', False), \
             patch('src.ocr_engine.TESSERACT_AVAILABLE', True), \
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False), \
             patch('src.ocr_engine.check_tesseract_runtime', return_value=_MOCK_TESSERACT_RUNTIME):
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

    def test_discover_no_allowlist_blocks_plugins(self, tmp_path):
        """Test that plugins are blocked when no allowlist exists."""
        EngineRegistry.clear()
        EngineRegistry._allowed_plugins = set()
        EngineRegistry._hash_store = {}

        # Create a fake engines directory
        engines_dir = tmp_path / '.ocr-vibe' / 'engines'
        engines_dir.mkdir(parents=True)

        # Create a minimal plugin file
        plugin_file = engines_dir / 'test_plugin.py'
        plugin_file.write_text("from src.ocr_engine import BaseOCREngine\n")

        with patch('pathlib.Path.home', return_value=tmp_path):
            EngineRegistry.discover()

        # Should not load any plugins without allowlist
        assert len(EngineRegistry.get_all()) == 0

    def test_add_remove_allowlist(self, tmp_path):
        """Test adding and removing plugins from allowlist."""
        EngineRegistry.clear()
        EngineRegistry._allowed_plugins = set()
        EngineRegistry._hash_store = {}

        # Create a test plugin file
        engines_dir = tmp_path / '.ocr-vibe' / 'engines'
        engines_dir.mkdir(parents=True)
        plugin_file = engines_dir / 'my_engine.py'
        plugin_file.write_text("# test plugin\n")

        # Add to allowlist
        EngineRegistry.add_to_allowlist('my_engine', str(plugin_file))
        assert 'my_engine' in EngineRegistry._allowed_plugins
        assert 'my_engine' in EngineRegistry._hash_store

        # Remove from allowlist
        EngineRegistry.remove_from_allowlist('my_engine')
        assert 'my_engine' not in EngineRegistry._allowed_plugins
        assert 'my_engine' not in EngineRegistry._hash_store

    def test_hash_verification(self, tmp_path):
        """Test hash verification catches tampered files."""
        EngineRegistry.clear()
        EngineRegistry._allowed_plugins = {'test_plugin'}
        EngineRegistry._hash_store = {'test_plugin': 'abc123'}

        # Create plugin file
        engines_dir = tmp_path / '.ocr-vibe' / 'engines'
        engines_dir.mkdir(parents=True)
        plugin_file = engines_dir / 'test_plugin.py'
        plugin_file.write_text("# tampered plugin\n")

        # Should fail hash check
        assert not EngineRegistry._verify_plugin_hash(plugin_file)

    def test_hash_rejects_unknown_plugin(self, tmp_path):
        """Test that hash verification rejects plugins with no stored hash.

        An unknown plugin (not in the hash store) must be rejected, not
        silently allowed — the allowlist is the gate, hash verification
        must not relax that gate.
        """
        EngineRegistry.clear()
        EngineRegistry._allowed_plugins = set()
        EngineRegistry._hash_store = {}

        engines_dir = tmp_path / '.ocr-vibe' / 'engines'
        engines_dir.mkdir(parents=True)
        plugin_file = engines_dir / 'unknown_plugin.py'
        plugin_file.write_text("# unknown plugin\n")

        assert not EngineRegistry._verify_plugin_hash(plugin_file)

    def test_load_save_allowlist(self, tmp_path):
        """Test loading and saving allowlist to disk."""
        EngineRegistry.clear()
        EngineRegistry._allowed_plugins = {'plugin1', 'plugin2'}
        EngineRegistry._hash_store = {'plugin1': 'hash1', 'plugin2': 'hash2'}

        allowlist_path = tmp_path / 'allowed_plugins.json'
        with patch.object(EngineRegistry, '_allowlist_path', allowlist_path):
            EngineRegistry.save_allowlist()
            assert allowlist_path.exists()

            # Load it back
            EngineRegistry._allowed_plugins = set()
            EngineRegistry._hash_store = {}
            EngineRegistry.load_allowlist()

            assert EngineRegistry._allowed_plugins == {'plugin1', 'plugin2'}
            assert EngineRegistry._hash_store == {'plugin1': 'hash1', 'plugin2': 'hash2'}

    def test_blocked_imports_in_sandbox(self, tmp_path, monkeypatch):
        """Test that dangerous modules are blocked in plugin context."""
        EngineRegistry.clear()
        EngineRegistry._allowed_plugins = {'safe_plugin'}
        EngineRegistry._hash_store = {'safe_plugin': 'dummy'}

        # Create plugin that tries to import subprocess
        engines_dir = tmp_path / '.ocr-vibe' / 'engines'
        engines_dir.mkdir(parents=True)
        plugin_file = engines_dir / 'safe_plugin.py'
        plugin_file.write_text("""
import subprocess  # This should be blocked
from src.ocr_engine import BaseOCREngine, EngineRegistry

class BadPlugin(BaseOCREngine):
    @property
    def name(self): return 'bad'
    @property
    def supported_languages(self): return ['en']
    def initialize(self, languages, use_gpu=False, **kwargs): return True
    def extract_text(self, image): return ('', [])

EngineRegistry.register('bad', BadPlugin())
""")

        with patch('pathlib.Path.home', return_value=tmp_path):
            # Should fail due to blocked import
            EngineRegistry.discover()
            # Plugin should not be registered
            assert 'bad' not in EngineRegistry.get_all()


class TestGetAvailableEngines:
    """Tests for get_available_engines function."""

    def test_returns_list_of_available_engines(self):
        """Test get_available_engines returns correct list."""
        with patch('src.ocr_engine.EASYOCR_AVAILABLE', True), \
             patch('src.ocr_engine.TESSERACT_AVAILABLE', True), \
             patch('src.ocr_engine.PADDLEOCR_AVAILABLE', False), \
             patch('src.ocr_engine.check_tesseract_runtime', return_value=_MOCK_TESSERACT_RUNTIME):
            engines = get_available_engines()
            assert 'easyocr' in engines
            assert 'tesseract' in engines
            assert 'paddleocr' not in engines
