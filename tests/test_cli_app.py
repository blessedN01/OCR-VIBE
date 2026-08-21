#!/usr/bin/env python3
"""
Unit tests for CLI Application Module.

Tests Click CLI invocation with CliRunner and format parsing.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from click.testing import CliRunner
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli_app import cli, parse_page_ranges, OutputFormatter


class TestParsePageRanges:
    """Tests for page range parsing."""

    def test_single_page(self):
        """Test parsing a single page."""
        result = parse_page_ranges("5")
        assert result == [5]

    def test_page_range(self):
        """Test parsing a page range."""
        result = parse_page_ranges("1-5")
        assert result == [1, 2, 3, 4, 5]

    def test_mixed_ranges(self):
        """Test parsing mixed single pages and ranges."""
        result = parse_page_ranges("1-3,5,7-9")
        assert result == [1, 2, 3, 5, 7, 8, 9]

    def test_invalid_range_start_greater_than_end(self):
        """Test invalid range where start > end."""
        with pytest.raises(ValueError):
            parse_page_ranges("5-3")

    def test_invalid_format(self):
        """Test invalid page range format."""
        with pytest.raises(ValueError):
            parse_page_ranges("1-2-3")


class TestOutputFormatter:
    """Tests for OutputFormatter class."""

    def test_format_txt(self):
        """Test TXT formatting."""
        result = OutputFormatter.format_txt("Hello world", {'file_path': 'test.jpg'})
        assert "Hello world" in result
        assert "test.jpg" in result

    def test_format_json(self):
        """Test JSON formatting."""
        result = OutputFormatter.format_json("Hello", [{'text': 'Hello', 'confidence': 0.9}])
        import json
        data = json.loads(result)
        assert data['text'] == "Hello"
        assert len(data['results']) == 1

    def test_format_csv(self):
        """Test CSV formatting."""
        results = [{'text': 'Hello', 'confidence': 0.9, 'bbox': [[0, 0], [10, 0], [10, 10], [0, 10]]}]
        result = OutputFormatter.format_csv("Hello", results)
        assert "Hello" in result
        assert "0.9" in result

    def test_format_md(self):
        """Test Markdown formatting."""
        result = OutputFormatter.format_md("Hello world", {'file_path': 'test.jpg'})
        assert "# OCR Results" in result
        assert "test.jpg" in result
        assert "Hello world" in result

    def test_format_html(self):
        """Test HTML formatting with confidence coloring."""
        results = [
            {'text': 'Good', 'confidence': 0.8},
            {'text': 'Bad', 'confidence': 0.3},
        ]
        result = OutputFormatter.format_html("Good Bad", results)
        assert "<!DOCTYPE html>" in result
        assert "high-conf" in result
        assert "low-conf" in result

    def test_format_txt_no_metadata(self):
        """Test TXT formatting without metadata."""
        result = OutputFormatter.format_txt("Hello world")
        assert result == "Hello world"


class TestCLICommands:
    """Tests for CLI commands using Click's test runner."""

    def test_cli_help(self):
        """Test CLI help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert "OCR Text Extractor" in result.output

    def test_engines_command(self):
        """Test engines command."""
        runner = CliRunner()
        with patch('src.cli_app.get_available_engines', return_value=['easyocr', 'tesseract']):
            result = runner.invoke(cli, ['engines'])
            assert result.exit_code == 0
            assert 'easyocr' in result.output

    def test_engines_command_no_engines(self):
        """Test engines command when no engines available."""
        runner = CliRunner()
        with patch('src.cli_app.get_available_engines', return_value=[]):
            result = runner.invoke(cli, ['engines'])
            assert result.exit_code == 1

    def test_process_file_help(self):
        """Test process-file command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['process-file', '--help'])
        assert result.exit_code == 0
        assert '--languages' in result.output
        assert '--engine' in result.output

    def test_process_dir_help(self):
        """Test process-dir command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['process-dir', '--help'])
        assert result.exit_code == 0
        assert '--parallel' in result.output

    def test_batch_process_help(self):
        """Test batch-process command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['batch-process', '--help'])
        assert result.exit_code == 0

    def test_version(self):
        """Test version output."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
