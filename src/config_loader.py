#!/usr/bin/env python3
"""
Configuration Loader Module

This module provides YAML/JSON configuration file support with a layered merge strategy:
- Project config (ocr-vibe.yaml / ocr-vibe.json): discovered by walking up from CWD to filesystem root
- Home config (~/.ocr-vibe/config.yaml): overrides project defaults
- CLI flags: highest precedence, applied via apply_cli_overrides()

Shallow merge is used (top-level keys only). CLI flags completely replace config values when present.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .exceptions import OCRError

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG: dict[str, Any] = {
    'languages': ['en'],
    'confidence_threshold': 0.5,
    'output_format': 'txt',
    'preprocess_images': True,
    'engine': None,
    'use_gpu': False,
    'memory_limit_mb': None,
    'post_process': False,
    'extract_tables': False,
    'preserve_layout': False,
    'deskew': False,
    'clahe': False,
    'upscale': False,
    'remove_bg': False,
    'searchable_pdf': False,
    'auto_select': True,
    'verbose': False,
}

SUPPORTED_CONFIG_KEYS = set(DEFAULT_CONFIG.keys())


def _find_project_config(start_path: Path | None = None) -> Path | None:
    """
    Walk up from start_path (or CWD) to find ocr-vibe.yaml, ocr-vibe.yml, or ocr-vibe.json.

    Args:
        start_path: Starting directory for search, defaults to CWD

    Returns:
        Path to project config file if found, None otherwise
    """
    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()

    while True:
        for filename in ('ocr-vibe.yaml', 'ocr-vibe.yml', 'ocr-vibe.json'):
            config_path = current / filename
            if config_path.is_file():
                return config_path

        if current.parent == current:
            break
        current = current.parent

    return None


def _load_config_file(config_path: Path) -> dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary with configuration values

    Raises:
        OCRError: If file format is unsupported or parsing fails
    """
    if not config_path.is_file():
        raise OCRError(f"Configuration file not found: {config_path}")

    suffix = config_path.suffix.lower()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if suffix in ('.yaml', '.yml'):
                if not YAML_AVAILABLE:
                    raise OCRError(
                        f"YAML config file found ({config_path}) but PyYAML is not installed. "
                        "Install with: pip install pyyaml"
                    )
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
            elif suffix == '.json':
                data = json.load(f)
                return data if isinstance(data, dict) else {}
            else:
                raise OCRError(
                    f"Unsupported config file format: {suffix}. Use .yaml, .yml, or .json"
                )
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise OCRError(
            f"Failed to parse config file {config_path}: {e}",
            details={'original_error': str(e)}
        )
    except OSError as e:
        raise OCRError(
            f"Failed to read config file {config_path}: {e}",
            details={'original_error': str(e)}
        )


def _get_home_config_path() -> Path | None:
    """
    Get path to home configuration file.

    Returns:
        Path to ~/.ocr-vibe/config.yaml or None if home directory unavailable
    """
    home_dir = Path.home()
    if not home_dir or str(home_dir) in ('/', 'C:\\'):
        # Home directory not available or is root
        return None

    home_config_dir = home_dir / '.ocr-vibe'
    config_path = home_config_dir / 'config.yaml'

    return config_path if config_path.is_file() else None


def _shallow_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Shallow merge two dictionaries. Only top-level keys are merged.
    Nested dictionaries are completely replaced, not merged recursively.

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary

    Returns:
        Merged dictionary
    """
    merged = base.copy()
    for key, value in override.items():
        if key in SUPPORTED_CONFIG_KEYS:
            merged[key] = value
        else:
            logger.warning(f"Ignoring unsupported config key: {key}")
    return merged


def load_config(start_path: Path | None = None) -> dict[str, Any]:
    """
    Load configuration with layered merge strategy.

    Precedence (lowest to highest):
    1. Defaults
    2. Project config (ocr-vibe.yaml / ocr-vibe.json)
    3. Home config (~/.ocr-vibe/config.yaml)
    4. CLI flags (applied by caller via apply_cli_overrides)

    Args:
        start_path: Starting directory for project config search, defaults to CWD

    Returns:
        Merged configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()

    # Layer 1: Project config (lowest precedence)
    project_config_path = _find_project_config(start_path)
    if project_config_path:
        try:
            project_config = _load_config_file(project_config_path)
            config = _shallow_merge(config, project_config)
            logger.debug(f"Loaded project config from {project_config_path}")
        except OCRError as e:
            logger.warning(f"Failed to load project config: {e}")

    # Layer 2: Home config (overrides project config)
    home_config_path = _get_home_config_path()
    if home_config_path:
        try:
            home_config = _load_config_file(home_config_path)
            config = _shallow_merge(config, home_config)
            logger.debug(f"Loaded home config from {home_config_path}")
        except OCRError as e:
            logger.warning(f"Failed to load home config: {e}")

    return config


def apply_cli_overrides(config: dict[str, Any], cli_args: dict[str, Any]) -> dict[str, Any]:
    """
    Apply CLI flag overrides to configuration (highest precedence).

    CLI flags completely replace config values when present.

    Args:
        config: Current configuration dictionary
        cli_args: Dictionary of CLI argument overrides (None values are ignored)

    Returns:
        Updated configuration dictionary
    """
    updated_config = config.copy()

    for key, value in cli_args.items():
        if value is not None and key in SUPPORTED_CONFIG_KEYS:
            updated_config[key] = value

    return updated_config


def get_config_parser(start_path: Path | None = None) -> dict[str, Any]:
    """
    Get default configuration loaded from project and home configs.

    This is a convenience function that returns a configuration dictionary
    ready to be overridden by CLI args.

    Args:
        start_path: Starting directory for project config search, defaults to CWD

    Returns:
        Merged configuration dictionary
    """
    return load_config(start_path)
