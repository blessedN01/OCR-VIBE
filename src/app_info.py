#!/usr/bin/env python3
"""
Application Identity Module

Single source of truth for the application's name, version, and per-user
data directory. Deriving these from one module (rather than hard-coding
the string in each file) keeps the window title, config directory, model
cache, and package metadata in sync.

The display name (``APP_NAME``) and the filesystem-safe slug
(``APP_NAME_LOWER``) are kept separate because Windows and PyInstaller
expect a directory name without dashes/special characters, while the
window title may use the stylized form.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME: str = "OCR-VIBE"
APP_NAME_LOWER: str = "ocr-vibe"
APP_VERSION: str = "2.1.0"


def get_app_dir() -> Path:
    """
    Return the per-user application directory, creating it if needed.

    Uses ``$APPDATA`` on Windows (matching the legacy behavior) and
    the application data directory on other platforms. Honors an
    ``OCR_VIBE_APP_DIR``
    override for testing and sandboxed runs.

    Returns:
        Path to the application directory.
    """
    override = os.environ.get("OCR_VIBE_APP_DIR")
    if override:
        app_dir = Path(override)
    else:
        home_dir = Path.home()
        if os.name == "nt":
            base = Path(os.environ.get("APPDATA", str(home_dir)))
        else:
            base = home_dir
        app_dir = base / f".{APP_NAME_LOWER}"

    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir