#!/usr/bin/env python3
"""
GUI Controller Module

Business logic for the OCR GUI, decoupled from Flet UI controls.
All shared state is protected by thread locks for thread safety.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

from .ocr_engine import OCREngine

logger = logging.getLogger(__name__)


class GUIController:
    """
    Controller class for GUI business logic.

    Holds shared state and provides methods for processing operations.
    All state access is protected by threading.Lock.
    """

    def __init__(
        self,
        ocr_engine: OCREngine | None = None,
        on_state_change: Callable[[], None] | None = None,
        on_log: Callable[[str], None] | None = None,
    ):
        """
        Initialize the GUI controller.

        Args:
            ocr_engine: OCR engine instance
            on_state_change: Callback when state changes
            on_log: Callback for logging messages
        """
        self._lock = threading.Lock()
        self._ocr_engine = ocr_engine
        self.on_state_change = on_state_change or (lambda: None)
        self.on_log = on_log or (lambda msg: None)

        self._processing = False
        self._current_file: Path | None = None
        self._temp_files: list[str] = []
        self._log_messages: list[str] = []

    @property
    def processing(self) -> bool:
        """Get processing state (thread-safe)."""
        with self._lock:
            return self._processing

    @processing.setter
    def processing(self, value: bool) -> None:
        """Set processing state (thread-safe)."""
        with self._lock:
            self._processing = bool(value)
        self.on_state_change()

    @property
    def current_file(self) -> Path | None:
        """Get current file being processed (thread-safe)."""
        with self._lock:
            return self._current_file

    @current_file.setter
    def current_file(self, value: Path | None) -> None:
        """Set current file being processed (thread-safe)."""
        with self._lock:
            self._current_file = Path(value) if value is not None else None
        self.on_state_change()

    @property
    def ocr_engine(self) -> OCREngine | None:
        """Get OCR engine instance (thread-safe)."""
        with self._lock:
            return self._ocr_engine

    @ocr_engine.setter
    def ocr_engine(self, value: OCREngine | None) -> None:
        """Set OCR engine instance (thread-safe)."""
        with self._lock:
            self._ocr_engine = value
        self.on_state_change()

    @property
    def temp_files(self) -> list[str]:
        """Get list of temporary files (thread-safe copy)."""
        with self._lock:
            return list(self._temp_files)

    def add_temp_file(self, temp_file: str) -> None:
        """Add a temporary file (thread-safe)."""
        with self._lock:
            self._temp_files.append(temp_file)
        self.on_state_change()

    def clear_temp_files(self) -> None:
        """Clear temporary files list (thread-safe)."""
        with self._lock:
            self._temp_files.clear()
        self.on_state_change()

    @property
    def log_messages(self) -> list[str]:
        """Get log messages (thread-safe copy)."""
        with self._lock:
            return list(self._log_messages)

    def log(self, message: str) -> None:
        """Add a log message (thread-safe)."""
        with self._lock:
            self._log_messages.append(message)
            if len(self._log_messages) > 1000:
                self._log_messages = self._log_messages[-500:]
        self.on_log(message)
        self.on_state_change()

    def start_processing(self, file_path: Path | str) -> bool:
        """
        Start processing a file.

        Args:
            file_path: Path to file to process

        Returns:
            True if processing started, False if already processing
        """
        with self._lock:
            if self._processing:
                return False
            self._processing = True
            self._current_file = Path(file_path) if file_path is not None else None
        self.log(f"Starting processing of {self.current_file.name if self.current_file else 'file'}")
        self.on_state_change()
        return True

    def cancel(self) -> None:
        """Cancel current processing."""
        with self._lock:
            if self._processing:
                self._processing = False
        self.log("Processing cancelled")
        self.on_state_change()

    def change_language(self, lang: str) -> None:
        """
        Change UI language.

        Args:
            lang: Language code
        """
        self.log(f"Language changed to {lang}")
        self.on_state_change()

    def toggle_theme(self) -> None:
        """Toggle UI theme."""
        self.log("Theme toggled")
        self.on_state_change()

    def reset(self) -> None:
        """Reset controller state."""
        with self._lock:
            self._processing = False
            self._current_file = None
        self.on_state_change()
