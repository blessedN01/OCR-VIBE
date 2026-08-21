#!/usr/bin/env python3
"""
Engine Stats Tracker Module

Tracks per-engine performance statistics for dynamic engine selection.
Uses a UCB-style scoring formula to balance exploitation (confidence)
with exploration of under-tried engines.

Stats are stored in ~/.ocr-vibe/engine_stats.json and persisted atomically.
"""

import json
import logging
import math
import os
import atexit
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .exceptions import OCRError

logger = logging.getLogger(__name__)

EXPLORATION_WEIGHT = 1.0

DEFAULT_STATS: Dict[str, Any] = {
    'engines': {},
}


def _get_stats_path() -> Path:
    """
    Get the path to the engine stats file in the user's home config directory.
    """
    home_dir = Path.home()
    if not home_dir or str(home_dir) in ('/', 'C:\\'):
        return Path(tempfile.gettempdir()) / '.ocr-vibe' / 'engine_stats.json'
    return home_dir / '.ocr-vibe' / 'engine_stats.json'


def get_stats_path() -> Path:
    """
    Public accessor for the stats file path (used by tests and callers).
    """
    return _get_stats_path()


class EngineStatsTracker:
    """
    Tracks per-engine performance statistics for dynamic engine selection.

    Stats are keyed by language set + file type. For each combination,
    we track per-engine:
      - average confidence
      - success/failure counts
      - average processing time (seconds)
      - page-count ranges for PDFs

    The scoring formula (UCB-style) is:
        score(e) = avg_confidence(e) + EXPLORATION_WEIGHT * sqrt(ln(total_runs + 1) / (runs_for_e + 1))

    Engines with no runs get a pure exploration bonus:
        EXPLORATION_WEIGHT * sqrt(ln(total_runs + 1))
    """

    def __init__(self, stats_path: Optional[Path] = None):
        """
        Initialize the stats tracker.

        Args:
            stats_path: Optional path to stats file. Defaults to ~/.ocr-vibe/engine_stats.json
        """
        self.stats_path = stats_path or _get_stats_path()
        self.stats: Dict[str, Any] = {}
        self._loaded = False
        self._dirty = False
        self._load()
        atexit.register(self.flush)

    def _ensure_stats_dir(self):
        """Ensure the directory for the stats file exists."""
        try:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _load(self):
        """Load stats from disk if available."""
        if not self.stats_path.exists():
            self.stats = self._deep_copy_default()
            self._loaded = True
            return

        try:
            with open(self.stats_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'engines' in data:
                self.stats = data
            else:
                self.stats = self._deep_copy_default()
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load engine stats from {self.stats_path}: {e}")
            self.stats = self._deep_copy_default()
        except Exception as e:
            logger.warning(f"Unexpected error loading engine stats: {e}")
            self.stats = self._deep_copy_default()

        self._loaded = True

    def _deep_copy_default(self) -> Dict[str, Any]:
        """Create a fresh copy of the default stats structure."""
        return {'engines': {}}

    @staticmethod
    def _lang_key(languages: List[str]) -> str:
        """Create a hashable key from a list of languages."""
        return '+'.join(sorted(languages))

    @staticmethod
    def _file_type_key(file_type: str) -> str:
        """Normalize file type key."""
        return file_type.lower()

    @staticmethod
    def _page_count_range(page_count: int) -> str:
        """
        Bucket page count into ranges for stats tracking.
        """
        if page_count <= 1:
            return '1'
        elif page_count <= 3:
            return '2-3'
        elif page_count <= 5:
            return '4-5'
        elif page_count <= 10:
            return '6-10'
        elif page_count <= 20:
            return '11-20'
        elif page_count <= 50:
            return '21-50'
        else:
            return '50+'

    def _get_combo(self, lang_key: str, file_type_key: str, page_count_range: str) -> Dict[str, Any]:
        """
        Get or create the stats entry for a specific combination of
        languages, file type, and page count range.
        """
        engines = self.stats.setdefault('engines', {})

        combo_key = f"{lang_key}|{file_type_key}|{page_count_range}"
        combo = engines.setdefault(combo_key, {})
        return combo

    def _get_engine_stats(self, combo: Dict[str, Any], engine: str) -> Dict[str, Any]:
        """
        Get or create the stats entry for a specific engine within a combo.
        """
        return combo.setdefault(engine, {
            'total_confidence': 0.0,
            'successful_runs': 0,
            'failed_runs': 0,
            'total_time': 0.0,
            'last_used': None,
        })

    def get_score(self, engine: str, languages: List[str], file_type: str,
                  page_count: int = 1) -> float:
        """
        Compute the UCB-style score for an engine given the input context.

        Args:
            engine: Engine name (e.g., 'easyocr', 'paddleocr', 'tesseract')
            languages: List of language codes
            file_type: File type (e.g., 'image', 'pdf')
            page_count: Page count for PDFs (ignored for images)

        Returns:
            Score (higher is better)
        """
        lang_key = self._lang_key(languages)
        file_type_key = self._file_type_key(file_type)
        page_range = self._page_count_range(page_count)
        combo = self._get_combo(lang_key, file_type_key, page_range)
        engine_stats = self._get_engine_stats(combo, engine)

        total_runs = engine_stats['successful_runs'] + engine_stats['failed_runs']

        if total_runs == 0:
            # Untried engine: pure exploration bonus
            all_runs = self._total_runs(combo)
            if all_runs == 0:
                return EXPLORATION_WEIGHT * math.sqrt(1.0)  # ln(0+1)=0, sqrt(0)=0, but we add 1
            return EXPLORATION_WEIGHT * math.sqrt(math.log(all_runs + 1))

        if engine_stats['successful_runs'] == 0:
            # Engine has only failed runs: low confidence, but still explore
            avg_confidence = 0.0
        else:
            avg_confidence = engine_stats['total_confidence'] / engine_stats['successful_runs']
        all_runs = self._total_runs(combo)
        exploitation = avg_confidence
        exploration = EXPLORATION_WEIGHT * math.sqrt(max(0, math.log(all_runs + 1) / (total_runs + 1)))
        return exploitation + exploration

    @staticmethod
    def _total_runs(combo: Dict[str, Any]) -> int:
        """Total runs across all engines in a combo."""
        total = 0
        for engine_stats in combo.values():
            if isinstance(engine_stats, dict):
                total += engine_stats.get('successful_runs', 0) + engine_stats.get('failed_runs', 0)
        return total

    def select_engine(self, available_engines: List[str], languages: List[str],
                      file_type: str, page_count: int = 1,
                      forced_engine: Optional[str] = None,
                      first_run_order: Optional[List[str]] = None) -> List[str]:
        """
        Select the best engine for the given context, ranked by score.

        Args:
            available_engines: List of engine names that are available
            languages: List of language codes
            file_type: File type ('image' or 'pdf')
            page_count: Page count for PDFs
            forced_engine: If provided, forces selection of this engine (bypasses scoring)
            first_run_order: Default order when no stats exist (default: easyocr, paddleocr, tesseract)

        Returns:
            List of engine names ranked by preference (highest score first)
        """
        if first_run_order is None:
            first_run_order = ['easyocr', 'paddleocr', 'tesseract']

        if forced_engine is not None:
            # User forces an engine; put it first, then the rest by score
            remaining = [e for e in available_engines if e != forced_engine]
            remaining = self._rank_engines(remaining, languages, file_type, page_count)
            return [forced_engine] + remaining

        # Check if there are any stats at all (first run)
        lang_key = self._lang_key(languages)
        file_type_key = self._file_type_key(file_type)
        page_range = self._page_count_range(page_count)
        combo_key = f"{lang_key}|{file_type_key}|{page_range}"

        engines = self.stats.get('engines', {})
        if combo_key not in engines or not engines.get(combo_key):
            # First run: use default order, filtered by availability
            result = [e for e in first_run_order if e in available_engines]
            # Add any remaining available engines not in the default order
            for e in available_engines:
                if e not in result:
                    result.append(e)
            return result

        return self._rank_engines(available_engines, languages, file_type, page_count)

    def _rank_engines(self, engines: List[str], languages: List[str],
                      file_type: str, page_count: int) -> List[str]:
        """
        Rank engines by score (descending).
        """
        scores = []
        for engine in engines:
            score = self.get_score(engine, languages, file_type, page_count)
            scores.append((engine, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scores]

    def record_result(self, engine: str, languages: List[str], file_type: str,
                      page_count: int, success: bool, confidence: float,
                      processing_time: float):
        """
        Record the result of an OCR extraction attempt.

        Args:
            engine: Engine name that was used
            languages: List of language codes
            file_type: File type ('image' or 'pdf')
            page_count: Page count for PDFs
            success: Whether the extraction succeeded
            confidence: The confidence value to record
            processing_time: Time taken in seconds
        """
        lang_key = self._lang_key(languages)
        file_type_key = self._file_type_key(file_type)
        page_range = self._page_count_range(page_count)
        combo = self._get_combo(lang_key, file_type_key, page_range)
        engine_stats = self._get_engine_stats(combo, engine)

        if success:
            engine_stats['successful_runs'] += 1
            engine_stats['total_confidence'] += confidence
            engine_stats['total_time'] += processing_time
        else:
            engine_stats['failed_runs'] += 1

        engine_stats['last_used'] = datetime.utcnow().isoformat() + 'Z'
        self._dirty = True

    def flush(self):
        """
        Persist stats to disk atomically.

        Writes to a temp file first, then uses os.replace() for atomic rename.
        """
        if not self._dirty:
            return

        self._ensure_stats_dir()

        try:
            tmp_path = str(self.stats_path) + '.tmp'
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, default=str)
            os.replace(tmp_path, str(self.stats_path))
            self._dirty = False
        except (OSError, TypeError, ValueError) as e:
            logger.warning(f"Failed to persist engine stats: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error persisting engine stats: {e}")

    def clear(self):
        """Reset all stats to defaults."""
        self.stats = self._deep_copy_default()
        self._dirty = True
        self.flush()
