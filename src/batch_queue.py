#!/usr/bin/env python3
"""
Batch Queue Manager Module

Provides a queue-based batch processing system for the GUI, managing a list of files
to be processed sequentially with per-file progress bars and cancel functionality.
Integrates with the OCR engine via callbacks.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchTask:
    """Represents a single file in the batch queue."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.status = TaskStatus.PENDING
        self.progress: float = 0.0
        self.error_message: str = ""
        self.result_text: str = ""
        self.start_time: float | None = None
        self.end_time: float | None = None

    @property
    def duration(self) -> float | None:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        if self.start_time:
            return time.time() - self.start_time
        return None


class BatchQueueManager:
    """
    Manages a queue of files for batch processing in the GUI.

    Processes files sequentially with per-file progress tracking and cancel support.
    Communicates with the GUI via callbacks to update the UI without blocking.
    """

    def __init__(
        self,
        ocr_engine: Any,
        pdf_handler: Any,
        config: dict[str, Any],
        on_progress: Callable | None = None,
        on_status_change: Callable | None = None,
        on_task_complete: Callable | None = None,
        on_queue_complete: Callable | None = None,
        on_log: Callable | None = None,
    ):
        """
        Initialize the batch queue manager.

        Args:
            ocr_engine: OCREngine instance
            pdf_handler: PDFHandler instance
            config: Configuration dictionary
            on_progress: Callback(task_id, progress_percent) for progress updates
            on_status_change: Callback(task_id, status) for status changes
            on_task_complete: Callback(task_id, success, result_text) for task completion
            on_queue_complete: Callback(total, successful, failed) for queue completion
            on_log: Callback(message) for log messages
        """
        self.ocr_engine = ocr_engine
        self.pdf_handler = pdf_handler
        self.config = config

        self.on_progress = on_progress or (lambda *args: None)
        self.on_status_change = on_status_change or (lambda *args: None)
        self.on_task_complete = on_task_complete or (lambda *args: None)
        self.on_queue_complete = on_queue_complete or (lambda *args: None)
        self.on_log = on_log or (lambda *args: None)

        self._tasks: dict[str, BatchTask] = {}
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._processing_thread: threading.Thread | None = None
        self._current_task_id: str | None = None
        self._running = False

    def add_file(self, file_path: Path) -> str:
        """
        Add a file to the batch queue.

        Args:
            file_path: Path to the file to process

        Returns:
            Task ID for the added file
        """
        task_id = str(file_path.resolve())
        with self._lock:
            if task_id in self._tasks:
                return task_id
            task = BatchTask(file_path)
            self._tasks[task_id] = task
            self._queue.put(task_id)
            self.on_log(f"Added to queue: {file_path.name}")
            return task_id

    def add_files(self, file_paths: list[Path]) -> list[str]:
        """Add multiple files to the batch queue."""
        return [self.add_file(fp) for fp in file_paths]

    def remove_task(self, task_id: str) -> bool:
        """
        Remove a task from the queue if it hasn't started processing yet.

        Args:
            task_id: Task ID to remove

        Returns:
            True if removed, False if already processing or completed
        """
        with self._lock:
            if task_id not in self._tasks:
                return False
            task = self._tasks[task_id]
            if task.status == TaskStatus.PROCESSING:
                return False
            task.status = TaskStatus.CANCELLED
            self.on_status_change(task_id, TaskStatus.CANCELLED.value)
            return True

    def get_task(self, task_id: str) -> BatchTask | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[BatchTask]:
        """Get all tasks as a list."""
        return list(self._tasks.values())

    def get_pending_count(self) -> int:
        """Get number of pending tasks."""
        return sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)

    def get_completed_count(self) -> int:
        """Get number of completed tasks."""
        return sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED)

    def get_failed_count(self) -> int:
        """Get number of failed tasks."""
        return sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED)

    def is_running(self) -> bool:
        """Check if the queue is currently processing."""
        return self._running

    def start(self):
        """Start processing the batch queue."""
        if self._running:
            return
        self._running = True
        self._cancel_event.clear()
        self._processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._processing_thread.start()
        self.on_log("Batch processing started")

    def cancel(self):
        """Cancel the current batch processing."""
        self._cancel_event.set()
        self.on_log("Batch processing cancelled")

    def stop(self):
        """Stop processing immediately."""
        self._cancel_event.set()
        self._running = False
        self.on_log("Batch processing stopped")

    def clear_completed(self) -> list[str]:
        """Remove all completed and failed tasks from the queue.
        
        Returns:
            List of cleared task IDs
        """
        with self._lock:
            to_remove = [
                tid for tid, task in self._tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            ]
            for tid in to_remove:
                del self._tasks[tid]
            self.on_log(f"Cleared {len(to_remove)} completed tasks")
            return to_remove

    def clear_all(self):
        """Clear all tasks and stop processing."""
        self.stop()
        with self._lock:
            self._tasks.clear()
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self.on_log("All tasks cleared")

    def _process_queue(self):
        """Main processing loop for the batch queue."""
        total = len(self._tasks)
        successful = 0
        failed = 0

        try:
            while not self._cancel_event.is_set():
                try:
                    task_id = self._queue.get(timeout=0.5)
                except queue.Empty:
                    if self.get_pending_count() == 0:
                        break
                    continue

                with self._lock:
                    if task_id not in self._tasks:
                        continue
                    task = self._tasks[task_id]
                    if task.status != TaskStatus.PENDING:
                        continue
                    task.status = TaskStatus.PROCESSING
                    task.start_time = time.time()
                    self._current_task_id = task_id

                self.on_status_change(task_id, TaskStatus.PROCESSING.value)
                self.on_log(f"Processing: {task.file_path.name}")

                success = self._process_task(task_id)

                with self._lock:
                    task.end_time = time.time()
                    if task.status == TaskStatus.PROCESSING:
                        task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
                    if task.status == TaskStatus.COMPLETED:
                        successful += 1
                    elif task.status == TaskStatus.FAILED:
                        failed += 1

                self.on_status_change(task_id, task.status.value)
                self.on_task_complete(task_id, success, task.result_text)

                self._current_task_id = None

                if self._cancel_event.is_set():
                    break

        finally:
            self._running = False
            self._current_task_id = None
            self.on_log(f"Batch processing finished: {successful}/{total} successful")
            self.on_queue_complete(total, successful, failed)

    def _process_task(self, task_id: str) -> bool:
        """
        Process a single task from the queue.

        Args:
            task_id: Task ID to process

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

        def progress_callback(progress: float):
            task.progress = progress
            self.on_progress(task_id, progress)

        try:
            is_pdf = task.file_path.suffix.lower() == '.pdf'

            if is_pdf:
                return self._process_pdf(task, progress_callback)
            else:
                return self._process_image(task, progress_callback)

        except Exception as e:
            task.error_message = str(e)
            self.on_log(f"Error processing {task.file_path.name}: {e}")
            return False

    def _process_image(self, task: BatchTask, progress_callback: Callable[[float], None]) -> bool:
        """Process an image file."""
        progress_callback(10)

        if self._cancel_event.is_set():
            task.status = TaskStatus.CANCELLED
            return False

        text, _results = self.ocr_engine.extract_text(
            task.file_path,
            file_type='image',
            page_count=1
        )

        progress_callback(80)

        if self.config.get('post_process', False):
            text = self.ocr_engine.post_process_text(text, self.ocr_engine.get_supported_languages())

        progress_callback(100)
        task.result_text = text
        return True

    def _process_pdf(self, task: BatchTask, progress_callback: Callable[[float], None]) -> bool:
        """Process a PDF file."""
        progress_callback(10)

        if self._cancel_event.is_set():
            task.status = TaskStatus.CANCELLED
            return False

        page_count = self.pdf_handler.get_page_count(task.file_path)
        pdf_type = self.pdf_handler.detect_pdf_type(task.file_path)
        self.on_log(f"{task.file_path.name}: {pdf_type} PDF, {page_count} page(s)")

        progress_callback(30)

        text = self.pdf_handler.extract_text(task.file_path, engine=self.config.get('engine'))

        progress_callback(80)

        if self.config.get('post_process', False):
            text = self.ocr_engine.post_process_text(text, self.ocr_engine.get_supported_languages())

        if self.config.get('extract_tables', False):
            try:
                tables = self.pdf_handler.extract_tables(task.file_path)
                if tables:
                    table_texts = []
                    for i, table in enumerate(tables):
                        try:
                            t = table.df.to_string(index=False)
                            table_texts.append(f"--- Table {i + 1} ---\n{t}")
                        except Exception as e:
                            self.on_log(f"Skipped table {i + 1} of {task.file_path.name}: {e}")
                    if table_texts:
                        text = text + "\n\n" + "\n\n".join(table_texts)
            except Exception as e:
                self.on_log(f"Table extraction failed for {task.file_path.name}: {e}")

        progress_callback(100)
        task.result_text = text
        return True
