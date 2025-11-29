#!/usr/bin/env python3
"""
Memory Management Module

This module provides memory usage monitoring and management utilities
for the OCR application, including memory limits, cleanup, and graceful
degradation for memory-constrained environments.
"""

import logging
import psutil
import gc
import os
from typing import Optional, Callable, Any
from contextlib import contextmanager
from functools import wraps

try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False
    logging.warning("resource module not available. Memory limits may not work on Windows.")

from .exceptions import MemoryError

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Memory management utility for OCR operations.

    Provides memory monitoring, limits, and cleanup functionality.
    """

    def __init__(self, memory_limit_mb: Optional[int] = None):
        """
        Initialize memory manager.

        Args:
            memory_limit_mb: Memory limit in MB, None for no limit
        """
        self.memory_limit_mb = memory_limit_mb
        self.process = psutil.Process(os.getpid())

    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in MB
        """
        try:
            return self.process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0

    def check_memory_limit(self, operation: str = "operation") -> None:
        """
        Check if memory usage exceeds limit.

        Args:
            operation: Description of current operation

        Raises:
            MemoryError: If memory limit exceeded
        """
        if self.memory_limit_mb is None:
            return

        current_usage = self.get_memory_usage()
        if current_usage > self.memory_limit_mb:
            raise MemoryError(
                operation=operation,
                memory_used=int(current_usage),
                memory_limit=self.memory_limit_mb
            )

    def force_gc(self) -> None:
        """Force garbage collection to free memory."""
        gc.collect()

    @contextmanager
    def memory_context(self, operation: str = "operation"):
        """
        Context manager for memory-monitored operations.

        Args:
            operation: Description of operation for error messages

        Raises:
            MemoryError: If memory limit exceeded during operation
        """
        try:
            self.check_memory_limit(f"start of {operation}")
            yield
        finally:
            self.force_gc()
            self.check_memory_limit(f"end of {operation}")


def memory_limited(memory_limit_mb: Optional[int] = None):
    """
    Decorator to limit memory usage for functions.

    Args:
        memory_limit_mb: Memory limit in MB

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            manager = MemoryManager(memory_limit_mb)
            with manager.memory_context(func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def temporary_memory_limit(limit_mb: int):
    """
    Temporarily set memory limit for a block of code.

    Args:
        limit_mb: Memory limit in MB
    """
    if not RESOURCE_AVAILABLE:
        logger.warning("Memory limits not supported on this platform")
        yield
        return

    try:
        # Get current limits
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)

        # Set new limit (convert MB to bytes)
        limit_bytes = limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))

        yield

    except Exception as e:
        logger.warning(f"Failed to set memory limit: {e}")
        yield
    finally:
        # Restore original limits
        try:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        except Exception as e:
            logger.error(f"Failed to restore memory limits: {e}")


def get_optimal_batch_size(image_count: int, available_memory_mb: Optional[int] = None) -> int:
    """
    Calculate optimal batch size based on available memory.

    Args:
        image_count: Total number of images to process
        available_memory_mb: Available memory in MB

    Returns:
        Optimal batch size
    """
    if available_memory_mb is None:
        # Conservative default
        return min(image_count, 10)

    # Estimate memory per image (rough estimate)
    memory_per_image_mb = 50  # Conservative estimate

    max_batch_size = available_memory_mb // memory_per_image_mb
    return max(1, min(image_count, max_batch_size))


def cleanup_resources():
    """Clean up system resources."""
    try:
        gc.collect()
        # Force cleanup of any cached objects
        import sys
        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()
    except Exception as e:
        logger.warning(f"Resource cleanup failed: {e}")


class ResourceManager:
    """
    Context manager for proper resource cleanup.
    """

    def __init__(self, cleanup_on_exit: bool = True):
        self.cleanup_on_exit = cleanup_on_exit
        self.resources = []

    def add_resource(self, resource, cleanup_func: Optional[Callable] = None):
        """
        Add a resource to be managed.

        Args:
            resource: The resource object
            cleanup_func: Function to call for cleanup, if None uses resource.close()
        """
        self.resources.append((resource, cleanup_func))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_on_exit:
            self.cleanup()

    def cleanup(self):
        """Clean up all managed resources."""
        for resource, cleanup_func in self.resources:
            try:
                if cleanup_func:
                    cleanup_func(resource)
                elif hasattr(resource, 'close'):
                    resource.close()
                elif hasattr(resource, '__del__'):
                    resource.__del__()
            except Exception as e:
                logger.warning(f"Failed to cleanup resource {resource}: {e}")

        self.resources.clear()
        cleanup_resources()