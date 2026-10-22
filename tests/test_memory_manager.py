#!/usr/bin/env python3
"""
Unit tests for Memory Manager Module.

Tests memory limit enforcement and garbage collection.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exceptions import MemoryError
from src.memory_manager import MemoryManager, ResourceManager, get_optimal_batch_size


class TestMemoryManager:
    """Tests for the MemoryManager class."""

    def test_initialization_no_limit(self):
        """Test MemoryManager initializes with no limit."""
        manager = MemoryManager()
        assert manager.memory_limit_mb is None

    def test_initialization_with_limit(self):
        """Test MemoryManager initializes with a limit."""
        manager = MemoryManager(memory_limit_mb=1024)
        assert manager.memory_limit_mb == 1024

    def test_check_memory_limit_no_limit(self):
        """Test memory limit check passes when no limit set."""
        manager = MemoryManager()
        manager.check_memory_limit("test")  # Should not raise

    def test_check_memory_limit_under_limit(self):
        """Test memory limit check passes when under limit."""
        manager = MemoryManager(memory_limit_mb=100000)
        manager.check_memory_limit("test")  # Should not raise

    def test_check_memory_limit_exceeded(self):
        """Test memory limit check raises when exceeded."""
        manager = MemoryManager(memory_limit_mb=1)
        with patch.object(manager, 'get_memory_usage', return_value=2048.0), \
             pytest.raises(MemoryError):
            manager.check_memory_limit("test")

    def test_force_gc(self):
        """Test force_gc calls garbage collection."""
        manager = MemoryManager()
        with patch('gc.collect') as mock_gc:
            manager.force_gc()
            mock_gc.assert_called_once()

    def test_memory_context(self):
        """Test memory_context context manager."""
        manager = MemoryManager(memory_limit_mb=100000)
        with manager.memory_context("test operation"):
            pass  # Should not raise

    def test_memory_context_raises_on_exceed(self):
        """Test memory_context raises when limit exceeded."""
        manager = MemoryManager(memory_limit_mb=1)
        with patch.object(manager, 'get_memory_usage', return_value=2048.0), \
             pytest.raises(MemoryError), manager.memory_context("test"):
            pass


class TestResourceManager:
    """Tests for the ResourceManager class."""

    def test_add_and_cleanup_resource(self):
        """Test adding and cleaning up a resource."""
        manager = ResourceManager()
        mock_resource = MagicMock()
        manager.add_resource(mock_resource)
        manager.cleanup()
        mock_resource.close.assert_called_once()

    def test_cleanup_with_custom_func(self):
        """Test cleanup with custom cleanup function."""
        manager = ResourceManager()
        mock_resource = MagicMock()
        cleanup_func = MagicMock()
        manager.add_resource(mock_resource, cleanup_func)
        manager.cleanup()
        cleanup_func.assert_called_once_with(mock_resource)

    def test_cleanup_handles_errors(self):
        """Test cleanup handles errors gracefully."""
        manager = ResourceManager()
        mock_resource = MagicMock()
        mock_resource.close.side_effect = Exception("Cleanup error")
        manager.add_resource(mock_resource)
        manager.cleanup()  # Should not raise


class TestGetOptimalBatchSize:
    """Tests for get_optimal_batch_size function."""

    def test_without_available_memory(self):
        """Test optimal batch size without memory info."""
        result = get_optimal_batch_size(20)
        assert result == 10  # Conservative default

    def test_with_available_memory(self):
        """Test optimal batch size with memory info."""
        result = get_optimal_batch_size(20, available_memory_mb=1000)
        assert result > 0
        assert result <= 20

    def test_small_batch(self):
        """Test optimal batch size for small batches."""
        result = get_optimal_batch_size(3, available_memory_mb=1000)
        assert result == 3

    def test_minimum_batch_size(self):
        """Test minimum batch size is 1."""
        result = get_optimal_batch_size(10, available_memory_mb=1)
        assert result == 1
