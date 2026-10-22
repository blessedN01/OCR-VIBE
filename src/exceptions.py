#!/usr/bin/env python3
"""
OCR Exceptions Module

This module defines custom exception classes for OCR-related errors,
providing standardized error handling across the OCR application.
"""
from __future__ import annotations

from typing import Any


class OCRError(Exception):
    """Base exception class for OCR-related errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class OCREngineNotAvailableError(OCRError):
    """Raised when no OCR engines are available."""

    def __init__(self, engines_tried: list[str] | None = None,
                 details: dict[str, Any] | None = None):
        message = "No OCR engines available for processing."
        if engines_tried:
            message += f" Engines tried: {', '.join(engines_tried)}"

        merged_details = {
            'engines_tried': engines_tried or [],
            'suggestion': "Install at least one OCR engine: pip install easyocr paddleocr or pip install pytesseract"
        }
        if details:
            merged_details.update(details)
        super().__init__(message, merged_details)


class OCREngineInitializationError(OCRError):
    """Raised when OCR engine fails to initialize."""

    def __init__(self, engine_name: str, original_error: Exception | None = None,
                 details: dict[str, Any] | None = None):
        message = f"Failed to initialize {engine_name} OCR engine."
        merged_details = {
            'engine': engine_name,
            'original_error': str(original_error) if original_error else None,
            'suggestion': f"Check {engine_name} installation and dependencies."
        }
        if details:
            merged_details.update(details)
        super().__init__(message, merged_details)


class OCRProcessingError(OCRError):
    """Raised when OCR processing fails."""

    def __init__(self, file_path: str | None = None, engine_name: str | None = None,
                 original_error: Exception | None = None, engines_tried: list[str] | None = None,
                 is_fatal: bool = False, details: dict[str, Any] | None = None):
        message = "OCR processing failed."
        if file_path:
            message += f" File: {file_path}"
        if engine_name:
            message += f" Engine: {engine_name}"
        if engines_tried:
            message += f" Engines tried: {', '.join(engines_tried)}"
        if original_error:
            message += f" Error: {original_error}"

        merged_details = {
            'file_path': file_path,
            'engine': engine_name,
            'original_error': str(original_error) if original_error else None,
            'engines_tried': engines_tried or [],
            'is_fatal': is_fatal,
        }
        if details:
            merged_details.update(details)
        super().__init__(message, merged_details)


class ImageProcessingError(OCRError):
    """Raised when image preprocessing fails."""

    def __init__(self, file_path: str | None = None, operation: str | None = None,
                 original_error: Exception | None = None,
                 details: dict[str, Any] | None = None):
        message = "Image processing failed."
        if operation:
            message += f" Operation: {operation}"
        if file_path:
            message += f" File: {file_path}"
        if original_error:
            message += f" Error: {original_error}"

        merged_details = {
            'file_path': file_path,
            'operation': operation,
            'original_error': str(original_error) if original_error else None,
            'suggestion': "Ensure image file is valid and supported format (PNG, JPG, BMP, TIFF)."
        }
        if details:
            merged_details.update(details)
        super().__init__(message, merged_details)


class PDFProcessingError(OCRError):
    """Raised when PDF processing fails."""

    def __init__(self, file_path: str | None = None, operation: str | None = None,
                 original_error: Exception | None = None,
                 details: dict[str, Any] | None = None):
        message = "PDF processing failed."
        if operation:
            message += f" Operation: {operation}"
        if file_path:
            message += f" File: {file_path}"
        if original_error:
            message += f" Error: {original_error}"

        merged_details = {
            'file_path': file_path,
            'operation': operation,
            'original_error': str(original_error) if original_error else None,
            'suggestion': "Ensure PDF file is not corrupted and is accessible."
        }
        if details:
            merged_details.update(details)
        super().__init__(message, merged_details)


class MemoryError(OCRError):
    """Raised when memory limits are exceeded."""

    def __init__(self, operation: str, memory_used: int | None = None,
                 memory_limit: int | None = None,
                 details: dict[str, Any] | None = None):
        message = f"Memory limit exceeded during {operation}."
        merged_details = {
            'operation': operation,
            'memory_used': memory_used,
            'memory_limit': memory_limit,
            'suggestion': "Try processing smaller files or increase memory limits."
        }
        if details:
            merged_details.update(details)
        super().__init__(message, merged_details)


class ResourceCleanupError(OCRError):
    """Raised when resource cleanup fails."""

    def __init__(self, resource_type: str, original_error: Exception | None = None,
                 details: dict[str, Any] | None = None):
        message = f"Failed to cleanup {resource_type} resources."
        merged_details = {
            'resource_type': resource_type,
            'original_error': str(original_error) if original_error else None,
            'suggestion': "Check system resources and permissions."
        }
        if details:
            merged_details.update(details)
        super().__init__(message, merged_details)


class OCREngineFallbackError(OCRError):
    """Raised when all OCR engines fail during fallback attempts."""

    def __init__(self, file_path: str | None = None, engines_tried: list[str] | None = None,
                 original_errors: dict[str, str] | None = None,
                 details: dict[str, Any] | None = None):
        message = "All OCR engines failed."
        if file_path:
            message += f" File: {file_path}"
        if engines_tried:
            message += f" Engines tried: {', '.join(engines_tried)}"

        merged_details = {
            'file_path': file_path,
            'engines_tried': engines_tried or [],
            'original_errors': original_errors or {},
            'suggestion': "Try different preprocessing options or check image quality."
        }
        if details:
            merged_details.update(details)
        super().__init__(message, merged_details)