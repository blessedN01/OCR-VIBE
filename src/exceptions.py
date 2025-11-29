#!/usr/bin/env python3
"""
OCR Exceptions Module

This module defines custom exception classes for OCR-related errors,
providing standardized error handling across the OCR application.
"""

from typing import Optional, List


class OCRError(Exception):
    """Base exception class for OCR-related errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class OCREngineNotAvailableError(OCRError):
    """Raised when no OCR engines are available."""

    def __init__(self, engines_tried: Optional[List[str]] = None):
        message = "No OCR engines available for processing."
        details = {
            'engines_tried': engines_tried or [],
            'suggestion': "Install at least one OCR engine: pip install easyocr or pip install pytesseract"
        }
        super().__init__(message, details)


class OCREngineInitializationError(OCRError):
    """Raised when OCR engine fails to initialize."""

    def __init__(self, engine_name: str, original_error: Optional[Exception] = None):
        message = f"Failed to initialize {engine_name} OCR engine."
        details = {
            'engine': engine_name,
            'original_error': str(original_error) if original_error else None,
            'suggestion': f"Check {engine_name} installation and dependencies."
        }
        super().__init__(message, details)


class OCRProcessingError(OCRError):
    """Raised when OCR processing fails."""

    def __init__(self, file_path: Optional[str] = None, engine_name: Optional[str] = None,
                 original_error: Optional[Exception] = None):
        message = "OCR processing failed."
        if file_path:
            message += f" File: {file_path}"
        if engine_name:
            message += f" Engine: {engine_name}"

        details = {
            'file_path': file_path,
            'engine': engine_name,
            'original_error': str(original_error) if original_error else None
        }
        super().__init__(message, details)


class ImageProcessingError(OCRError):
    """Raised when image preprocessing fails."""

    def __init__(self, file_path: Optional[str] = None, operation: Optional[str] = None,
                 original_error: Optional[Exception] = None):
        message = "Image processing failed."
        if operation:
            message += f" Operation: {operation}"
        if file_path:
            message += f" File: {file_path}"

        details = {
            'file_path': file_path,
            'operation': operation,
            'original_error': str(original_error) if original_error else None,
            'suggestion': "Ensure image file is valid and supported format (PNG, JPG, BMP, TIFF)."
        }
        super().__init__(message, details)


class PDFProcessingError(OCRError):
    """Raised when PDF processing fails."""

    def __init__(self, file_path: Optional[str] = None, operation: Optional[str] = None,
                 original_error: Optional[Exception] = None):
        message = "PDF processing failed."
        if operation:
            message += f" Operation: {operation}"
        if file_path:
            message += f" File: {file_path}"

        details = {
            'file_path': file_path,
            'operation': operation,
            'original_error': str(original_error) if original_error else None,
            'suggestion': "Ensure PDF file is not corrupted and is accessible."
        }
        super().__init__(message, details)


class MemoryError(OCRError):
    """Raised when memory limits are exceeded."""

    def __init__(self, operation: str, memory_used: Optional[int] = None,
                 memory_limit: Optional[int] = None):
        message = f"Memory limit exceeded during {operation}."

        details = {
            'operation': operation,
            'memory_used': memory_used,
            'memory_limit': memory_limit,
            'suggestion': "Try processing smaller files or increase memory limits."
        }
        super().__init__(message, details)


class ResourceCleanupError(OCRError):
    """Raised when resource cleanup fails."""

    def __init__(self, resource_type: str, original_error: Optional[Exception] = None):
        message = f"Failed to cleanup {resource_type} resources."

        details = {
            'resource_type': resource_type,
            'original_error': str(original_error) if original_error else None,
            'suggestion': "Check system resources and permissions."
        }
        super().__init__(message, details)