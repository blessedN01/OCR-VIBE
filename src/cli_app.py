#!/usr/bin/env python3
"""
CLI Application Module

This module provides a command-line interface for the OCR application using Click.
It supports processing single files, directories, and batch operations with various
output formats and configuration options.
"""

import json
import csv
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import sys

try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False
    print("ERROR: Click is required but not installed. Install with: pip install click")
    sys.exit(1)

from .ocr_engine import OCREngine, get_available_engines
from .pdf_handler import PDFHandler
from .exceptions import (
    OCRError, OCREngineNotAvailableError, OCRProcessingError,
    ImageProcessingError, PDFProcessingError, MemoryError
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"


class OutputFormatter:
    """Handles different output formats for OCR results."""

    @staticmethod
    def format_txt(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Format as plain text."""
        if metadata:
            header = f"# OCR Results\n# File: {metadata.get('file_path', 'Unknown')}\n"
            header += f"# Languages: {', '.join(metadata.get('languages', ['en']))}\n"
            header += f"# Confidence Threshold: {metadata.get('confidence_threshold', 0.5)}\n\n"
            return header + text
        return text

    @staticmethod
    def format_json(text: str, results: List[Dict], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Format as JSON."""
        output = {
            "text": text,
            "results": results,
            "metadata": metadata or {}
        }
        return json.dumps(output, indent=2, ensure_ascii=False)

    @staticmethod
    def format_csv(text: str, results: List[Dict], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Format as CSV."""
        if not results:
            return "No results found"

        # Create CSV output
        output_lines = []
        output_lines.append("text,confidence,x1,y1,x2,y2,x3,y3")

        for result in results:
            if len(result) >= 3:  # EasyOCR format: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, confidence
                coords = result[0]
                if len(coords) == 4:
                    x1, y1 = coords[0]
                    x2, y2 = coords[1]
                    x3, y3 = coords[2]
                    x4, y4 = coords[3]
                    text_part = result[1].replace('"', '""')  # Escape quotes
                    confidence = result[2]
                    output_lines.append(f'"{text_part}",{confidence},{x1},{y1},{x2},{y2},{x3},{y3}')
                else:
                    # Fallback for other formats
                    text_part = str(result.get('text', '')).replace('"', '""')
                    confidence = result.get('confidence', 0.0)
                    output_lines.append(f'"{text_part}",{confidence},,,,,')

        return "\n".join(output_lines)


def save_output(content: str, output_path: Optional[Path], output_format: str):
    """Save content to file or print to stdout."""
    if output_path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Results saved to {output_path}")
        except PermissionError as e:
            logger.error(f"Permission denied writing to {output_path}. Please check directory permissions.")
            raise
        except OSError as e:
            logger.error(f"OS error writing to {output_path}: {e}. This may be due to disk space or file system issues.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving to {output_path}: {e}")
            raise
    else:
        print(content)


def process_single_file(file_path: Path, ocr_engine: OCREngine, pdf_handler: PDFHandler,
                       output_format: str, output_path: Optional[Path] = None,
                       engine: Optional[str] = None) -> bool:
    """Process a single file (image or PDF) with improved error handling."""
    try:
        metadata = {
            'file_path': str(file_path),
            'languages': ocr_engine.get_supported_languages(),
            'confidence_threshold': ocr_engine.confidence_threshold
        }

        if file_path.suffix.lower() in ['.pdf']:
            # PDF processing
            text = pdf_handler.extract_text(file_path, engine)
            results = []  # PDFs don't have detailed results like images
        else:
            # Image processing
            text, results = ocr_engine.extract_text(file_path, engine)

        # Format output
        if output_format == 'txt':
            content = OutputFormatter.format_txt(text, metadata)
        elif output_format == 'json':
            content = OutputFormatter.format_json(text, results, metadata)
        elif output_format == 'csv':
            content = OutputFormatter.format_csv(text, results, metadata)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        save_output(content, output_path, output_format)
        return True

    except OCREngineNotAvailableError as e:
        logger.error(f"No OCR engines available: {e}")
        if e.details and 'suggestion' in e.details:
            logger.error(f"Suggestion: {e.details['suggestion']}")
        return False
    except OCRProcessingError as e:
        logger.error(f"OCR processing failed for {file_path}: {e}")
        if e.details and 'suggestion' in e.details:
            logger.error(f"Suggestion: {e.details['suggestion']}")
        return False
    except ImageProcessingError as e:
        logger.error(f"Image processing failed for {file_path}: {e}")
        if e.details and 'suggestion' in e.details:
            logger.error(f"Suggestion: {e.details['suggestion']}")
        return False
    except PDFProcessingError as e:
        logger.error(f"PDF processing failed for {file_path}: {e}")
        if e.details and 'suggestion' in e.details:
            logger.error(f"Suggestion: {e.details['suggestion']}")
        return False
    except MemoryError as e:
        logger.error(f"Memory limit exceeded processing {file_path}: {e}")
        if e.details and 'suggestion' in e.details:
            logger.error(f"Suggestion: {e.details['suggestion']}")
        return False
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}. Please check the file path and ensure it exists.")
        return False
    except PermissionError as e:
        logger.error(f"Permission denied accessing {file_path}. Please check file permissions.")
        return False
    except IsADirectoryError as e:
        logger.error(f"Path {file_path} is a directory, not a file. Please specify a file path.")
        return False
    except OSError as e:
        logger.error(f"OS error processing {file_path}: {e}. This may be due to file system issues or unsupported file format.")
        return False
    except ValueError as e:
        logger.error(f"Invalid file format or corrupted file: {file_path}. Error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing {file_path}: {e}")
        logger.error("This may be due to missing dependencies or corrupted installation.")
        return False


def process_directory(directory_path: Path, ocr_engine: OCREngine, pdf_handler: PDFHandler,
                     output_format: str, output_dir: Optional[Path] = None,
                     recursive: bool = False, engine: Optional[str] = None) -> int:
    """Process all supported files in a directory."""
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.pdf'}

    if recursive:
        files = [f for f in directory_path.rglob('*') if f.is_file() and f.suffix.lower() in supported_extensions]
    else:
        files = [f for f in directory_path.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]

    if not files:
        logger.warning(f"No supported files found in {directory_path}")
        return 0

    successful = 0
    for file_path in files:
        logger.info(f"Processing {file_path}")

        # Determine output path
        if output_dir:
            relative_path = file_path.relative_to(directory_path)
            output_file = output_dir / relative_path.with_suffix(f'.{output_format}')
        else:
            output_file = None

        if process_single_file(file_path, ocr_engine, pdf_handler, output_format, output_file, engine):
            successful += 1

    logger.info(f"Processed {successful}/{len(files)} files successfully")
    return successful


# Click CLI Definition

@click.group()
@click.version_option(__version__)
def cli():
    """OCR Text Extractor - Extract text from images and PDFs via command line."""
    pass


@cli.command()
@click.argument('file_path', type=click.Path(exists=True, path_type=Path))
@click.option('--languages', '-l', multiple=True, default=['en'],
              help='OCR languages (can be specified multiple times)')
@click.option('--confidence', '-c', type=float, default=0.5,
              help='Confidence threshold (0.0-1.0)')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output file path')
@click.option('--format', '-f', type=click.Choice(['txt', 'json', 'csv']), default='txt',
              help='Output format')
@click.option('--engine', type=click.Choice(['easyocr', 'tesseract']),
              help='Force specific OCR engine')
@click.option('--no-preprocess', is_flag=True,
              help='Disable image preprocessing')
@click.option('--use-gpu', is_flag=True,
              help='Enable GPU acceleration for OCR')
def process_file(file_path, languages, confidence, output, format, engine, no_preprocess, use_gpu):
    """Process a single image or PDF file."""
    # Initialize engines
    ocr_engine = OCREngine(languages=list(languages), confidence_threshold=confidence,
                          use_gpu=use_gpu, preprocess_images=not no_preprocess)
    pdf_handler = PDFHandler(ocr_engine=ocr_engine)

    success = process_single_file(file_path, ocr_engine, pdf_handler, format, output)
    if not success:
        sys.exit(1)


@cli.command()
@click.argument('directory_path', type=click.Path(exists=True, path_type=Path))
@click.option('--languages', '-l', multiple=True, default=['en'],
              help='OCR languages (can be specified multiple times)')
@click.option('--confidence', '-c', type=float, default=0.5,
              help='Confidence threshold (0.0-1.0)')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path),
              help='Output directory for processed files')
@click.option('--format', '-f', type=click.Choice(['txt', 'json', 'csv']), default='txt',
              help='Output format')
@click.option('--recursive', '-r', is_flag=True,
              help='Process directories recursively')
@click.option('--engine', type=click.Choice(['easyocr', 'tesseract']),
              help='Force specific OCR engine')
@click.option('--no-preprocess', is_flag=True,
              help='Disable image preprocessing')
@click.option('--use-gpu', is_flag=True,
              help='Enable GPU acceleration for OCR')
def process_dir(directory_path, languages, confidence, output_dir, format, recursive, engine, no_preprocess, use_gpu):
    """Process all supported files in a directory."""
    if not directory_path.is_dir():
        click.echo(f"Error: {directory_path} is not a directory", err=True)
        sys.exit(1)

    # Initialize engines
    ocr_engine = OCREngine(languages=list(languages), confidence_threshold=confidence,
                          use_gpu=use_gpu, preprocess_images=not no_preprocess)
    pdf_handler = PDFHandler(ocr_engine=ocr_engine)

    successful = process_directory(directory_path, ocr_engine, pdf_handler, format,
                                 output_dir, recursive, engine)
    if successful == 0:
        sys.exit(1)


@cli.command()
@click.argument('batch_file', type=click.Path(exists=True, path_type=Path))
@click.option('--languages', '-l', multiple=True, default=['en'],
              help='OCR languages (can be specified multiple times)')
@click.option('--confidence', '-c', type=float, default=0.5,
              help='Confidence threshold (0.0-1.0)')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path),
              help='Output directory for processed files')
@click.option('--format', '-f', type=click.Choice(['txt', 'json', 'csv']), default='txt',
              help='Output format')
@click.option('--engine', type=click.Choice(['easyocr', 'tesseract']),
              help='Force specific OCR engine')
@click.option('--no-preprocess', is_flag=True,
              help='Disable image preprocessing')
@click.option('--use-gpu', is_flag=True,
              help='Enable GPU acceleration for OCR')
def batch_process(batch_file, languages, confidence, output_dir, format, engine, no_preprocess, use_gpu):
    """Process files listed in a batch file (one file path per line)."""
    # Read batch file
    try:
        with open(batch_file, 'r', encoding='utf-8') as f:
            file_paths = [Path(line.strip()) for line in f if line.strip()]
    except FileNotFoundError as e:
        click.echo(f"Batch file not found: {batch_file}. Please check the file path.", err=True)
        sys.exit(1)
    except PermissionError as e:
        click.echo(f"Permission denied reading batch file: {batch_file}. Please check file permissions.", err=True)
        sys.exit(1)
    except UnicodeDecodeError as e:
        click.echo(f"Batch file encoding error: {batch_file}. Ensure the file is UTF-8 encoded text.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error reading batch file {batch_file}: {e}", err=True)
        sys.exit(1)

    # Filter valid files
    valid_files = [fp for fp in file_paths if fp.exists()]
    if len(valid_files) != len(file_paths):
        invalid = len(file_paths) - len(valid_files)
        logger.warning(f"{invalid} files from batch file do not exist")

    if not valid_files:
        click.echo("No valid files found in batch file", err=True)
        sys.exit(1)

    # Initialize engines
    ocr_engine = OCREngine(languages=list(languages), confidence_threshold=confidence,
                          use_gpu=use_gpu, preprocess_images=not no_preprocess)
    pdf_handler = PDFHandler(ocr_engine=ocr_engine)

    # Process each file
    successful = 0
    for file_path in valid_files:
        logger.info(f"Processing {file_path}")

        # Determine output path
        if output_dir:
            output_file = output_dir / f"{file_path.stem}.{format}"
        else:
            output_file = None

        if process_single_file(file_path, ocr_engine, pdf_handler, format, output_file, engine):
            successful += 1

    logger.info(f"Processed {successful}/{len(valid_files)} files successfully")
    if successful == 0:
        sys.exit(1)


@cli.command()
def engines():
    """List available OCR engines."""
    available = get_available_engines()
    if available:
        click.echo("Available OCR engines:")
        for engine in available:
            click.echo(f"  - {engine}")
    else:
        click.echo("No OCR engines available. Install easyocr or pytesseract.")
        sys.exit(1)


if __name__ == '__main__':
    cli()