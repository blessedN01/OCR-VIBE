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
import multiprocessing
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import sys

try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False
    print("ERROR: Click is required but not installed. Install with: pip install click")
    sys.exit(1)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .ocr_engine import OCREngine, get_available_engines
from .pdf_handler import PDFHandler
from .config_loader import load_config, apply_cli_overrides, get_config_parser
from .exceptions import (
    OCRError, OCREngineNotAvailableError, OCREngineInitializationError,
    OCRProcessingError, ImageProcessingError, PDFProcessingError, MemoryError,
    OCREngineFallbackError
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Version information
__version__ = "2.0.0"


def parse_page_ranges(page_spec: str) -> List[int]:
    """
    Parse a page range string like '1-3,5,7-9' into a list of 1-based page numbers.

    Args:
        page_spec: String specifying page ranges

    Returns:
        List of page numbers (1-based)

    Raises:
        ValueError: If page specification is invalid
    """
    pages = []
    for part in page_spec.split(','):
        part = part.strip()
        if '-' in part:
            bounds = part.split('-')
            if len(bounds) != 2:
                raise ValueError(f"Invalid page range: {part}")
            start, end = int(bounds[0]), int(bounds[1])
            if start > end:
                raise ValueError(f"Invalid page range: {part} (start > end)")
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return pages


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

        output_lines = []
        output_lines.append("text,confidence,x1,y1,x2,y2,x3,y3")

        for result in results:
            # Handle both dict and tuple formats
            if isinstance(result, dict):
                text_part = str(result.get('text', '')).replace('"', '""')
                confidence = result.get('confidence', 0.0)
                bbox = result.get('bbox', [])
                if bbox and len(bbox) >= 4:
                    x1, y1 = bbox[0]
                    x2, y2 = bbox[1]
                    x3, y3 = bbox[2]
                    output_lines.append(f'"{text_part}",{confidence},{x1},{y1},{x2},{y2},{x3},{y3}')
                else:
                    output_lines.append(f'"{text_part}",{confidence},,,,,,')
            elif len(result) >= 3:
                coords = result[0]
                if len(coords) == 4:
                    x1, y1 = coords[0]
                    x2, y2 = coords[1]
                    x3, y3 = coords[2]
                    x4, y4 = coords[3]
                    text_part = str(result[1]).replace('"', '""')
                    confidence = result[2]
                    output_lines.append(f'"{text_part}",{confidence},{x1},{y1},{x2},{y2},{x3},{y3}')
                else:
                    text_part = str(result[1]).replace('"', '""')
                    confidence = result[2]
                    output_lines.append(f'"{text_part}",{confidence},,,,,,')
            else:
                continue

        return "\n".join(output_lines)

    @staticmethod
    def format_md(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Format as Markdown with metadata header."""
        lines = []
        lines.append("# OCR Results\n")
        if metadata:
            lines.append("## Metadata\n")
            lines.append(f"- **File**: {metadata.get('file_path', 'Unknown')}")
            lines.append(f"- **Languages**: {', '.join(metadata.get('languages', ['en']))}")
            lines.append(f"- **Confidence Threshold**: {metadata.get('confidence_threshold', 0.5)}")
            lines.append(f"- **Output Format**: {metadata.get('output_format', 'txt')}")
            if metadata.get('file_type') == 'PDF':
                lines.append(f"- **Page Count**: {metadata.get('page_count', 'Unknown')}")
                lines.append(f"- **PDF Type**: {metadata.get('type', 'Unknown')}")
            lines.append("")
        lines.append("## Extracted Text\n")
        lines.append("```")
        lines.append(text)
        lines.append("```")
        return "\n".join(lines)

    @staticmethod
    def format_docx(text: str, metadata: Optional[Dict[str, Any]] = None, output_path: Optional[Path] = None) -> str:
        """Format as DOCX (Word document)."""
        try:
            from docx import Document
            from docx.shared import Pt, Inches
            DOCX_AVAILABLE = True
        except ImportError:
            DOCX_AVAILABLE = False

        if not DOCX_AVAILABLE:
            raise ImportError("python-docx is not installed. Install with: pip install python-docx")

        doc = Document()

        if metadata:
            doc.add_heading('OCR Results', level=1)
            doc.add_heading('Metadata', level=2)
            for key, value in metadata.items():
                doc.add_paragraph(f"{key}: {value}")
            doc.add_heading('Extracted Text', level=2)

        doc.add_paragraph(text)

        if output_path:
            doc.save(str(output_path))
            return str(output_path)
        return ""

    @staticmethod
    def format_html(text: str, results: List[Dict], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Format as annotated HTML with confidence-colored bounding box overlays."""
        lines = []
        lines.append("<!DOCTYPE html>")
        lines.append("<html lang='en'>")
        lines.append("<head>")
        lines.append("<meta charset='UTF-8'>")
        lines.append("<title>OCR Results</title>")
        lines.append("<style>")
        lines.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        lines.append(".result-item { display: inline-block; margin: 2px; padding: 2px 4px; border-radius: 3px; }")
        lines.append(".high-conf { background-color: #d4edda; color: #155724; }")
        lines.append(".medium-conf { background-color: #fff3cd; color: #856404; }")
        lines.append(".low-conf { background-color: #f8d7da; color: #721c24; }")
        lines.append("</style>")
        lines.append("</head>")
        lines.append("<body>")
        lines.append("<h1>OCR Results</h1>")

        if metadata:
            lines.append("<h2>Metadata</h2>")
            lines.append("<ul>")
            for key, value in metadata.items():
                lines.append(f"<li><strong>{key}</strong>: {value}</li>")
            lines.append("</ul>")

        lines.append("<h2>Extracted Text</h2>")
        lines.append("<div class='ocr-results'>")

        if results:
            for result in results:
                confidence = result.get('confidence', 0.0)
                text_part = result.get('text', '')
                if confidence >= 0.7:
                    css_class = 'high-conf'
                elif confidence >= 0.4:
                    css_class = 'medium-conf'
                else:
                    css_class = 'low-conf'
                lines.append(f"<span class='result-item {css_class}' title='Confidence: {confidence:.2f}'>{text_part}</span>")
        else:
            lines.append(f"<p>{text}</p>")

        lines.append("</div>")
        lines.append("</body>")
        lines.append("</html>")
        return "\n".join(lines)


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


def _create_engine_from_config(config: dict) -> OCREngine:
    """
    Create an OCREngine instance from a config dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        OCREngine instance
    """
    return OCREngine(
        languages=config.get('languages', ['en']),
        confidence_threshold=config.get('confidence_threshold', 0.5),
        use_gpu=config.get('use_gpu', False),
        preprocess_images=config.get('preprocess_images', True),
        memory_limit_mb=config.get('memory_limit_mb'),
        deskew=config.get('deskew', False),
        clahe=config.get('clahe', False),
        upscale=config.get('upscale', False),
        remove_bg=config.get('remove_bg', False),
        preserve_layout=config.get('preserve_layout', False),
        auto_select=config.get('auto_select', True),
        stats_path=config.get('stats_path'),
    )


def _process_single_file_task(args: Tuple[Path, dict, Optional[str]]) -> Tuple[Path, bool]:
    """
    Process a single file. Designed to be used with multiprocessing.Pool.

    Args:
        args: Tuple of (file_path, config_dict, engine)

    Returns:
        Tuple of (file_path, success)
    """
    file_path, config, engine = args

    try:
        ocr_engine = _create_engine_from_config(config)
        pdf_handler = PDFHandler(ocr_engine=ocr_engine)
        output_format = config.get('output_format', 'txt')

        metadata = {
            'file_path': str(file_path),
            'languages': ocr_engine.get_supported_languages(),
            'confidence_threshold': ocr_engine.confidence_threshold,
            'output_format': output_format,
        }

        if file_path.suffix.lower() in ['.pdf']:
            text = pdf_handler.extract_text(file_path, engine)
            results = []
            metadata['file_type'] = 'PDF'
        else:
            text, results = ocr_engine.extract_text(file_path, engine,
                                                    file_type='image', page_count=1)
            metadata['file_type'] = 'Image'

        if output_format == 'txt':
            content = OutputFormatter.format_txt(text, metadata)
        elif output_format == 'json':
            content = OutputFormatter.format_json(text, results, metadata)
        elif output_format == 'csv':
            content = OutputFormatter.format_csv(text, results, metadata)
        elif output_format == 'md':
            content = OutputFormatter.format_md(text, metadata)
        elif output_format == 'docx':
            if config.get('_output_file'):
                OutputFormatter.format_docx(text, metadata, config.get('_output_file'))
                return (file_path, True)
            else:
                raise ValueError("DOCX format requires an output file path (-o)")
        elif output_format == 'html':
            content = OutputFormatter.format_html(text, results, metadata)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        # Post-process if requested
        if config.get('post_process', False):
            text = ocr_engine.post_process_text(text, config.get('languages', ['en']))
            if output_format == 'md':
                content = OutputFormatter.format_md(text, metadata)
            elif output_format == 'txt':
                content = OutputFormatter.format_txt(text, metadata)

        # Table extraction for PDFs
        if config.get('extract_tables', False) and file_path.suffix.lower() == '.pdf':
            try:
                tables = pdf_handler.extract_tables(file_path, pages=config.get('pages'))
                table_output = _format_tables(tables, config.get('output_format', 'txt'))
                if table_output:
                    content = content + "\n\n" + table_output
            except Exception as e:
                logger.warning(f"Table extraction failed for {file_path}: {e}")

        # Searchable PDF generation
        if config.get('searchable_pdf', False) and file_path.suffix.lower() == '.pdf':
            try:
                search_output = config.get('_searchable_output')
                search_path = pdf_handler.generate_searchable_pdf(
                    file_path, ocr_engine,
                    output_path=search_output
                )
                logger.info(f"Searchable PDF generated: {search_path}")
            except Exception as e:
                logger.warning(f"Searchable PDF generation failed for {file_path}: {e}")

        output_file = config.get('_output_file')
        if output_file:
            save_output(content, output_file, output_format)

        ocr_engine.cleanup()
        return (file_path, True)

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return (file_path, False)


def _format_tables(tables: list, output_format: str) -> str:
    """
    Format extracted tables as text.

    Args:
        tables: List of camelot Table objects
        output_format: Output format ('txt', 'json', 'csv')

    Returns:
        Formatted table text
    """
    if not tables:
        return ""

    if output_format in ('json', 'csv'):
        # For JSON/CSV, export tables to their respective formats
        import io
        output_parts = []
        for i, table in enumerate(tables):
            try:
                if output_format == 'csv':
                    csv_str = table.df.to_csv(index=False)
                    output_parts.append(f"--- Table {i + 1} (CSV) ---\n{csv_str}")
                else:
                    json_str = table.df.to_json(orient='records', indent=2, force_ascii=False)
                    output_parts.append(f"--- Table {i + 1} (JSON) ---\n{json_str}")
            except Exception as e:
                logger.warning(f"Failed to format table {i + 1}: {e}")
                output_parts.append(f"--- Table {i + 1} (error) ---")
        return '\n\n'.join(output_parts)
    else:
        # Plain text output
        output_parts = []
        for i, table in enumerate(tables):
            try:
                text = table.df.to_string(index=False)
                output_parts.append(f"--- Table {i + 1} ---\n{text}")
            except Exception as e:
                logger.warning(f"Failed to format table {i + 1}: {e}")
        return '\n\n'.join(output_parts)


def process_single_file(file_path: Path, ocr_engine: OCREngine, pdf_handler: PDFHandler,
                        output_format: str, output_path: Optional[Path] = None,
                        engine: Optional[str] = None, post_process: bool = False,
                        extract_tables: bool = False, preserve_layout: bool = False,
                        searchable_pdf: bool = False,
                        pages: Optional[List[int]] = None) -> bool:
    """Process a single file (image or PDF) with improved error handling."""
    try:
        is_pdf = file_path.suffix.lower() in ['.pdf']

        metadata = {
            'file_path': str(file_path),
            'languages': ocr_engine.get_supported_languages(),
            'confidence_threshold': ocr_engine.confidence_threshold,
            'output_format': output_format,
        }

        if is_pdf:
            if pages:
                page_texts = pdf_handler.extract_text_by_page(file_path, engine, pages)
                text = '\n\n'.join(page_texts)
                page_count = len(page_texts)
            else:
                text = pdf_handler.extract_text(file_path, engine)
                page_count = pdf_handler.get_page_count(file_path)
            results = []
            metadata['file_type'] = 'PDF'
            metadata['page_count'] = page_count
        else:
            text, results = ocr_engine.extract_text(file_path, engine,
                                                    file_type='image', page_count=1)
            metadata['file_type'] = 'Image'

        if post_process:
            text = ocr_engine.post_process_text(text, ocr_engine.get_supported_languages())

        if output_format == 'txt':
            content = OutputFormatter.format_txt(text, metadata)
        elif output_format == 'json':
            content = OutputFormatter.format_json(text, results, metadata)
        elif output_format == 'csv':
            content = OutputFormatter.format_csv(text, results, metadata)
        elif output_format == 'md':
            content = OutputFormatter.format_md(text, metadata)
        elif output_format == 'docx':
            if output_path:
                OutputFormatter.format_docx(text, metadata, output_path)
                return True
            else:
                raise ValueError("DOCX format requires an output file path (-o)")
        elif output_format == 'html':
            content = OutputFormatter.format_html(text, results, metadata)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        # Table extraction for PDFs
        if extract_tables and is_pdf:
            try:
                tables = pdf_handler.extract_tables(file_path)
                table_output = _format_tables(tables, output_format)
                if table_output:
                    content = content + "\n\n" + table_output
            except Exception as e:
                logger.warning(f"Table extraction failed for {file_path}: {e}")

        # Searchable PDF generation
        if searchable_pdf and is_pdf:
            try:
                search_output = output_path.parent / f"{file_path.stem}_searchable.pdf" if output_path else None
                search_path = pdf_handler.generate_searchable_pdf(file_path, ocr_engine,
                                                                     output_path=search_output)
                logger.info(f"Searchable PDF generated: {search_path}")
            except Exception as e:
                logger.warning(f"Searchable PDF generation failed for {file_path}: {e}")

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
                      recursive: bool = False, engine: Optional[str] = None,
                      parallel: bool = False, post_process: bool = False,
                      extract_tables: bool = False, preserve_layout: bool = False,
                      searchable_pdf: bool = False,
                      pages: Optional[List[int]] = None) -> int:
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

    if parallel and len(files) > 1:
        cpu_count = multiprocessing.cpu_count()
        try:
            import psutil
            available_memory_mb = psutil.virtual_memory().available // (1024 * 1024)
            max_workers = min(cpu_count, available_memory_mb // 500)
        except Exception:
            max_workers = cpu_count

        max_workers = max(1, max_workers)
        logger.info(f"Processing {len(files)} files in parallel with {max_workers} workers")

        config = {
            'languages': ocr_engine.get_supported_languages(),
            'confidence_threshold': ocr_engine.confidence_threshold,
            'use_gpu': ocr_engine.use_gpu,
            'preprocess_images': ocr_engine.preprocess_images,
            'deskew': ocr_engine.deskew,
            'clahe': ocr_engine.clahe,
            'upscale': ocr_engine.upscale,
            'remove_bg': ocr_engine.remove_bg,
            'preserve_layout': ocr_engine.preserve_layout,
            'memory_limit_mb': ocr_engine.memory_limit_mb,
            'output_format': output_format,
            'post_process': post_process,
            'extract_tables': extract_tables,
            'searchable_pdf': searchable_pdf,
        }

        if TQDM_AVAILABLE:
            files_iter = tqdm(files, desc="Processing", unit="file")
        else:
            files_iter = files

        tasks = []
        for file_path in files:
            task_config = config.copy()
            if output_dir:
                relative_path = file_path.relative_to(directory_path)
                task_config['_output_file'] = output_dir / relative_path.with_suffix(f'.{output_format}')
            else:
                task_config['_output_file'] = None
            if searchable_pdf and file_path.suffix.lower() == '.pdf':
                if output_dir:
                    task_config['_searchable_output'] = output_dir / f"{file_path.stem}_searchable.pdf"
                else:
                    task_config['_searchable_output'] = None
            tasks.append((file_path, task_config, engine))

        with multiprocessing.Pool(processes=max_workers) as pool:
            if TQDM_AVAILABLE:
                results = list(tqdm(
                    pool.imap(_process_single_file_task, tasks),
                    total=len(tasks),
                    desc="Processing",
                    unit="file"
                ))
            else:
                results = pool.map(_process_single_file_task, tasks)

        successful = sum(1 for _, success in results if success)
    else:
        if TQDM_AVAILABLE:
            files_iter = tqdm(files, desc="Processing", unit="file")
        else:
            files_iter = files

        for file_path in files_iter:
            logger.info(f"Processing {file_path}")

            if output_dir:
                relative_path = file_path.relative_to(directory_path)
                output_file = output_dir / relative_path.with_suffix(f'.{output_format}')
            else:
                output_file = None

            if process_single_file(file_path, ocr_engine, pdf_handler, output_format,
                                  output_file, engine, post_process,
                                  extract_tables, preserve_layout, searchable_pdf, pages):
                successful += 1

    logger.info(f"Processed {successful}/{len(files)} files successfully")
    return successful


# Click CLI Definition

@click.group()
@click.version_option(__version__)
@click.option('--verbose', '-v', is_flag=True, default=False,
              help='Enable verbose (DEBUG) logging')
@click.pass_context
def cli(ctx, verbose):
    """OCR Text Extractor - Extract text from images and PDFs via command line."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)


def _get_ocr_engine(config: dict) -> OCREngine:
    """Create an OCREngine from a config dict (shared by all CLI commands)."""
    return OCREngine(
        languages=config['languages'],
        confidence_threshold=config['confidence_threshold'],
        use_gpu=config['use_gpu'],
        preprocess_images=config['preprocess_images'],
        memory_limit_mb=config['memory_limit_mb'],
        deskew=config.get('deskew', False),
        clahe=config.get('clahe', False),
        upscale=config.get('upscale', False),
        remove_bg=config.get('remove_bg', False),
        preserve_layout=config.get('preserve_layout', False),
        auto_select=config.get('auto_select', True),
    )


@cli.command()
@click.argument('file_path', type=click.Path(exists=True, path_type=Path))
@click.option('--languages', '-l', multiple=True, default=None,
              help='OCR languages (can be specified multiple times)')
@click.option('--confidence', '-c', type=float, default=None,
              help='Confidence threshold (0.0-1.0)')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output file path')
@click.option('--format', '-f', type=click.Choice(['txt', 'json', 'csv', 'md', 'docx', 'html']), default=None,
              help='Output format')
@click.option('--engine', type=click.Choice(['easyocr', 'paddleocr', 'tesseract']), default=None,
              help='Force specific OCR engine')
@click.option('--no-preprocess', is_flag=True, default=None,
              help='Disable image preprocessing')
@click.option('--use-gpu', is_flag=True, default=None,
              help='Enable GPU acceleration for OCR')
@click.option('--post-process', is_flag=True, default=None,
              help='Apply language-aware post-processing to extracted text')
@click.option('--memory-limit', type=int, default=None,
              help='Memory limit in MB for processing')
@click.option('--extract-tables', is_flag=True, default=None,
              help='Extract tables from PDFs (requires camelot-py[cv])')
@click.option('--preserve-layout', is_flag=True, default=None,
              help='Preserve text layout using DBSCAN clustering')
@click.option('--deskew', is_flag=True, default=None,
              help='Apply deskewing correction')
@click.option('--clahe', is_flag=True, default=None,
              help='Apply CLAHE contrast enhancement')
@click.option('--upscale', is_flag=True, default=None,
              help='Upscale images below 150 DPI')
@click.option('--remove-bg', is_flag=True, default=None,
              help='Remove background using morphological opening')
@click.option('--searchable-pdf', is_flag=True, default=None,
              help='Generate searchable PDF with embedded OCR text layer')
def process_file(file_path, languages, confidence, output, format, engine, no_preprocess, use_gpu, post_process,
                 memory_limit, extract_tables, preserve_layout, deskew, clahe, upscale, remove_bg, searchable_pdf):
    """Process a single image or PDF file."""
    config = get_config_parser()

    cli_overrides = {
        'languages': list(languages) if languages else None,
        'confidence_threshold': confidence,
        'output_format': format,
        'engine': engine,
        'preprocess_images': not no_preprocess if no_preprocess is not None else None,
        'use_gpu': use_gpu,
        'memory_limit_mb': memory_limit,
        'post_process': post_process,
        'extract_tables': extract_tables,
        'preserve_layout': preserve_layout,
        'deskew': deskew,
        'clahe': clahe,
        'upscale': upscale,
        'remove_bg': remove_bg,
        'searchable_pdf': searchable_pdf,
    }
    config = apply_cli_overrides(config, cli_overrides)

    ocr_engine = _get_ocr_engine(config)
    pdf_handler = PDFHandler(ocr_engine=ocr_engine)

    success = process_single_file(file_path, ocr_engine, pdf_handler, config['output_format'],
                                  output, config['engine'], config.get('post_process', False),
                                  config.get('extract_tables', False),
                                  config.get('preserve_layout', False),
                                  config.get('searchable_pdf', False))
    if not success:
        sys.exit(1)


@cli.command()
@click.argument('directory_path', type=click.Path(exists=True, path_type=Path))
@click.option('--languages', '-l', multiple=True, default=None,
              help='OCR languages (can be specified multiple times)')
@click.option('--confidence', '-c', type=float, default=None,
              help='Confidence threshold (0.0-1.0)')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path),
              help='Output directory for processed files')
@click.option('--format', '-f', type=click.Choice(['txt', 'json', 'csv', 'md', 'docx', 'html']), default=None,
              help='Output format')
@click.option('--recursive', '-r', is_flag=True, default=False,
              help='Process directories recursively')
@click.option('--engine', type=click.Choice(['easyocr', 'paddleocr', 'tesseract']), default=None,
              help='Force specific OCR engine')
@click.option('--no-preprocess', is_flag=True, default=None,
              help='Disable image preprocessing')
@click.option('--use-gpu', is_flag=True, default=None,
              help='Enable GPU acceleration for OCR')
@click.option('--parallel', '-p', is_flag=True, default=False,
              help='Process files in parallel using multiple CPU cores')
@click.option('--post-process', is_flag=True, default=None,
              help='Apply language-aware post-processing to extracted text')
@click.option('--memory-limit', type=int, default=None,
              help='Memory limit in MB for processing')
@click.option('--extract-tables', is_flag=True, default=None,
              help='Extract tables from PDFs (requires camelot-py[cv])')
@click.option('--preserve-layout', is_flag=True, default=None,
              help='Preserve text layout using DBSCAN clustering')
@click.option('--deskew', is_flag=True, default=None,
              help='Apply deskewing correction')
@click.option('--clahe', is_flag=True, default=None,
              help='Apply CLAHE contrast enhancement')
@click.option('--upscale', is_flag=True, default=None,
              help='Upscale images below 150 DPI')
@click.option('--remove-bg', is_flag=True, default=None,
              help='Remove background using morphological opening')
@click.option('--searchable-pdf', is_flag=True, default=None,
              help='Generate searchable PDF with embedded OCR text layer')
@click.option('--pages', default=None,
              help='Page range to extract (e.g., 1-3,5,7-9)')
def process_dir(directory_path, languages, confidence, output_dir, format, recursive, engine,
                no_preprocess, use_gpu, parallel, post_process, memory_limit,
                extract_tables, preserve_layout, deskew, clahe, upscale, remove_bg,
                searchable_pdf, pages):
    """Process all supported files in a directory."""
    if not directory_path.is_dir():
        click.echo(f"Error: {directory_path} is not a directory", err=True)
        sys.exit(1)

    config = get_config_parser()

    cli_overrides = {
        'languages': list(languages) if languages else None,
        'confidence_threshold': confidence,
        'output_format': format,
        'engine': engine,
        'preprocess_images': not no_preprocess if no_preprocess is not None else None,
        'use_gpu': use_gpu,
        'memory_limit_mb': memory_limit,
        'post_process': post_process,
        'extract_tables': extract_tables,
        'preserve_layout': preserve_layout,
        'deskew': deskew,
        'clahe': clahe,
        'upscale': upscale,
        'remove_bg': remove_bg,
        'searchable_pdf': searchable_pdf,
    }
    config = apply_cli_overrides(config, cli_overrides)

    ocr_engine = _get_ocr_engine(config)
    pdf_handler = PDFHandler(ocr_engine=ocr_engine)

    page_numbers = None
    if pages:
        try:
            page_numbers = parse_page_ranges(pages)
        except ValueError as e:
            click.echo(f"Invalid page range: {e}", err=True)
            sys.exit(1)

    successful = process_directory(directory_path, ocr_engine, pdf_handler, config['output_format'],
                                  output_dir, recursive, config['engine'], parallel,
                                  config.get('post_process', False),
                                  config.get('extract_tables', False),
                                  config.get('preserve_layout', False),
                                  config.get('searchable_pdf', False),
                                  page_numbers)
    if successful == 0:
        sys.exit(1)


@cli.command()
@click.argument('batch_file', type=click.Path(exists=True, path_type=Path))
@click.option('--languages', '-l', multiple=True, default=None,
              help='OCR languages (can be specified multiple times)')
@click.option('--confidence', '-c', type=float, default=None,
              help='Confidence threshold (0.0-1.0)')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path),
              help='Output directory for processed files')
@click.option('--format', '-f', type=click.Choice(['txt', 'json', 'csv', 'md', 'docx', 'html']), default=None,
              help='Output format')
@click.option('--engine', type=click.Choice(['easyocr', 'paddleocr', 'tesseract']), default=None,
              help='Force specific OCR engine')
@click.option('--no-preprocess', is_flag=True, default=None,
              help='Disable image preprocessing')
@click.option('--use-gpu', is_flag=True, default=None,
              help='Enable GPU acceleration for OCR')
@click.option('--parallel', '-p', is_flag=True, default=False,
              help='Process files in parallel using multiple CPU cores')
@click.option('--post-process', is_flag=True, default=None,
              help='Apply language-aware post-processing to extracted text')
@click.option('--memory-limit', type=int, default=None,
              help='Memory limit in MB for processing')
@click.option('--extract-tables', is_flag=True, default=None,
              help='Extract tables from PDFs (requires camelot-py[cv])')
@click.option('--preserve-layout', is_flag=True, default=None,
              help='Preserve text layout using DBSCAN clustering')
@click.option('--deskew', is_flag=True, default=None,
              help='Apply deskewing correction')
@click.option('--clahe', is_flag=True, default=None,
              help='Apply CLAHE contrast enhancement')
@click.option('--upscale', is_flag=True, default=None,
              help='Upscale images below 150 DPI')
@click.option('--remove-bg', is_flag=True, default=None,
              help='Remove background using morphological opening')
@click.option('--searchable-pdf', is_flag=True, default=None,
              help='Generate searchable PDF with embedded OCR text layer')
def batch_process(batch_file, languages, confidence, output_dir, format, engine,
                  no_preprocess, use_gpu, parallel, post_process, memory_limit,
                  extract_tables, preserve_layout, deskew, clahe, upscale, remove_bg,
                  searchable_pdf):
    """Process files listed in a batch file (one file path per line)."""
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

    valid_files = [fp for fp in file_paths if fp.exists()]
    if len(valid_files) != len(file_paths):
        invalid = len(file_paths) - len(valid_files)
        logger.warning(f"{invalid} files from batch file do not exist")

    if not valid_files:
        click.echo("No valid files found in batch file", err=True)
        sys.exit(1)

    config = get_config_parser()

    cli_overrides = {
        'languages': list(languages) if languages else None,
        'confidence_threshold': confidence,
        'output_format': format,
        'engine': engine,
        'preprocess_images': not no_preprocess if no_preprocess is not None else None,
        'use_gpu': use_gpu,
        'memory_limit_mb': memory_limit,
        'post_process': post_process,
        'extract_tables': extract_tables,
        'preserve_layout': preserve_layout,
        'deskew': deskew,
        'clahe': clahe,
        'upscale': upscale,
        'remove_bg': remove_bg,
        'searchable_pdf': searchable_pdf,
    }
    config = apply_cli_overrides(config, cli_overrides)

    ocr_engine = _get_ocr_engine(config)
    pdf_handler = PDFHandler(ocr_engine=ocr_engine)

    successful = 0
    if TQDM_AVAILABLE:
        files_iter = tqdm(valid_files, desc="Processing", unit="file")
    else:
        files_iter = valid_files

    for file_path in files_iter:
        logger.info(f"Processing {file_path}")

        if output_dir:
            output_file = output_dir / f"{file_path.stem}.{config['output_format']}"
        else:
            output_file = None

        search_output = None
        if config.get('searchable_pdf', False) and file_path.suffix.lower() == '.pdf':
            if output_dir:
                search_output = output_dir / f"{file_path.stem}_searchable.pdf"

        if process_single_file(file_path, ocr_engine, pdf_handler, config['output_format'],
                              output_file, config['engine'], config.get('post_process', False),
                              config.get('extract_tables', False),
                              config.get('preserve_layout', False),
                              config.get('searchable_pdf', False)):
            successful += 1

    logger.info(f"Processed {successful}/{len(valid_files)} files successfully")
    if successful == 0:
        sys.exit(1)


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True, path_type=Path))
@click.option('--languages', '-l', multiple=True, default=None,
              help='OCR languages (can be specified multiple times)')
@click.option('--confidence', '-c', type=float, default=None,
              help='Confidence threshold (0.0-1.0)')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output file path')
@click.option('--format', '-f', type=click.Choice(['txt', 'json', 'csv', 'md', 'docx', 'html']), default=None,
              help='Output format')
@click.option('--engine', type=click.Choice(['easyocr', 'paddleocr', 'tesseract']), default=None,
              help='Force specific OCR engine')
@click.option('--pages', '-p', default=None,
              help='Page range to extract (e.g., 1-3,5,7-9)')
@click.option('--no-preprocess', is_flag=True, default=None,
              help='Disable image preprocessing')
@click.option('--use-gpu', is_flag=True, default=None,
              help='Enable GPU acceleration for OCR')
@click.option('--post-process', is_flag=True, default=None,
              help='Apply language-aware post-processing to extracted text')
@click.option('--extract-tables', is_flag=True, default=None,
              help='Extract tables from PDF (requires camelot-py[cv])')
@click.option('--searchable-pdf', is_flag=True, default=None,
              help='Generate searchable PDF with embedded OCR text layer')
@click.option('--preserve-layout', is_flag=True, default=None,
              help='Preserve text layout using DBSCAN clustering')
@click.option('--deskew', is_flag=True, default=None,
              help='Apply deskewing correction')
@click.option('--clahe', is_flag=True, default=None,
              help='Apply CLAHE contrast enhancement')
@click.option('--upscale', is_flag=True, default=None,
              help='Upscale images below 150 DPI')
@click.option('--remove-bg', is_flag=True, default=None,
              help='Remove background using morphological opening')
def process_pdf(pdf_path, languages, confidence, output, format, engine, pages,
                no_preprocess, use_gpu, post_process, extract_tables, searchable_pdf,
                preserve_layout, deskew, clahe, upscale, remove_bg):
    """Process a PDF file with optional page range selection."""
    config = get_config_parser()

    cli_overrides = {
        'languages': list(languages) if languages else None,
        'confidence_threshold': confidence,
        'output_format': format,
        'engine': engine,
        'preprocess_images': not no_preprocess if no_preprocess is not None else None,
        'use_gpu': use_gpu,
        'post_process': post_process,
        'extract_tables': extract_tables,
        'searchable_pdf': searchable_pdf,
        'preserve_layout': preserve_layout,
        'deskew': deskew,
        'clahe': clahe,
        'upscale': upscale,
        'remove_bg': remove_bg,
    }
    config = apply_cli_overrides(config, cli_overrides)

    ocr_engine = _get_ocr_engine(config)
    pdf_handler = PDFHandler(ocr_engine=ocr_engine)

    page_numbers = None
    if pages:
        try:
            page_numbers = parse_page_ranges(pages)
        except ValueError as e:
            click.echo(f"Invalid page range: {e}", err=True)
            sys.exit(1)

    try:
        if page_numbers:
            page_texts = pdf_handler.extract_text_by_page(pdf_path, engine, page_numbers)
            text = '\n\n'.join(page_texts)
            results = []
        else:
            text = pdf_handler.extract_text(pdf_path, engine)
            results = []

        if config.get('post_process', False):
            text = ocr_engine.post_process_text(text, config['languages'])

        metadata = {
            'file_path': str(pdf_path),
            'file_type': 'PDF',
            'page_count': pdf_handler.get_page_count(pdf_path),
            'type': pdf_handler.detect_pdf_type(pdf_path),
            'languages': config['languages'],
            'confidence_threshold': config['confidence_threshold'],
            'output_format': config['output_format'],
        }

        if config['output_format'] == 'txt':
            content = OutputFormatter.format_txt(text, metadata)
        elif config['output_format'] == 'json':
            content = OutputFormatter.format_json(text, results, metadata)
        elif config['output_format'] == 'csv':
            content = OutputFormatter.format_csv(text, results, metadata)
        elif config['output_format'] == 'md':
            content = OutputFormatter.format_md(text, metadata)
        elif config['output_format'] == 'docx':
            if output:
                OutputFormatter.format_docx(text, metadata, output)
                return True
            else:
                raise ValueError("DOCX format requires an output file path (-o)")
        elif config['output_format'] == 'html':
            content = OutputFormatter.format_html(text, results, metadata)
        else:
            raise ValueError(f"Unsupported output format: {config['output_format']}")

        # Table extraction for PDFs
        if config.get('extract_tables', False):
            try:
                tables = pdf_handler.extract_tables(pdf_path)
                table_output = _format_tables(tables, config['output_format'])
                if table_output:
                    content = content + "\n\n" + table_output
            except Exception as e:
                logger.warning(f"Table extraction failed for {pdf_path}: {e}")

        # Searchable PDF generation
        if config.get('searchable_pdf', False):
            try:
                search_output = output.parent / f"{pdf_path.stem}_searchable.pdf" if output else None
                search_path = pdf_handler.generate_searchable_pdf(pdf_path, ocr_engine,
                                                                   output_path=search_output)
                click.echo(f"Searchable PDF generated: {search_path}")
            except Exception as e:
                click.echo(f"Warning: Searchable PDF generation failed: {e}", err=True)

        save_output(content, output, config['output_format'])

    except PDFProcessingError as e:
        click.echo(f"PDF processing failed: {e}", err=True)
        if e.details and 'suggestion' in e.details:
            click.echo(f"Suggestion: {e.details['suggestion']}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error processing PDF: {e}", err=True)
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
