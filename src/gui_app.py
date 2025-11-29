#!/usr/bin/env python3
"""
GUI Application Module

This module provides a graphical user interface for the OCR application using tkinter
with drag-and-drop functionality via tkinterdnd2. It includes a main window with file
drop zone, results display area, save functionality, and progress indicators.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import threading
import queue
import os

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    TKINTERDND_AVAILABLE = True
except ImportError:
    TKINTERDND_AVAILABLE = False
    logging.error("tkinterdnd2 not available. Drag-and-drop functionality will be disabled. Install with: pip install tkinterdnd2")

from .ocr_engine import OCREngine
from .pdf_handler import PDFHandler
from .exceptions import (
    OCRError, OCREngineNotAvailableError, OCRProcessingError,
    ImageProcessingError, PDFProcessingError, MemoryError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRGUIApp:
    """
    Main GUI application class for OCR processing.

    Features:
    - Drag-and-drop file support
    - Progress indication during processing
    - Results display with save functionality
    - Error handling and status updates
    - Support for images and PDFs
    """

    def __init__(self, root: tk.Tk):
        """
        Initialize the GUI application.

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("OCR Text Extractor")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)

        # Initialize OCR components
        self.ocr_engine = OCREngine()
        self.pdf_handler = PDFHandler(ocr_engine=self.ocr_engine)

        # Processing queue for thread communication
        self.processing_queue = queue.Queue()

        # Current processing state
        self.current_file = None
        self.processing = False

        # Setup UI
        self.setup_ui()

        # Setup drag and drop if available
        if TKINTERDND_AVAILABLE:
            self.setup_drag_drop()

        # Start queue processing
        self.root.after(100, self.process_queue)

    def setup_ui(self):
        """Setup the main user interface."""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="OCR Text Extractor",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))

        # Drop zone frame
        drop_frame = ttk.LabelFrame(main_frame, text="Drop Files Here", padding="10")
        drop_frame.pack(fill=tk.X, pady=(0, 10))

        self.drop_label = ttk.Label(drop_frame,
                                   text="Drag and drop image files (PNG, JPG, BMP, TIFF) or PDF files here\n"
                                        "Or click 'Browse Files' to select files",
                                   justify=tk.CENTER,
                                   background="#f0f0f0",
                                   relief="sunken")
        self.drop_label.pack(fill=tk.X, expand=True, pady=20, padx=20)

        # Control buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        self.browse_button = ttk.Button(button_frame, text="Browse Files",
                                       command=self.browse_files)
        self.browse_button.pack(side=tk.LEFT, padx=(0, 10))

        self.clear_button = ttk.Button(button_frame, text="Clear Results",
                                      command=self.clear_results, state=tk.DISABLED)
        self.clear_button.pack(side=tk.LEFT, padx=(0, 10))

        self.save_button = ttk.Button(button_frame, text="Save Results",
                                     command=self.save_results, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var,
                                           maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.pack(anchor=tk.W, pady=(0, 10))

        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Extracted Text", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD,
                                                     font=("Courier", 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # File info label
        self.file_info_label = ttk.Label(results_frame, text="")
        self.file_info_label.pack(anchor=tk.W, pady=(5, 0))

    def setup_drag_drop(self):
        """Setup drag and drop functionality."""
        if not TKINTERDND_AVAILABLE:
            return

        # Register drop target
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.on_drop)

        # Change cursor on hover
        self.drop_label.bind('<Enter>', lambda e: self.drop_label.config(cursor="hand2"))
        self.drop_label.bind('<Leave>', lambda e: self.drop_label.config(cursor=""))

    def on_drop(self, event):
        """Handle file drop event."""
        files = self.root.splitlist(event.data)
        if files:
            file_path = Path(files[0])  # Take first file
            self.process_file(file_path)

    def browse_files(self):
        """Open file browser dialog."""
        filetypes = [
            ('All supported files', '*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.pdf'),
            ('Image files', '*.png *.jpg *.jpeg *.bmp *.tiff *.tif'),
            ('PDF files', '*.pdf'),
            ('All files', '*.*')
        ]

        filename = filedialog.askopenfilename(
            title="Select file to process",
            filetypes=filetypes
        )

        if filename:
            self.process_file(Path(filename))

    def process_file(self, file_path: Path):
        """Process a single file in a separate thread."""
        if self.processing:
            messagebox.showwarning("Processing", "Please wait for current processing to complete.")
            return

        try:
            if not file_path.exists():
                messagebox.showerror("File Error", f"File not found: {file_path}\n\nPlease check the file path and ensure the file exists.")
                return
        except OSError as e:
            messagebox.showerror("File Access Error", f"Cannot access file {file_path}: {e}\n\nThis may be due to permission issues or file system problems.")
            return

        # Check file extension
        supported_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.pdf'}
        if file_path.suffix.lower() not in supported_exts:
            messagebox.showerror("Unsupported File",
                                f"Unsupported file type: {file_path.suffix}\n\n"
                                "Supported formats: PNG, JPG, JPEG, BMP, TIFF, TIF, PDF\n\n"
                                "Please select a file with one of these extensions.")
            return

        # Start processing
        self.processing = True
        self.current_file = file_path
        self.status_label.config(text=f"Processing: {file_path.name}")
        self.progress_var.set(0)
        self.results_text.delete(1.0, tk.END)
        self.file_info_label.config(text="")
        self.clear_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)

        # Process in separate thread
        thread = threading.Thread(target=self._process_file_thread, args=(file_path,))
        thread.daemon = True
        thread.start()

    def _process_file_thread(self, file_path: Path):
        """Process file in background thread with improved error handling."""
        try:
            # Update progress
            self.processing_queue.put(('progress', 10))

            # Determine file type and process
            if file_path.suffix.lower() == '.pdf':
                # PDF processing
                self.processing_queue.put(('status', f"Detecting PDF type: {file_path.name}"))
                pdf_info = self.pdf_handler.get_page_count(file_path)
                self.processing_queue.put(('progress', 30))

                self.processing_queue.put(('status', f"Extracting text from {pdf_info} pages"))
                text = self.pdf_handler.extract_text(file_path)
                self.processing_queue.put(('progress', 90))

                results = []  # PDFs don't have detailed OCR results
                metadata = {
                    'file_path': str(file_path),
                    'file_type': 'PDF',
                    'page_count': pdf_info,
                    'type': self.pdf_handler.detect_pdf_type(file_path)
                }
            else:
                # Image processing
                self.processing_queue.put(('status', f"Preprocessing image: {file_path.name}"))
                self.processing_queue.put(('progress', 30))

                self.processing_queue.put(('status', "Extracting text with OCR"))
                text, results = self.ocr_engine.extract_text(file_path)
                self.processing_queue.put(('progress', 90))

                metadata = {
                    'file_path': str(file_path),
                    'file_type': 'Image',
                    'languages': self.ocr_engine.get_supported_languages(),
                    'confidence_threshold': self.ocr_engine.confidence_threshold
                }

            self.processing_queue.put(('progress', 100))
            self.processing_queue.put(('status', "Processing complete"))
            self.processing_queue.put(('results', text, results, metadata))

        except OCREngineNotAvailableError as e:
            error_msg = f"No OCR engines available: {e}"
            if e.details and 'suggestion' in e.details:
                error_msg += f"\n\nSuggestion: {e.details['suggestion']}"
            logger.error(error_msg)
            self.processing_queue.put(('error', error_msg))
        except OCRProcessingError as e:
            error_msg = f"OCR processing failed: {e}"
            if e.details and 'suggestion' in e.details:
                error_msg += f"\n\nSuggestion: {e.details['suggestion']}"
            logger.error(error_msg)
            self.processing_queue.put(('error', error_msg))
        except ImageProcessingError as e:
            error_msg = f"Image processing failed: {e}"
            if e.details and 'suggestion' in e.details:
                error_msg += f"\n\nSuggestion: {e.details['suggestion']}"
            logger.error(error_msg)
            self.processing_queue.put(('error', error_msg))
        except PDFProcessingError as e:
            error_msg = f"PDF processing failed: {e}"
            if e.details and 'suggestion' in e.details:
                error_msg += f"\n\nSuggestion: {e.details['suggestion']}"
            logger.error(error_msg)
            self.processing_queue.put(('error', error_msg))
        except MemoryError as e:
            error_msg = f"Memory limit exceeded: {e}"
            if e.details and 'suggestion' in e.details:
                error_msg += f"\n\nSuggestion: {e.details['suggestion']}"
            logger.error(error_msg)
            self.processing_queue.put(('error', error_msg))
        except FileNotFoundError as e:
            error_msg = f"File not found: {file_path}. Please check the file path and ensure it exists."
            logger.error(error_msg)
            self.processing_queue.put(('error', error_msg))
        except PermissionError as e:
            error_msg = f"Permission denied accessing {file_path}. Please check file permissions."
            logger.error(error_msg)
            self.processing_queue.put(('error', error_msg))
        except IsADirectoryError as e:
            error_msg = f"Path {file_path} is a directory, not a file. Please select a file."
            logger.error(error_msg)
            self.processing_queue.put(('error', error_msg))
        except OSError as e:
            error_msg = f"OS error processing {file_path}: {e}. This may be due to file system issues."
            logger.error(error_msg)
            self.processing_queue.put(('error', error_msg))
        except ValueError as e:
            error_msg = f"Invalid file format or corrupted file: {file_path}. Error: {e}"
            logger.error(error_msg)
            self.processing_queue.put(('error', error_msg))
        except RuntimeError as e:
            error_msg = f"OCR engine error: {e}"
            logger.error(error_msg)
            self.processing_queue.put(('error', error_msg))
        except Exception as e:
            error_msg = f"Unexpected error processing {file_path}: {e}"
            logger.error(error_msg)
            self.processing_queue.put(('error', error_msg))

    def process_queue(self):
        """Process messages from the background thread."""
        try:
            while True:
                message = self.processing_queue.get_nowait()
                msg_type = message[0]

                if msg_type == 'progress':
                    self.progress_var.set(message[1])
                elif msg_type == 'status':
                    self.status_label.config(text=message[1])
                elif msg_type == 'results':
                    text, results, metadata = message[1], message[2], message[3]
                    self.display_results(text, results, metadata)
                elif msg_type == 'error':
                    error_msg = message[1]
                    self.status_label.config(text=f"Error: {error_msg}")
                    messagebox.showerror("Processing Error", error_msg)
                    self.reset_ui()

        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.process_queue)

    def display_results(self, text: str, results: List[Dict], metadata: Dict[str, Any]):
        """Display processing results in the UI."""
        # Display text
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)

        # Display file info
        info_parts = []
        if metadata.get('file_type') == 'PDF':
            info_parts.append(f"Pages: {metadata.get('page_count', 'Unknown')}")
            info_parts.append(f"Type: {metadata.get('type', 'Unknown')}")
        else:
            info_parts.append(f"Languages: {', '.join(metadata.get('languages', ['en']))}")
            info_parts.append(f"Confidence: {metadata.get('confidence_threshold', 0.5)}")

        self.file_info_label.config(text=" | ".join(info_parts))

        # Update UI state
        self.clear_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.processing = False
        self.status_label.config(text="Ready")

    def clear_results(self):
        """Clear the results display."""
        self.results_text.delete(1.0, tk.END)
        self.file_info_label.config(text="")
        self.clear_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_label.config(text="Ready")

    def save_results(self):
        """Save the current results to a file."""
        text = self.results_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Save Results", "No results to save.")
            return

        filetypes = [
            ('Text files', '*.txt'),
            ('All files', '*.*')
        ]

        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=filetypes
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    # Add header with file info
                    if self.current_file:
                        f.write(f"# OCR Results\n")
                        f.write(f"# Original file: {self.current_file}\n")
                        f.write(f"# Processed with OCR Text Extractor\n\n")
                    f.write(text)
                messagebox.showinfo("Save Results", f"Results saved to {filename}")
            except PermissionError as e:
                messagebox.showerror("Save Error", f"Permission denied saving to {filename}.\n\nPlease check directory permissions and try a different location.")
            except OSError as e:
                messagebox.showerror("Save Error", f"OS error saving to {filename}: {e}\n\nThis may be due to disk space or file system issues.")
            except Exception as e:
                messagebox.showerror("Save Error", f"Unexpected error saving to {filename}: {e}")

    def reset_ui(self):
        """Reset UI to ready state after error."""
        self.processing = False
        self.progress_var.set(0)
        self.status_label.config(text="Ready")


def main():
    """Main entry point for the GUI application."""
    if not TKINTERDND_AVAILABLE:
        print("WARNING: tkinterdnd2 not available. Drag-and-drop functionality will be disabled.")
        print("Install with: pip install tkinterdnd2")
        root = tk.Tk()
    else:
        root = TkinterDnD.Tk()

    app = OCRGUIApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()