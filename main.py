#!/usr/bin/env python3
"""
OCR Text Extractor - Main Entry Point
Standalone OCR application for extracting text from images and PDFs
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

def main():
    """Main entry point for the OCR application"""
    try:
        # Check command line arguments
        if len(sys.argv) > 1:
            # CLI mode
            from src.cli_app import cli
            cli()
        else:
            # GUI mode
            try:
                import tkinter as tk
                from src.gui_app import OCRGUIApp
                try:
                    from tkinterdnd2 import TkinterDnD
                    root = TkinterDnD.Tk()
                except ImportError:
                    root = tk.Tk()
                app = OCRGUIApp(root)
                root.mainloop()
            except ImportError as e:
                print(f"GUI mode not available: {e}")
                print("Required dependencies for GUI mode are missing.")
                print("Install GUI dependencies with: pip install tkinterdnd2")
                print("Falling back to CLI mode. Use --help for usage.")
                from src.cli_app import cli
                cli()
            except Exception as e:
                print(f"Unexpected error starting GUI: {e}")
                print("Falling back to CLI mode. Use --help for usage.")
                from src.cli_app import cli
                cli()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Critical error starting OCR application: {e}")
        print("This may be due to missing dependencies or corrupted installation.")
        print("Check the README for installation instructions.")
        sys.exit(1)

if __name__ == "__main__":
    main()