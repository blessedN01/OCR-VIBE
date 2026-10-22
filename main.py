#!/usr/bin/env python3
"""
OCR Text Extractor - Main Entry Point
Standalone OCR application for extracting text from images and PDFs
"""

import logging
import sys
from pathlib import Path

src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))


def main():
    """Main entry point for the OCR application"""
    try:
        if len(sys.argv) > 1:
            from src.cli_app import cli
            cli()
        else:
            logging.basicConfig(level=logging.INFO)

            import flet as ft

            def main_gui(page: ft.Page):
                from src.gui_flet import FletGUIApp
                app = FletGUIApp(page)
                page.add(app.build())

            ft.run(main_gui)
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
