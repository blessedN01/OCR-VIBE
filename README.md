# OCR Text Extractor

A powerful, standalone OCR (Optical Character Recognition) application that extracts text from images and PDF documents. Built with Python and multiple OCR engines, featuring both GUI and CLI interfaces for maximum flexibility.

## Features

- **Multi-format Support**: Process images (PNG, JPG, JPEG, TIFF, BMP, GIF) and PDF documents
- **Multiple OCR Engines**: EasyOCR, PaddleOCR, and Tesseract with automatic engine selection
- **Dual Interface**: Choose between intuitive GUI or powerful command-line interface
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Batch Processing**: Process multiple files with queue management
- **Multiple Export Formats**: Save results as TXT, JSON, CSV, Markdown, DOCX, or HTML
- **Drag & Drop**: Easy file handling in GUI mode
- **Multi-language Support**: 80+ languages supported via EasyOCR and PaddleOCR
- **PDF Intelligence**: Auto-detects text vs image-based PDFs for optimal processing
- **Table Extraction**: Extract tables from PDFs using camelot-py
- **Searchable PDF**: Generate searchable PDFs with embedded OCR text layer
- **Layout Preservation**: DBSCAN-based layout detection for complex documents
- **Plugin System**: Extensible architecture for custom OCR engines
- **Dark Theme**: Modern dark theme in GUI mode
- **Configuration**: YAML/JSON config files with layered merge strategy

## Installation

### Option 1: Standalone Executable (Recommended)

1. Download the appropriate package for your platform from the releases
2. Extract the archive
3. Run the executable:
   - **GUI Mode**: Double-click `OCR_Text_Extractor` (or `OCR_Text_Extractor.exe` on Windows)
   - **CLI Mode**: Run `OCR_Text_Extractor_CLI` (or `OCR_Text_Extractor_CLI.exe` on Windows)

### Option 2: From Source

```bash
# Clone or download the source code
cd OCR-VIBE

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py                    # GUI mode
python main.py --help            # CLI help
```

### Option 3: pip install

```bash
pip install ocr-vibe[all]
```

## Usage

### GUI Mode

1. Launch the application (no command-line arguments)
2. Drag and drop files onto the drop zone, or click "Browse Files"
3. Select language, confidence threshold, and OCR engine
4. View extracted text in the results area
5. Save results using "Save Results" button or copy to clipboard
6. Use Batch Queue tab for processing multiple files

### CLI Mode

```bash
# Process a single file
ocr-vibe process-file image.jpg --output ./results --format txt

# Process a directory
ocr-vibe process-dir ./images --output-dir ./results --recursive --parallel

# Batch process files listed in a text file
ocr-vibe batch-process batch_list.txt --output-dir ./results

# Process PDF with page range
ocr-vibe process-pdf document.pdf --pages 1-5,8 --output result.txt

# Extract tables from PDF
ocr-vibe process-pdf document.pdf --extract-tables --output tables.json

# Generate searchable PDF
ocr-vibe process-pdf document.pdf --searchable-pdf --output searchable.pdf

# List available engines
ocr-vibe engines

# Show help
ocr-vibe --help
```

### CLI Commands

- `process-file <file>` - Process a single file
- `process-dir <directory>` - Process all files in a directory
- `batch-process <batch_file>` - Process files listed in a text file
- `process-pdf <pdf>` - Process a PDF with advanced options
- `engines` - List available OCR engines

### CLI Options

- `-l, --languages` - OCR languages (can be specified multiple times, default: en)
- `-o, --output` / `--output-dir` - Output file/directory path
- `-c, --confidence` - Confidence threshold (0.0-1.0, default: 0.5)
- `-f, --format` - Output format (txt, json, csv, md, docx, html, default: txt)
- `-r, --recursive` - Process directories recursively (process-dir only)
- `--engine` - Force specific OCR engine (easyocr, paddleocr, tesseract)
- `--no-preprocess` - Disable image preprocessing
- `--use-gpu` - Enable GPU acceleration for OCR
- `--post-process` - Apply language-aware post-processing
- `--extract-tables` - Extract tables from PDFs
- `--preserve-layout` - Preserve text layout using DBSCAN clustering
- `--deskew` - Apply deskewing correction
- `--clahe` - Apply CLAHE contrast enhancement
- `--upscale` - Upscale images below 150 DPI
- `--remove-bg` - Remove background using morphological opening
- `--searchable-pdf` - Generate searchable PDF with embedded OCR text layer
- `--pages` - Page range to extract (e.g., 1-3,5,7-9)
- `--parallel` - Process files in parallel using multiple CPU cores
- `--memory-limit` - Memory limit in MB for processing
- `-v, --verbose` - Enable verbose (DEBUG) logging

## Configuration

OCR-VIBE supports YAML/JSON configuration files with a layered merge strategy:

1. **Project config** (`ocr-vibe.yaml`): discovered by walking up from CWD
2. **Home config** (`~/.ocr-vibe/config.yaml`): overrides project defaults
3. **CLI flags**: highest precedence

Example `ocr-vibe.yaml`:

```yaml
languages:
  - en
  - fr
confidence_threshold: 0.6
output_format: json
preprocess_images: true
use_gpu: false
post_process: true
extract_tables: false
preserve_layout: true
deskew: true
clahe: true
```

## Supported Languages

The application supports 80+ languages including:
English, French, German, Spanish, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, and many more.

Use `ocr-vibe engines` to see the available OCR engines.

## Output Formats

- **TXT**: Plain text file with extracted content
- **JSON**: Structured data with metadata and per-word confidence
- **CSV**: Tabular format with bounding box coordinates
- **Markdown**: Formatted text with metadata header
- **DOCX**: Editable Word document (requires python-docx)
- **HTML**: Annotated HTML with confidence-colored bounding boxes

## Plugin System

OCR-VIBE supports custom OCR engine plugins. Create a Python file in `~/.ocr-vibe/engines/`:

```python
from src.ocr_engine import BaseOCREngine, EngineRegistry

class MyCustomEngine(BaseOCREngine):
    @property
    def name(self) -> str:
        return "my_engine"

    @property
    def supported_languages(self) -> List[str]:
        return ['en']

    def initialize(self, languages, use_gpu=False, **kwargs):
        return True

    def extract_text(self, image):
        return "extracted text", []

def register(registry):
    registry.register("my_engine", MyCustomEngine())
```

## System Requirements

- **Python**: 3.9+ (for source installation)
- **Memory**: 4GB+ recommended for large PDFs
- **Storage**: 2GB+ for model downloads on first run

## Building from Source

### Prerequisites

```bash
# Install PyInstaller for packaging
pip install pyinstaller

# For PDF processing (Linux)
sudo apt-get install poppler-utils

# For PDF processing (macOS)
brew install poppler

# For PDF processing (Windows)
# Download and install poppler from https://blog.alivate.com.au/poppler-windows/
```

### Build Commands

```bash
# Make build script executable
chmod +x build.sh build_macos.sh build_linux.sh

# Run build script (platform-specific)
./build.sh          # Windows
./build_macos.sh    # macOS
./build_linux.sh    # Linux
```

This creates a standalone executable in the `dist/` directory.

## Troubleshooting

### Common Issues

1. **"OCR engines not available"**
   - Wait for the application to download models on first run
   - Ensure internet connection for model downloads
   - Check that easyocr, paddleocr, or pytesseract is installed

2. **"File not found" errors**
   - Check file paths and permissions
   - Use absolute paths for CLI operations
   - Ensure supported file formats (.png, .jpg, .jpeg, .bmp, .tiff, .tif, .pdf)

3. **Poor OCR accuracy**
   - Try different confidence thresholds (default: 0.5)
   - Ensure good image quality (300+ DPI recommended)
   - Use preprocessing options (deskew, CLAHE, upscale)
   - Try different OCR engines

4. **PDF processing issues**
   - For image-based PDFs, ensure poppler-utils is installed
   - Check that pdf2image and pdfplumber are available
   - Try different PDF extraction methods (automatic detection)

### Performance Tips

- Use GPU acceleration if available (`--use-gpu` flag)
- Process directories recursively for bulk operations
- Use parallel processing (`--parallel`) for multiple files
- Close other applications to free up memory
- Use appropriate image resolutions (not too large)
- For large PDFs, ensure sufficient RAM (4GB+ recommended)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style guidelines, and contribution process.

## License

This project is open source. See LICENSE file for details.

## Changelog

### v2.0.0
- Added PaddleOCR engine integration with dynamic engine selection
- Added PDF table extraction using camelot-py
- Added layout preservation with DBSCAN clustering
- Added searchable PDF generation
- Added enhanced preprocessing (deskew, CLAHE, upscale, background removal)
- Added DOCX and HTML output formats
- Added plugin system for custom OCR engines
- Added batch queue processing in GUI
- Added language selector and confidence slider in GUI
- Added preprocessing preview in GUI
- Added dark theme support
- Added configuration file support (YAML/JSON)
- Added parallel batch processing
- Added CLI progress bars with tqdm
- Added language-aware post-processing
- Added Markdown output format
- Added PDF page range selection
- Added clipboard copy in GUI
- Added memory monitor in status bar
- Added CI/CD with GitHub Actions
- Added comprehensive test suite
- Added cross-platform build scripts

### v1.1.0
- Added configuration file support
- Added parallel batch processing
- Added CLI progress bars
- Added language-aware post-processing
- Added Markdown output format
- Added PDF page range selection
- Added clipboard copy in GUI

### v1.0.0
- Initial release
- GUI and CLI interfaces
- Multi-format support (images + PDFs)
- Cross-platform compatibility
- Directory and batch processing capabilities
- Multiple export formats (TXT, JSON, CSV)
- Drag & drop functionality in GUI
- Multi-language OCR support (80+ languages)
- Automatic PDF type detection (text vs image-based)
- GPU acceleration support
- Image preprocessing for better accuracy
