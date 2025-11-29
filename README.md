# OCR Text Extractor

A powerful, standalone OCR (Optical Character Recognition) application that extracts text from images and PDF documents. Built with Python and EasyOCR, featuring both GUI and CLI interfaces for maximum flexibility.

## Features

- **Multi-format Support**: Process images (PNG, JPG, JPEG, TIFF, BMP, GIF) and PDF documents
- **Dual Interface**: Choose between intuitive GUI or powerful command-line interface
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Batch Processing**: Process multiple files at once
- **Multiple Export Formats**: Save results as TXT, JSON, or CSV
- **Drag & Drop**: Easy file handling in GUI mode
- **Multi-language Support**: 80+ languages supported via EasyOCR
- **PDF Intelligence**: Auto-detects text vs image-based PDFs for optimal processing

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
cd ocr-app

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py                    # GUI mode
python main.py --help            # CLI help
```

## Usage

### GUI Mode

1. Launch the application (no command-line arguments)
2. Drag and drop files onto the drop zone, or click "Browse File"
3. View extracted text in the results area
4. Save results using "Save Text" button

### CLI Mode

```bash
# Process a single file
python main.py process-file image.jpg --output ./results --format txt

# Process a directory
python main.py process-dir ./images --output-dir ./results --recursive

# Batch process files listed in a text file
python main.py batch-process batch_list.txt --output-dir ./results

# Show help
python main.py --help
```

### CLI Commands

- `process-file <file>` - Process a single file
- `process-dir <directory>` - Process all files in a directory
- `batch-process <batch_file>` - Process files listed in a text file
- `engines` - List available OCR engines

### CLI Options

- `-l, --languages` - OCR languages (can be specified multiple times, default: en)
- `-o, --output` / `--output-dir` - Output file/directory path
- `-c, --confidence` - Confidence threshold (0.0-1.0, default: 0.5)
- `-f, --format` - Output format (txt, json, csv, default: txt)
- `-r, --recursive` - Process directories recursively (process-dir only)
- `--engine` - Force specific OCR engine (easyocr or tesseract)
- `--no-preprocess` - Disable image preprocessing
- `--use-gpu` - Enable GPU acceleration for OCR

## Supported Languages

The application supports 80+ languages including:
English, French, German, Spanish, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, and many more.

Use `python main.py engines` to see the available OCR engines.

## Output Formats

- **TXT**: Plain text file with extracted content
- **JSON**: Structured data with metadata
- **CSV**: Tabular format for batch processing results

## System Requirements

- **Python**: 3.7+ (for source installation)
- **Memory**: 4GB+ recommended for large PDFs
- **Storage**: 2GB+ for model downloads on first run

## Dependencies

- easyocr: OCR engine
- pdfplumber: PDF text extraction
- PyPDF2: PDF processing
- PyMuPDF: Advanced PDF handling
- pdf2image: PDF to image conversion
- tkinter: GUI framework
- click: CLI framework
- pillow: Image processing
- opencv-python: Computer vision
- numpy: Numerical computing

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
chmod +x build.sh

# Run build script
./build.sh
```

This creates a standalone executable in the `dist/` directory.

## Troubleshooting

### Common Issues

1. **"OCR engines not initialized"**
    - Wait for the application to download models on first run
    - Ensure internet connection for model downloads
    - Check that easyocr or pytesseract is installed

2. **"File not found" errors**
    - Check file paths and permissions
    - Use absolute paths for CLI operations
    - Ensure supported file formats (.png, .jpg, .jpeg, .bmp, .tiff, .tif, .pdf)

3. **Poor OCR accuracy**
    - Try different confidence thresholds (default: 0.5)
    - Ensure good image quality (300+ DPI recommended)
    - Use preprocessing options (enabled by default)
    - Try different OCR engines if available

4. **PDF processing issues**
    - For image-based PDFs, ensure poppler-utils is installed
    - Check that pdf2image and pdfplumber are available
    - Try different PDF extraction methods (automatic detection)

### Performance Tips

- Use GPU acceleration if available (`--use-gpu` flag)
- Process directories recursively for bulk operations
- Close other applications to free up memory
- Use appropriate image resolutions (not too large)
- For large PDFs, ensure sufficient RAM (4GB+ recommended)

## License

This project is open source. See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Support

For support and questions:
- Check the troubleshooting section above
- Review the CLI help: `python main.py --help`
- List available engines: `python main.py engines`
- Create an issue on the project repository

## Changelog

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