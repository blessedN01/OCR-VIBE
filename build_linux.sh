#!/bin/bash
# Build script for OCR Text Extractor on Linux
# Creates standalone executables using PyInstaller

set -e

echo "Building OCR Text Extractor for Linux..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "main.py" ] || [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the project root directory containing main.py and requirements.txt"
    exit 1
fi

# Check for Python
if ! command -v python3 >/dev/null 2>&1; then
    print_error "python3 is not installed or not in PATH"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
print_status "Installing dependencies..."
pip install -r requirements.txt
pip install pyinstaller

# Build with PyInstaller
print_status "Building executable with PyInstaller..."

# Linux GUI
pyinstaller --clean --onedir --name "OCR_Text_Extractor" \
    --add-data "src:src" \
    main.py

# Linux CLI
pyinstaller --clean --onedir --console --name "OCR_Text_Extractor_CLI" \
    --add-data "src:src" \
    main.py

# Create distribution package
DIST_DIR="dist/OCR_Text_Extractor_linux"
mkdir -p "$DIST_DIR"

# Copy executables
cp -r dist/OCR_Text_Extractor "$DIST_DIR/"
cp -r dist/OCR_Text_Extractor_CLI "$DIST_DIR/"

# Copy documentation
[ -f "README.md" ] && cp README.md "$DIST_DIR/"

# Create run scripts
cat > "$DIST_DIR/run_gui.sh" << 'EOF'
#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
"$DIR/OCR_Text_Extractor/OCR_Text_Extractor"
EOF

cat > "$DIST_DIR/run_cli.sh" << 'EOF'
#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
"$DIR/OCR_Text_Extractor_CLI/OCR_Text_Extractor_CLI" "$@"
EOF

chmod +x "$DIST_DIR/run_gui.sh"
chmod +x "$DIST_DIR/run_cli.sh"

# Create archive
ARCHIVE_NAME="OCR_Text_Extractor_linux.tar.gz"
cd dist
tar -czf "$ARCHIVE_NAME" "OCR_Text_Extractor_linux"
cd ..

print_success "Build completed successfully!"
print_status "Distribution package: dist/$ARCHIVE_NAME"
print_status "Size: $(du -sh "dist/$ARCHIVE_NAME" 2>/dev/null | cut -f1 || echo 'unknown')"
print_status ""
print_status "To use the application:"
print_status "1. Extract the archive: tar -xzf dist/$ARCHIVE_NAME"
print_status "2. Run GUI: ./OCR_Text_Extractor_linux/run_gui.sh"
print_status "3. Run CLI: ./OCR_Text_Extractor_linux/run_cli.sh --help"
