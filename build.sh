#!/bin/bash
# Build script for OCR Text Extractor
# Creates standalone executables for Windows, macOS, and Linux

set -e

echo "🔨 Building OCR Text Extractor for $PLATFORM_NAME..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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
    print_error "Please run this script from the ocr-app directory containing main.py and requirements.txt"
    exit 1
fi

# Check for required dependencies
print_status "Checking dependencies..."

if ! command -v python3 >/dev/null 2>&1; then
    print_error "python3 is not installed or not in PATH"
    exit 1
fi

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 6) else 1)" 2>/dev/null; then
    print_error "Python 3.6 or higher is required"
    exit 1
fi

if ! command -v pip >/dev/null 2>&1; then
    print_error "pip is not installed or not in PATH"
    exit 1
fi

if ! pip show pyinstaller >/dev/null 2>&1; then
    print_warning "PyInstaller not found in current environment. It will be installed if needed."
fi

# Create dist directory
if ! mkdir -p dist; then
    print_error "Failed to create dist directory"
    exit 1
fi

# Detect platform
if ! PLATFORM=$(uname -s 2>/dev/null); then
    print_error "Failed to detect platform using uname"
    exit 1
fi

case $PLATFORM in
    Linux)
        PLATFORM_NAME="linux"
        ;;
    Darwin)
        PLATFORM_NAME="macos"
        ;;
    CYGWIN*|MINGW32*|MSYS*|MINGW*)
        PLATFORM_NAME="windows"
        ;;
    *)
        print_error "Unsupported platform: $PLATFORM"
        print_error "Supported platforms: Linux, macOS, Windows (via Cygwin/MSYS/MinGW)"
        exit 1
        ;;
esac

print_status "Detected platform: $PLATFORM_NAME"

# Install dependencies if virtual environment doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    if ! python3 -m venv venv; then
        print_error "Failed to create virtual environment"
        exit 1
    fi
fi

print_status "Activating virtual environment..."
if [ "$PLATFORM_NAME" = "windows" ]; then
    if ! source venv/Scripts/activate; then
        print_error "Failed to activate virtual environment"
        exit 1
    fi
else
    if ! source venv/bin/activate; then
        print_error "Failed to activate virtual environment"
        exit 1
    fi
fi

print_status "Installing dependencies..."
if ! pip install -r requirements.txt; then
    print_error "Failed to install dependencies from requirements.txt"
    exit 1
fi

# Ensure PyInstaller is available
if ! python -c "import PyInstaller" >/dev/null 2>&1; then
    print_status "Installing PyInstaller..."
    if ! pip install pyinstaller; then
        print_error "Failed to install PyInstaller"
        exit 1
    fi
fi

# Build with PyInstaller
print_status "Building executable with PyInstaller..."

if [ "$PLATFORM_NAME" = "windows" ]; then
    # Windows specific build
    if ! pyinstaller --clean --onedir -d noarchive --windowed --name OCR_Text_Extractor --debug=all main.py; then
        print_error "Failed to build GUI executable"
        exit 1
    fi
    if ! pyinstaller --clean --onedir -d noarchive --console --name OCR_Text_Extractor_CLI main.py; then
        print_error "Failed to build CLI executable"
        exit 1
    fi
else
    # Unix-like systems
    if ! pyinstaller --clean --onedir -d noarchive --name OCR_Text_Extractor main.py; then
        print_error "Failed to build GUI executable"
        exit 1
    fi
    if ! pyinstaller --clean --onedir -d noarchive --console --name OCR_Text_Extractor_CLI main.py; then
        print_error "Failed to build CLI executable"
        exit 1
    fi
fi

# Create distribution package
print_status "Creating distribution package..."

DIST_DIR="dist/OCR_Text_Extractor_$PLATFORM_NAME"
if ! mkdir -p "$DIST_DIR"; then
    print_error "Failed to create distribution directory: $DIST_DIR"
    exit 1
fi

# Copy executables
if [ "$PLATFORM_NAME" = "windows" ]; then
    if [ ! -f "dist/OCR_Text_Extractor.exe" ]; then
        print_error "GUI executable not found: dist/OCR_Text_Extractor.exe"
        exit 1
    fi
    if ! cp dist/OCR_Text_Extractor.exe "$DIST_DIR/"; then
        print_error "Failed to copy GUI executable"
        exit 1
    fi
    if [ ! -f "dist/OCR_Text_Extractor_CLI.exe" ]; then
        print_error "CLI executable not found: dist/OCR_Text_Extractor_CLI.exe"
        exit 1
    fi
    if ! cp dist/OCR_Text_Extractor_CLI.exe "$DIST_DIR/"; then
        print_error "Failed to copy CLI executable"
        exit 1
    fi
else
    if [ ! -d "dist/OCR_Text_Extractor" ]; then
        print_error "GUI executable directory not found: dist/OCR_Text_Extractor"
        exit 1
    fi
    if ! cp -r dist/OCR_Text_Extractor "$DIST_DIR/"; then
        print_error "Failed to copy GUI executable directory"
        exit 1
    fi
    if [ ! -d "dist/OCR_Text_Extractor_CLI" ]; then
        print_error "CLI executable directory not found: dist/OCR_Text_Extractor_CLI"
        exit 1
    fi
    if ! cp -r dist/OCR_Text_Extractor_CLI "$DIST_DIR/"; then
        print_error "Failed to copy CLI executable directory"
        exit 1
    fi
fi

# Copy documentation and examples
if [ -f "README.md" ]; then
    cp README.md "$DIST_DIR/" || print_warning "Failed to copy README.md"
fi
if [ -f "LICENSE" ]; then
    cp LICENSE "$DIST_DIR/" || print_warning "Failed to copy LICENSE"
fi

# Create run scripts
if [ "$PLATFORM_NAME" = "windows" ]; then
    # Windows batch files
    cat > "$DIST_DIR/run_gui.bat" << 'EOF'
@echo off
REM Run OCR Text Extractor GUI
"%~dp0OCR_Text_Extractor.exe"
EOF

    cat > "$DIST_DIR/run_cli.bat" << 'EOF'
@echo off
REM Run OCR Text Extractor CLI
"%~dp0OCR_Text_Extractor_CLI.exe" %*
EOF
else
    # Unix-like shell scripts
    cat > "$DIST_DIR/run_gui.sh" << 'EOF'
#!/bin/bash
# Run OCR Text Extractor GUI
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
"$DIR/OCR_Text_Extractor/OCR_Text_Extractor"
EOF

    cat > "$DIST_DIR/run_cli.sh" << 'EOF'
#!/bin/bash
# Run OCR Text Extractor CLI
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
"$DIR/OCR_Text_Extractor_CLI/OCR_Text_Extractor_CLI" "$@"
EOF

    if ! chmod +x "$DIST_DIR/run_gui.sh"; then
        print_warning "Failed to make run_gui.sh executable"
    fi
    if ! chmod +x "$DIST_DIR/run_cli.sh"; then
        print_warning "Failed to make run_cli.sh executable"
    fi
fi

# Create archive
if [ "$PLATFORM_NAME" = "windows" ]; then
    ARCHIVE_NAME="OCR_Text_Extractor_$PLATFORM_NAME.zip"
    cd dist || { print_error "Failed to change to dist directory"; exit 1; }
    # Use zip for Windows
    if command -v zip >/dev/null 2>&1; then
        if ! zip -r "$ARCHIVE_NAME" "OCR_Text_Extractor_$PLATFORM_NAME"; then
            print_error "Failed to create zip archive"
            cd ..
            exit 1
        fi
    else
        print_warning "zip command not found, creating tar.gz instead"
        ARCHIVE_NAME="${ARCHIVE_NAME%.zip}.tar.gz"
        if ! tar -czf "$ARCHIVE_NAME" "OCR_Text_Extractor_$PLATFORM_NAME"; then
            print_error "Failed to create tar.gz archive"
            cd ..
            exit 1
        fi
    fi
    cd ..
else
    ARCHIVE_NAME="OCR_Text_Extractor_$PLATFORM_NAME.tar.gz"
    cd dist || { print_error "Failed to change to dist directory"; exit 1; }
    if ! tar -czf "$ARCHIVE_NAME" "OCR_Text_Extractor_$PLATFORM_NAME"; then
        print_error "Failed to create tar.gz archive"
        cd ..
        exit 1
    fi
    cd ..
fi

print_success "Build completed successfully!"
print_status "Distribution package created: dist/$ARCHIVE_NAME"
print_status "Size: $(du -sh "dist/$ARCHIVE_NAME" 2>/dev/null | cut -f1 || echo 'unknown')"
print_status ""
print_status "To use the application:"
print_status "1. Extract the archive: tar -xzf dist/$ARCHIVE_NAME (or unzip on Windows)"
if [ "$PLATFORM_NAME" = "windows" ]; then
    print_status "2. Run GUI: ./OCR_Text_Extractor_$PLATFORM_NAME/run_gui.bat"
    print_status "3. Run CLI: ./OCR_Text_Extractor_$PLATFORM_NAME/run_cli.bat --help"
else
    print_status "2. Run GUI: ./OCR_Text_Extractor_$PLATFORM_NAME/run_gui.sh"
    print_status "3. Run CLI: ./OCR_Text_Extractor_$PLATFORM_NAME/run_cli.sh --help"
fi

# Cleanup
print_status "Cleaning up temporary files..."
if [ -d "build" ]; then
    rm -rf build || print_warning "Failed to remove build directory"
fi
if [ -d "dist" ]; then
    # Keep the final archive, remove everything else in dist
    if [ -f "dist/$ARCHIVE_NAME" ]; then
        # Move archive to current directory temporarily
        mv "dist/$ARCHIVE_NAME" . || print_warning "Failed to move archive"
    fi
    rm -rf dist || print_warning "Failed to remove dist directory"
    if [ -f "$ARCHIVE_NAME" ]; then
        mkdir -p dist || print_warning "Failed to recreate dist directory"
        mv "$ARCHIVE_NAME" "dist/" || print_warning "Failed to restore archive"
    fi
fi

print_success "Build process completed!"