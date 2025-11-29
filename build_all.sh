#!/bin/bash
# Build script for OCR Text Extractor for all platforms
# Uses Docker to build standalone executables for Linux and Windows

set -e

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

# Check if Docker is installed
if ! command -v docker >/dev/null 2>&1; then
    print_error "Docker is not installed or not in PATH"
    print_error "Please install Docker to use this build script"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "main.py" ] || [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the ocr-app directory containing main.py and requirements.txt"
    exit 1
fi

# Create output directory
OUTPUT_DIR="dist_all"
if ! mkdir -p "$OUTPUT_DIR"; then
    print_error "Failed to create output directory: $OUTPUT_DIR"
    exit 1
fi

# Function to build for a platform
build_platform() {
    local platform=$1
    local dockerfile=$2
    local image_name="ocr-builder-$platform"

    print_status "Building Docker image for $platform..."
    if ! docker build -f "$dockerfile" -t "$image_name" .; then
        print_error "Failed to build Docker image for $platform"
        return 1
    fi

    print_status "Running build container for $platform..."
    local container_name="ocr-build-$platform-$$"
    if ! docker run --name "$container_name" "$image_name"; then
        print_error "Failed to run build container for $platform"
        docker rm -f "$container_name" >/dev/null 2>&1 || true
        return 1
    fi

    print_status "Copying build artifacts for $platform..."
    local platform_dir="$OUTPUT_DIR/$platform"
    if ! mkdir -p "$platform_dir"; then
        print_error "Failed to create platform directory: $platform_dir"
        docker rm -f "$container_name" >/dev/null 2>&1 || true
        return 1
    fi

    # Copy the dist directory from container
    if docker cp "$container_name:/app/dist/." "$platform_dir/"; then
        print_success "Successfully built for $platform"
    else
        print_warning "No dist artifacts found for $platform"
    fi

    # Clean up container
    docker rm -f "$container_name" >/dev/null 2>&1 || true

    return 0
}

# Build for Linux
print_status "Starting multi-platform build..."
if build_platform "linux" "Dockerfile.linux"; then
    print_success "Linux build completed"
else
    print_warning "Linux build failed, continuing with other platforms"
fi

# Build for Windows
if build_platform "windows" "Dockerfile.windows"; then
    print_success "Windows build completed"
else
    print_warning "Windows build failed, continuing"
fi

# Note about macOS
print_warning "macOS build requires a macOS environment and is not supported in this script"
print_warning "To build for macOS, run build.sh on a macOS machine"

print_success "Multi-platform build process completed!"
print_status "Build artifacts are available in: $OUTPUT_DIR"
print_status "Contents:"
ls -la "$OUTPUT_DIR" || true