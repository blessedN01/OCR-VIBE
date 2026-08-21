# Contributing to OCR-VIBE

Thank you for your interest in contributing to OCR-VIBE! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Setting Up the Development Environment

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/your-username/OCR-VIBE.git
   cd OCR-VIBE
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Install optional dependencies (for DOCX support):**
   ```bash
   pip install -r requirements-optional.txt
   ```

## Project Structure

```
OCR-VIBE/
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── cli_app.py         # CLI application
│   ├── gui_app.py         # GUI application
│   ├── ocr_engine.py      # OCR engine with plugin system
│   ├── pdf_handler.py     # PDF processing
│   ├── config_loader.py   # Configuration management
│   ├── memory_manager.py  # Memory management
│   ├── stats_tracker.py   # Engine stats tracking
│   ├── batch_queue.py     # Batch processing queue
│   └── exceptions.py      # Custom exceptions
├── tests/                  # Test suite
├── .github/workflows/      # CI/CD configuration
├── requirements.txt        # Core dependencies
├── requirements-dev.txt    # Development dependencies
├── requirements-optional.txt # Optional dependencies
├── pyproject.toml          # Project metadata
├── build.sh               # Windows build script
├── build_macos.sh         # macOS build script
├── build_linux.sh         # Linux build script
└── main.py                # Entry point
```

## Code Style

This project uses:
- **Ruff** for linting and formatting
- **Mypy** for type checking
- **Black-compatible** formatting (100 character line length)

### Linting

```bash
# Check code style
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/
```

### Type Checking

```bash
mypy src/ --ignore-missing-imports
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_ocr_engine.py -v
```

### Test Structure

- `tests/test_ocr_engine.py` - OCR engine tests
- `tests/test_pdf_handler.py` - PDF handler tests
- `tests/test_memory_manager.py` - Memory manager tests
- `tests/test_cli_app.py` - CLI application tests

### Writing Tests

- Use pytest fixtures for common setup
- Mock external dependencies (OCR engines, file I/O)
- Aim for high coverage of core functionality
- Test both success and error cases

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clean, well-documented code
   - Follow existing code conventions
   - Add tests for new functionality

3. **Run quality checks:**
   ```bash
   ruff check src/ tests/
   mypy src/ --ignore-missing-imports
   pytest tests/ -v
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

## Commit Message Convention

Follow conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Test additions or changes
- `chore:` - Build process or auxiliary tool changes

## Plugin System

OCR-VIBE supports custom OCR engine plugins. To create a plugin:

1. Create a Python file in `~/.ocr-vibe/engines/`
2. Define a class that inherits from `BaseOCREngine`
3. Implement the required methods
4. Define a `register(registry)` function

Example plugin:

```python
# ~/.ocr-vibe/engines/my_engine.py
from src.ocr_engine import BaseOCREngine, EngineRegistry

class MyCustomEngine(BaseOCREngine):
    @property
    def name(self) -> str:
        return "my_engine"

    @property
    def supported_languages(self) -> List[str]:
        return ['en']

    def initialize(self, languages, use_gpu=False, **kwargs):
        # Initialize your engine
        return True

    def extract_text(self, image):
        # Extract text from image
        return "extracted text", []

def register(registry):
    registry.register("my_engine", MyCustomEngine())
```

## Documentation

When adding new features:
- Update the README.md with new CLI flags and features
- Add docstrings to new functions and classes
- Update type hints

## Release Process

1. Update version in `pyproject.toml` and `src/__init__.py`
2. Update `README.md` changelog
3. Create a git tag: `git tag v2.0.0`
4. Push tags: `git push --tags`
5. GitHub Actions will build and create a release

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Respect different viewpoints and experiences

## License

By contributing to OCR-VIBE, you agree that your contributions will be licensed under the MIT License.
