# OCR-VIBE Enhancement Plan

## 1. Goal

Deliver a comprehensive, production-ready OCR application across three implementation phases. Enhancements must preserve backward compatibility with the existing CLI/GUI interfaces and the current `src/` package layout.

## 2. Current State

- **Version**: 1.0.0
- **OCR Engines**: EasyOCR (primary) + pytesseract (fallback)
- **PDF Pipeline**: pdfplumber, PyPDF2, PyMuPDF, pdf2image
- **Preprocessing**: Basic OpenCV pipeline (grayscale → blur → threshold → morphology)
- **Interfaces**: Tkinter GUI with drag-and-drop; Click CLI
- **Output Formats**: TXT, JSON, CSV
- **Packaging**: PyInstaller for Windows only; Dockerfiles exist but are incomplete
- **Testing**: None
- **CI/CD**: None

## 3. Phase 1 — Quick Wins (Immediate Value, Low Effort)

### 3.1 Config File Support
- Add YAML-based config with layered merge strategy:
  - **Project config** (`ocr-vibe.yaml`): discovered by walking up from CWD to filesystem root (like `git`); lowest precedence
  - **Home config** (`~/.ocr-vibe/config.yaml`): overrides project defaults
  - **CLI flags**: highest precedence, overrides everything
- Store defaults: `languages`, `confidence_threshold`, `output_format`, `preprocess_images`, `engine`, `use_gpu`, `memory_limit_mb`
- Shallow merge (top-level keys only); CLI flags completely replace config values when present
- **Files**: `src/config_loader.py`, update `src/cli_app.py`, update `src/gui_app.py`, update `requirements.txt` (add `pyyaml`)

### 3.2 Dependency Organization
- Split `requirements.txt` into three files:
  - `requirements.txt` — core runtime dependencies (existing list minus dev/optional)
  - `requirements-dev.txt` — pytest, pytest-cov, mypy, ruff, pyinstaller
  - `requirements-optional.txt` — python-docx, any future optional extras
- `pyproject.toml` references these groups for PyPI extras (`ocr-vibe[all]`, `ocr-vibe[dev]`)
- **Files**: `requirements.txt`, `requirements-dev.txt`, `requirements-optional.txt`, `pyproject.toml`

### 3.2 Parallel Batch Processing
- Add `--parallel` / `-p` flag to `process-dir` and `batch-process` CLI commands
- Use `multiprocessing.Pool` with capped worker count: `min(cpu_count, available_memory_mb // 500)`, falling back to `get_optimal_batch_size()` from `memory_manager.py`
- Workers lazily initialize OCR engines on first task via module-level singleton pattern inside each process
- Default: sequential (preserve existing behavior when flag absent)
- **Files**: `src/cli_app.py`, `src/memory_manager.py`

### 3.3 CLI Progress Bars
- Integrate `tqdm` for batch operations (`process-dir`, `batch-process`)
- Show per-file progress and overall ETA
- **Files**: `src/cli_app.py`, `requirements.txt`

### 3.4 Language-Aware Post-Processing
- Replace `--spell-check` with `--post-process` flag
- Behavior depends on detected language(s):
  - **English (`en`) only**: use `pyspellchecker` for spell correction + filter single-character noise tokens
  - **All languages**: universal cleanup that works on any script:
    - Remove repeated character runs (e.g., `aaa` → `a`)
    - Collapse multiple spaces/tabs/newlines to single spaces
    - Strip isolated punctuation-only tokens
    - Remove tokens shorter than 2 chars unless they're language-specific whitelist items
- GUI: checkbox for "Post-process output"
- **Files**: `src/ocr_engine.py`, `src/cli_app.py`, `src/gui_app.py`, `requirements.txt`

### 3.5 Markdown Output Format
- Add `md` as output format choice
- Wrap extracted text in markdown with metadata header
- **Files**: `src/cli_app.py` (OutputFormatter)

### 3.6 PDF Page Range Selection
- Add `--pages` option to CLI (e.g., `--pages 1-3,5,7-9`)
- Parse page ranges and pass to `PDFHandler.extract_text_by_page()` with filter
- **Files**: `src/cli_app.py`, `src/pdf_handler.py`

### 3.7 GUI Clipboard Copy
- Add "Copy to Clipboard" button next to "Save Results"
- Use `root.clipboard_clear()` / `root.clipboard_append()`
- **Files**: `src/gui_app.py`

### 3.8 Version Bump
- Bump `__version__` to `1.1.0` in `src/__init__.py` and `src/cli_app.py`

## 4. Phase 2 — Core Quality (Accuracy & Robustness)

### 4.1 PaddleOCR Engine Integration
- Add `paddleocr` as third OCR engine option (installed by default)
- PaddleOCR excels at CJK, complex layouts, and low-quality scans
- **Dynamic engine selection**: choose engine based on historical performance for the input's language(s) and file type
  - Track per-engine stats in `~/.ocr-vibe/engine_stats.json`:
    - Average confidence per language per engine
    - Success/failure rate per engine per file type (image extension, PDF)
    - Average processing time per engine per page-count range (for PDFs)
  - **Scoring formula (UCB-style)**:
    - For each engine *e* and combination of language(s) + file type, compute:
      `score(e) = avg_confidence(e) + exploration_weight * sqrt(ln(total_runs + 1) / (runs_for_e + 1))`
    - Where `exploration_weight = 1.0` (tunable constant), `total_runs` is total extractions for this combination across all engines, and `runs_for_e` is runs for engine *e*
    - Untried engines (runs_for_e = 0) get a finite exploration bonus: `1.0 * sqrt(ln(total_runs + 1))`
    - Select engine with highest score; if it fails, fall back to next-highest score
  - Update stats after each successful/failed extraction; persist to disk after each file and on cleanup
    - Write to `engine_stats.json.tmp`, then atomic `os.replace()` to `engine_stats.json`
    - In `--parallel` mode, each worker writes independently; last-write-wins is acceptable (stats are advisory)
    - On app exit / `atexit`, force one final write to flush in-memory stats
  - On first run (no stats), use fixed order: EasyOCR → PaddleOCR → Tesseract
  - On subsequent runs, select the engine with highest expected confidence for the current language(s) and file type
  - If selected engine fails at runtime, fall back to next best in the ranked order
  - User can still force a specific engine with the existing `--engine` flag (bypasses auto-selection)
- **Files**: `src/ocr_engine.py`, `src/exceptions.py`, `requirements.txt`, `src/stats_tracker.py` (new)

### 4.2 PDF Table Extraction
- Use `camelot-py[cv]` for table extraction from PDFs (OpenCV flavor avoids Ghostscript/Java system deps)
- Expose via `--extract-tables` CLI flag and GUI checkbox
- Output tables as CSV or JSON alongside text
- **Files**: `src/pdf_handler.py`, `src/cli_app.py`, `src/gui_app.py`, `requirements.txt`

### 4.3 Layout Preservation
- Use DBSCAN clustering (`scikit-learn`) on bounding box centroids to detect columns/paragraphs
- Sort clusters top-to-bottom, left-to-right within each cluster
- Detect headers/footers via position heuristics (top/bottom 10% of page)
- Add `--preserve-layout` flag
- **Files**: `src/ocr_engine.py` (post-processing logic), `requirements.txt` (add `scikit-learn`)

### 4.4 Enhanced Preprocessing Pipeline
- Add configurable preprocessing steps:
  - **Deskewing**: detect skew angle with OpenCV `minAreaRect` and rotate
  - **CLAHE**: Contrast Limited Adaptive Histogram Equalization
  - **Upscaling**: 2x upscale for images < 150 DPI (using OpenCV `INTER_CUBIC`)
  - **Background removal**: morphological opening to isolate foreground
- Expose toggles via CLI flags (`--deskew`, `--clahe`, `--upscale`, `--remove-bg`)
- **Files**: `src/ocr_engine.py`, `src/cli_app.py`, `src/gui_app.py`

### 4.5 Searchable PDF Output
- Rebuild PDF with embedded OCR text layer using `PyMuPDF`
- Output as `_searchable.pdf` alongside text extraction
- CLI flag: `--searchable-pdf`
- **Files**: `src/pdf_handler.py`, `src/cli_app.py`

### 4.6 Improved Error Diagnostics
- Add `--verbose` / `-v` flag to CLI for detailed logging (DEBUG level)
- In GUI, add a "Log" tab or expandable section showing processing details
- **Files**: `src/cli_app.py`, `src/gui_app.py`

## 5. Phase 3 — Full Feature Set (GUI, Distribution, Testing)

### 5.1 GUI Improvements
- **Language Selector**: Dropdown populated from `OCREngine.get_supported_languages()` (EasyOCR/PaddleOCR)
- **Confidence Slider**: `ttk.Scale` bound to `OCREngine.confidence_threshold` with live label
- **Preprocessing Preview**: Side-by-side canvas showing original vs. processed image
- **Engine Selector**: Radio buttons or dropdown for EasyOCR / Tesseract / PaddleOCR
- **Dark Theme**: Use `ttk.Style` with `clam` theme and custom color palette
- **Batch Queue View**: Separate `BatchQueueManager` class (`src/batch_queue.py`) managing a `ttk.Treeview` queue; process files sequentially with per-file progress bars and cancel button; GUI delegates to queue manager via callbacks
- **Memory Monitor**: Live memory usage label in status bar via `psutil`
- **Files**: `src/batch_queue.py`, `src/gui_app.py`

### 5.2 Additional Output Formats
- **DOCX**: Use `python-docx` to create editable Word documents with extracted text
- **HTML**: Annotated HTML with confidence-colored bounding box overlays for each word/phrase using `<span>` elements with inline `style` positioning; Tesseract engine requires `pytesseract.image_to_data()` for bounding box data (update `extract_text_tesseract` to return boxes alongside text)
- **Files**: `src/cli_app.py` (OutputFormatter), `src/ocr_engine.py` (Tesseract bbox support), `requirements.txt`

### 5.3 Testing Infrastructure
- Add `pytest` and `pytest-cov`
- Create `tests/` directory:
  - `tests/test_ocr_engine.py`: Unit tests for preprocessing, engine fallback, confidence filtering
  - `tests/test_pdf_handler.py`: Mock-based tests for PDF type detection, page counting
  - `tests/test_memory_manager.py`: Memory limit and GC tests
  - `tests/test_cli_app.py`: Click CLI invocation tests with `CliRunner`
  - `tests/test_gui_app.py`: GUI smoke tests (optional, harder to automate)
- Add test fixtures: sample images, small PDFs in `tests/fixtures/`
- **Files**: `tests/`, `requirements-dev.txt`, `pyproject.toml` (or `setup.cfg`)

### 5.4 CI/CD
- GitHub Actions workflow (`.github/workflows/ci.yml`)
- Matrix: Python 3.9, 3.11, 3.13 on ubuntu-latest, windows-latest, macos-latest
- Steps: lint (ruff), typecheck (mypy), test (pytest), build (PyInstaller on Windows)
- **Files**: `.github/workflows/ci.yml`

### 5.5 Packaging & Distribution
- **PyPI package**: Add `pyproject.toml` with `[build-system]` (setuptools + wheel), package name `ocr-vibe`
- **macOS build script**: `build_macos.sh` (similar to `build.sh` but macOS-specific)
- **Linux build script**: `build_linux.sh`
- **pip installable**: `pip install ocr-vibe[all]` with extras for optional engines
- **Files**: `pyproject.toml`, `build_macos.sh`, `build_linux.sh`, `requirements.txt`, `requirements-optional.txt`

### 5.6 Plugin System for OCR Engines
- Define `BaseOCREngine` abstract class in `src/ocr_engine.py`
- Engine registry: `register_engine(name, class)`
- **Dynamic discovery**: at startup, scan `~/.ocr-vibe/engines/` for Python files with `register()` function; import and register them
- Works identically in source mode and PyInstaller bundles (no entry points dependency)
- **Files**: `src/ocr_engine.py`

### 5.7 Documentation Updates
- Update `README.md` with new features, config file docs, new CLI flags
- Add `CONTRIBUTING.md` with dev setup, test commands, linting rules
- **Files**: `README.md`, `CONTRIBUTING.md`

## 6. Dependency Graph

```
Phase 1 (config, parallel, progress, spell-check, markdown, page ranges, clipboard)
  └── Phase 2 (PaddleOCR, tables, layout, preprocessing, searchable PDF)
       └── Phase 3 (GUI polish, DOCX/HTML, tests, CI/CD, PyPI, plugin system)
```

- Phase 1 is fully independent.
- Phase 2 depends on Phase 1's config file for new engine/table/preprocessing toggles.
- Phase 3 depends on Phase 2's new engines and features being exposed in the GUI/CLI.

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| PaddleOCR model download (~100MB) fails in restricted environments | Cache models in `_get_easyocr_cache_dir()` equivalent; show clear error |
| camelot-py[cv] image parsing fails on certain PDFs | Document limitation; users can fall back to image-based OCR via `--engine` flag |
| PyInstaller bundle size grows with new engines | Use `--onedir` (already used); document `--onedir` vs `--onefile` tradeoffs |
| Tkinter dark theme looks inconsistent across platforms | Use `ttk.Style` with `clam` theme; test on Windows/macOS/Linux |
| Multiprocessing on Windows requires `if __name__ == '__main__'` guard | Ensure `cli_app.py` wraps pool execution properly |

## 8. Validation Plan

- **Per-phase**: Run existing CLI help, verify no regressions in basic image/PDF processing
- **Unit tests**: `pytest tests/ -v` with coverage report
- **Lint**: `ruff check src/ tests/`
- **Type check**: `mypy src/`
- **Build**: `python -m PyInstaller` on Windows, verify executables run
- **Manual GUI**: Test drag-and-drop, save, copy, new toggles
- **Cross-platform**: GitHub Actions matrix must pass

## 9. Out of Scope

- Video frame OCR / subtitle generation
- Handwriting recognition (trOCR)
- Barcode / QR code reading
- Receipt parsing
- Web interface / REST API
- Mobile apps
- Real-time camera OCR

## 10. Implementation Order (Task List)

### Phase 1
1. Create `src/config_loader.py` (YAML/JSON config, merge with CLI args)
2. Add `pyyaml` to `requirements.txt`
3. Update `src/cli_app.py` — config integration, `--parallel`, `tqdm` progress, `--pages`, `--post-process`, `md` format
4. Update `src/gui_app.py` — config defaults, clipboard copy, spell-check toggle
5. Update `src/ocr_engine.py` — language-aware post-processing, page-range-aware extraction helper
6. Bump version to `1.1.0`

### Phase 2
1. Add `paddleocr` to `requirements.txt`
2. Update `src/ocr_engine.py` — PaddleOCR engine class, dynamic engine selection with stats tracking
3. Update `src/exceptions.py` — engine-specific error details
4. Add `camelot-py[cv]` to `requirements.txt`
5. Update `src/pdf_handler.py` — table extraction, searchable PDF generation
6. Update `src/ocr_engine.py` — enhanced preprocessing steps (deskew, CLAHE, upscale, bg removal)
7. Update `src/cli_app.py` — new flags for tables, layout, preprocessing, searchable PDF
8. Update `src/gui_app.py` — GUI toggles for Phase 2 features

### Phase 3
1. Create `pyproject.toml` with build system and PyPI metadata
2. Create `requirements-dev.txt` (pytest, pytest-cov, mypy, ruff, etc.)
3. Split `requirements.txt` into `requirements.txt` (core) and `requirements-optional.txt` (python-docx and future extras)
4. Create `src/batch_queue.py` (BatchQueueManager class)
5. Update `src/gui_app.py` — language selector, confidence slider, preview canvas, engine selector, dark theme, batch queue integration, memory monitor
6. Update `src/ocr_engine.py` — Tesseract `image_to_data` integration for HTML bbox support
7. Update `src/cli_app.py` — DOCX and annotated HTML output formatters
8. Create `tests/` with unit and integration tests
9. Create `.github/workflows/ci.yml`
10. Create `build_macos.sh` and `build_linux.sh`
11. Add plugin system (`BaseOCREngine`, dynamic discovery) to `src/ocr_engine.py`
12. Update `README.md` and `CONTRIBUTING.md`
13. Bump version to `2.0.0`
