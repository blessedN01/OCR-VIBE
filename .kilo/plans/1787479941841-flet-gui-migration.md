# Plan: Replace OCR-VIBE GUI with Flet

## Goal

Replace `src/gui_app.py` entirely with a modern Flet desktop GUI. Remove all tkinter/tkinterdnd2 dependencies. The new GUI must preserve every feature the current one offers: single-file processing, batch queue, file browsing, preview, save/copy, preprocessing toggles, and dark theme.

## Constraints

1. **Do not modify** `src/ocr_engine.py`, `src/pdf_handler.py`, `src/batch_queue.py`, `src/config_loader.py`, `src/exceptions.py`, or `src/cli_app.py`.
2. **Do not break** the existing CLI entry point (`ocr-vibe`).
3. Flet version target: `>=0.24.0` (stable desktop support, `ft.FilePicker`, `ft.ProgressBar`, `ft.Tabs`, `ft.Dialog`).
4. Target platforms: Windows, macOS, Linux.

## What Changes

| File | Action | Summary |
|---|---|---|
| `requirements.txt` | Modify | Add `flet>=0.24.0`, remove `tkinterdnd2>=0.3.0` |
| `pyproject.toml` | Modify | Add `flet>=0.24.0` to `dependencies`, remove `tkinterdnd2>=0.3.0` |
| `main.py` | Modify | Replace tkinter GUI branch with `ft.app(target=...)`. CLI branch untouched. No legacy fallback. |
| `src/gui_flet.py` | New | Complete Flet GUI implementation |
| `src/gui_app.py` | Delete | Remove old tkinter GUI entirely |

## Architecture: `src/gui_flet.py`

**Class:** `FletGUIApp(page: ft.Page)`

**State held by the class:**
- `page` — the Flet page
- `config` — from `get_config_parser()`
- `ocr_engine` — `OCREngine(...)`
- `pdf_handler` — `PDFHandler(ocr_engine=self.ocr_engine)`
- `batch_manager` — `BatchQueueManager(...)`, lazily initialized
- `processing` — bool, guard against double-processing
- `current_file` — `Path | None`
- `preview_original_photo` / `preview_processed_photo` — `ft.FileImage | None`
- `queue` — `queue.Queue` for worker → UI messages

**Layout (top to bottom):**
1. **Title row** — `ft.Text("OCR-VIBE", size=24, weight=ft.FontWeight.BOLD)`
2. **Controls row** — `ft.Row` containing:
   - `ft.Dropdown` for language
   - `ft.Slider` for confidence (0.0–1.0, 100 steps)
   - `ft.RadioGroup` for engine selection (Auto / EasyOCR / PaddleOCR / Tesseract)
3. **Drop zone** — `ft.Container` with dashed border, padding, centered text. `on_click` opens `ft.FilePicker`. Visual feedback on hover via `on_hover`.
4. **Action buttons** — `ft.Row`: Browse, Clear Results, Save Results, Copy to Clipboard, Clear Log
5. **Batch buttons** — `ft.Row`: Start Batch, Cancel Batch, Clear Completed
6. **Options row** — `ft.Row`: Post-process, Extract tables, Preserve layout, Searchable PDF (checkboxes)
7. **Preprocessing row** — `ft.Row`: Deskew, CLAHE, Upscale, Remove background (switches)
8. **Preview row** — `ft.Row` with two `ft.Image` widgets (original / processed). Hidden when no file loaded.
9. **Progress bar** — `ft.ProgressBar(value=0.0)`
10. **Tabs** — `ft.Tabs`:
    - **Results** — `ft.TextField(multiline=True, read_only=True, border=ft.InputBorder.NONE)`
    - **Log** — same style, read-only
    - **Batch Queue** — `ft.Column`:
      - `ft.DataTable` columns: File, Status, Progress
      - `ft.TextField(multiline=True, read_only=True)` for batch results output
11. **Status bar** — `ft.Row`: status text left, memory usage right

## Data Flow

1. User clicks drop zone or Browse → `ft.FilePicker` returns file path(s).
2. `process_file(path)` disables UI, clears results, spawns `threading.Thread(target=_process_file_thread, args=(path,))`.
3. Worker thread runs OCR/PDF extraction, pushes tuples into `self.queue`: `('progress', 0-100)`, `('status', str)`, `('results', text, results, metadata)`, `('error', str)`, `('log', str)`.
4. `page.add_periodic_task(drain_queue, 100)` runs every 100ms on the main thread, drains the queue, updates Flet controls directly.
5. Batch processing reuses `BatchQueueManager` with callbacks that push into the same `self.queue`.

## Widget Mapping (tkinter → Flet)

| Old | New |
|---|---|
| `ttk.Frame` | `ft.Column` / `ft.Row` |
| `ttk.Label` | `ft.Text` |
| `ttk.Button` | `ft.ElevatedButton` |
| `ttk.Combobox` | `ft.Dropdown` |
| `ttk.Checkbutton` | `ft.Checkbox` |
| `ttk.Radiobutton` | `ft.RadioGroup` |
| `ttk.Scale` | `ft.Slider` |
| `ttk.Progressbar` | `ft.ProgressBar` |
| `ttk.Notebook` | `ft.Tabs` |
| `ttk.Treeview` | Custom `ft.Column` of row containers in Batch Queue tab |
| `scrolledtext.ScrolledText` | `ft.TextField(multiline=True, read_only=True)` |
| `tkinterdnd2` drop zone | `ft.Container` + `ft.FilePicker` |
| Dark theme | `page.theme_mode = ft.ThemeMode.DARK` |

## Key Implementation Details

### File Picking
- Single file: `ft.FilePicker(on_result=lambda e: process_file(Path(e.files[0].path)))`
- Multiple files: same picker, iterate `e.files`, add to batch queue
- Save: `ft.FilePicker.save_file(on_result=...)`

### Preview Images
- Original: `ft.Image(src=str(file_path), width=300, height=200, fit=ft.ImageFit.CONTAIN)`
- Processed: write OpenCV preview to temp PNG (`tempfile.NamedTemporaryFile(suffix=".png", delete=False)`), pass path to `ft.Image`. Clean up temp file after display.

### Threading Safety
- All UI updates happen inside `drain_queue()`, called by `page.add_periodic_task`.
- Worker threads only push to `queue.Queue`. They never touch Flet controls.
- This mirrors the existing `root.after(100, self.process_queue)` pattern.

### Batch Queue Display
- Use a `ft.Column` of custom rows instead of `ft.DataTable` (which lacks per-cell progress bars).
- Each row: `ft.Row([ft.Text(file_name, expand=True), ft.Text(status), ft.ProgressBar(width=120, value=0.0)])`
- Update by keeping a `dict[task_id, ft.Row]` mapping.

### Memory Monitor
- Same `psutil` logic as current GUI. Update a `ft.Text` in the status bar via the periodic task.

### Error Handling
- Worker catches all exceptions, pushes `('error', msg)` to queue.
- `drain_queue()` shows `page.show_snack_bar(ft.SnackBar(ft.Text(msg)))` and resets UI state.

## `main.py` — New Behavior

```python
def main():
    if len(sys.argv) > 1:
        from src.cli_app import cli
        cli()
    else:
        import flet as ft

        def main_gui(page: ft.Page):
            page.title = "OCR-VIBE"
            page.theme_mode = ft.ThemeMode.DARK
            page.window_width = 1100
            page.window_height = 850
            page.window_min_width = 900
            page.window_min_height = 700
            page.padding = 20
            page.add_periodic_task(lambda: app.drain_queue(), milliseconds=100)
            page.add(app.build())

        from src.gui_flet import FletGUIApp
        app = FletGUIApp(page)
        ft.app(target=main_gui)
```

No CLI flag for legacy GUI. If Flet is not installed, the user gets a clear error message telling them to `pip install flet`.

## Step-by-Step Tasks

1. **Dependencies** — Update `requirements.txt` and `pyproject.toml`: add `flet>=0.24.0`, remove `tkinterdnd2>=0.3.0`.
2. **Scaffold** — Create `src/gui_flet.py` with `FletGUIApp` class, `__init__`, `build()`, and the full layout skeleton.
3. **Controls wiring** — Connect language dropdown, confidence slider, engine radio group, and preprocessing switches to `self.ocr_engine` state.
4. **File input** — Implement `ft.FilePicker` for browse and drop-zone click. Wire into `process_file()`.
5. **Processing thread** — Implement `_process_file_thread` that pushes to `self.queue`. Wire progress/status/results/error/log messages.
6. **Queue drain** — Implement `drain_queue()` that reads from `self.queue` and updates all Flet controls.
7. **Preview** — Implement original + processed image display with temp-file conversion for processed preview.
8. **Save / Copy** — Implement save (file picker + write) and copy (`page.set_clipboard`).
9. **Batch queue** — Initialize `BatchQueueManager`, build the batch tab UI, wire progress/status/completion callbacks to the queue.
10. **Memory monitor** — Add `page.add_periodic_task` for `psutil` memory update.
11. **Wire main.py** — Replace tkinter branch with Flet launch.
12. **Delete old GUI** — Remove `src/gui_app.py`.
13. **Validation** — Run the checks below.

## Validation

- [ ] `pip install -r requirements.txt` succeeds without `tkinterdnd2`.
- [ ] `python main.py` opens the Flet window.
- [ ] Browse button opens file picker; selecting an image processes it and shows text in Results tab.
- [ ] Save button writes a `.txt` file via save-file dialog.
- [ ] Copy to clipboard works.
- [ ] Batch Queue: add multiple files via browse, start batch, observe progress and completion in the tab.
- [ ] Preprocessing switches affect processing output.
- [ ] `python main.py --help` still shows CLI help.
- [ ] `pytest tests/` passes.
