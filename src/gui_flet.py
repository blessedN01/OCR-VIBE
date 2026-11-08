#!/usr/bin/env python3
"""
Flet GUI Application Module

Modern desktop GUI for the OCR application using Flet, replacing the
legacy tkinter GUI.
"""
from __future__ import annotations

import asyncio
import atexit
import datetime
import logging
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import asyncio
import flet as ft
import psutil

# Compatibility shim: flet >= 1.0 renamed ElevatedButton to FilledButton.
# The rest of this module references ft.ElevatedButton, so alias it back.
if not hasattr(ft, "ElevatedButton"):
    ft.ElevatedButton = ft.FilledButton  # type: ignore[attr-defined]

from .app_info import APP_NAME, APP_NAME_LOWER
from .batch_queue import BatchQueueManager
from .cli_app import OutputFormatter, parse_page_ranges
from .config_loader import get_config_parser, save_config
from .exceptions import (
    ImageProcessingError,
    MemoryError,
    OCREngineNotAvailableError,
    OCRProcessingError,
    PDFProcessingError,
)
from .ocr_engine import OCREngine, get_available_engines
from .pdf_handler import PDFHandler

logger = logging.getLogger(__name__)


TRANSLATIONS = {
    "en": {
        "title": APP_NAME,
        "browse": "Browse",
        "clear_results": "Clear Results",
        "save_results": "Save Results",
        "copy_clipboard": "Copy to Clipboard",
        "clear_log": "Clear Log",
        "cancel": "Cancel",
        "browse_folder": "Browse Folder",
        "recursive": "Recursive",
        "start_batch": "Start Batch",
        "cancel_batch": "Cancel Batch",
        "clear_completed": "Clear Completed",
        "parallel": "Parallel processing",
        "drop_zone": "Drop files here or click Browse\nSupports PNG, JPG, BMP, TIFF, PDF",
        "language": "Language",
        "confidence": "Confidence",
        "engine": "Engine",
        "output_format": "Output format",
        "pages_pdf": "PDF pages (e.g. 1-3,5,7-9)",
        "pages_hint": "All pages if empty",
        "ocr_options": "OCR Options",
        "pdf_options": "PDF Options",
        "preprocessing": "Preprocessing",
        "preview": "Preview",
        "system": "System",
        "about": "About",
        "results": "Results",
        "log": "Log",
        "batch_queue": "Batch Queue",
        "post_process": "Post-process output",
        "extract_tables": "Extract tables from PDF",
        "preserve_layout": "Preserve layout (DBSCAN)",
        "searchable_pdf": "Searchable PDF",
        "deskew": "Deskew",
        "clahe": "CLAHE",
        "upscale": "Upscale (2x)",
        "remove_bg": "Remove background",
        "use_gpu": "Use GPU",
        "memory_limit": "Memory limit (MB)",
        "enable_preprocess": "Enable preprocessing",
        "original_image": "Original image will appear here",
        "processed_image": "Processed image will appear here",
        "original": "Original",
        "processed": "Processed",
        "ready": "Ready",
        "memory_na": "Memory: N/A",
        "processing": "Processing",
        "cancelled": "Processing cancelled",
        "select_files": "Select files to process",
        "select_folder": "Select folder to process",
        "no_files_found": "No supported files found in folder.",
        "please_wait": "Please wait for current processing to complete.",
        "file_not_found": "File not found",
        "file_not_found_with_path": "File not found: {{path}}",
        "unsupported": "Unsupported file type",
        "unsupported_with_path": "Unsupported file type: {{suffix}}",
        "no_results_save": "No results to save.",
        "save_error": "Save error",
        "no_results_copy": "No results to copy.",
        "copied": "Results copied to clipboard.",
        "no_batch_files": "No files in batch queue. Use 'Browse' to add multiple files.",
        "batch_complete": "Batch complete",
        "results_saved": "Results saved to",
        "cancelling": "Cancelling...",
        "dark_mode": "Switch to dark mode",
        "light_mode": "Switch to light mode",
        "python_docx_required": "python-docx is required to save as .docx",
        "format_txt": "TXT",
        "format_json": "JSON",
        "format_csv": "CSV",
        "format_md": "Markdown",
        "format_html": "HTML",
        "format_docx": "DOCX",
        "engine_auto": "Auto",
        "settings": "Settings",
        "help": "Help",
        "bounding_box": "Bounding Box Viewer",
        "pdf_thumbnails": "PDF Thumbnails",
        "spell_check": "Spell Check",
        "statistics": "Statistics",
    },
    "fr": {
        "title": APP_NAME,
        "browse": "Parcourir",
        "clear_results": "Effacer les résultats",
        "save_results": "Enregistrer les résultats",
        "copy_clipboard": "Copier dans le presse-papiers",
        "clear_log": "Vider le journal",
        "cancel": "Annuler",
        "browse_folder": "Parcourir un dossier",
        "recursive": "Récursif",
        "start_batch": "Démarrer le lot",
        "cancel_batch": "Annuler le lot",
        "clear_completed": "Effacer terminés",
        "parallel": "Traitement parallèle",
        "drop_zone": "Déposer des fichiers ici ou cliquer sur Parcourir\nPrend en charge PNG, JPG, BMP, TIFF, PDF",
        "language": "Langue",
        "confidence": "Confiance",
        "engine": "Moteur",
        "output_format": "Format de sortie",
        "pages_pdf": "Pages PDF (ex. 1-3,5,7-9)",
        "pages_hint": "Toutes les pages si vide",
        "ocr_options": "Options OCR",
        "pdf_options": "Options PDF",
        "preprocessing": "Prétraitement",
        "preview": "Aperçu",
        "system": "Système",
        "about": "À propos",
        "results": "Résultats",
        "log": "Journal",
        "batch_queue": "File d'attente",
        "post_process": "Post-traitement du texte",
        "extract_tables": "Extraire les tableaux du PDF",
        "preserve_layout": "Préserver la mise en page (DBSCAN)",
        "searchable_pdf": "PDF interrogeable",
        "deskew": "Redresser",
        "clahe": "CLAHE",
        "upscale": "Agrandir (x2)",
        "remove_bg": "Supprimer l'arrière-plan",
        "use_gpu": "Utiliser le GPU",
        "memory_limit": "Limite mémoire (Mo)",
        "enable_preprocess": "Activer le prétraitement",
        "original_image": "L'image originale s'affichera ici",
        "processed_image": "L'image traitée s'affichera ici",
        "original": "Original",
        "processed": "Traité",
        "ready": "Prêt",
        "memory_na": "Mémoire : N/D",
        "processing": "Traitement en cours",
        "cancelled": "Traitement annulé",
        "select_files": "Sélectionner des fichiers à traiter",
        "select_folder": "Sélectionner un dossier à traiter",
        "no_files_found": "Aucun fichier pris en charge trouvé dans le dossier.",
        "please_wait": "Veuillez attendre la fin du traitement en cours.",
        "file_not_found": "Fichier introuvable",
        "file_not_found_with_path": "Fichier introuvable : {{path}}",
        "unsupported": "Type de fichier non pris en charge",
        "unsupported_with_path": "Type de fichier non pris en charge : {{suffix}}",
        "no_results_save": "Aucun résultat à enregistrer.",
        "save_error": "Erreur d'enregistrement",
        "no_results_copy": "Aucun résultat à copier.",
        "copied": "Résultats copiés dans le presse-papiers.",
        "no_batch_files": "Aucun fichier dans la file d'attente. Utilisez 'Parcourir' pour ajouter des fichiers.",
        "batch_complete": "Lot terminé",
        "results_saved": "Résultats enregistrés dans",
        "cancelling": "Annulation...",
        "dark_mode": "Passer en mode sombre",
        "light_mode": "Passer en mode clair",
        "python_docx_required": "python-docx est requis pour sauvegarder en .docx",
        "format_txt": "TXT",
        "format_json": "JSON",
        "format_csv": "CSV",
        "format_md": "Markdown",
        "format_html": "HTML",
        "format_docx": "DOCX",
        "engine_auto": "Auto",
    },
}

AVAILABLE_LANGUAGES = [
    ('en', 'English'),
    ('fr', 'French'),
    ('de', 'German'),
    ('es', 'Spanish'),
    ('it', 'Italian'),
    ('pt', 'Portuguese'),
    ('nl', 'Dutch'),
    ('ru', 'Russian'),
    ('ja', 'Japanese'),
    ('ko', 'Korean'),
    ('zh', 'Chinese'),
    ('ar', 'Arabic'),
    ('hi', 'Hindi'),
    ('tr', 'Turkish'),
    ('pl', 'Polish'),
    ('cs', 'Czech'),
    ('sv', 'Swedish'),
    ('da', 'Danish'),
    ('fi', 'Finnish'),
    ('el', 'Greek'),
]


class FletGUIApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = APP_NAME
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.window_width = 1100
        self.page.window_height = 850
        self.page.window_min_width = 900
        self.page.window_min_height = 700
        self.page.padding = 20

        self.config = get_config_parser()
        self.lang = self.config.get('ui_language', 'fr')
        self.ocr_engine = OCREngine(
            languages=self.config.get('languages', ['en']),
            confidence_threshold=self.config.get('confidence_threshold', 0.5),
            use_gpu=self.config.get('use_gpu', False),
            preprocess_images=self.config.get('preprocess_images', True),
            memory_limit_mb=self.config.get('memory_limit_mb'),
            deskew=self.config.get('deskew', False),
            clahe=self.config.get('clahe', False),
            upscale=self.config.get('upscale', False),
            remove_bg=self.config.get('remove_bg', False),
            preserve_layout=self.config.get('preserve_layout', False),
            auto_select=self.config.get('auto_select', True),
        )
        self.pdf_handler = PDFHandler(ocr_engine=self.ocr_engine)

        self.processing = False
        self.current_file: Path | None = None
        self.log_messages: list[str] = []
        self.batch_manager: BatchQueueManager | None = None
        self.batch_rows: dict[str, ft.Row] = {}
        self._temp_files: list[str] = []
        self._cancel_event = threading.Event()

        self.queue: queue.Queue = queue.Queue()
        self._ui_dirty = False
        self._state_lock = threading.Lock()
        # Signal for the background loops to stop. Cleared by _on_close so
        # the loops exit cleanly instead of running until interpreter
        # shutdown (which also avoids "import of sys halted" errors from
        # resource cleanup happening during teardown).
        self._running = True
        self._build_ui()
        # Register close handler BEFORE starting background threads so a
        # close event always tears them down in the right order.
        self.page.on_close = self._on_close
        # Flet's Windows IPC pipe transport raises a spurious
        # ConnectionResetError when the session is garbage-collected after a
        # clean close. It is harmless (the session IS collected right after),
        # but it prints a traceback. Suppress it at the asyncio level rather
        # than hiding a real bug.
        self._install_asyncio_exception_filter()
        self.page.run_thread(self._drain_queue_loop)
        self.page.run_thread(self._update_memory_usage_loop)
        atexit.register(self._cleanup_temp_files)

    def t(self, key: str, **kwargs) -> str:
        """Translate a key, optionally interpolating keyword arguments."""
        text = TRANSLATIONS.get(self.lang, TRANSLATIONS["fr"]).get(key, key)
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return text

    def _show_snack(self, message: str):
        self.page.snack_bar = ft.Snackbar(ft.Text(message), open=True)
        self.page.update()

    def _build_ui(self):
        self.title_text = ft.Text(APP_NAME, size=24, weight=ft.FontWeight.BOLD)

        self.theme_button = ft.IconButton(
            icon=ft.Icons.LIGHT_MODE,
            tooltip=self.t("light_mode"),
            on_click=self._toggle_theme,
        )

        self.ui_language_dropdown = ft.Dropdown(
            label=self.t("language"),
            value=self.lang,
            options=[
                ft.dropdown.Option("fr", "Français"),
                ft.dropdown.Option("en", "English"),
            ],
            width=120,
            on_select=self._change_language,
        )

        self.title_row = ft.Row(
            [self.title_text, ft.Container(expand=True), self.ui_language_dropdown, self.theme_button],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        self.language_dropdown = ft.Dropdown(
            label=self.t("language"),
            value=self.config.get('languages', ['en'])[0] if self.config.get('languages') else 'en',
            options=[ft.dropdown.Option(code, name) for code, name in AVAILABLE_LANGUAGES],
            width=220,
        )
        self.confidence_slider = ft.Slider(
            min=0.0,
            max=1.0,
            value=self.config.get('confidence_threshold', 0.5),
            label="{value:.2f}",
            width=220,
        )
        engines = get_available_engines()
        engine_options = [ft.Radio(value="auto", label=self.t("engine_auto"))] + [
            ft.Radio(value=e, label=e.capitalize()) for e in engines
        ]
        self.engine_radio_group = ft.RadioGroup(content=ft.Row(engine_options), value="auto")

        self.output_format_dropdown = ft.Dropdown(
            label=self.t("output_format"),
            value=self.config.get('output_format', 'txt'),
            options=[
                ft.dropdown.Option("txt", self.t("format_txt")),
                ft.dropdown.Option("json", self.t("format_json")),
                ft.dropdown.Option("csv", self.t("format_csv")),
                ft.dropdown.Option("md", self.t("format_md")),
                ft.dropdown.Option("html", self.t("format_html")),
                ft.dropdown.Option("docx", self.t("format_docx")),
            ],
            width=140,
        )

        self.page_range_field = ft.TextField(
            label=self.t("pages_pdf"),
            value="",
            width=200,
            hint_text=self.t("pages_hint"),
        )

        ocr_config_tab = ft.Column(
            [
                ft.Text(self.t("ocr_options"), size=16, weight=ft.FontWeight.BOLD),
                ft.Row(
                    [
                        ft.Column([self.language_dropdown], tight=True),
                        ft.Column([ft.Text(self.t("confidence")), self.confidence_slider], tight=True),
                        ft.Column([ft.Text(self.t("engine")), self.engine_radio_group], tight=True),
                        ft.Column([self.output_format_dropdown], tight=True),
                        ft.Column([self.page_range_field], tight=True),
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=20,
                    wrap=True,
                ),
            ],
            spacing=10,
        )

        self.drop_container = ft.Container(
            content=ft.Text(
                self.t("drop_zone"),
                text_align=ft.TextAlign.CENTER,
                size=14,
            ),
            border=ft.Border.all(2, ft.Colors.with_opacity(0.5, ft.Colors.ON_SURFACE)),
            border_radius=8,
            padding=12,
            ink=True,
            on_click=self._browse_files,
            on_hover=self._on_hover_drop_zone,
        )

        self.browse_button = ft.ElevatedButton(self.t("browse"), on_click=self._browse_files)
        self.clear_button = ft.ElevatedButton(self.t("clear_results"), on_click=self._clear_results, disabled=True)
        self.save_button = ft.ElevatedButton(self.t("save_results"), on_click=self._save_results, disabled=True)
        self.copy_button = ft.ElevatedButton(self.t("copy_clipboard"), on_click=self._copy_to_clipboard, disabled=True)
        self.clear_log_button = ft.ElevatedButton(self.t("clear_log"), on_click=self._clear_log)
        self.cancel_button = ft.ElevatedButton(self.t("cancel"), on_click=self._cancel_processing, disabled=True)

        action_row = ft.Row(
            [self.browse_button, self.clear_button, self.save_button, self.copy_button, self.clear_log_button, self.cancel_button],
            spacing=10,
            wrap=True,
        )

        self.folder_button = ft.ElevatedButton(self.t("browse_folder"), on_click=self._browse_folder)
        self.recursive_checkbox = ft.Checkbox(label=self.t("recursive"), value=False)
        self.folder_row = ft.Row(
            [self.folder_button, self.recursive_checkbox],
            spacing=10,
            visible=False,
        )

        self.gpu_checkbox = ft.Checkbox(label=self.t("use_gpu"), value=self.config.get('use_gpu', False))
        self.memory_limit_field = ft.TextField(
            label=self.t("memory_limit"),
            value=str(self.config.get('memory_limit_mb', '')),
            width=140,
        )
        self.preprocess_global_checkbox = ft.Checkbox(label=self.t("enable_preprocess"), value=self.config.get('preprocess_images', True))

        system_tab = ft.Column(
            [
                ft.Text(self.t("system"), size=16, weight=ft.FontWeight.BOLD),
                ft.Row(
                    [self.gpu_checkbox, self.memory_limit_field, self.preprocess_global_checkbox],
                    spacing=20,
                ),
            ],
            spacing=10,
        )

        self.batch_start_button = ft.ElevatedButton(self.t("start_batch"), on_click=self._start_batch)
        self.batch_cancel_button = ft.ElevatedButton(self.t("cancel_batch"), on_click=self._cancel_batch, disabled=True)
        self.batch_clear_button = ft.ElevatedButton(self.t("clear_completed"), on_click=self._clear_batch_completed)
        self.batch_parallel_checkbox = ft.Checkbox(label=self.t("parallel"), value=False)

        batch_row = ft.Row(
            [self.batch_start_button, self.batch_cancel_button, self.batch_clear_button, self.batch_parallel_checkbox],
            spacing=10,
            wrap=True,
        )

        self.post_process_checkbox = ft.Checkbox(label=self.t("post_process"), value=self.config.get('post_process', False))
        self.extract_tables_checkbox = ft.Checkbox(label=self.t("extract_tables"), value=self.config.get('extract_tables', False))
        self.preserve_layout_checkbox = ft.Checkbox(label=self.t("preserve_layout"), value=self.config.get('preserve_layout', False))
        self.searchable_pdf_checkbox = ft.Checkbox(label=self.t("searchable_pdf"), value=self.config.get('searchable_pdf', False))

        pdf_options_tab = ft.Column(
            [
                ft.Text(self.t("pdf_options"), size=16, weight=ft.FontWeight.BOLD),
                ft.Row(
                    [self.post_process_checkbox, self.extract_tables_checkbox, self.preserve_layout_checkbox, self.searchable_pdf_checkbox],
                    spacing=20,
                    wrap=True,
                ),
            ],
            spacing=10,
        )

        self.deskew_checkbox = ft.Checkbox(label=self.t("deskew"), value=self.config.get('deskew', False))
        self.clahe_checkbox = ft.Checkbox(label=self.t("clahe"), value=self.config.get('clahe', False))
        self.upscale_checkbox = ft.Checkbox(label=self.t("upscale"), value=self.config.get('upscale', False))
        self.remove_bg_checkbox = ft.Checkbox(label=self.t("remove_bg"), value=self.config.get('remove_bg', False))

        preprocessing_tab = ft.Column(
            [
                ft.Text(self.t("preprocessing"), size=16, weight=ft.FontWeight.BOLD),
                ft.Row(
                    [self.deskew_checkbox, self.clahe_checkbox, self.upscale_checkbox, self.remove_bg_checkbox],
                    spacing=20,
                ),
            ],
            spacing=10,
        )

        self.preview_original_image = ft.Image(src="", width=300, height=200, fit=ft.BoxFit.CONTAIN, visible=False)
        self.preview_processed_image = ft.Image(src="", width=300, height=200, fit=ft.BoxFit.CONTAIN, visible=False)
        self.preview_original_text = ft.Text(self.t("original_image"), size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.preview_processed_text = ft.Text(self.t("processed_image"), size=12, color=ft.Colors.ON_SURFACE_VARIANT)

        preview_row = ft.Row(
            [
                ft.Column([self.preview_original_text, self.preview_original_image]),
                ft.Column([self.preview_processed_text, self.preview_processed_image]),
            ],
            spacing=20,
            visible=True,
        )
        self.preview_row = preview_row

        preview_tab = ft.Column(
            [
                ft.Text(self.t("preview"), size=16, weight=ft.FontWeight.BOLD),
                preview_row,
            ],
            spacing=10,
        )
        self.preview_tab = preview_tab
        self.ocr_config_tab = ocr_config_tab
        self.pdf_options_tab = pdf_options_tab
        self.preprocessing_tab = preprocessing_tab
        self.system_tab = system_tab
        self.about_tab = self._build_about_tab()

        self.progress_bar = ft.ProgressBar(value=0.0, width=float("inf"))

        self.status_text = ft.Text(self.t("ready"), size=12)
        self.memory_text = ft.Text(self.t("memory_na"), size=12)
        status_bar = ft.Row([self.status_text, ft.Container(expand=True), self.memory_text], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        self.results_textfield = ft.TextField(
            multiline=True,
            read_only=True,
            border=ft.InputBorder.NONE,
            bgcolor=ft.Colors.with_opacity(0.0, ft.Colors.SURFACE),
            text_style=ft.TextStyle(font_family="Courier", size=12),
            expand=True,
        )

        self.log_textfield = ft.TextField(
            multiline=True,
            read_only=True,
            border=ft.InputBorder.NONE,
            bgcolor=ft.Colors.with_opacity(0.0, ft.Colors.SURFACE),
            text_style=ft.TextStyle(font_family="Courier", size=11),
            expand=True,
        )

        self.batch_column = ft.Column([], spacing=5, expand=True)
        self.batch_results_textfield = ft.TextField(
            multiline=True,
            read_only=True,
            border=ft.InputBorder.NONE,
            bgcolor=ft.Colors.with_opacity(0.0, ft.Colors.SURFACE),
            text_style=ft.TextStyle(font_family="Courier", size=11),
            expand=True,
        )

        results_tab = ft.Column(
            [ft.Container(self.results_textfield, padding=10, expand=True)],
            expand=True,
            scroll=ft.ScrollMode.AUTO,
        )
        log_tab = ft.Column(
            [ft.Container(self.log_textfield, padding=10, expand=True)],
            expand=True,
            scroll=ft.ScrollMode.AUTO,
        )
        batch_tab = ft.Column(
            [
                ft.Text(self.t("batch_queue"), size=16, weight=ft.FontWeight.BOLD),
                self.batch_column,
                self.batch_results_textfield,
            ],
            expand=True,
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
        )

        self.tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            expand=True,
            length=9,
            content=ft.Column(
                expand=True,
                controls=[
                    ft.TabBar(
                        tabs=[
                            ft.Tab(label=self.t("results")),
                            ft.Tab(label=self.t("log")),
                            ft.Tab(label=self.t("batch_queue")),
                            ft.Tab(label=self.t("ocr_options")),
                            ft.Tab(label=self.t("pdf_options")),
                            ft.Tab(label=self.t("preprocessing")),
                            ft.Tab(label=self.t("preview")),
                            ft.Tab(label=self.t("system")),
                            ft.Tab(label=self.t("about")),
                        ],
                    ),
                    ft.TabBarView(
                        expand=True,
                        controls=[
                            results_tab,
                            log_tab,
                            batch_tab,
                            ocr_config_tab,
                            pdf_options_tab,
                            preprocessing_tab,
                            preview_tab,
                            system_tab,
                            self.about_tab,
                        ],
                    ),
                ],
            ),
        )

        self.file_picker = ft.FilePicker()
        self.page.services.append(self.file_picker)

        self.save_file_picker = ft.FilePicker()
        self.page.services.append(self.save_file_picker)

        self.folder_picker = ft.FilePicker()
        self.page.services.append(self.folder_picker)

        top_section = ft.Column(
            [
                self.drop_container,
                action_row,
                self.folder_row,
                batch_row,
                self.progress_bar,
                status_bar,
            ],
            spacing=6,
        )

        self.column = ft.Column(
            [
                self.title_row,
                ft.Container(self.tabs, expand=1),
                top_section,
            ],
            spacing=0,
            expand=True,
        )

    def _change_language(self, e: ft.ControlEvent):
        if hasattr(e, 'control') and e.control:
            self.lang = e.control.value or self.lang or "en"
        else:
            self.lang = getattr(self.ui_language_dropdown, 'value', None) or self.lang or "en"
        try:
            self.config['ui_language'] = self.lang
            save_config(self.config)
        except (AttributeError, OSError):
            pass
        self._apply_translations()
        self.page.update()

    def _apply_translations(self):
        self.page.title = APP_NAME
        if hasattr(self, 'ui_language_dropdown'):
            self.ui_language_dropdown.label = self.t("language")
        if hasattr(self, 'drop_container') and self.drop_container.content and isinstance(self.drop_container.content, ft.Text):
            self.drop_container.content.value = self.t("drop_zone")
        if hasattr(self, 'browse_button'):
            self.browse_button.text = self.t("browse")
        if hasattr(self, 'clear_button'):
            self.clear_button.text = self.t("clear_results")
        if hasattr(self, 'save_button'):
            self.save_button.text = self.t("save_results")
        if hasattr(self, 'copy_button'):
            self.copy_button.text = self.t("copy_clipboard")
        if hasattr(self, 'clear_log_button'):
            self.clear_log_button.text = self.t("clear_log")
        if hasattr(self, 'cancel_button'):
            self.cancel_button.text = self.t("cancel")
        if hasattr(self, 'folder_button'):
            self.folder_button.text = self.t("browse_folder")
        if hasattr(self, 'recursive_checkbox'):
            self.recursive_checkbox.label = self.t("recursive")
        if hasattr(self, 'batch_start_button'):
            self.batch_start_button.text = self.t("start_batch")
        if hasattr(self, 'batch_cancel_button'):
            self.batch_cancel_button.text = self.t("cancel_batch")
        if hasattr(self, 'batch_clear_button'):
            self.batch_clear_button.text = self.t("clear_completed")
        if hasattr(self, 'batch_parallel_checkbox'):
            self.batch_parallel_checkbox.label = self.t("parallel")
        if hasattr(self, 'output_format_dropdown'):
            self.output_format_dropdown.label = self.t("output_format")
        if hasattr(self, 'page_range_field'):
            self.page_range_field.label = self.t("pages_pdf")
            self.page_range_field.hint_text = self.t("pages_hint")
        if hasattr(self, 'post_process_checkbox'):
            self.post_process_checkbox.label = self.t("post_process")
        if hasattr(self, 'extract_tables_checkbox'):
            self.extract_tables_checkbox.label = self.t("extract_tables")
        if hasattr(self, 'preserve_layout_checkbox'):
            self.preserve_layout_checkbox.label = self.t("preserve_layout")
        if hasattr(self, 'searchable_pdf_checkbox'):
            self.searchable_pdf_checkbox.label = self.t("searchable_pdf")
        if hasattr(self, 'deskew_checkbox'):
            self.deskew_checkbox.label = self.t("deskew")
        if hasattr(self, 'clahe_checkbox'):
            self.clahe_checkbox.label = self.t("clahe")
        if hasattr(self, 'upscale_checkbox'):
            self.upscale_checkbox.label = self.t("upscale")
        if hasattr(self, 'remove_bg_checkbox'):
            self.remove_bg_checkbox.label = self.t("remove_bg")
        if hasattr(self, 'gpu_checkbox'):
            self.gpu_checkbox.label = self.t("use_gpu")
        if hasattr(self, 'memory_limit_field'):
            self.memory_limit_field.label = self.t("memory_limit")
        if hasattr(self, 'preprocess_global_checkbox'):
            self.preprocess_global_checkbox.label = self.t("enable_preprocess")
        if hasattr(self, 'preview_original_text'):
            self.preview_original_text.value = self.t("original_image")
        if hasattr(self, 'preview_processed_text'):
            self.preview_processed_text.value = self.t("processed_image")
        if hasattr(self, 'status_text') and self.status_text.value == self.t("ready"):
            self.status_text.value = self.t("ready")
        if hasattr(self, 'memory_text') and self.memory_text.value == self.t("memory_na"):
            self.memory_text.value = self.t("memory_na")
        if hasattr(self, 'theme_button'):
            is_dark = self.page.theme_mode == ft.ThemeMode.DARK
            self.theme_button.tooltip = self.t("light_mode") if is_dark else self.t("dark_mode")

        try:
            if hasattr(self, 'tabs') and self.tabs.content and self.tabs.content.controls:
                tab_bar = self.tabs.content.controls[0]
                labels = [self.t("results"), self.t("log"), self.t("batch_queue"), self.t("ocr_options"), self.t("pdf_options"), self.t("preprocessing"), self.t("preview"), self.t("system"), self.t("about")]
                for i, label in enumerate(labels):
                    if i < len(tab_bar.tabs):
                        tab_bar.tabs[i].label = label
                if hasattr(self, 'ocr_config_tab') and self.ocr_config_tab.controls:
                    self.ocr_config_tab.controls[0].value = self.t("ocr_options")
                if hasattr(self, 'pdf_options_tab') and self.pdf_options_tab.controls:
                    self.pdf_options_tab.controls[0].value = self.t("pdf_options")
                if hasattr(self, 'preprocessing_tab') and self.preprocessing_tab.controls:
                    self.preprocessing_tab.controls[0].value = self.t("preprocessing")
                if hasattr(self, 'preview_tab') and self.preview_tab.controls:
                    self.preview_tab.controls[0].value = self.t("preview")
                if hasattr(self, 'system_tab') and self.system_tab.controls:
                    self.system_tab.controls[0].value = self.t("system")
                if hasattr(self, 'about_tab') and self.about_tab.content and self.about_tab.content.controls:
                    self.about_tab.content.controls[0].value = self.t("about")
        except (AttributeError, IndexError, TypeError):
            pass

    def _toggle_theme(self, e: ft.ControlEvent):
        self.page.theme_mode = (
            ft.ThemeMode.LIGHT if self.page.theme_mode == ft.ThemeMode.DARK else ft.ThemeMode.DARK
        )
        is_dark = self.page.theme_mode == ft.ThemeMode.DARK
        self.theme_button.icon = ft.Icons.LIGHT_MODE if is_dark else ft.Icons.DARK_MODE
        self.theme_button.tooltip = self.t("light_mode") if is_dark else self.t("dark_mode")
        self.page.update()

        self._apply_translations()
        self.page.update()

    def _on_hover_drop_zone(self, e: ft.HoverEvent):
        if e.data == "true":
            self.drop_container.border = ft.Border.all(2, ft.Colors.PRIMARY)
            self.drop_container.bgcolor = ft.Colors.with_opacity(0.1, ft.Colors.PRIMARY)
        else:
            self.drop_container.border = ft.Border.all(2, ft.Colors.with_opacity(0.5, ft.Colors.ON_SURFACE))
            self.drop_container.bgcolor = ft.Colors.with_opacity(0.0, ft.Colors.SURFACE)
        self.page.update()

    async def _browse_files(self, e: ft.ControlEvent):
        try:
            files = await self.file_picker.pick_files(
                dialog_title="Select files to process",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["png", "jpg", "jpeg", "bmp", "tiff", "tif", "pdf"],
            )
            if files:
                paths = [Path(f.path) for f in files]
                if len(paths) == 1:
                    self._process_file(paths[0])
                else:
                    for fp in paths:
                        self._add_to_batch(fp)
        except Exception as ex:
            logger.error(f"File picker error: {ex}")
            self._show_snack(f"Error selecting files: {ex}")

    async def _browse_folder(self, e: ft.ControlEvent):
        try:
            result = await self.folder_picker.get_directory_path(
                dialog_title="Select folder to process",
            )
            if result:
                folder_path = Path(result)
                recursive = self.recursive_checkbox.value
                self._process_folder(folder_path, recursive)
        except Exception as ex:
            logger.error(f"Folder picker error: {ex}")
            self._show_snack(f"Error selecting folder: {ex}")

    def _process_folder(self, folder_path: Path, recursive: bool):
        if self.processing:
            self._show_snack(self.t("please_wait"))
            return
        self._cancel_event.clear()
        supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.pdf'}
        if recursive:
            files = [f for f in folder_path.rglob('*') if f.is_file() and f.suffix.lower() in supported_extensions]
        else:
            files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]

        if not files:
            self._show_snack(self.t("no_files_found"))
            return

        self._set_processing(True)
        self.status_text.value = f"Processing folder: {folder_path.name} ({len(files)} files)"
        self.progress_bar.value = 0.0
        self.results_textfield.value = ""
        self.clear_button.disabled = True
        self.save_button.disabled = True
        self.copy_button.disabled = True
        self.cancel_button.disabled = False
        self.page.update()

        try:
            thread = threading.Thread(target=self._process_folder_thread, args=(folder_path, files), daemon=True)
            thread.start()
        except Exception:
            self._set_processing(False)
            raise

    def _process_file(self, file_path: Path):
        if self.processing:
            self._show_snack(self.t("please_wait"))
            return
        self._cancel_event.clear()
        if not file_path.exists():
            self._show_snack(self.t("file_not_found_with_path", path=file_path))
            return
        if file_path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.pdf'}:
            self._show_snack(self.t("unsupported_with_path", suffix=file_path.suffix))
            return

        self._set_processing(True)
        self.current_file = file_path
        self.status_text.value = f"Processing: {file_path.name}"
        self.progress_bar.value = 0.0
        self.results_textfield.value = ""
        self.clear_button.disabled = True
        self.save_button.disabled = True
        self.copy_button.disabled = True
        self.cancel_button.disabled = False
        self.preview_original_image.visible = False
        self.preview_processed_image.visible = False
        self.preview_original_text.value = self.t("original_image")
        self.preview_processed_text.value = self.t("processed_image")
        self.page.update()

        # Snapshot UI state to avoid reading Flet widgets from a worker thread
        opts = {
            'engine': self.engine_radio_group.value,
            'deskew': self.deskew_checkbox.value,
            'clahe': self.clahe_checkbox.value,
            'upscale': self.upscale_checkbox.value,
            'remove_bg': self.remove_bg_checkbox.value,
            'preserve_layout': self.preserve_layout_checkbox.value,
            'confidence': self.confidence_slider.value,
            'page_range': self.page_range_field.value.strip() or None,
            'post_process': self.post_process_checkbox.value,
            'extract_tables': self.extract_tables_checkbox.value,
            'searchable_pdf': self.searchable_pdf_checkbox.value,
        }
        try:
            self._update_preview(file_path)
            thread = threading.Thread(
                target=self._process_file_thread,
                args=(file_path, opts),
                daemon=True,
            )
            thread.start()
        except Exception:
            self._set_processing(False)
            raise

    def _update_preview(self, file_path: Path):
        try:
            if file_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}:
                import cv2
                from PIL import Image

                with Image.open(file_path) as img:
                    img.thumbnail((300, 200))
                self.preview_original_image.src = str(file_path)
                self.preview_original_image.visible = True
                self.preview_original_text.value = self.t("original")
                self.preview_original_text.visible = True

                img_cv = cv2.imread(str(file_path))
                if img_cv is not None:
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    thresh = cv2.adaptiveThreshold(
                        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                    )
                    img_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    img_pil.thumbnail((300, 200))
                    # delete=False: the file must outlive this scope, it is
                    # referenced by the image control and removed in _cleanup_temp_files
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)  # noqa: SIM115
                    img_pil.save(tmp.name)
                    img_pil.close()
                    self.preview_processed_image.src = tmp.name
                    self.preview_processed_image.visible = True
                    self.preview_processed_text.value = self.t("processed")
                    self.preview_processed_text.visible = True
                    self.preview_processed_image.data = tmp.name
                    self._temp_files.append(tmp.name)

                self.page.update()
        except Exception as ex:
            logger.debug(f"Preview update failed: {ex}")

    def _process_file_thread(self, file_path: Path, opts: dict):
        try:
            if self._cancel_event.is_set():
                self._reset_ui()
                return
            self.queue.put(("progress", 10))
            self._log(f"Starting processing of {file_path.name}")

            engine = None if opts['engine'] == "auto" else opts['engine']

            # Engine configuration mutations must be synchronized: the worker
            # thread and the UI thread may both touch the engine instance.
            with self._state_lock:
                self.ocr_engine.deskew = opts['deskew']
                self.ocr_engine.clahe = opts['clahe']
                self.ocr_engine.upscale = opts['upscale']
                self.ocr_engine.remove_bg = opts['remove_bg']
                self.ocr_engine.preserve_layout = opts['preserve_layout']
                self.ocr_engine.set_confidence_threshold(opts['confidence'])

            page_numbers = None
            if opts['page_range']:
                try:
                    page_numbers = parse_page_ranges(opts['page_range'])
                except ValueError as ex:
                    self.queue.put(("error", f"Invalid page range: {ex}"))
                    self._reset_ui()
                    return

            if self._cancel_event.is_set():
                self.queue.put(("log", "[Cancelled] Processing stopped."))
                self._reset_ui()
                return

            text = None
            metadata = {}
            if file_path.suffix.lower() == '.pdf':
                self.queue.put(("status", f"Detecting PDF type: {file_path.name}"))
                self._log("Detecting PDF type...")

            if self._cancel_event.is_set():
                self.queue.put(("log", "[Cancelled] Processing stopped."))
                self._reset_ui()
                return

            if file_path.suffix.lower() == '.pdf':
                page_count = self.pdf_handler.get_page_count(file_path)
                pdf_type = self.pdf_handler.detect_pdf_type(file_path)
                self._log(f"PDF type: {pdf_type}, pages: {page_count}")
                self.queue.put(("progress", 40))

                if self._cancel_event.is_set():
                    self.queue.put(("log", "[Cancelled] Processing stopped."))
                    self._reset_ui()
                    return

                def _page_progress(page: int, total: int) -> None:
                    """Map per-page progress onto the 40→70 progress band."""
                    if total <= 0:
                        return
                    fraction = page / total
                    # Map [0, 1] → [40, 70]
                    progress_value = int(40 + fraction * 30)
                    self.queue.put(("progress", progress_value))

                text = self.pdf_handler.extract_text(
                    file_path, engine=engine, pages=page_numbers,
                    page_progress_callback=_page_progress,
                )
                self.queue.put(("progress", 70))

                if self._cancel_event.is_set():
                    self.queue.put(("log", "[Cancelled] Processing stopped."))
                    self._reset_ui()
                    return

                self.queue.put(("status", f"Extracting text from {page_count} pages"))
                self._log("Extracting text from PDF...")

                self.queue.put(("progress", 90))

                results = []
                metadata = {
                    'file_path': str(file_path),
                    'file_type': 'PDF',
                    'page_count': page_count,
                    'type': pdf_type,
                    'languages': self.ocr_engine.get_supported_languages(),
                    'confidence_threshold': self.ocr_engine.confidence_threshold,
                }

                if opts['extract_tables']:
                    self._log("Extracting tables from PDF...")
                    try:
                        tables = self.pdf_handler.extract_tables(file_path, pages=','.join(str(p) for p in page_numbers) if page_numbers else None)
                        table_output = self._format_gui_tables(tables)
                        if table_output:
                            text = text + "\n\n" + table_output
                    except Exception as ex:
                        self._log(f"Table extraction failed: {ex}")

                if opts['searchable_pdf']:
                    if self._cancel_event.is_set():
                        self.queue.put(("log", "[Cancelled] Processing stopped."))
                        self._reset_ui()
                        return
                    self._log("Generating searchable PDF...")
                    try:
                        search_path = self.pdf_handler.generate_searchable_pdf(file_path, self.ocr_engine, pages=page_numbers)
                        self._log(f"Searchable PDF saved to: {search_path}")
                    except Exception as ex:
                        self._log(f"Searchable PDF generation failed: {ex}")
            else:
                self.queue.put(("status", f"Preprocessing image: {file_path.name}"))
                self.queue.put(("progress", 30))

                self.queue.put(("status", "Extracting text with OCR"))
                self._log(f"Extracting text using engine: {engine or 'auto'}")
                text, results = self.ocr_engine.extract_text(
                    file_path, engine=engine, file_type='image', page_count=1
                )
                self.queue.put(("progress", 90))

                metadata = {
                    'file_path': str(file_path),
                    'file_type': 'Image',
                    'languages': self.ocr_engine.get_supported_languages(),
                    'confidence_threshold': self.ocr_engine.confidence_threshold,
                }

            if self._cancel_event.is_set():
                self.queue.put(("log", "[Cancelled] Processing stopped."))
                self._reset_ui()
                return

            self.queue.put(("status", "Finalizing results"))
            self._log("Finalizing results...")

            if opts['post_process']:
                self.queue.put(("status", "Post-processing text"))
                self._log("Applying post-processing...")
                text = self.ocr_engine.post_process_text(text, self.ocr_engine.get_supported_languages())

            if self._cancel_event.is_set():
                self.queue.put(("log", "[Cancelled] Processing stopped."))
                self._reset_ui()
                return
            self.queue.put(("progress", 100))
            self.queue.put(("status", "Processing complete"))
            self._log("Processing complete.")
            self.queue.put(("results", text, results if 'results' in locals() else [], metadata))
        except Exception as ex:
            if self._cancel_event.is_set():
                self.queue.put(("log", "[Cancelled] Processing stopped."))
                self._reset_ui()
                return
            if isinstance(ex, (OCREngineNotAvailableError, OCRProcessingError, ImageProcessingError, PDFProcessingError, MemoryError)):
                error_msg = f"Processing error: {ex}"
                if getattr(ex, 'details', None) and 'suggestion' in ex.details:
                    error_msg += f"\n\nSuggestion: {ex.details['suggestion']}"
            elif isinstance(ex, (OSError, ValueError, RuntimeError)):
                error_msg = f"Error processing {file_path}: {ex}"
            elif isinstance(ex, FileNotFoundError):
                error_msg = f"File not found: {file_path}. Please check the file path and ensure it exists."
            elif isinstance(ex, PermissionError):
                error_msg = f"Permission denied accessing {file_path}. Please check file permissions."
            elif isinstance(ex, IsADirectoryError):
                error_msg = f"Path {file_path} is a directory, not a file. Please select a file."
            else:
                error_msg = f"Unexpected error processing {file_path}: {ex}"
            logger.error(error_msg)
            self._log(f"ERROR: {error_msg}")
            self.queue.put(("error", error_msg))

    def _process_folder_thread(self, folder_path: Path, files: list[Path]):
        try:
            output_format = self.output_format_dropdown.value
            page_range = self.page_range_field.value.strip() if self.page_range_field.value else None
            page_numbers = None
            if page_range:
                try:
                    page_numbers = parse_page_ranges(page_range)
                except ValueError as ex:
                    self.queue.put(("error", f"Invalid page range: {ex}"))
                    self._reset_ui()
                    return

            ocr_engine = self.ocr_engine
            pdf_handler = self.pdf_handler

            # Snapshot UI state to avoid reading Flet widgets from a worker thread
            opts = {
                'engine': self.engine_radio_group.value,
                'post_process': self.post_process_checkbox.value,
                'output_format': output_format,
                'page_numbers': page_numbers,
            }

            all_results = []
            successful = 0
            total = len(files)
            file_path = None

            for i, file_path in enumerate(files):
                if self._cancel_event.is_set():
                    self.queue.put(("log", "[Cancelled] Processing stopped."))
                    self._reset_ui()
                    return
                if not self.processing:
                    break
                progress = (i / total) * 100.0
                self.queue.put(("progress", progress))
                self.queue.put(("status", f"Processing {file_path.name} ({i+1}/{total})"))
                self._log(f"Processing {file_path.name}...")

                try:
                    if self._cancel_event.is_set():
                        self.queue.put(("log", "[Cancelled] Processing stopped."))
                        self._reset_ui()
                        return

                    engine = None if opts['engine'] == "auto" else opts['engine']

                    text = ""
                    if file_path.suffix.lower() == ".pdf":
                        self.queue.put(("status", f"Detecting PDF: {file_path.name}"))
                        text = pdf_handler.extract_text(
                            file_path, engine=engine, pages=opts['page_numbers']
                        )
                    else:
                        self.queue.put(("status", f"Preprocessing image: {file_path.name}"))
                        with self._state_lock:
                            text, _img_results = ocr_engine.extract_text(
                                file_path, engine=engine, file_type="image", page_count=1
                            )

                    if opts['post_process'] and text:
                        text = ocr_engine.post_process_text(
                            text, ocr_engine.get_supported_languages()
                        )

                    successful += 1
                    all_results.append(f"=== {file_path.name} ===\n{text}\n")
                except Exception as ex:
                    if self._cancel_event.is_set():
                        self.queue.put(("log", "[Cancelled] Processing stopped."))
                        self._reset_ui()
                        return
                    self._log(f"Failed {file_path.name}: {ex}")

            self.queue.put(("progress", 100))
            self.queue.put(("status", f"Folder complete: {successful}/{total} successful"))
            combined = "\n\n".join(all_results)
            self.queue.put(("results", combined, [], {
                'file_path': str(folder_path),
                'file_type': 'Folder',
                'languages': ocr_engine.get_supported_languages(),
                'confidence_threshold': ocr_engine.confidence_threshold,
                'output_format': output_format,
            }))
        except Exception as ex:
            if self._cancel_event.is_set():
                self.queue.put(("log", "[Cancelled] Processing stopped."))
                self._reset_ui()
                return
            name = file_path.name if file_path else "<unknown>"
            error_msg = f"Unexpected error processing {name}: {ex}"
            logger.error(error_msg)
            self._log(f"ERROR: {error_msg}")
            self.queue.put(("error", error_msg))

    def _drain_queue(self):
        try:
            while True:
                message = self.queue.get_nowait()
                msg_type = message[0]
                self._ui_dirty = True

                if msg_type == "progress":
                    self.progress_bar.value = message[1] / 100.0
                elif msg_type == "status":
                    self.status_text.value = message[1]
                elif msg_type == "memory":
                    self.memory_text.value = message[1]
                elif msg_type == "results":
                    text, results, metadata = message[1], message[2], message[3]
                    self._display_results(text, results, metadata)
                elif msg_type == "error":
                    error_msg = message[1]
                    self.status_text.value = f"Error: {error_msg}"
                    self._show_snack(error_msg)
                    self._reset_ui()
                elif msg_type == "log":
                    log_entry = message[1]
                    if self.log_textfield.value:
                        self.log_textfield.value += log_entry + "\n"
                    else:
                        self.log_textfield.value = log_entry + "\n"
                elif msg_type == "batch_progress":
                    task_id, progress = message[1], message[2]
                    row = self.batch_rows.get(task_id)
                    if row:
                        row.controls[2].value = progress / 100.0
                elif msg_type == "batch_status":
                    task_id, status = message[1], message[2]
                    row = self.batch_rows.get(task_id)
                    if row:
                        row.controls[1].value = status
                elif msg_type == "batch_complete":
                    task_id, success, result_text = message[1], message[2], message[3]
                    row = self.batch_rows.get(task_id)
                    if row:
                        row.controls[1].value = "completed" if success else "failed"
                        row.controls[2].value = 1.0
                    if result_text:
                        if self.batch_results_textfield.value:
                            self.batch_results_textfield.value += "\n\n" + result_text
                        else:
                            self.batch_results_textfield.value = result_text
                elif msg_type == "batch_queue_complete":
                    self.batch_start_button.disabled = False
                    self.batch_cancel_button.disabled = True
                    total, successful, failed = message[1], message[2], message[3]
                    self._show_snack(f"{self.t('batch_complete')}: {successful}/{total} successful, {failed} failed")
                elif msg_type == "batch_log":
                    if self.log_textfield.value:
                        self.log_textfield.value += message[1] + "\n"
                    else:
                        self.log_textfield.value = message[1] + "\n"
        except queue.Empty:
            pass

        if self._ui_dirty:
            self.page.update()
            self._ui_dirty = False

    def _drain_queue_loop(self):
        while self._running:
            self._drain_queue()
            time.sleep(0.1)

    def _update_memory_usage_loop(self):
        while self._running:
            self._update_memory_usage()
            time.sleep(5.0)

    def _display_results(self, text: str, results: list[dict], metadata: dict[str, Any]):
        self.results_textfield.value = text
        info_parts = []
        if metadata.get('file_type') == 'PDF':
            info_parts.append(f"Pages: {metadata.get('page_count', 'Unknown')}")
            info_parts.append(f"Type: {metadata.get('type', 'Unknown')}")
        else:
            info_parts.append(f"Languages: {', '.join(metadata.get('languages', ['en']))}")
            info_parts.append(f"Confidence: {metadata.get('confidence_threshold', 0.5)}")

        if self.post_process_checkbox.value:
            info_parts.append("Post-processed")

        self.status_text.value = " | ".join(info_parts)
        self.clear_button.disabled = False
        self.save_button.disabled = False
        self.copy_button.disabled = False
        self.processing = False
        self.progress_bar.value = 1.0
        self.page.update()

    def _clear_results(self, e: ft.ControlEvent):
        self._cleanup_temp_files()
        self.results_textfield.value = ""
        self.status_text.value = self.t("ready")
        self.clear_button.disabled = True
        self.save_button.disabled = True
        self.copy_button.disabled = True
        self.progress_bar.value = 0.0
        self.page.update()

    def _clear_log(self, e: ft.ControlEvent):
        self.log_messages.clear()
        self.log_textfield.value = ""
        self.page.update()

    def _cleanup_temp_files(self):
        for tmp_path in self._temp_files:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        self._temp_files.clear()

    def _cancel_processing(self, e: ft.ControlEvent):
        if self.processing:
            self._cancel_event.set()
            self.status_text.value = "Cancelling..."
            self.cancel_button.disabled = True
            self.page.update()
            self._log("Cancellation requested...")
            # NOTE: do NOT clear the log here. Clearing it immediately after
            # logging the cancellation request erases the only evidence that
            # cancellation was triggered, and on close it also wipes the
            # shutdown log. Leave the log intact.

    def _install_asyncio_exception_filter(self):
        """
        Suppress Flet's spurious ConnectionResetError during session teardown.

        Flet's Windows IPC pipe transport raises this when the session is
        garbage-collected after a clean close. It is harmless (the session IS
        collected right after), but it prints a traceback. We only suppress
        the specific teardown noise; everything else still propagates.
        """
        # Flet's ft.run creates the event loop before invoking the page callback,
        # so a running loop is available here. Prefer get_running_loop() to
        # avoid the deprecation warning from get_event_loop() when no loop is
        # set yet.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        def _default_handler(loop, context):
            exc = context.get("exception")
            if isinstance(exc, ConnectionResetError):
                logger.debug(
                    f"Suppressed Flet teardown noise: {context.get('message', exc)}"
                )
                return
            loop.default_exception_handler(context)

        loop.set_exception_handler(_default_handler)

    def _on_close(self, e=None):
        """
        Clean shutdown when the window is closed.

        Tears down the background loops, cancels any running processing,
        stops the batch queue, and releases OCR engine resources — instead
        of relying on __del__ during interpreter teardown (which can raise
        "import of sys halted" errors).
        """
        logger.info("Window closing, performing clean shutdown...")
        self._log("Shutting down...")

        # Stop background loops first so they don't touch the page after it's
        # gone. The loops check self._running; set it False and wait briefly
        # for them to notice.
        self._running = False
        for _ in range(50):  # up to ~5s
            time.sleep(0.1)

        # Cancel any in-flight processing and stop the batch queue so worker
        # threads don't keep the process alive after the window closes.
        if self.processing:
            self._cancel_event.set()
        if self.batch_manager is not None:
            try:
                self.batch_manager.stop()
            except Exception as ex:
                logger.debug(f"Batch manager stop error: {ex}")

        # Release OCR engine resources explicitly. __del__ is unreliable
        # during interpreter shutdown, so don't depend on it.
        try:
            self.ocr_engine.cleanup()
        except Exception as ex:
            logger.debug(f"OCR engine cleanup error: {ex}")

        # Remove temporary preview images.
        self._cleanup_temp_files()

        # Shut down Flet's page executor. run_thread() schedules handlers on
        # it, so without this the executor's worker threads keep the process
        # alive after the window closes and the terminal won't exit.
        try:
            executor = self.page.session.connection.executor
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
        except Exception as ex:
            logger.debug(f"Executor shutdown error: {ex}")

    async def _save_results(self, e: ft.ControlEvent):
        text = self.results_textfield.value.strip()
        if not text:
            self._show_snack(self.t("no_results_save"))
            return
        path = await self.save_file_picker.save_file(
            dialog_title="Save Results",
            file_name="ocr_results.txt",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["txt", "md", "json", "csv", "html", "docx"],
        )
        if not path:
            return
        try:
            suffix = Path(path).suffix.lower()
            output_format = self.output_format_dropdown.value
            if suffix == ".docx":
                try:
                    from docx import Document
                except ImportError:
                    self._show_snack(self.t("python_docx_required"))
                    return
                doc = Document()
                if self.current_file:
                    doc.add_heading("OCR Results", level=1)
                    doc.add_paragraph(f"Original file: {self.current_file}")
                    doc.add_paragraph(f"Languages: {', '.join(self.ocr_engine.get_supported_languages())}")
                doc.add_paragraph(text)
                doc.save(path)
            else:
                metadata = {
                    'file_path': str(self.current_file) if self.current_file else 'Unknown',
                    'languages': self.ocr_engine.get_supported_languages(),
                    'confidence_threshold': self.ocr_engine.confidence_threshold,
                    'output_format': output_format,
                }
                if output_format == 'json':
                    content = OutputFormatter.format_json(text, [], metadata)
                elif output_format == 'csv':
                    content = OutputFormatter.format_csv(text, [], metadata)
                elif output_format == 'md':
                    content = OutputFormatter.format_md(text, metadata)
                elif output_format == 'html':
                    content = OutputFormatter.format_html(text, [], metadata)
                else:
                    content = OutputFormatter.format_txt(text, metadata)

                def _write_output():
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)

                await asyncio.to_thread(_write_output)
            self._show_snack(self.t("results_saved") + f" {path}")
        except Exception as ex:
            self._show_snack(self.t("save_error") + f": {ex}")

    def _copy_to_clipboard(self, e: ft.ControlEvent):
        text = self.results_textfield.value.strip()
        if not text:
            self._show_snack(self.t("no_results_copy"))
            return
        self.page.set_clipboard(text)
        self._show_snack(self.t("copied"))

    def _add_to_batch(self, file_path: Path):
        if not self.batch_manager:
            self._init_batch_manager()

        task_id = self.batch_manager.add_file(file_path)
        row = ft.Row(
            [
                ft.Text(file_path.name, expand=True, size=12),
                ft.Text("pending", size=12, width=100),
                ft.ProgressBar(value=0.0, width=120),
            ],
            spacing=10,
        )
        self.batch_rows[task_id] = row
        self.batch_column.controls.append(row)
        self._log(f"Added to batch queue: {file_path.name}")
        self.page.update()

    def _init_batch_manager(self):
        memory_limit = None
        if self.memory_limit_field.value.strip():
            try:
                memory_limit = int(self.memory_limit_field.value.strip())
            except ValueError:
                pass

        config = {
            'post_process': self.post_process_checkbox.value,
            'extract_tables': self.extract_tables_checkbox.value,
            'engine': self.engine_radio_group.value if self.engine_radio_group.value != "auto" else None,
            'preserve_layout': self.preserve_layout_checkbox.value,
            'searchable_pdf': self.searchable_pdf_checkbox.value,
            'deskew': self.deskew_checkbox.value,
            'clahe': self.clahe_checkbox.value,
            'upscale': self.upscale_checkbox.value,
            'remove_bg': self.remove_bg_checkbox.value,
            'preprocess_images': self.preprocess_global_checkbox.value,
            'use_gpu': self.gpu_checkbox.value,
            'memory_limit_mb': memory_limit,
            'languages': self.ocr_engine.languages,
            'confidence_threshold': self.confidence_slider.value,
            'output_format': self.output_format_dropdown.value,
            'parallel': self.batch_parallel_checkbox.value,
        }
        self.batch_manager = BatchQueueManager(
            ocr_engine=self.ocr_engine,
            pdf_handler=self.pdf_handler,
            config=config,
            on_progress=self._on_batch_progress,
            on_status_change=self._on_batch_status_change,
            on_task_complete=self._on_batch_task_complete,
            on_queue_complete=self._on_batch_queue_complete,
            on_log=self._log,
        )

    def _start_batch(self, e: ft.ControlEvent):
        if not self.batch_manager or self.batch_manager.get_pending_count() == 0:
            self._show_snack(self.t("no_batch_files"))
            return
        self.batch_start_button.disabled = True
        self.batch_cancel_button.disabled = False
        self.batch_manager.start()

    def _cancel_batch(self, e: ft.ControlEvent):
        if self.batch_manager:
            self.batch_manager.cancel()
        self.batch_start_button.disabled = False
        self.batch_cancel_button.disabled = True
        self.page.update()

    def _clear_batch_completed(self, e: ft.ControlEvent):
        if self.batch_manager:
            cleared_ids = self.batch_manager.clear_completed()
            for tid in cleared_ids:
                row = self.batch_rows.pop(tid, None)
                if row and row in self.batch_column.controls:
                    self.batch_column.controls.remove(row)
        self.batch_results_textfield.value = ""
        self.page.update()

    def _on_batch_progress(self, task_id: str, progress: float):
        self.queue.put(("batch_progress", task_id, progress))

    def _on_batch_status_change(self, task_id: str, status: str):
        self.queue.put(("batch_status", task_id, status))

    def _on_batch_task_complete(self, task_id: str, success: bool, result_text: str):
        self.queue.put(("batch_complete", task_id, success, result_text))

    def _on_batch_queue_complete(self, total: int, successful: int, failed: int):
        self.queue.put(("batch_queue_complete", total, successful, failed))

    def _log(self, message: str):
        timestamp = datetime.datetime.now().astimezone().strftime('%H:%M:%S')
        entry = f"[{timestamp}] {message}"
        self.log_messages.append(entry)
        if len(self.log_messages) > 1000:
            self.log_messages = self.log_messages[-500:]
        self.queue.put(("log", entry))

    def _update_memory_usage(self):
        try:
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            mem_percent = process.memory_percent()
            self.queue.put(("memory", f"Memory: {mem_mb:.0f} MB ({mem_percent:.1f}%)"))
        except Exception:
            self.queue.put(("memory", self.t("memory_na")))

    def _set_processing(self, value: bool):
        with self._state_lock:
            self.processing = value

    def _reset_ui(self):
        self._set_processing(False)
        self.progress_bar.value = 0.0
        self.status_text.value = self.t("ready")
        self.clear_button.disabled = True
        self.save_button.disabled = True
        self.copy_button.disabled = True
        self.cancel_button.disabled = True
        self.page.update()

    def _show_snack(self, message: str):
        self.page.snack_bar = ft.SnackBar(ft.Text(message), open=True)
        self.page.update()

    @staticmethod
    def _format_gui_tables(tables: list) -> str:
        if not tables:
            return ""
        output_parts = []
        for i, table in enumerate(tables):
            try:
                if not hasattr(table, 'df') or table.df is None:
                    continue
                text = table.df.to_string(index=False)
                output_parts.append(f"--- Table {i + 1} ---\n{text}")
            except Exception as ex:
                logger.warning(f"Failed to format table {i + 1}: {ex}")
        return "\n\n".join(output_parts)

    def _build_about_tab(self):
        engines = get_available_engines()
        engine_rows = []
        for e in engines:
            initialized = (
                (e == 'easyocr' and self.ocr_engine.easyocr_reader is not None)
                or (e == 'paddleocr' and self.ocr_engine.paddleocr_reader is not None)
            )
            if not initialized and e == 'tesseract':
                from .ocr_engine import TESSERACT_AVAILABLE
                initialized = TESSERACT_AVAILABLE
            status = self.t("ready") if initialized else "Not initialized"
            color = ft.Colors.GREEN if initialized else ft.Colors.ORANGE
            engine_rows.append(
                ft.Row([
                    ft.Text(e, width=120),
                    ft.Text(status, color=color, width=120),
                ], spacing=10)
            )

        about_content = ft.Column([
            ft.Text(f"About {APP_NAME}", size=18, weight=ft.FontWeight.BOLD),
            ft.Text("Version 2.0.0", size=14),
            ft.Divider(),
            ft.Text("Available OCR Engines", size=16, weight=ft.FontWeight.BOLD),
            ft.Column(engine_rows, spacing=5),
            ft.Divider(),
            ft.Text("Features", size=16, weight=ft.FontWeight.BOLD),
            ft.Text("• Multi-format: PNG, JPG, JPEG, TIFF, BMP, PDF"),
            ft.Text("• Multiple OCR engines: EasyOCR, PaddleOCR, Tesseract"),
            ft.Text("• Batch and folder processing"),
            ft.Text("• Export: TXT, JSON, CSV, Markdown, HTML, DOCX"),
            ft.Text("• PDF: table extraction, searchable PDF, page range"),
            ft.Text("• Preprocessing: deskew, CLAHE, upscale, background removal"),
            ft.Text("• Layout preservation, language post-processing"),
            ft.Divider(),
            ft.Text(f"Configuration: {APP_NAME_LOWER}.yaml / app data dir / config.yaml"),
        ], spacing=10, scroll=ft.ScrollMode.AUTO)
        return ft.Container(about_content, padding=20, expand=True)

    def build(self) -> ft.Column:
        self._apply_translations()
        if not hasattr(self, 'column'):
            self._build_ui()
        return self.column
