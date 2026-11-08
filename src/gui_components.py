"""Reusable GUI components for OCR-VIBE Phase 3 UI redesign."""
from __future__ import annotations

from typing import Callable
import flet as ft


class NavRail:
    def __init__(self, on_nav_change: Callable[[str], None], t_func: Callable[[str], str] = None):
        self.on_nav_change = on_nav_change
        self.t_func = t_func or (lambda x: x)
        self.selected = "results"
        
        self.items = [
            {"id": "results", "icon": ft.Icons.DESCRIPTION, "label": self.t_func("results")},
            {"id": "settings", "icon": ft.Icons.SETTINGS, "label": self.t_func("settings")},
            {"id": "batch", "icon": ft.Icons.QUEUE, "label": self.t_func("batch_queue")},
            {"id": "help", "icon": ft.Icons.HELP_OUTLINE, "label": self.t_func("help")},
        ]
        
        self.buttons = []
        for item in self.items:
            btn = ft.IconButton(
                icon=item["icon"],
                tooltip=item["label"],
                on_click=lambda e, id=item["id"]: self._select(id),
            )
            self.buttons.append(btn)
        
        content = ft.Column(
            controls=self.buttons,
            spacing=4,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
        
        self.container = ft.Container(
            content=content,
            width=64,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            padding=ft.padding.only(top=20),
            border=ft.Border(right=ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT)),
        )
    
    def _select(self, nav_id: str):
        self.selected = nav_id
        self.on_nav_change(nav_id)
    
    def build(self):
        return self.container


class AccordionSection:
    def __init__(self, title: str, content: ft.Control, expanded: bool = True):
        self.title = title
        self.content = content
        self.expanded = expanded
        
        self.header = ft.Row(
            [
                ft.Text(title, weight=ft.FontWeight.BOLD, size=14),
                ft.Container(expand=True),
                ft.Icon(ft.Icons.EXPAND_LESS if expanded else ft.Icons.EXPAND_MORE),
            ],
            on_click=self._toggle,
            ink=True,
        )
        
        self.content_container = ft.Container(
            content=content,
            visible=expanded,
            padding=ft.padding.only(left=10, right=10, top=5, bottom=10),
        )
        
        col = ft.Column(
            [
                self.header,
                ft.Divider(height=1, color=ft.Colors.OUTLINE_VARIANT),
                self.content_container,
            ],
            spacing=2,
        )
        
        self.container = ft.Container(
            content=col,
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
            border_radius=8,
            margin=ft.margin.only(bottom=10),
        )
    
    def _toggle(self, e):
        self.expanded = not self.expanded
        self.content_container.visible = self.expanded
        self.header.controls[-1].name = ft.Icons.EXPAND_LESS if self.expanded else ft.Icons.EXPAND_MORE
        if hasattr(e, 'page') and e.page:
            e.page.update()
    
    def build(self):
        return self.container


class DropZone:
    def __init__(self, on_click: Callable, t_func: Callable[[str], str] = None):
        self.t_func = t_func or (lambda x: x)
        content = ft.Column(
            [
                ft.Icon(ft.Icons.CLOUD_UPLOAD_OUTLINED, size=48, color=ft.Colors.PRIMARY),
                ft.Text(self.t_func("drop_zone"), text_align=ft.TextAlign.CENTER, size=14),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        )
        
        self.container = ft.Container(
            content=content,
            border=ft.Border.all(2, ft.Colors.with_opacity(0.5, ft.Colors.ON_SURFACE)),
            border_radius=12,
            padding=40,
            ink=True,
            on_click=on_click,
            on_hover=self._on_hover,
            alignment=ft.alignment.center,
        )
    
    def _on_hover(self, e: ft.HoverEvent):
        if e.data == "true":
            self.container.border = ft.Border.all(2, ft.Colors.PRIMARY)
            self.container.bgcolor = ft.Colors.with_opacity(0.08, ft.Colors.PRIMARY)
        else:
            self.container.border = ft.Border.all(2, ft.Colors.with_opacity(0.5, ft.Colors.ON_SURFACE))
            self.container.bgcolor = None
        if e.page:
            e.page.update()
    
    def build(self):
        return self.container


class PreviewPanel:
    def __init__(self, t_func: Callable[[str], str] = None):
        self.t_func = t_func or (lambda x: x)
        self.original_image = ft.Image(src="", fit=ft.BoxFit.CONTAIN, expand=True, visible=False)
        self.processed_image = ft.Image(src="", fit=ft.BoxFit.CONTAIN, expand=True, visible=False)
        self.original_text = ft.Text(self.t_func("original_image"), size=12)
        self.processed_text = ft.Text(self.t_func("processed_image"), size=12)
        
        original_col = ft.Column([ft.Row([ft.Icon(ft.Icons.IMAGE), self.original_text]), ft.Container(content=self.original_image, expand=True, alignment=ft.alignment.center)], expand=True)
        processed_col = ft.Column([ft.Row([ft.Icon(ft.Icons.AUTO_FIX_HIGH), self.processed_text]), ft.Container(content=self.processed_image, expand=True, alignment=ft.alignment.center)], expand=True)
        row = ft.Row([original_col, ft.VerticalDivider(), processed_col], expand=True, spacing=10)
        self.container = ft.Container(content=row, border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT), border_radius=8, padding=10, expand=True)
    
    def build(self):
        return self.container


class ActionStrip:
    def __init__(self, buttons: list):
        row = ft.Row(buttons, spacing=8, wrap=True)
        self.container = ft.Container(content=row, padding=ft.padding.only(top=8, bottom=8))
    
    def build(self):
        return self.container


class ResultsPanel:
    def __init__(self, results_textfield: ft.TextField = None):
        self.results_textfield = results_textfield or ft.TextField(multiline=True, read_only=True, border=ft.InputBorder.NONE, expand=True, text_style=ft.TextStyle(font_family="Courier New", size=12))
        self.container = ft.Container(content=ft.Column([ft.Container(self.results_textfield, padding=15, expand=True)], expand=True), border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT), border_radius=8, expand=True)
    
    def build(self):
        return self.container


class LogConsole:
    def __init__(self, log_textfield: ft.TextField = None):
        self.log_textfield = log_textfield or ft.TextField(multiline=True, read_only=True, border=ft.InputBorder.NONE, expand=True, text_style=ft.TextStyle(font_family="Courier New", size=11))
        self.container = ft.Container(content=ft.Column([ft.Container(self.log_textfield, padding=10, expand=True)], expand=True), border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT), border_radius=8, expand=True)
    
    def build(self):
        return self.container


class SettingsPanel:
    def __init__(self, sections: list = None, theme_toggle: ft.Control = None):
        controls = []
        if theme_toggle:
            controls.append(ft.Row([ft.Text("Theme", weight=ft.FontWeight.BOLD), ft.Container(expand=True), theme_toggle]))
            controls.append(ft.Divider())
        if sections:
            controls.extend(sections)
        col = ft.Column(controls, spacing=8, scroll=ft.ScrollMode.AUTO, expand=True)
        self.container = ft.Container(content=ft.Container(col, padding=15), expand=True)
    
    def build(self):
        return self.container


class BatchPanel:
    def __init__(self, batch_column: ft.Column = None, batch_results: ft.TextField = None, controls_row: ft.Row = None):
        self.batch_column = batch_column or ft.Column([], spacing=5)
        self.batch_results = batch_results or ft.TextField(multiline=True, read_only=True, border=ft.InputBorder.NONE, expand=True)
        self.controls_row = controls_row or ft.Row([])
        content = ft.Column([self.controls_row, ft.Text("Batch Queue", weight=ft.FontWeight.BOLD), ft.Container(self.batch_column, expand=True), ft.Text("Batch Results", weight=ft.FontWeight.BOLD), ft.Container(self.batch_results, padding=10, border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT), border_radius=8, expand=True)], spacing=10, expand=True)
        self.container = ft.Container(content=ft.Container(content, padding=15), expand=True)
    
    def build(self):
        return self.container


class HelpPanel:
    def __init__(self):
        search = ft.TextField(prefix_icon=ft.Icons.SEARCH, hint_text="Search features...", border_radius=8)
        categories = ft.Tabs(selected_index=0, tabs=[ft.Tab(text="Features"), ft.Tab(text="Shortcuts"), ft.Tab(text="Changelog"), ft.Tab(text="About")])
        features = ft.Column([
            ft.Card(content=ft.Container(ft.Column([ft.Text("Multi-format Support", weight=ft.FontWeight.BOLD), ft.Text("PNG, JPG, BMP, TIFF, PDF")]), padding=10)),
            ft.Card(content=ft.Container(ft.Column([ft.Text("Multiple OCR Engines", weight=ft.FontWeight.BOLD), ft.Text("EasyOCR, PaddleOCR, Tesseract")]), padding=10)),
            ft.Card(content=ft.Container(ft.Column([ft.Text("Batch Processing", weight=ft.FontWeight.BOLD), ft.Text("Process multiple files efficiently")]), padding=10)),
            ft.Card(content=ft.Container(ft.Column([ft.Text("PDF Features", weight=ft.FontWeight.BOLD), ft.Text("Table extraction, searchable PDF, page ranges")]), padding=10)),
        ], spacing=8)
        content = ft.Column([search, categories, ft.Container(features, expand=True, padding=5)], spacing=10, expand=True)
        self.container = ft.Container(content=ft.Container(content, padding=15), expand=True)
    
    def build(self):
        return self.container