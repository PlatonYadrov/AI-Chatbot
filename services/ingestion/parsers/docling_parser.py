"""Unified document parser using Docling library.

Docling provides high-quality document understanding for PDF, DOCX, PPTX, images,
and other formats with built-in OCR, table extraction, and structure preservation.

All models run locally - no external API calls.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, TextIO
import time

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from .base import BaseParser, RawBlock, ensure_dependency, ParserError

LOGGER = logging.getLogger(__name__)


class DoclingParser(BaseParser):
    """
    Универсальный парсер документов на базе Docling.

    Возможности:
        - Преобразует документ в поток структурированных блоков (:class:`RawBlock`)
        - Извлекает текст и таблицы, добавляет метаданные (страница, bbox, заголовки)
        - Поддерживает локальный OCR (для PDF) и работает офлайн

    Поддерживаемые форматы:
        PDF, DOCX, PPTX, HTML, XLSX (автодетект входного формата Docling).

    Сохранение Markdown:
        Если указан `output_md_dir`, для каждого входного файла создаётся
        Markdown-файл `{stem}_{ext}.md` в заданной директории.
    """

    def __init__(
        self,
        *,
        extract_tables: bool = True,
        extract_images: bool = True,
        ocr_enabled: bool = True,
        preserve_structure: bool = True,  # зарезервировано
        offline_mode: bool = True,        # зарезервировано
        cache_dir: Optional[str] = None,
        use_vlm: bool = False,
        output_md_dir: Optional[str] = None,
    ) -> None:
        """
        Инициализация парсера Docling.

        Аргументы:
            extract_tables: Извлекать таблицы (если доступны в структуре)
            extract_images: Извлекать изображения/превью страниц (для PDF)
            ocr_enabled: Включить OCR для PDF-сканов
            preserve_structure: (зарезервировано) Сохранять структуру документа
            offline_mode: (зарезервировано) Принудительный офлайн-режим
            cache_dir: Папка кэша для моделей (установит HF_HOME/TRANSFORMERS_CACHE)
            use_vlm: Включить VLM для описания изображений (требуется 'docling[vlm]')
            output_md_dir: Каталог, куда сохранять результат в Markdown
        """
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled
        self.preserve_structure = preserve_structure
        self.offline_mode = offline_mode
        self.use_vlm = use_vlm

        self.cache_dir = cache_dir or os.getenv("DOCLING_CACHE_DIR")
        if self.cache_dir:
            os.environ.setdefault("HF_HOME", self.cache_dir)

        # каталог для .md: создадим по требованию в parse()
        self.output_md_dir = Path(output_md_dir) if output_md_dir else None

    # ---------- Публичные методы ----------

    def parse_to_document(
        self,
        path: str | os.PathLike[str],
        *,
        doc_id: Optional[str] = None,
        markdown: bool = False,
        markdown_only: bool = False,
    ):
        """
        Спарсить файл и вернуть DoclingDocument. Опционально сформировать Markdown.

        Аргументы:
            path: путь к файлу
            doc_id: необязательный идентификатор документа
            markdown: если True — дополнительно вернуть Markdown-текст
            markdown_only: если True — вернуть только Markdown-текст

        Возвращает:
            - DoclingDocument                (по умолчанию)
            - (DoclingDocument, str)         (если markdown=True)
            - str (Markdown)                 (если markdown_only=True)
        """
        ensure_dependency("docling", "pip install docling")
        path_obj = Path(path)
        resolved_id = self._resolve_doc_id(path_obj, doc_id)
        LOGGER.info("docling_parse_to_document_start", extra={"doc_id": resolved_id, "path": str(path_obj), "suffix": path_obj.suffix.lower(), "ocr": self.ocr_enabled, "tables": self.extract_tables, "images": self.extract_images, "vlm": self.use_vlm, "cache_dir": self.cache_dir, "return_markdown": markdown or markdown_only})
        try:
            pipeline_options = PdfPipelineOptions()
            if self.ocr_enabled:
                pipeline_options.ocr_options = TesseractOcrOptions(
                    force_full_page_ocr=False,
                    lang=["rus", "eng"]
                )
            
            pdf_fmt_option = PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
            converter = DocumentConverter(
                format_options={InputFormat.PDF: pdf_fmt_option}
            )
            result = converter.convert(str(path_obj))
            md_text = result.document.export_to_markdown()
            return md_text, result.document
        except FileNotFoundError as exc:
            LOGGER.exception("docling_file_not_found", extra={"path": str(path_obj)})
            raise ParserError(f"Файл не найден: {path_obj}") from exc
        except Exception as exc:
            LOGGER.exception("docling_parse_to_document_failed", extra={"path": str(path_obj)})
            raise ParserError(f"Ошибка Docling при обработке: {path_obj}") from exc
