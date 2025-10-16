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
            os.environ.setdefault("TRANSFORMERS_CACHE", self.cache_dir)

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

        LOGGER.info(
            "docling_parse_to_document_start",
            extra={
                "doc_id": resolved_id,
                "path": str(path_obj),
                "suffix": path_obj.suffix.lower(),
                "ocr": self.ocr_enabled,
                "tables": self.extract_tables,
                "images": self.extract_images,
                "vlm": self.use_vlm,
                "cache_dir": self.cache_dir,
                "return_markdown": markdown or markdown_only,
            },
        )

        try:
            converter = self._build_converter_for_path(path_obj)
            result = converter.convert(str(path_obj))
            docling_doc = result.document

            if not (markdown or markdown_only):
                return docling_doc

            # Пытаемся воспользоваться «родным» экспортом, если он есть
            md_text = None
            for attr in ("to_markdown", "as_markdown", "export_markdown", "dump_markdown", "export_to_markdown"):
                print("okak")
                if hasattr(docling_doc, attr):
                    fn = getattr(docling_doc, attr)
                    try:
                        md_text = fn() if callable(fn) else None
                    except Exception:
                        md_text = None
                if md_text:
                    break

            # Фоллбек: построим Markdown из iterate_items()
            if not md_text:
                md_text = self._document_to_markdown(docling_doc)

            if markdown_only:
                return md_text
            return docling_doc, md_text

        except FileNotFoundError as exc:
            LOGGER.exception("docling_file_not_found", extra={"path": str(path_obj)})
            raise ParserError(f"Файл не найден: {path_obj}") from exc
        except Exception as exc:
            LOGGER.exception("docling_parse_to_document_failed", extra={"path": str(path_obj)})
            raise ParserError(f"Ошибка Docling при обработке: {path_obj}") from exc

    # --- НИЖЕ: новый helper ---
    def _document_to_markdown(self, docling_doc) -> str:
        """
        Построить Markdown из структуры Docling: заголовки/параграфы/таблицы.
        Использует _format_table и простые правила для заголовков.
        """
        parts: list[str] = []
        # по желанию — можно добавить шапку: parts.append(f"# {имя_файла}\n")
        for item, level in docling_doc.iterate_items():
            d = item.model_dump() if hasattr(item, "model_dump") else {}
            label = (d.get("label") or "text").lower()

            # Таблица
            if label == "table" and self.extract_tables:
                tbl = d.get("data")
                md_tbl = self._format_table(tbl) if tbl else ""
                if md_tbl:
                    parts.append(md_tbl.strip() + "\n")
                continue

            # Текст
            text = self._clean_text(self._normalise(d.get("text") or ""))
            if not text:
                continue
            if label in {"title", "section_header", "heading"}:
                try:
                    lvl = int(level) if level is not None else 2
                except Exception:
                    lvl = 2
                lvl = min(max(1, lvl + 1), 6)
                parts.append(f"{'#' * lvl} {text}\n")
            else:
                parts.append(text + "\n")
        return "\n".join(parts).strip() + "\n"

    def parse(self, path: str | os.PathLike[str], *, doc_id: Optional[str] = None) -> Iterator[RawBlock]:
        """
        Спарсить документ и отдать поток структурированных блоков (текст/таблицы).
        Если задан `output_md_dir`, параллельно создаётся Markdown-файл.

        Аргументы:
            path: путь к файлу
            doc_id: необязательный идентификатор документа

        Генерирует:
            RawBlock: текст и метаданные по каждому элементу.
        """
        ensure_dependency("docling", "pip install docling")

        path_obj = Path(path)
        resolved_id = self._resolve_doc_id(path_obj, doc_id)
        ext = path_obj.suffix.lower().lstrip(".") or "filetype"

        LOGGER.info(
            "docling_parse_start",
            extra={
                "doc_id": resolved_id,
                "path": str(path_obj),
                "suffix": f".{ext}",
                "ocr": self.ocr_enabled,
                "tables": self.extract_tables,
                "images": self.extract_images,
                "vlm": self.use_vlm,
                "cache_dir": self.cache_dir,
                "offline_mode": self.offline_mode,
            },
        )

        md_fp: Optional[TextIO] = None
        md_path: Optional[Path] = None

        # подготовим markdown-файл, если нужно
        if self.output_md_dir:
            try:
                self.output_md_dir.mkdir(parents=True, exist_ok=True)
                md_filename = f"{path_obj.stem}_{ext}.md"
                md_path = (self.output_md_dir / md_filename).resolve()
                md_fp = md_path.open("w", encoding="utf-8")
                LOGGER.info("md_file_opened", extra={"doc_id": resolved_id, "md_path": str(md_path)})
            except Exception as e:
                LOGGER.exception("md_file_open_failed", extra={"target_dir": str(self.output_md_dir)})
                # не падаем: просто продолжаем без записи MD

        try:
            converter = self._build_converter_for_path(path_obj)
            t0 = time.perf_counter()
            result = converter.convert(str(path_obj))
            docling_doc = result.document
            conv_time = time.perf_counter() - t0

            LOGGER.info(
                "docling_conversion_complete",
                extra={
                    "doc_id": resolved_id,
                    "elapsed_sec": round(conv_time, 3),
                    "num_pages": getattr(docling_doc, "num_pages", None),
                },
            )

            heading_stack: list[str] = []
            block_index = 0
            text_blocks_total = 0
            tables_total = 0

            # заголовок файла в MD
            if md_fp:
                self._md_write_header(md_fp, path_obj.name, ext)

            # Один проход по элементам: текст и таблицы
            for item, level in docling_doc.iterate_items():
                item_dict = item.model_dump() if hasattr(item, "model_dump") else {}
                label = (item_dict.get("label") or "text").lower()

                # Табличные элементы
                if label == "table" and self.extract_tables:
                    table_data = item_dict.get("data")
                    table_text = self._format_table(table_data) if table_data else ""
                    table_text = self._normalise(table_text)
                    if table_text:
                        meta = self._build_meta(
                            doc_id=resolved_id,
                            path=str(path_obj),
                            block_index=block_index,
                            label="table",
                            level=level,
                            item=item,
                        )
                        if md_fp:
                            self._md_write_table(md_fp, table_text, meta)
                        yield RawBlock(text=table_text, meta=meta)
                        block_index += 1
                        tables_total += 1
                    continue

                # Текстовые элементы
                text = item_dict.get("text") or ""
                text = self._normalise(text)
                text = self._clean_text(text)
                if not text:
                    continue

                # Обновление стека заголовков
                if label in {"title", "section_header", "heading"}:
                    heading_stack.append(text)
                    if len(heading_stack) > 3:
                        heading_stack.pop(0)

                meta = self._build_meta(
                    doc_id=resolved_id,
                    path=str(path_obj),
                    block_index=block_index,
                    label=label,
                    level=level,
                    item=item,
                )
                if heading_stack and label not in {"title", "section_header", "heading"}:
                    meta["headings"] = list(heading_stack)

                if md_fp:
                    self._md_write_text(md_fp, text, label, level, meta)

                yield RawBlock(text=text, meta=meta)
                block_index += 1
                text_blocks_total += 1

            LOGGER.info(
                "docling_parse_complete",
                extra={
                    "doc_id": resolved_id,
                    "blocks_total": block_index,
                    "text_blocks": text_blocks_total,
                    "tables_total": tables_total,
                    "md_saved": bool(md_fp),
                    "md_path": str(md_path) if md_path else None,
                },
            )

        except FileNotFoundError as exc:
            LOGGER.exception("docling_file_not_found", extra={"path": str(path_obj)})
            raise ParserError(f"Файл не найден: {path_obj}") from exc
        except Exception as exc:
            LOGGER.exception("docling_parse_failed", extra={"doc_id": resolved_id, "path": str(path_obj)})
            raise ParserError(f"Ошибка Docling при разборе файла: {path_obj}") from exc
        finally:
            if md_fp:
                try:
                    md_fp.close()
                except Exception:
                    LOGGER.exception("md_file_close_failed", extra={"md_path": str(md_path)})

    # ---------- Вспомогательные методы ----------

    def _build_converter_for_path(self, path: Path):
        """
        Создать и сконфигурировать DocumentConverter под входной файл.

        Стратегия:
            - Для PDF настраиваем PdfPipelineOptions/PdfFormatOption (OCR/таблицы/изображения/VLM)
            - Для других форматов (DOCX/PPTX/HTML/XLSX) полагаемся на автодетект Docling
              (без явных format_options), чтобы не зависеть от внутренних классов опций.
        """
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TesseractCliOcrOptions
        from docling.datamodel.base_models import InputFormat
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

        suffix = path.suffix.lower()

        # PDF — настраиваем явно и богато
        if suffix == ".pdf":
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = self.ocr_enabled
            pipeline_options.do_table_structure = self.extract_tables
            if self.ocr_enabled:
                pipeline_options.ocr_options = TesseractCliOcrOptions(
                    force_full_page_ocr=False,
                    lang=["rus", "eng"],
                )

            if self.extract_tables:
                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
                pipeline_options.table_structure_options.do_cell_matching = True

            if self.extract_images:
                pipeline_options.images_scale = 2.0
                pipeline_options.generate_page_images = True
                pipeline_options.generate_picture_images = True

            if self.use_vlm:
                try:
                    pipeline_options.use_vlm = True
                    LOGGER.info("vlm_enabled", extra={"path": str(path)})
                except Exception as e:
                    LOGGER.warning("vlm_unavailable", extra={"path": str(path), "error": str(e)})

            pdf_fmt_option = PdfFormatOption(
                pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
            )
            return DocumentConverter(format_options={InputFormat.PDF: pdf_fmt_option})

        # Иные форматы — отдадим на автодетект Docling.
        # При необходимости можно будет добавить Docx/Pptx/Html/Xlsx FormatOption здесь.
        LOGGER.debug("using_autodetect_converter", extra={"path": str(path), "suffix": suffix})
        return DocumentConverter()

    @staticmethod
    def _build_meta(
        *, doc_id: str, path: str, block_index: int, label: str, level: Any, item: Any
    ) -> Dict[str, Any]:
        """
        Сформировать метаданные блока, включая страницу и bbox (если доступны).

        Возвращает:
            dict с полями: doc_id, type, path, block_index, label, level, page?, bbox?
        """
        meta: Dict[str, Any] = {
            "doc_id": doc_id,
            "type": f"docling_{label}",
            "path": path,
            "block_index": block_index,
            "label": label,
            "level": level,
        }

        prov_list = []
        if hasattr(item, "prov") and item.prov:
            prov_list = item.prov if isinstance(item.prov, list) else [item.prov]

        for prov in prov_list:
            if hasattr(prov, "page_no"):
                meta["page"] = getattr(prov, "page_no", None)
                break

        for prov in prov_list:
            if hasattr(prov, "bbox") and getattr(prov, "bbox", None):
                bbox = prov.bbox
                meta["bbox"] = (bbox.l, bbox.t, bbox.r, bbox.b)
                break

        return meta

    @staticmethod
    def _format_table(table_data) -> str:
        """
        Отформатировать таблицу в Markdown (простой формат).
        Если сетка отсутствует — вернуть строковое представление объекта.

        Возвращает:
            Строку Markdown-таблицы либо строковый фоллбек.
        """
        try:
            if hasattr(table_data, "grid"):
                grid = table_data.grid
                if not grid:
                    return ""
                lines = []
                header = " | ".join(str(cell) for cell in grid[0])
                lines.append(header)
                lines.append(" | ".join(["---"] * len(grid[0])))
                for row in grid[1:]:
                    lines.append(" | ".join(str(cell) for cell in row))
                return "\n".join(lines)
            return str(table_data) if table_data is not None else ""
        except Exception:
            LOGGER.exception("table_format_failed")
            return ""

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Дополнительная нормализация текста:
            - схлопывает повторяющиеся пробелы/табуляции
            - ограничивает подряд идущие пустые строки
        """
        if not text:
            return ""
        import re
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    # ---------- Markdown helpers ----------

    @staticmethod
    def _md_write_header(fp: TextIO, filename: str, ext: str) -> None:
        """Записать заголовок документа в Markdown."""
        fp.write(f"# {filename}\n\n")
        fp.write(f"> Источник: `{filename}` ({ext.upper()})\n\n")

    @staticmethod
    def _md_write_text(fp: TextIO, text: str, label: str, level: Any, meta: Dict[str, Any]) -> None:
        """Записать текстовый блок в Markdown с учётом типа/уровня."""
        # подберём уровень заголовка
        if label in {"title", "section_header", "heading"}:
            # level может быть None/число; поумолчанию H2
            try:
                lvl = int(level) if level is not None else 2
            except Exception:
                lvl = 2
            lvl = min(max(1, lvl + 1), 6)  # немного смягчим уровни
            fp.write(f"{'#' * lvl} {text}\n\n")
        else:
            # обычный параграф
            fp.write(text.rstrip() + "\n\n")

        # служебная приписка со страницей/bbox (если есть)
        page = meta.get("page")
        bbox = meta.get("bbox")
        if page is not None or bbox is not None:
            tail = []
            if page is not None:
                tail.append(f"стр. {page}")
            if bbox is not None:
                l, t, r, b = bbox
                tail.append(f"bbox=({l:.1f},{t:.1f},{r:.1f},{b:.1f})")
            fp.write("> " + " · ".join(tail) + "\n\n")

    @staticmethod
    def _md_write_table(fp: TextIO, md_table: str, meta: Dict[str, Any]) -> None:
        """Записать таблицу в Markdown, с подписью-метаданными."""
        fp.write("\n")
        page = meta.get("page")
        idx = meta.get("table_index")
        caption = f"Таблица{'' if idx is None else f' #{idx}'}"
        if page is not None:
            caption += f" (стр. {page})"
        fp.write(f"**{caption}**\n\n")
        fp.write(md_table.strip() + "\n\n")


__all__ = ["DoclingParser"]

