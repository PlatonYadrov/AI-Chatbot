"""Unified document parser using Docling library.

Docling provides high-quality document understanding for PDF, DOCX, PPTX, images,
and other formats with built-in OCR, table extraction, and structure preservation.

All models run locally - no external API calls.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator, Optional, Dict, Any

from .base import BaseParser, RawBlock, ensure_dependency

LOGGER = logging.getLogger(__name__)


class DoclingParser(BaseParser):
    """Universal document parser powered by Docling.
    
    Supports: PDF, DOCX, PPTX, XLSX, images (PNG, JPG, TIFF), HTML, and more.
    Provides automatic OCR, table extraction, layout analysis, and semantic structure.
    
    Works completely offline with local models.
    """

    def __init__(
        self,
        *,
        extract_tables: bool = True,
        extract_images: bool = True,
        ocr_enabled: bool = True,
        preserve_structure: bool = True,
        offline_mode: bool = True,
        cache_dir: Optional[str] = None,
        use_vlm: bool = False,  # Enable VLM for image understanding
    ) -> None:
        """Initialize DoclingParser.
        
        Args:
            extract_tables: Extract tables from documents
            extract_images: Extract and OCR images
            ocr_enabled: Enable OCR for scanned documents
            preserve_structure: Preserve document structure (headings, sections)
            offline_mode: Force offline mode (no external downloads)
            cache_dir: Custom directory for model cache
        """
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled
        self.preserve_structure = preserve_structure
        self.offline_mode = offline_mode
        self.use_vlm = use_vlm
        self.cache_dir = cache_dir or os.getenv("DOCLING_CACHE_DIR")
        
        # Set cache directory if specified
        if self.cache_dir:
            os.environ["HF_HOME"] = self.cache_dir
            os.environ["TRANSFORMERS_CACHE"] = self.cache_dir

    def parse(self, path: str, *, doc_id: Optional[str] = None) -> Iterator[RawBlock]:
        """Parse document using Docling and yield structured blocks.
        
        Args:
            path: Path to document file
            doc_id: Optional document identifier
            
        Yields:
            RawBlock objects with text and metadata
        """
        ensure_dependency("docling", "pip install docling")
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        
        resolved_id = self._resolve_doc_id(path, doc_id)
        path_obj = Path(path)
        
        LOGGER.info(
            "docling_parse_start",
            extra={
                "doc_id": resolved_id,
                "path": str(path),
                "suffix": path_obj.suffix,
                "offline_mode": self.offline_mode,
            },
        )

        try:
            # Configure pipeline for local operation with ALL Docling models
            pipeline_options = PdfPipelineOptions()
            
            # OCR settings (Tesseract + EasyOCR)
            pipeline_options.do_ocr = self.ocr_enabled
            
            # Table structure detection (TableFormer model)
            pipeline_options.do_table_structure = self.extract_tables
            if self.extract_tables:
                # Use ACCURATE mode for best quality (uses TableFormer model)
                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
                pipeline_options.table_structure_options.do_cell_matching = True
            
            # Document layout analysis (DocLayNet model)
            # Automatically enabled - detects:
            # - titles, section_header, paragraph
            # - list_item, table, figure, caption
            # - formula, code, footnote, header, footer
            
            # Image and figure extraction
            if self.extract_images:
                pipeline_options.images_scale = 2.0  # Higher quality
                pipeline_options.generate_page_images = True
                pipeline_options.generate_picture_images = True
            
            # Vision Language Model (VLM) for image understanding
            if self.use_vlm:
                try:
                    # Enable VLM for image descriptions and understanding
                    # Requires: pip install "docling[vlm]" and GPU recommended
                    pipeline_options.use_vlm = True
                    LOGGER.info("VLM enabled for image understanding")
                except Exception as e:
                    LOGGER.warning(f"VLM not available: {e}")
                    LOGGER.warning("Install with: pip install 'docling[vlm]'")
            
            # Initialize converter with all models (new Docling API)
            # Backend is automatically selected by Docling
            converter = DocumentConverter(
                format_options={
                    "pdf": pipeline_options,
                }
            )
            
            # Convert document (all processing is local)
            # This will use:
            # 1. DocLayNet layout model (~200MB)
            # 2. TableFormer model (~150MB)
            # 3. Tesseract OCR (if enabled)
            # 4. Formula detection model (auto)
            # 5. Code detection model (auto)
            # 6. VLM model (if enabled, ~3-5GB)
            result = converter.convert(str(path))
            docling_doc = result.document
            
            LOGGER.info(
                "docling_conversion_complete",
                extra={
                    "doc_id": resolved_id,
                    "num_pages": getattr(docling_doc, "num_pages", None),
                },
            )
            
            # Track current heading hierarchy for context
            heading_stack: list[str] = []
            page_num = 1
            block_index = 0
            
            # Iterate through document items
            for item, level in docling_doc.iterate_items():
                # Extract item data
                item_dict = item.model_dump() if hasattr(item, "model_dump") else {}
                label = item_dict.get("label", "text")
                text = item_dict.get("text", "")
                
                # Skip empty items
                if not text or not text.strip():
                    continue
                
                # Clean and normalize text
                text = self._clean_text(text)
                if not text:
                    continue
                
                # Update heading stack for context
                if label in ("title", "section_header", "heading"):
                    heading_stack.append(text)
                    # Keep only last 3 levels
                    if len(heading_stack) > 3:
                        heading_stack.pop(0)
                
                # Extract metadata
                metadata: Dict[str, Any] = {
                    "doc_id": resolved_id,
                    "type": f"docling_{label}",
                    "path": str(path),
                    "block_index": block_index,
                    "label": label,
                    "level": level,
                }
                
                # Add page number if available
                if hasattr(item, "prov"):
                    prov_list = item.prov if isinstance(item.prov, list) else [item.prov]
                    for prov in prov_list:
                        if hasattr(prov, "page_no"):
                            page_num = prov.page_no
                            metadata["page"] = page_num
                            break
                
                # Add heading context
                if heading_stack and label not in ("title", "section_header", "heading"):
                    metadata["headings"] = list(heading_stack)
                
                # Add bounding box if available
                if hasattr(item, "prov") and item.prov:
                    prov_list = item.prov if isinstance(item.prov, list) else [item.prov]
                    for prov in prov_list:
                        if hasattr(prov, "bbox"):
                            bbox = prov.bbox
                            metadata["bbox"] = (bbox.l, bbox.t, bbox.r, bbox.b)
                            break
                
                yield RawBlock(text=text, meta=metadata)
                block_index += 1
            
            # Extract tables if enabled
            if self.extract_tables:
                yield from self._extract_tables(docling_doc, resolved_id, str(path))
            
            LOGGER.info(
                "docling_parse_complete",
                extra={"doc_id": resolved_id, "blocks": block_index},
            )
            
        except Exception as exc:
            LOGGER.exception(
                "docling_parse_failed",
                extra={"doc_id": resolved_id, "path": str(path)},
            )
            raise

    def _extract_tables(
        self,
        docling_doc,
        doc_id: str,
        path: str,
    ) -> Iterator[RawBlock]:
        """Extract tables from Docling document.
        
        Args:
            docling_doc: Docling Document object
            doc_id: Document identifier
            path: Document path
            
        Yields:
            RawBlock objects with table data
        """
        try:
            table_index = 0
            for item, level in docling_doc.iterate_items():
                item_dict = item.model_dump() if hasattr(item, "model_dump") else {}
                label = item_dict.get("label", "")
                
                if label != "table":
                    continue
                
                # Try to get table data
                table_data = item_dict.get("data", None)
                if table_data:
                    # Convert table to markdown or CSV format
                    text = self._format_table(table_data)
                    if text:
                        metadata = {
                            "doc_id": doc_id,
                            "type": "docling_table",
                            "path": path,
                            "table_index": table_index,
                            "label": "table",
                        }
                        
                        # Add page if available
                        if hasattr(item, "prov") and item.prov:
                            prov_list = item.prov if isinstance(item.prov, list) else [item.prov]
                            for prov in prov_list:
                                if hasattr(prov, "page_no"):
                                    metadata["page"] = prov.page_no
                                    break
                        
                        yield RawBlock(text=text, meta=metadata)
                        table_index += 1
        except Exception:
            LOGGER.exception("docling_table_extraction_failed", extra={"doc_id": doc_id})

    @staticmethod
    def _format_table(table_data) -> str:
        """Format table data as markdown or CSV.
        
        Args:
            table_data: Table data from Docling
            
        Returns:
            Formatted table string
        """
        try:
            # If table_data has grid attribute (TableData object)
            if hasattr(table_data, "grid"):
                grid = table_data.grid
                if not grid:
                    return ""
                
                # Simple markdown table formatting
                lines = []
                header = " | ".join(str(cell) for cell in grid[0])
                lines.append(header)
                lines.append(" | ".join(["---"] * len(grid[0])))
                
                for row in grid[1:]:
                    lines.append(" | ".join(str(cell) for cell in row))
                
                return "\n".join(lines)
            
            # Fallback: convert to string
            return str(table_data)
        except Exception:
            LOGGER.exception("table_format_failed")
            return ""

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Basic cleaning (Docling already does most of the work)
        text = text.strip()
        
        # Remove excessive whitespace
        import re
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text


__all__ = ["DoclingParser"]

