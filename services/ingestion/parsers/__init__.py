"""Collection of ingestion parsers exposed for pipeline wiring."""

from .base import BaseParser, ParserError, RawBlock
from .docx_parser import DocxParser
from .ebook_parser import EbookParser
from .ocr import OcrCache, OcrEngine, merge_ocr_text
from .pdf_parser import PdfParser
from .pptx_parser import PptxParser
from .scorm_parser import ScormParser
from .xlsx_parser import XlsxParser

__all__ = [
    "BaseParser",
    "ParserError",
    "RawBlock",
    "DocxParser",
    "EbookParser",
    "PdfParser",
    "OcrCache",
    "OcrEngine",
    "merge_ocr_text",
    "PptxParser",
    "ScormParser",
    "XlsxParser",
]   