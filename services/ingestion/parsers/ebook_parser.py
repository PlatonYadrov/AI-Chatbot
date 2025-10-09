"""EPUB and FB2 parser implementation."""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path
from typing import Iterator, Optional
from xml.etree import ElementTree

from .base import BaseParser, ParserError, RawBlock

LOGGER = logging.getLogger(__name__)


class EbookParser(BaseParser):
    """Parse popular e-book formats (EPUB, FB2) into :class:`RawBlock` objects."""

    def parse(self, path: str, *, doc_id: Optional[str] = None) -> Iterator[RawBlock]:
        resolved_id = self._resolve_doc_id(path, doc_id)
        suffix = Path(path).suffix.lower()
        LOGGER.debug("Parsing e-book", extra={"doc_id": resolved_id, "path": path, "suffix": suffix})

        if suffix == ".epub":
            yield from self._parse_epub(path, resolved_id)
        elif suffix == ".fb2":
            yield from self._parse_fb2(path, resolved_id)
        else:  # pragma: no cover - defensive branch for unsupported formats
            raise ParserError(f"Unsupported e-book format: {suffix}")

    def _parse_epub(self, path: str, doc_id: str) -> Iterator[RawBlock]:
        with zipfile.ZipFile(path, "r") as archive:
            for name in archive.namelist():
                if not name.lower().endswith((".xhtml", ".html")):
                    continue
                text = archive.read(name).decode("utf-8", errors="ignore")
                cleaned = self._strip_html(text)
                if not cleaned:
                    continue
                yield RawBlock(
                    text=cleaned,
                    meta={
                        "doc_id": doc_id,
                        "type": "epub",
                        "path": path,
                        "resource": name,
                    },
                )

    def _parse_fb2(self, path: str, doc_id: str) -> Iterator[RawBlock]:
        tree = ElementTree.parse(path)
        root = tree.getroot()
        namespaces = {k if k else "default": v for k, v in root.attrib.items() if k.startswith("xmlns")}
        for section in root.findall(".//{*}section", namespaces):
            paragraphs = []
            for paragraph in section.findall("{*}p", namespaces):
                text = "".join(paragraph.itertext())
                cleaned = self._strip_html(text)
                if cleaned:
                    paragraphs.append(cleaned)
            if not paragraphs:
                continue
            yield RawBlock(
                text="\n\n".join(paragraphs),
                meta={
                    "doc_id": doc_id,
                    "type": "fb2",
                    "path": path,
                },
            )

    @staticmethod
    def _strip_html(payload: str) -> str:
        payload = re.sub(r"<script.*?>.*?</script>", " ", payload, flags=re.S | re.I)
        payload = re.sub(r"<style.*?>.*?</style>", " ", payload, flags=re.S | re.I)
        payload = re.sub(r"<[^>]+>", " ", payload)
        payload = re.sub(r"\s+", " ", payload)
        return payload.strip()


__all__ = ["EbookParser"]