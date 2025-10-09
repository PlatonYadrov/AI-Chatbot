"""Utilities for extracting learning content from SCORM packages."""

from __future__ import annotations

import logging
import re
import zipfile
from typing import Iterator, Optional
from xml.etree import ElementTree

from .base import BaseParser, RawBlock

LOGGER = logging.getLogger(__name__)


class ScormParser(BaseParser):
    """Parse zipped SCORM packages and emit HTML resources as text blocks."""

    MANIFEST = "imsmanifest.xml"

    def parse(self, path: str, *, doc_id: Optional[str] = None) -> Iterator[RawBlock]:
        resolved_id = self._resolve_doc_id(path, doc_id)
        LOGGER.debug("Parsing SCORM package", extra={"doc_id": resolved_id, "path": path})

        with zipfile.ZipFile(path, "r") as archive:
            if self.MANIFEST not in archive.namelist():
                LOGGER.warning("SCORM package missing manifest", extra={"path": path})
                return

            manifest = ElementTree.fromstring(archive.read(self.MANIFEST))
            ns = self._manifest_namespaces(manifest)
            resources = manifest.findall(".//{*}resource", ns)
            for resource in resources:
                href = resource.get("href")
                if not href:
                    continue
                try:
                    with archive.open(href) as handle:
                        raw = handle.read().decode("utf-8", errors="ignore")
                except KeyError:
                    LOGGER.debug("Resource not found in SCORM package", extra={"href": href})
                    continue

                text = self._strip_html(raw)
                if not text:
                    continue

                yield RawBlock(
                    text=text,
                    meta={
                        "doc_id": resolved_id,
                        "type": "scorm",
                        "path": path,
                        "resource": href,
                    },
                )

    @staticmethod
    def _manifest_namespaces(manifest: ElementTree.Element) -> dict[str, str]:
        namespaces = {k if k else "default": v for k, v in manifest.attrib.items() if k.startswith("xmlns")}
        return namespaces

    @staticmethod
    def _strip_html(payload: str) -> str:
        payload = re.sub(r"<script.*?>.*?</script>", " ", payload, flags=re.S | re.I)
        payload = re.sub(r"<style.*?>.*?</style>", " ", payload, flags=re.S | re.I)
        payload = re.sub(r"<[^>]+>", " ", payload)
        payload = re.sub(r"\s+", " ", payload)
        return payload.strip()


__all__ = ["ScormParser"]