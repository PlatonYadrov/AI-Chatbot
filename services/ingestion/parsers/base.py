from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Iterator, Optional


@dataclass
class RawBlock:
    text: str
    meta: Dict[str, Any]


class BaseParser:
    def parse(self, path: str, *, doc_id: Optional[str] = None) -> Iterator[RawBlock]:  # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def _resolve_doc_id(path: str, doc_id: Optional[str]) -> str:
        import hashlib
        if doc_id:
            return doc_id
        return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]


def ensure_dependency(module_name: str, install_hint: str) -> None:
    try:
        __import__(module_name)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Missing dependency '{module_name}'. Try: {install_hint}") from exc

"""Common parser primitives used across ingestion connectors.

The original scaffolding only exposed placeholder functions which made it
impossible to experiment with the parsing layer or to connect subsequent
pipeline stages.  This module introduces a small but fully functional parsing
framework that can be shared by the different document type specific parsers.

The design follows the practical guidance that was added in the documentation
previously: every parser emits :class:`RawBlock` objects that preserve both the
textual payload and rich structural metadata (location within the document,
headings, sheet names, etc.).  Downstream processors such as normalisation,
chunking and embedding can therefore rely on a consistent data model.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RawBlock:
    """Atomic unit extracted from a source document.

    Attributes
    ----------
    text:
        Normalised textual representation of the block.  The text should be
        ready for downstream chunking and must not contain leading/trailing
        whitespace.
    meta:
        Arbitrary metadata payload describing where the block originated from.
        Typical fields include ``doc_id``, ``page``/``slide`` indices and a
        hierarchical list of headings.
    """

    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Return a stable fingerprint for deduplication purposes."""

        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()


class BaseParser:
    """Base class for ingestion parsers.

    Parsers are intentionally kept stateless.  They accept a path-like object
    and lazily yield :class:`RawBlock` instances.  A thin helper around
    :func:`_resolve_doc_id` ensures a deterministic document identifier when it
    is not provided by the caller (e.g. when ingesting from an SMB share).
    """

    def parse(self, path: str | os.PathLike[str], *, doc_id: Optional[str] = None) -> Iterator[RawBlock]:
        raise NotImplementedError

    @staticmethod
    def _resolve_doc_id(path: str | os.PathLike[str], provided: Optional[str]) -> str:
        if provided:
            return provided
        return Path(path).stem

    @staticmethod
    def _normalise(text: str) -> str:
        return text.replace("\u00a0", " ").strip()


class ParserError(RuntimeError):
    """Domain specific error thrown by parsers when they fail irrecoverably."""


def ensure_dependency(module_name: str, install_hint: str) -> None:
    """Raise a helpful error message when an optional dependency is missing."""

    try:
        __import__(module_name)
    except Exception as exc:  # pragma: no cover - executed only when missing
        raise ParserError(
            f"Optional dependency '{module_name}' is required for this parser. "
            f"Install it with `{install_hint}`"
        ) from exc


__all__ = ["RawBlock", "BaseParser", "ParserError", "ensure_dependency"]