"""OCR utilities shared by the rich document parsers.

The practical guides that ship with the repository already describe how OCR
extracted text should be merged back into the surrounding logical blocks.  This
module turns those snippets into reusable code so that parsers can opt in to
image processing without duplicating helpers or forcing hard dependencies on
Pillow/pytesseract in environments where they are not installed.
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from typing import Dict, Optional

LOGGER = logging.getLogger(__name__)


def _lazy_imports():  # pragma: no cover - exercised only when dependencies exist
    try:
        from PIL import Image, ImageOps, ImageFilter  # type: ignore
        import pytesseract  # type: ignore
    except Exception as exc:  # noqa: BLE001 - intentional broad catch for optional deps
        LOGGER.debug("OCR dependencies unavailable", exc_info=exc)
        return None
    return Image, ImageOps, ImageFilter, pytesseract


@dataclass(slots=True)
class OcrCache:
    """A tiny in-memory cache keyed by the fingerprint of image bytes."""

    store: Dict[str, str]

    def __init__(self) -> None:
        self.store = {}

    @staticmethod
    def _fingerprint(data: bytes) -> str:
        return hashlib.sha1(data).hexdigest()

    def get(self, data: bytes) -> Optional[str]:
        return self.store.get(self._fingerprint(data))

    def set(self, data: bytes, value: str) -> None:
        self.store[self._fingerprint(data)] = value


class OcrEngine:
    """Thin wrapper around pytesseract with dependency fallbacks."""

    def __init__(self, *, languages: str = "rus+eng", cache: Optional[OcrCache] = None) -> None:
        self.languages = languages
        self.cache = cache or OcrCache()
        self._imports = None

    def is_available(self) -> bool:
        if self._imports is None:
            self._imports = _lazy_imports()
        return self._imports is not None

    def extract_text(self, image_bytes: bytes) -> str:
        """Return OCR text for the provided image bytes.

        When the optional dependencies are missing the method returns an empty
        string which allows parsers to degrade gracefully in constrained
        environments (e.g. CI or unit tests).
        """

        if not image_bytes:
            return ""

        cached = self.cache.get(image_bytes)
        if cached is not None:
            return cached

        if not self.is_available():
            text = ""
        else:
            text = self._perform_ocr(image_bytes)

        self.cache.set(image_bytes, text)
        return text

    def _perform_ocr(self, image_bytes: bytes) -> str:
        Image, ImageOps, ImageFilter, pytesseract = self._imports  # type: ignore[assignment]
        with Image.open(io.BytesIO(image_bytes)) as image:  # type: ignore[attr-defined]
            image = self._preprocess_image(image, ImageOps, ImageFilter)
            try:
                text = pytesseract.image_to_string(image, lang=self.languages)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - robustness for flaky OCR
                LOGGER.warning("OCR extraction failed", exc_info=exc)
                return ""
        return _normalise_text(text)

    @staticmethod
    def _preprocess_image(image, ImageOps, ImageFilter):  # pragma: no cover - simple plumbing
        width, height = image.size
        target = max(800, min(1600, max(width, height)))
        if max(width, height) < target:
            scale = target / max(width, height)
            image = image.resize((int(width * scale), int(height * scale)))
        image = ImageOps.grayscale(image)
        image = ImageOps.autocontrast(image)
        image = image.filter(ImageFilter.MedianFilter(3))
        return image


def merge_ocr_text(base: str, *, ocr_text: str = "", caption: str | None = None) -> str:
    """Merge OCR output and optional captions with a textual block."""

    parts = []
    if base and base.strip():
        parts.append(base.strip())
    if caption and caption.strip():
        parts.append(f"[Caption] {caption.strip()}")
    if ocr_text and ocr_text.strip():
        parts.append(f"[OCR] {ocr_text.strip()}")
    return "\n\n".join(parts)


def _normalise_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines()).strip()


__all__ = ["OcrCache", "OcrEngine", "merge_ocr_text"]
