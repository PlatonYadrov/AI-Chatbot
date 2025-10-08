"""Text normalization utilities."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(raw_text: str) -> str:
    """
    Normalize text: unicode cleanup, whitespace, OCR artifacts.
    
    Args:
        raw_text: raw extracted text
    
    Returns:
        Normalized text
    """
    if not raw_text:
        return ""
    
    # Unicode normalization (NFKC: compatibility decomposition + canonical composition)
    text = unicodedata.normalize("NFKC", raw_text)
    
    # Remove control characters except newline/tab
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    
    # Fix common OCR artifacts
    text = re.sub(r"[﻿​‌‍]", "", text)  # zero-width chars
    text = re.sub(r"[\u200b-\u200f\u202a-\u202f]", "", text)  # other invisible chars
    
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)  # multiple spaces/tabs → single space
    text = re.sub(r"\n{3,}", "\n\n", text)  # excessive newlines → double
    
    # Remove leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    
    return text.strip()


__all__ = ["normalize_text"]
