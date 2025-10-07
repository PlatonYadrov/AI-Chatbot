"""Chunk level deduplication helpers."""

from __future__ import annotations

import hashlib
from typing import Iterable, Iterator, Optional, Set

from ..parsers import RawBlock


class Deduplicator:
    """Simple SHA1 based deduplicator with optional similarity fingerprinting.

    The production system is expected to use MinHash/SimHash, however the SHA1
    fallback already removes exact duplicates and identical paragraphs across
    documents which is sufficient for local testing.
    """

    def __init__(self, *, seen_hashes: Optional[Set[str]] = None) -> None:
        self.seen_hashes: Set[str] = seen_hashes or set()

    def filter_blocks(self, blocks: Iterable[RawBlock]) -> Iterator[RawBlock]:
        for block in blocks:
            digest = self._fingerprint(block.text)
            if digest in self.seen_hashes:
                continue
            self.seen_hashes.add(digest)
            yield block

    @staticmethod
    def _fingerprint(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()


__all__ = ["Deduplicator"]
