from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document


class CustomQdrant(Qdrant):
    """Qdrant adapter that preserves full payload metadata and robustly extracts text.

    Rationale: Our ingestion stores chunk text under the `text` key (plus sometimes
    `page_content`/`content`). LangChain's default mapping may drop payload fields
    when building `Document`. This adapter keeps all metadata (except text fields)
    while setting `Document.page_content` from the best available text key.
    """

    def _document_from_scored_point(self, *args: Any, **kwargs: Any) -> Document | None:
        try:
            # Newer LC variants: (scored_point, content_payload_key, metadata_payload_key, score_threshold)
            if len(args) >= 1 and len(args) <= 4:
                scored_point = args[0]
                content_key = kwargs.get("content_payload_key") or (
                    args[1] if len(args) >= 2 else getattr(self, "content_payload_key", "page_content")
                )
                _ = kwargs.get("metadata_payload_key") or (args[2] if len(args) >= 3 else getattr(self, "metadata_payload_key", "metadata"))
            else:
                # Legacy variant: (scored_point[, score_threshold])
                scored_point = args[0]
                content_key = getattr(self, "content_payload_key", "page_content")

            if not scored_point:
                return None

            payload: Dict[str, Any] = getattr(scored_point, "payload", None) or {}

            text_content = (
                payload.get(content_key)
                or payload.get("text")
                or payload.get("page_content")
                or payload.get("enriched_text")
                or payload.get("content")
            )

            if not text_content or not isinstance(text_content, str) or not text_content.strip():
                return None

            # Preserve all non-text payload keys as metadata
            metadata: Dict[str, Any] = {
                k: v
                for k, v in payload.items()
                if k not in {content_key, "text", "page_content", "enriched_text", "content"}
            }

            return Document(page_content=text_content, metadata=metadata)
        except Exception as e:
            logging.error(f"Error in CustomQdrant._document_from_scored_point: {e}")
            return None

    def _document_from_point(self, *args: Any, **kwargs: Any) -> Document | None:
        try:
            if len(args) >= 1 and len(args) <= 3:
                point = args[0]
                content_key = kwargs.get("content_payload_key") or (
                    args[1] if len(args) >= 2 else getattr(self, "content_payload_key", "page_content")
                )
                _ = kwargs.get("metadata_payload_key") or (args[2] if len(args) >= 3 else getattr(self, "metadata_payload_key", "metadata"))
            else:
                point = args[0]
                content_key = getattr(self, "content_payload_key", "page_content")

            if not point:
                return None

            payload: Dict[str, Any] = getattr(point, "payload", None) or {}

            text_content = (
                payload.get(content_key)
                or payload.get("text")
                or payload.get("page_content")
                or payload.get("enriched_text")
                or payload.get("content")
            )

            if not text_content or not isinstance(text_content, str) or not text_content.strip():
                return None

            metadata: Dict[str, Any] = {
                k: v
                for k, v in payload.items()
                if k not in {content_key, "text", "page_content", "enriched_text", "content"}
            }

            return Document(page_content=text_content, metadata=metadata)
        except Exception as e:
            logging.error(f"Error in CustomQdrant._document_from_point: {e}")
            return None


__all__ = ["CustomQdrant"]


