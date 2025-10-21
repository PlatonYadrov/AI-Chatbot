"""
Гибридный поиск с использованием LangChain EnsembleRetriever.

Объединяет dense (Qdrant семантический поиск) и sparse (BM25 keyword поиск)
с автоматическим RRF (Reciprocal Rank Fusion).
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Qdrant
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

# Импортируем CustomQdrant из локального модуля, чтобы не создавать циклы зависимостей
try:
    from .custom_qdrant import CustomQdrant
    USE_CUSTOM_QDRANT = True
except Exception:
    USE_CUSTOM_QDRANT = False
    logger.warning("CustomQdrant не найден, используем стандартный Qdrant")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_chunks")


class HybridRetriever:
    """
    Гибридный ретривер с использованием LangChain EnsembleRetriever.
    
    Автоматически объединяет:
    - Dense retrieval (Qdrant семантический поиск)
    - Sparse retrieval (BM25 поиск по ключевым словам)
    
    Со встроенным RRF (Reciprocal Rank Fusion).
    """
    
    def __init__(
        self,
        embeddings: Embeddings,
        documents: Optional[List[Document]] = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        k: int = 10
    ):
        """
        Инициализация гибридного ретривера.
        
        Args:
            embeddings: Модель эмбеддингов для dense поиска
            documents: Предзагруженные документы для BM25 (опционально, загрузит из Qdrant если None)
            dense_weight: Вес для dense поиска (0-1)
            sparse_weight: Вес для sparse поиска (0-1)
            k: Количество результатов на каждый ретривер
        """
        self.embeddings = embeddings
        self.k = k
        
        logger.info("🔧 Инициализация HybridRetriever...")
        
        # 1. Dense ретривер (Qdrant)
        logger.info(f"   Dense: Qdrant @ {QDRANT_URL}")
        
        # Используем CustomQdrant если доступен (правильно обрабатывает поле 'text')
        QdrantClass = CustomQdrant if USE_CUSTOM_QDRANT else Qdrant
        
        # Строим явный QdrantClient и передаем его в vectorstore
        client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
        self.qdrant_vectorstore = QdrantClass(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embeddings=embeddings,
        )

        # Явно укажем ключ payload для текста, чтобы брать текст из поля 'text'
        try:
            setattr(self.qdrant_vectorstore, "content_payload_key", "text")
        except Exception:
            pass
        self.dense_retriever = self.qdrant_vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        
        # 2. Sparse ретривер (BM25)
        if documents is None:
            logger.info("   Sparse: Загрузка документов из Qdrant для BM25...")
            documents = self._load_documents_from_qdrant()
        
        logger.info(f"   Sparse: BM25 с {len(documents)} документами")
        self.bm25_retriever = BM25Retriever.from_documents(
            documents,
            k=k
        )
        
        # 3. Ensemble ретривер с RRF
        logger.info(f"   Fusion: RRF (веса: dense={dense_weight}, sparse={sparse_weight})")
        self.ensemble = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.bm25_retriever],
            weights=[dense_weight, sparse_weight]
        )
        
        logger.info("✅ HybridRetriever инициализирован")
    
    def _load_documents_from_qdrant(self) -> List[Document]:
        """
        Загрузка всех документов из Qdrant для BM25 индексации.
        
        Returns:
            Список LangChain Document объектов
        """
        from qdrant_client import QdrantClient
        
        client = QdrantClient(url=QDRANT_URL)
        
        # Прокручиваем все точки в коллекции
        documents = []
        offset = None
        batch_size = 100
        
        logger.info("📥 Загрузка документов из Qdrant...")
        
        while True:
            result = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False  # Векторы не нужны для BM25
            )
            
            points, offset = result
            
            if not points:
                break
            
            for point in points:
                payload = point.payload or {}
                
                # Извлекаем текст из различных полей
                text = (
                    payload.get("text") or
                    payload.get("page_content") or
                    payload.get("content") or
                    payload.get("enriched_text") or
                    ""
                )
                
                # Проверяем что текст не пустой и не None
                if text and isinstance(text, str) and text.strip():
                    # Убираем текстовые поля из метаданных (чтобы не дублировать)
                    metadata = {
                        k: v for k, v in payload.items()
                        if k not in ["text", "page_content", "content", "enriched_text"]
                    }
                    
                    documents.append(
                        Document(page_content=text.strip(), metadata=metadata)
                    )
            
            if offset is None:
                break
        
        logger.info(f"✅ Загружено {len(documents)} документов из Qdrant")
        
        # Проверяем что есть хотя бы несколько документов для BM25
        if len(documents) < 1:
            logger.warning("⚠️ Нет документов для BM25 индекса! Добавляем заглушку.")
            # Добавляем заглушку чтобы BM25 не упал
            documents.append(
                Document(
                    page_content="Placeholder document for BM25 initialization",
                    metadata={"is_placeholder": True}
                )
            )
        
        return documents
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Получить релевантные документы используя гибридный поиск.
        
        Args:
            query: Поисковый запрос
        
        Returns:
            Список релевантных документов (уже объединенных через RRF)
        """
        logger.info(f"🔍 Гибридный поиск: '{query[:50]}...'")
        results = self.ensemble.get_relevant_documents(query)
        
        # Фильтруем документы с пустым page_content
        valid_results = []
        for doc in results:
            if (doc and 
                hasattr(doc, 'page_content') and 
                doc.page_content and 
                isinstance(doc.page_content, str) and 
                doc.page_content.strip()):
                valid_results.append(doc)
        
        logger.info(f"✅ Найдено {len(valid_results)} валидных документов (из {len(results)})")
        return valid_results
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Асинхронная версия get_relevant_documents."""
        return await self.ensemble.aget_relevant_documents(query)
    
    def update_k(self, k: int):
        """
        Обновить количество возвращаемых результатов.
        
        Args:
            k: Новое количество результатов
        """
        self.k = k
        self.dense_retriever.search_kwargs["k"] = k
        self.bm25_retriever.k = k


def create_hybrid_retriever(
    embeddings: Embeddings,
    documents: Optional[List[Document]] = None,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    k: int = 10
) -> HybridRetriever:
    """
    Создать гибридный ретривер с настройками по умолчанию.
    
    Args:
        embeddings: Модель эмбеддингов
        documents: Предзагруженные документы (опционально)
        dense_weight: Вес для dense поиска (0-1)
        sparse_weight: Вес для sparse поиска (0-1)
        k: Количество результатов на ретривер
    
    Returns:
        Настроенный HybridRetriever
    """
    return HybridRetriever(
        embeddings=embeddings,
        documents=documents,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        k=k
    )


# Кэшированный глобальный ретривер для переиспользования
_global_retriever: Optional[HybridRetriever] = None


def get_global_hybrid_retriever(embeddings: Embeddings) -> HybridRetriever:
    """
    Получить глобальный закэшированный гибридный ретривер.
    
    Зачем: Избегаем переиндексации BM25 при каждом запросе.
    Документы загружаются один раз при первом вызове.
    
    Args:
        embeddings: Модель эмбеддингов
    
    Returns:
        Закэшированный HybridRetriever
    """
    global _global_retriever
    
    if _global_retriever is None:
        logger.info("🚀 Создание глобального гибридного ретривера...")
        _global_retriever = create_hybrid_retriever(
            embeddings=embeddings,
            dense_weight=0.5,
            sparse_weight=0.5,
            k=10
        )
    
    return _global_retriever


def reset_global_retriever():
    """
    Сбросить глобальный ретривер.
    
    Зачем: Использовать после добавления новых документов
    для переиндексации BM25.
    """
    global _global_retriever
    _global_retriever = None
    logger.info("🔄 Глобальный ретривер сброшен")


__all__ = [
    "HybridRetriever",
    "create_hybrid_retriever",
    "get_global_hybrid_retriever",
    "reset_global_retriever"
]
