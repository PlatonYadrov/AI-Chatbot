from __future__ import annotations

import os
import uuid
import logging
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from langchain_community.vectorstores import Qdrant
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import requests
from generation.llm_service import generate_response

# Import reranking service
try:
    from reranker.cross_encoder_service import rerank_candidates
    RERANKING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Reranking not available: {e}")
    RERANKING_AVAILABLE = False

# Import chat components for conversational RAG
# Зачем: Добавляем диалоговый режим с уточняющими вопросами
from chat.session_manager import SessionManager, ChatSession
from chat.clarification_service import ClarificationService


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_chunks")


class CustomQdrant(Qdrant):
    """Кастомный Qdrant класс для работы с нашей структурой данных."""
    
    def _document_from_scored_point(self, *args, **kwargs):
        """
        Универсальная обёртка: аккуратно распарсить аргументы из разных версий LC.
        Совместим со старыми и новыми вызовами.
        """
        try:
            # Попытка разобрать «новую» сигнатуру (передают 4 аргумента + self)
            # Примерно: (scored_point, content_payload_key, metadata_payload_key, score_threshold)
            if len(args) >= 1 and len(args) <= 4:
                scored_point = args[0]
                content_key = kwargs.get("content_payload_key") or (args[1] if len(args) >= 2 else getattr(self, "content_payload_key", "page_content"))
                meta_key    = kwargs.get("metadata_payload_key") or (args[2] if len(args) >= 3 else getattr(self, "metadata_payload_key", "metadata"))
                score_thr   = kwargs.get("score_threshold") or (args[3] if len(args) >= 4 else None)
            else:
                # Фоллбэк на «старую» (self, scored_point[, score_threshold])
                scored_point = args[0]
                score_thr    = args[1] if len(args) >= 2 else kwargs.get("score_threshold")
                content_key  = getattr(self, "content_payload_key", "page_content")
                meta_key     = getattr(self, "metadata_payload_key", "metadata")

            if not scored_point:
                return None
            
            payload = getattr(scored_point, "payload", None) or {}
            
            # Ищем текст в разных полях (поддерживаем разные ключи)
            text_content = (payload.get(content_key) or 
                          payload.get('page_content') or 
                          payload.get('text') or 
                          payload.get('enriched_text') or 
                          payload.get('content'))
            
            # Если текста нет, пропускаем документ
            if not text_content or not text_content.strip():
                return None
            
            # Создаем metadata без текстовых полей
            metadata = {}
            if payload:
                for key, value in payload.items():
                    if key not in [content_key, 'page_content', 'text', 'enriched_text', 'content']:
                        metadata[key] = value
            
            # Создаем Document с правильным page_content
            return Document(
                page_content=text_content,
                metadata=metadata
            )
        except Exception as e:
            logging.error(f"Error in _document_from_scored_point: {e}")
            return None
    
    def _document_from_point(self, *args, **kwargs):
        """
        Универсальная обёртка для _document_from_point: совместим со старыми и новыми вызовами.
        """
        try:
            # Попытка разобрать «новую» сигнатуру
            if len(args) >= 1 and len(args) <= 3:
                point = args[0]
                content_key = kwargs.get("content_payload_key") or (args[1] if len(args) >= 2 else getattr(self, "content_payload_key", "page_content"))
                meta_key    = kwargs.get("metadata_payload_key") or (args[2] if len(args) >= 3 else getattr(self, "metadata_payload_key", "metadata"))
            else:
                # Фоллбэк на «старую» (self, point)
                point = args[0]
                content_key  = getattr(self, "content_payload_key", "page_content")
                meta_key     = getattr(self, "metadata_payload_key", "metadata")

            if not point:
                return None
            
            payload = getattr(point, "payload", None) or {}
            
            # Ищем текст в разных полях (поддерживаем разные ключи)
            text_content = (payload.get(content_key) or 
                          payload.get('page_content') or 
                          payload.get('text') or 
                          payload.get('enriched_text') or 
                          payload.get('content'))
            
            # Если текста нет, пропускаем документ
            if not text_content or not text_content.strip():
                return None
            
            # Создаем metadata без текстовых полей
            metadata = {}
            if payload:
                for key, value in payload.items():
                    if key not in [content_key, 'page_content', 'text', 'enriched_text', 'content']:
                        metadata[key] = value
            
            # Создаем Document с правильным page_content
            return Document(
                page_content=text_content,
                metadata=metadata
            )
        except Exception as e:
            logging.error(f"Error in _document_from_point: {e}")
            return None


class TEIEmbeddings(Embeddings):
    def __init__(self, base_url: str = os.getenv("EMBEDDINGS_BASE_URL", "http://embeddings:80")):
        self.base_url = base_url.rstrip("/")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        resp = requests.post(f"{self.base_url}/embed", json={"inputs": texts}, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, list):
            return [p.get("embedding", []) if isinstance(p, dict) else p for p in payload]
        if isinstance(payload, dict):
            if "data" in payload:
                return [item.get("embedding", []) for item in payload.get("data", [])]
            if "embeddings" in payload:
                return payload.get("embeddings", [])
            if "embedding" in payload:
                return [payload.get("embedding", [])]
        return []


class QueryRequest(BaseModel):
    """Модель запроса для обычного RAG (без диалога)."""
    query: str
    k: int = 4
    include_vectors: bool = False
    vector_top_n: int = 10
    use_reranking: bool = True
    rerank_top_k: Optional[int] = None


class ChatRequest(BaseModel):
    """
    Модель запроса для диалогового RAG с уточняющими вопросами.
    
    Зачем: Поддержка сессий и уточнений для более точных ответов.
    """
    query: str
    session_id: Optional[str] = None  # ID сессии для продолжения диалога
    skip_clarification: bool = False  # Пропустить уточнения (сразу ответить)
    clarification_answers: Optional[Dict[str, str]] = None  # Ответы на уточняющие вопросы
    k: int = 4  # Количество документов для поиска
    use_reranking: bool = True  # Использовать реранкинг
    rerank_top_k: Optional[int] = None  # Количество документов после реранкинга


class ChatResponse(BaseModel):
    """
    Модель ответа для диалогового RAG.
    
    Зачем: Структурированный ответ с поддержкой уточнений и истории.
    """
    session_id: str  # ID сессии для продолжения
    state: str  # "awaiting_clarification" или "completed"
    clarification_questions: Optional[List[str]] = None  # Уточняющие вопросы (если нужны)
    clarification_priorities: Optional[List[str]] = None  # Приоритеты вопросов
    answer: Optional[str] = None  # Финальный ответ (если state="completed")
    sources: Optional[List[Dict]] = None  # Источники (если state="completed")
    conversation_history: List[Dict] = []  # История диалога


app = FastAPI(title="RAG Gateway")

# Инициализация сервисов для диалогового режима
# Зачем: Создаем глобальные экземпляры для управления сессиями и уточнениями
session_manager = SessionManager()
clarification_service = ClarificationService()


@app.post("/query")
def query_endpoint(req: QueryRequest):
    embeddings = TEIEmbeddings()
    vs = CustomQdrant.from_existing_collection(
        embedding=embeddings,
        collection_name=QDRANT_COLLECTION,
        url=QDRANT_URL,
        prefer_grpc=False,
        path=None,
    )
    
    # Initial retrieval - get more candidates if reranking is enabled
    initial_k = req.k * 3 if (req.use_reranking and RERANKING_AVAILABLE) else req.k
    docs = vs.similarity_search(req.query, k=initial_k)
    
    # Дополнительная фильтрация (кастомный класс уже обрабатывает пустые документы)
    docs = [doc for doc in docs if doc and doc.page_content and doc.page_content.strip()]
    
    # Check if we have any valid documents
    if not docs:
        logging.warning("No valid documents found in search results")
        return {
            "answer": "Извините, не удалось найти релевантную информацию для вашего запроса.",
            "sources": [],
            "processing_time_ms": 0,
            "error": "No valid documents found"
        }
    
    # Apply reranking if enabled and available
    if req.use_reranking and RERANKING_AVAILABLE and docs:
        try:
            # Convert docs to format expected by reranker
            candidates = []
            for d in docs:
                meta = getattr(d, 'metadata', {}) or {}
                # Добавляем текст для реранкинга
                if hasattr(d, 'page_content') and d.page_content:
                    meta = {**meta, 'text': d.page_content}
                candidates.append(meta)
            
            # Rerank candidates
            rerank_k = req.rerank_top_k or req.k
            reranked_candidates = rerank_candidates(req.query, candidates, top_k=rerank_k)
            
            # Update docs with reranked order
            if reranked_candidates:
                docs = docs[:len(reranked_candidates)]  # Keep original docs but limit to reranked count
            else:
                logging.warning("Reranking returned no candidates, using original docs")
                docs = docs[:req.k]
            
            logging.info(f"Reranking applied: {len(candidates)} -> {len(reranked_candidates)} candidates")
            
        except Exception as e:
            logging.error(f"Reranking failed, using original order: {e}")
            # Fallback to original order
            docs = docs[:req.k]
    
    # Limit to requested number if no reranking
    elif not req.use_reranking or not RERANKING_AVAILABLE:
        docs = docs[:req.k]
    
    # Build context safely
    context_parts = []
    for d in docs:
        if hasattr(d, 'page_content') and d.page_content and d.page_content.strip():
            context_parts.append(d.page_content.strip())
    
    if not context_parts:
        logging.warning("No valid context found for generation")
        return {
            "answer": "Извините, не удалось найти релевантную информацию для вашего запроса.",
            "sources": [],
            "processing_time_ms": 0,
            "error": "No valid context found"
        }
    
    context = "\n\n".join(context_parts)
    prompt = f"Answer the question using the context.\n\nContext:\n{context}\n\nQuestion: {req.query}"
    answer, thoughts = generate_response(prompt)

    sources = []
    for d in docs:
        # Получаем metadata и text из Document (уже правильно обработанного)
        meta = getattr(d, 'metadata', {}) or {}
        text_content = getattr(d, 'page_content', '')
        
        # Добавляем текст в metadata для ответа
        meta = {**meta, 'text': text_content}
        sources.append(meta)
    if req.include_vectors:
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=QDRANT_URL)
            # fetch points with vectors by chunk_id
            enriched = []
            for meta in sources:
                cid = meta.get("chunk_id")
                if not cid:
                    enriched.append(meta)
                    continue
                # Compute deterministic UUID5 from chunk_id to match ingestion IDs
                try:
                    pid = str(uuid.uuid5(uuid.NAMESPACE_URL, str(cid)))
                except Exception:
                    pid = str(cid)
                res = client.retrieve(collection_name=QDRANT_COLLECTION, ids=[pid], with_vectors=True)
                if res:
                    record = res[0]
                    vec = getattr(record, "vector", None)
                    # Qdrant may return named vectors under 'vector' (dict) or 'vectors'
                    if isinstance(vec, dict):
                        vec = next(iter(vec.values()), None)
                    if vec is None:
                        vectors_dict = getattr(record, "vectors", None)
                        if isinstance(vectors_dict, dict):
                            vec = next(iter(vectors_dict.values()), None)
                    if isinstance(vec, (list, tuple)) and len(vec) > 0:
                        head_n = max(0, int(req.vector_top_n))
                        meta = {**meta, "vector_head": list(vec)[: head_n], "vector_dim": len(vec)}
                enriched.append(meta)
            sources = enriched
        except Exception:
            # if enrichment fails, return sources as-is
            pass

    return {"answer": answer, "thoughts": thoughts, "sources": sources}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    Диалоговый RAG endpoint с поддержкой уточняющих вопросов.
    
    Зачем: Улучшает качество ответов, запрашивая уточнения для расплывчатых запросов.
    
    Поток работы:
    1. Первый запрос: Анализ нужны ли уточнения → возврат вопросов
    2. Второй запрос (с ответами): Генерация финального ответа с учетом уточнений
    
    Args:
        req: ChatRequest с запросом и опциональными уточнениями
    
    Returns:
        ChatResponse с вопросами или финальным ответом
    """
    
    # Получаем или создаем сессию
    # Зачем: Сохраняем контекст диалога между запросами
    session = session_manager.get_or_create_session(req.session_id)
    session.add_message("user", req.query)
    
    # Обрабатываем ответы на уточняющие вопросы (если есть)
    # Зачем: Сохраняем уточнения для расширения запроса
    if req.clarification_answers:
        for question, answer in req.clarification_answers.items():
            session.add_clarification(question, answer)
            session.add_message("clarification", f"Q: {question}\nA: {answer}")
    
    # Проверяем нужны ли уточнения (если не пропущено явно)
    # Зачем: Определяем достаточно ли информации для качественного ответа
    if (not req.skip_clarification and 
        session.state == "initial" and 
        not req.clarification_answers):
        
        # Анализируем запрос
        conversation_context = session.get_conversation_context()
        needs_clarification, reason, confidence = clarification_service.should_clarify(
            req.query, 
            conversation_context
        )
        
        # Если нужны уточнения - генерируем вопросы
        if needs_clarification and confidence > 0.6:  # Порог уверенности
            logging.info(f"Требуются уточнения: {reason} (confidence={confidence})")
            
            # Генерируем 2-3 уточняющих вопроса
            questions, priorities = clarification_service.generate_clarification_questions(
                req.query,
                conversation_context,
                num_questions=3
            )
            
            session.set_pending_questions(questions)
            session.add_message("assistant", f"Нужны уточнения: {reason}")
            
            return ChatResponse(
                session_id=session.session_id,
                state="awaiting_clarification",
                clarification_questions=questions,
                clarification_priorities=priorities,
                conversation_history=[m for m in session.messages]
            )
    
    # Расширяем запрос уточнениями (если есть)
    # Зачем: Делаем запрос более конкретным для лучшего поиска
    enhanced_query = req.query
    if session.clarifications:
        enhanced_query = clarification_service.enhance_query_with_clarifications(
            req.query,
            session.clarifications
        )
        logging.info(f"Запрос расширен {len(session.clarifications)} уточнениями")
    
    # === Выполняем RAG поиск (аналогично /query endpoint) ===
    
    embeddings = TEIEmbeddings()
    vs = CustomQdrant.from_existing_collection(
        embedding=embeddings,
        collection_name=QDRANT_COLLECTION,
        url=QDRANT_URL,
        prefer_grpc=False,
        path=None,
    )
    
    # Начальный поиск (больше кандидатов если используем реранкинг)
    initial_k = req.k * 3 if (req.use_reranking and RERANKING_AVAILABLE) else req.k
    docs = vs.similarity_search(enhanced_query, k=initial_k)
    
    # Фильтруем валидные документы
    docs = [doc for doc in docs if doc and doc.page_content and doc.page_content.strip()]
    
    if not docs:
        session.add_message("assistant", "Информация не найдена")
        return ChatResponse(
            session_id=session.session_id,
            state="completed",
            answer="Извините, не удалось найти релевантную информацию для вашего запроса.",
            sources=[],
            conversation_history=[m for m in session.messages]
        )
    
    # Применяем реранкинг (если включен)
    if req.use_reranking and RERANKING_AVAILABLE and docs:
        try:
            candidates = []
            for d in docs:
                meta = getattr(d, 'metadata', {}) or {}
                if hasattr(d, 'page_content') and d.page_content:
                    meta = {**meta, 'text': d.page_content}
                candidates.append(meta)
            
            rerank_k = req.rerank_top_k or req.k
            reranked_candidates = rerank_candidates(enhanced_query, candidates, top_k=rerank_k)
            
            if reranked_candidates:
                docs = docs[:len(reranked_candidates)]
            else:
                docs = docs[:req.k]
                
        except Exception as e:
            logging.error(f"Reranking failed: {e}")
            docs = docs[:req.k]
    elif not req.use_reranking or not RERANKING_AVAILABLE:
        docs = docs[:req.k]
    
    # Строим контекст
    context_parts = []
    for d in docs:
        if hasattr(d, 'page_content') and d.page_content and d.page_content.strip():
            context_parts.append(d.page_content.strip())
    
    if not context_parts:
        session.add_message("assistant", "Контекст не найден")
        return ChatResponse(
            session_id=session.session_id,
            state="completed",
            answer="Извините, не удалось найти релевантную информацию.",
            sources=[],
            conversation_history=[m for m in session.messages]
        )
    
    context = "\n\n".join(context_parts)
    
    # Строим промпт с учетом истории диалога
    # Зачем: LLM видит предыдущие сообщения и уточнения
    conversation_history = session.get_conversation_context()
    clarifications_text = session.get_clarifications_text()
    
    prompt = f"""Ответь на вопрос используя контекст и историю диалога.

История диалога:
{conversation_history}

{f"Уточнения от пользователя:\n{clarifications_text}\n" if clarifications_text else ""}

Контекст из базы знаний:
{context}

Текущий вопрос: {req.query}

Предоставь развернутый и точный ответ на основе контекста и предыдущего диалога."""
    
    # Генерируем ответ
    answer, thoughts = generate_response(prompt, max_tokens=800)
    session.add_message("assistant", answer)
    
    # Собираем источники
    sources = []
    for d in docs:
        meta = getattr(d, 'metadata', {}) or {}
        text_content = getattr(d, 'page_content', '')
        meta = {**meta, 'text': text_content}
        sources.append(meta)
    
    return ChatResponse(
        session_id=session.session_id,
        state="completed",
        answer=answer,
        sources=sources,
        conversation_history=[m for m in session.messages]
    )


@app.delete("/chat/{session_id}")
def delete_chat_session(session_id: str):
    """
    Удалить сессию диалога.
    
    Зачем: Позволяет пользователю явно завершить диалог и очистить историю.
    
    Args:
        session_id: ID сессии для удаления
    
    Returns:
        Статус удаления
    """
    session_manager.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.get("/chat/{session_id}")
def get_chat_session(session_id: str):
    """
    Получить информацию о сессии диалога.
    
    Зачем: Позволяет просмотреть историю диалога и текущее состояние.
    
    Args:
        session_id: ID сессии
    
    Returns:
        Данные сессии
    
    Raises:
        HTTPException: Если сессия не найдена
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@app.get("/chat/stats")
def get_chat_stats():
    """
    Получить статистику по всем сессиям.
    
    Зачем: Для мониторинга и отладки системы.
    
    Returns:
        Статистика сессий
    """
    return session_manager.get_stats()


@app.post("/chat/cleanup")
def cleanup_expired_sessions():
    """
    Очистить истекшие сессии.
    
    Зачем: Периодическая очистка памяти от неактивных сессий.
    Рекомендуется вызывать по расписанию (например, каждые 10 минут).
    
    Returns:
        Результат очистки
    """
    session_manager.cleanup_expired()
    stats = session_manager.get_stats()
    return {"status": "cleaned", "current_sessions": stats["total_sessions"]}
