from __future__ import annotations

import os
import uuid
import logging
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Initialize logger
logger = logging.getLogger(__name__)
from langchain_community.vectorstores import Qdrant
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import requests
from generation.llm_service import generate_response

# Import reranking service
try:
    from reranker.cross_encoder_service import (
        rerank_candidates,
        get_reranking_info as get_reranking_info_impl  # Rename to avoid conflict with endpoint
    )
    RERANKING_AVAILABLE = True
    logger.info("✅ Reranking service imported successfully")
except ImportError as e:
    logger.warning(f"⚠️  Reranking not available: {e}")
    RERANKING_AVAILABLE = False
    get_reranking_info_impl = None

# Import chat components for conversational RAG
# Зачем: Добавляем диалоговый режим с уточняющими вопросами
from chat.session_manager import SessionManager, ChatSession
from chat.clarification_service import ClarificationService

# Импорт гибридного поиска
try:
    from search.hybrid_search import get_global_hybrid_retriever, reset_global_retriever
    HYBRID_SEARCH_AVAILABLE = True
    logger.info("✅ Гибридный поиск импортирован успешно")
except ImportError as e:
    logger.warning(f"⚠️  Гибридный поиск недоступен: {e}")
    HYBRID_SEARCH_AVAILABLE = False


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_chunks")


from search.custom_qdrant import CustomQdrant


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
    
    # 🆕 Метаданные процесса принятия решений
    metadata: Optional[Dict] = None  # Подробная информация о процессе


app = FastAPI(title="RAG Gateway")

# Инициализация сервисов для диалогового режима
# Зачем: Создаем глобальные экземпляры для управления сессиями и уточнениями
session_manager = SessionManager()
clarification_service = ClarificationService()


@app.post("/query")
def query_endpoint(req: QueryRequest):
    """
    Основной RAG endpoint с гибридным поиском.
    
    Args:
        req: QueryRequest с параметрами запроса
    
    Returns:
        Ответ с генерированным текстом и источниками
    """
    embeddings = TEIEmbeddings()
    
    # Определяем количество кандидатов для начального поиска
    initial_k = req.k * 3 if (req.use_reranking and RERANKING_AVAILABLE) else req.k
    
    # ✅ ГИБРИДНЫЙ ПОИСК (если доступен)
    if HYBRID_SEARCH_AVAILABLE:
        try:
            logger.info("🔍 Используем гибридный поиск (Dense + BM25 + RRF)")
            
            # Получаем глобальный кэшированный ретривер
            hybrid_retriever = get_global_hybrid_retriever(embeddings)
            
            # Обновляем k для текущего запроса
            hybrid_retriever.update_k(initial_k)
            
            # Выполняем гибридный поиск
            docs = hybrid_retriever.get_relevant_documents(req.query)
            
        except Exception as e:
            logger.error(f"❌ Ошибка гибридного поиска, fallback на обычный: {e}")
            # Fallback на обычный векторный поиск
            vs = CustomQdrant.from_existing_collection(
                embedding=embeddings,
                collection_name=QDRANT_COLLECTION,
                url=QDRANT_URL,
                prefer_grpc=False,
                path=None,
            )
            docs = vs.similarity_search(req.query, k=initial_k)
    
    # ❌ Обычный векторный поиск (если гибридный недоступен)
    else:
        logger.info("🔍 Используем обычный векторный поиск (Dense only)")
        vs = CustomQdrant.from_existing_collection(
            embedding=embeddings,
            collection_name=QDRANT_COLLECTION,
            url=QDRANT_URL,
            prefer_grpc=False,
            path=None,
        )
        docs = vs.similarity_search(req.query, k=initial_k)
    
    # Фильтрация пустых документов (кастомный класс уже обрабатывает, но перестраховка)
    docs = [doc for doc in docs if doc and doc.page_content and doc.page_content.strip()]
    
    # Проверка что нашли хоть что-то
    if not docs:
        logging.warning("Не найдено валидных документов в результатах поиска")
        return {
            "answer": "Извините, не удалось найти релевантную информацию для вашего запроса.",
            "sources": [],
            "processing_time_ms": 0,
            "error": "No valid documents found"
        }
    
    # Реранкинг (если включен и доступен)
    if req.use_reranking and RERANKING_AVAILABLE and docs:
        try:
            # Конвертируем docs в формат для реранкера
            candidates = []
            for d in docs:
                meta = getattr(d, 'metadata', {}) or {}
                # Добавляем текст для реранкинга
                if hasattr(d, 'page_content') and d.page_content:
                    meta = {**meta, 'text': d.page_content}
                candidates.append(meta)
            
            # Реранжируем кандидатов
            rerank_k = req.rerank_top_k or req.k
            logger.info(f"🔄 Вызов реранкера для {len(candidates)} кандидатов...")
            reranked_candidates = rerank_candidates(req.query, candidates, top_k=rerank_k)
            
            # Обновляем docs с отреранжированным порядком
            if reranked_candidates:
                docs = docs[:len(reranked_candidates)]  # Оставляем оригинальные docs но ограничиваем
                # Логируем детали реранжирования
                if reranked_candidates and isinstance(reranked_candidates[0], dict):
                    rerank_method = reranked_candidates[0].get('rerank_method', 'unknown')
                    rerank_model = reranked_candidates[0].get('rerank_model', 'unknown')
                    logger.info(f"✅ Реранкинг завершен: {len(candidates)} -> {len(reranked_candidates)} кандидатов")
                    logger.info(f"   Метод: {rerank_method}, Модель: {rerank_model}")
            else:
                logger.warning("⚠️  Реранкинг не вернул кандидатов, используем оригинальные docs")
                docs = docs[:req.k]
            
        except Exception as e:
            logging.error(f"Реранкинг не удался, используем оригинальный порядок: {e}")
            # Fallback на оригинальный порядок
            docs = docs[:req.k]
    
    # Ограничиваем до запрошенного количества если реранкинг выключен
    elif not req.use_reranking or not RERANKING_AVAILABLE:
        docs = docs[:req.k]
    
    # Строим контекст безопасно
    context_parts = []
    for d in docs:
        if hasattr(d, 'page_content') and d.page_content and d.page_content.strip():
            context_parts.append(d.page_content.strip())
    
    if not context_parts:
        logging.warning("Не найдено валидного контекста для генерации")
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
    
    # 🆕 Инициализируем метаданные для отслеживания процесса
    import time
    start_time = time.time()
    metadata = {
        "decision_process": {},
        "rag_pipeline": {},
        "timing": {}
    }
    
    # Инициализируем embeddings и vector store для поиска в БД
    # Зачем: Нужны для предварительного поиска контекста и основного RAG
    embeddings = TEIEmbeddings()
    vs = CustomQdrant.from_existing_collection(
        embedding=embeddings,
        collection_name=QDRANT_COLLECTION,
        url=QDRANT_URL,
        prefer_grpc=False,
        path=None,
    )
    
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
        
        metadata["decision_process"]["clarifications_provided"] = len(req.clarification_answers)
        metadata["decision_process"]["clarifications"] = req.clarification_answers
    
    # Проверяем нужны ли уточнения (если не пропущено явно)
    # Зачем: Определяем достаточно ли информации для качественного ответа
    if (not req.skip_clarification and 
        session.state == "initial" and 
        not req.clarification_answers):
        
        # 🆕 ШАГ 1: Выполняем предварительный векторный поиск для получения контекста из БД
        # Зачем: Анализ уточнений и генерация вопросов будут основаны на реальных данных из БД
        logging.info("Выполняем предварительный поиск в БД для анализа уточнений")
        kb_search_start = time.time()
        
        try:
            # Ищем небольшое количество документов для контекста (не нужно много)
            logging.info(f"Поиск в БД для запроса: '{req.query}'")
            kb_docs = vs.similarity_search(req.query, k=5)
            logging.info(f"Найдено документов до фильтрации: {len(kb_docs)}")
            
            # Фильтруем пустые документы
            kb_docs = [doc for doc in kb_docs if doc.page_content and doc.page_content.strip()]
            logging.info(f"Найдено документов после фильтрации: {len(kb_docs)}")
            
            # Формируем полный контекст из найденных документов
            kb_context_parts = []
            for i, doc in enumerate(kb_docs[:5], 1):  # Максимум 5 документов
                # Берем ВЕСЬ текст документа для более точного анализа
                content = doc.page_content.strip()
                if content:
                    kb_context_parts.append(f"Документ {i}:\n{content}")
                    logging.info(f"Документ {i}: {len(content)} символов")
            
            knowledge_base_context = "\n\n".join(kb_context_parts) if kb_context_parts else ""
            kb_docs_count = len(kb_docs)
            
            if kb_docs_count == 0:
                logging.warning("⚠️ В БД не найдено документов для анализа уточнений!")
            else:
                logging.info(f"✅ Сформирован контекст из {kb_docs_count} документов, общая длина: {len(knowledge_base_context)} символов")
            
        except Exception as e:
            logging.error(f"Ошибка при предварительном поиске в БД: {e}", exc_info=True)
            knowledge_base_context = ""
            kb_docs_count = 0
        
        kb_search_time = time.time() - kb_search_start
        
        # Сохраняем информацию о предварительном поиске
        metadata["decision_process"]["knowledge_base_search"] = {
            "documents_found": kb_docs_count,
            "search_time_ms": round(kb_search_time * 1000, 2),
            "context_length": len(knowledge_base_context)
        }
        
        # 🆕 ШАГ 2: Анализируем запрос с учетом контекста из БД
        clarification_start = time.time()
        conversation_context = session.get_conversation_context()
        needs_clarification, reason, confidence = clarification_service.should_clarify(
            req.query, 
            conversation_context,
            knowledge_base_context  # ← Передаем контекст из БД!
        )
        clarification_time = time.time() - clarification_start
        
        # 🆕 Сохраняем информацию об анализе
        metadata["decision_process"]["clarification_analysis"] = {
            "needs_clarification": needs_clarification,
            "reason": reason,
            "confidence": confidence,
            "threshold": 0.6,
            "analysis_time_ms": round(clarification_time * 1000, 2),
            "used_kb_context": kb_docs_count > 0  # Был ли использован контекст из БД
        }
        
        # 🆕 ШАГ 3: Если нужны уточнения - генерируем вопросы на основе контекста из БД
        if needs_clarification and confidence > 0.6:  # Порог уверенности
            logging.info(f"Требуются уточнения: {reason} (confidence={confidence})")
            
            # Генерируем 2-3 уточняющих вопроса с учетом контекста из БД
            questions_start = time.time()
            questions, priorities = clarification_service.generate_clarification_questions(
                req.query,
                conversation_context,
                knowledge_base_context,  # ← Передаем контекст из БД!
                num_questions=3
            )
            questions_time = time.time() - questions_start
            
            metadata["decision_process"]["questions_generation"] = {
                "num_questions": len(questions),
                "priorities": priorities,
                "generation_time_ms": round(questions_time * 1000, 2),
                "used_kb_context": kb_docs_count > 0  # Был ли использован контекст из БД
            }
            metadata["timing"]["total_ms"] = round((time.time() - start_time) * 1000, 2)
            
            session.set_pending_questions(questions)
            session.add_message("assistant", f"Нужны уточнения: {reason}")
            
            return ChatResponse(
                session_id=session.session_id,
                state="awaiting_clarification",
                clarification_questions=questions,
                clarification_priorities=priorities,
                conversation_history=[m for m in session.messages],
                metadata=metadata
            )
        else:
            # 🆕 Уточнения не нужны - продолжаем без них
            metadata["decision_process"]["clarification_skipped"] = {
                "reason": "Query is specific enough" if not needs_clarification else "Confidence below threshold",
                "confidence": confidence
            }
    
    # Расширяем запрос уточнениями (если есть)
    # Зачем: Делаем запрос более конкретным для лучшего поиска
    enhanced_query = req.query
    if session.clarifications:
        enhanced_query = clarification_service.enhance_query_with_clarifications(
            req.query,
            session.clarifications
        )
        logging.info(f"Запрос расширен {len(session.clarifications)} уточнениями")
        
        # 🆕 Сохраняем информацию о расширении запроса
        metadata["rag_pipeline"]["query_enhancement"] = {
            "original_query": req.query,
            "enhanced_query": enhanced_query,
            "clarifications_used": len(session.clarifications)
        }
    else:
        metadata["rag_pipeline"]["query_enhancement"] = {
            "enhanced": False,
            "reason": "No clarifications provided"
        }
    
    # === Выполняем RAG поиск (с гибридным если доступен) ===
    # Примечание: embeddings уже инициализирован в начале функции
    
    search_start = time.time()
    
    # Начальный поиск (больше кандидатов если используем реранкинг)
    initial_k = req.k * 3 if (req.use_reranking and RERANKING_AVAILABLE) else req.k
    
    # ✅ ГИБРИДНЫЙ ПОИСК (если доступен)
    if HYBRID_SEARCH_AVAILABLE:
        try:
            logger.info("🔍 Используем гибридный поиск для /chat")
            hybrid_retriever = get_global_hybrid_retriever(embeddings)
            hybrid_retriever.update_k(initial_k)
            docs = hybrid_retriever.get_relevant_documents(enhanced_query)
        except Exception as e:
            logger.error(f"❌ Ошибка гибридного поиска в /chat, fallback: {e}")
            docs = vs.similarity_search(enhanced_query, k=initial_k)
    else:
        # Обычный векторный поиск
        docs = vs.similarity_search(enhanced_query, k=initial_k)
    
    search_time = time.time() - search_start
    
    # Фильтруем валидные документы
    docs_before_filter = len(docs)
    docs = [doc for doc in docs if doc and doc.page_content and doc.page_content.strip()]
    
    # 🆕 Сохраняем информацию о поиске
    metadata["rag_pipeline"]["vector_search"] = {
        "initial_k": initial_k,
        "documents_found": docs_before_filter,
        "documents_after_filter": len(docs),
        "search_time_ms": round(search_time * 1000, 2)
    }
    
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
            rerank_start = time.time()
            candidates = []
            for d in docs:
                meta = getattr(d, 'metadata', {}) or {}
                if hasattr(d, 'page_content') and d.page_content:
                    meta = {**meta, 'text': d.page_content}
                candidates.append(meta)
            
            rerank_k = req.rerank_top_k or req.k
            logger.info(f"🔄 Calling reranker for {len(candidates)} candidates...")
            reranked_candidates = rerank_candidates(enhanced_query, candidates, top_k=rerank_k)
            rerank_time = time.time() - rerank_start
            
            docs_before_rerank = len(docs)
            rerank_method = "unknown"
            rerank_model = "unknown"
            
            if reranked_candidates:
                docs = docs[:len(reranked_candidates)]
                # Extract reranking metadata
                if isinstance(reranked_candidates[0], dict):
                    rerank_method = reranked_candidates[0].get('rerank_method', 'unknown')
                    rerank_model = reranked_candidates[0].get('rerank_model', 'unknown')
                logger.info(f"✅ Reranking completed: {docs_before_rerank} -> {len(docs)} candidates")
                logger.info(f"   Method: {rerank_method}, Model: {rerank_model}")
            else:
                docs = docs[:req.k]
                logger.warning("⚠️  Reranking returned no candidates")
            
            # 🆕 Сохраняем информацию о реранкинге
            metadata["rag_pipeline"]["reranking"] = {
                "enabled": True,
                "method": rerank_method,
                "model": rerank_model,
                "candidates_before": docs_before_rerank,
                "candidates_after": len(docs),
                "target_k": rerank_k,
                "rerank_time_ms": round(rerank_time * 1000, 2),
                "success": len(reranked_candidates) > 0 if reranked_candidates else False
            }
                
        except Exception as e:
            logging.error(f"Reranking failed: {e}")
            docs = docs[:req.k]
            
            # 🆕 Сохраняем информацию об ошибке реранкинга
            metadata["rag_pipeline"]["reranking"] = {
                "enabled": True,
                "method": "LangChain compression pipeline",
                "error": str(e),
                "fallback": "Using original order"
            }
    elif not req.use_reranking or not RERANKING_AVAILABLE:
        docs = docs[:req.k]
        
        # 🆕 Реранкинг отключен
        metadata["rag_pipeline"]["reranking"] = {
            "enabled": False,
            "reason": "Disabled by user" if not req.use_reranking else "Not available"
        }
    
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
    
    # Формируем секцию с уточнениями (если есть)
    clarifications_section = ""
    if clarifications_text:
        clarifications_section = f"\nУточнения от пользователя:\n{clarifications_text}\n"
    
    prompt = f"""Ответь на вопрос используя контекст и историю диалога.

История диалога:
{conversation_history}
{clarifications_section}
Контекст из базы знаний:
{context}

Текущий вопрос: {req.query}

Предоставь развернутый и точный ответ на основе контекста и предыдущего диалога."""
    
    # Генерируем ответ
    generation_start = time.time()
    answer, thoughts = generate_response(prompt, max_tokens=800)
    generation_time = time.time() - generation_start
    session.add_message("assistant", answer)
    
    # 🆕 Сохраняем информацию о генерации
    metadata["rag_pipeline"]["generation"] = {
        "model": "vLLM (Qwen2.5-14B-Instruct-AWQ)",
        "max_tokens": 800,
        "generation_time_ms": round(generation_time * 1000, 2),
        "context_length": len(context),
        "prompt_includes_history": len(conversation_history) > 0,
        "prompt_includes_clarifications": len(clarifications_text) > 0
    }
    
    # Собираем источники
    sources = []
    for d in docs:
        meta = getattr(d, 'metadata', {}) or {}
        text_content = getattr(d, 'page_content', '')
        meta = {**meta, 'text': text_content}
        sources.append(meta)
    
    # 🆕 Финальная статистика
    metadata["rag_pipeline"]["final_stats"] = {
        "sources_used": len(sources),
        "context_chunks": len(context_parts),
        "total_context_length": len(context)
    }
    
    metadata["timing"]["total_ms"] = round((time.time() - start_time) * 1000, 2)
    metadata["timing"]["breakdown"] = {
        "search": metadata["rag_pipeline"]["vector_search"]["search_time_ms"],
        "reranking": metadata["rag_pipeline"]["reranking"].get("rerank_time_ms", 0),
        "generation": metadata["rag_pipeline"]["generation"]["generation_time_ms"]
    }
    
    return ChatResponse(
        session_id=session.session_id,
        state="completed",
        answer=answer,
        sources=sources,
        conversation_history=[m for m in session.messages],
        metadata=metadata
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


@app.get("/reranker/info")
def get_reranker_info_endpoint():
    """
    Получить информацию о текущем реранкере.
    
    Показывает:
        - Какой реранкер используется (Infinity/Local/Fallback)
        - Доступность сервиса
        - Модель
        - Метод инференса
    
    Returns:
        Информация о реранкере
    """
    if not RERANKING_AVAILABLE or get_reranking_info_impl is None:
        return {
            "available": False,
            "method": "none",
            "error": "Reranking service not imported"
        }
    
    try:
        info = get_reranking_info_impl()
        return {
            "available": True,
            "info": info
        }
    except Exception as e:
        logger.error(f"Failed to get reranker info: {e}")
        return {
            "available": False,
            "method": "unknown",
            "error": str(e)
        }


@app.post("/admin/reset-retriever")
def reset_retriever():
    """
    Сбросить кэшированный гибридный ретривер.
    
    Зачем: Использовать после добавления новых документов в Qdrant,
    чтобы BM25 индекс обновился с новыми документами.
    
    Returns:
        Статус сброса
    """
    if not HYBRID_SEARCH_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Гибридный поиск недоступен"
        )
    
    try:
        reset_global_retriever()
        return {
            "status": "success",
            "message": "Гибридный ретривер сброшен, будет переиндексирован при следующем запросе"
        }
    except Exception as e:
        logger.error(f"❌ Ошибка сброса ретривера: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """
    Проверка здоровья сервиса и его зависимостей.
    
    Проверяет:
        - Qdrant подключение
        - LLM доступность
        - Reranker статус
        - Гибридный поиск статус
    
    Returns:
        Статус сервиса и зависимостей
    """
    health_status = {
        "service": "online-rag",
        "status": "healthy",
        "components": {}
    }
    
    # Check Qdrant
    try:
        embeddings = TEIEmbeddings()
        vs = CustomQdrant.from_existing_collection(
            embedding=embeddings,
            collection_name=QDRANT_COLLECTION,
            url=QDRANT_URL,
            prefer_grpc=False,
            path=None,
        )
        health_status["components"]["qdrant"] = {"status": "healthy", "url": QDRANT_URL}
    except Exception as e:
        health_status["components"]["qdrant"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check Reranker
    if RERANKING_AVAILABLE and get_reranking_info_impl:
        try:
            reranker_info = get_reranking_info_impl()
            health_status["components"]["reranker"] = {
                "status": "healthy" if reranker_info.get("service_available", False) else "unavailable",
                "method": reranker_info.get("method", "unknown"),
                "model": reranker_info.get("model_name", "unknown"),
                "inference_location": reranker_info.get("inference_location", "unknown")
            }
        except Exception as e:
            health_status["components"]["reranker"] = {"status": "error", "error": str(e)}
    else:
        health_status["components"]["reranker"] = {"status": "not_imported"}
    
    # Check Hybrid Search
    if HYBRID_SEARCH_AVAILABLE:
        health_status["components"]["hybrid_search"] = {
            "status": "available",
            "method": "LangChain EnsembleRetriever (Dense + BM25)",
            "fusion": "RRF (Reciprocal Rank Fusion)"
        }
    else:
        health_status["components"]["hybrid_search"] = {"status": "not_imported"}
    
    return health_status
