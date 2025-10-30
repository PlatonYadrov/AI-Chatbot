from __future__ import annotations

import os
import sys
import uuid
import logging
import re
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Initialize logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False
from langchain_community.vectorstores import Qdrant
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
import requests
from generation.llm_service import generate_response

# OpenAI client for conversational pipeline
from openai import OpenAI

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
try:
    from chat.db_session_manager import PostgresSessionManager
    USE_POSTGRES_SESSIONS = True
    logger.info("✅ Using PostgreSQL for session storage")
except ImportError as e:
    logger.warning(f"⚠️  PostgreSQL session manager not available, using in-memory: {e}")
    from chat.session_manager import SessionManager
    USE_POSTGRES_SESSIONS = False

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


def _format_qna(question: str, answer: str, sourses: List[Dict[str, str]] | None = None) -> str:
    lines = [question, "", f"- {answer}"]
    if sourses:
        seen = set()
        items = []
        for item in sourses:
            doc = item.get("document") or "unknown"
            # По требованию: выводим только название документа без пути
            src = doc
            if src not in seen:
                seen.add(src)
                items.append(src)
        if items:
            lines.append("")
            lines.append("источники: " + ", ".join(items))
    return "\n".join(lines)


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


# Глобальный менеджер сессий (инициализируется в lifespan)
session_manager = None
clarification_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler для инициализации и очистки ресурсов."""
    global session_manager, clarification_service
    
    # Startup: инициализация сервисов
    if USE_POSTGRES_SESSIONS:
        session_manager = PostgresSessionManager()
        try:
            await session_manager.initialize()
            logger.info("✅ Session manager initialized with PostgreSQL backend")
        except Exception as e:
            logger.error(f"❌ Failed to initialize PostgreSQL session manager: {e}")
            logger.warning("⚠️  Falling back to in-memory session manager")
            from chat.session_manager import SessionManager
            session_manager = SessionManager()
    else:
        from chat.session_manager import SessionManager
        session_manager = SessionManager()
        logger.info("⚠️  Session manager initialized with in-memory backend")
    
    clarification_service = ClarificationService()
    
    logger.info("✅ RAG Gateway startup complete")
    
    yield
    
    # Shutdown: очистка ресурсов
    logger.info("🔄 RAG Gateway shutting down...")


app = FastAPI(title="RAG Gateway", lifespan=lifespan)

# Configure CORS для работы с веб-интерфейсом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Conversational RAG Pipeline ====================
# Зачем: Полноценный диалоговый пайплайн с query condensation и автоматическими уточнениями

# Initialize OpenAI client for vLLM
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://llm:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
llm_client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")


def _call_llm_chat(messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 8000) -> str:
    """
    Вызов LLM через OpenAI-совместимый API.
    
    Args:
        messages: Список сообщений в формате [{"role": "system|user|assistant", "content": "..."}]
        temperature: Температура генерации
        max_tokens: Максимальное количество токенов
    
    Returns:
        Ответ LLM
    """
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


def _condense_question_with_history(
    question: str,
    history_messages: List[Dict],
    aux_summaries: Optional[List[str]] = None,
) -> str:
    """
    Переформулирует вопрос пользователя в самостоятельный с учетом истории диалога.
    
    Зачем: Делает запрос независимым от контекста предыдущих сообщений для лучшего поиска.
    
    Args:
        question: Оригинальный вопрос пользователя
        history_messages: История диалога из SessionManager
    
    Returns:
        Переформулированный самостоятельный вопрос
    """
    if not history_messages or len(history_messages) <= 1:
        return question
    
    # Формируем историю для промпта (последние 8 сообщений для контекста)
    history_text = ""
    for msg in history_messages[-8:-1]:  # Исключаем последнее (текущий вопрос)
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            history_text += f"Пользователь: {content}\n"
        elif role == "assistant":
            history_text += f"Ассистент: {content}\n"
    
    aux_section = ""
    if aux_summaries:
        try:
            top3 = aux_summaries[:3]
            joined = "\n".join([f"- {s}" for s in top3])
            aux_section = (
                "\nДополнительные сведения из прошлых веток (используй ТОЛЬКО при сильной неоднозначности,"
                " как вспомогательный контекст; не подменяй текущую историю):\n"
                f"{joined}\n"
            )
        except Exception:
            aux_section = ""

    messages = [
        {
            "role": "system",
            "content": (
                "Переформулируй последний вопрос пользователя в самостоятельный, используя контекст истории ТЕКУЩЕЙ ветки. "
                "Дополнительные сведения из других веток использовать только как справку при крайней необходимости. "
                "Верни ТОЛЬКО переформулированный вопрос без дополнительного текста."
            )
        },
        {
            "role": "user",
            "content": f"""История диалога:
{history_text}

{aux_section}
Последний вопрос пользователя: {question}

Переформулируй последний вопрос так, чтобы он был понятен без контекста истории. Ответь ТОЛЬКО переформулированным вопросом."""
        }
    ]
    
    condensed = _call_llm_chat(messages, temperature=0.1, max_tokens=200)
    logger.info(f"🔄 Query condensation: '{question}' → '{condensed}'")
    return condensed.strip()


def _summarize_thread(messages: List[Dict], final_answer: str) -> str:
    """Краткая суммаризация ветки (2–3 предложения или до 5 пунктов)."""
    try:
        parts = []
        for m in messages[-10:]:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            prefix = "П" if role == "user" else ("А" if role == "assistant" else "У")
            parts.append(f"{prefix}: {content}")
        parts_text = "\n".join(parts)

        messages_llm = [
            {
                "role": "system",
                "content": (
                    "Суммаризируй ветку диалога для будущего вспомогательного контекста. "
                    "Будь кратким: 2–3 предложения ИЛИ до 5 пунктов. Тема, ключевые факты, итог."
                ),
            },
            {
                "role": "user",
                "content": f"""Диалог ветки (усечённо):
{parts_text}

Финальный ответ ассистента:
{final_answer}

Дай краткую суммаризацию ветки.""",
            },
        ]
        summary = _call_llm_chat(messages_llm, temperature=0.1, max_tokens=220)
        return (summary or "").strip()
    except Exception as e:
        logger.warning(f"Thread summarization failed: {e}")
        return ""


def _check_clarification_needed(question: str, history_messages: List[Dict], kb_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Определяет нужны ли уточнения для ответа на вопрос.
    
    Зачем: Предотвращает галлюцинации при неполных/расплывчатых вопросах.
    
    Args:
        question: Вопрос пользователя
        history_messages: История диалога
    
    Returns:
        Dict с полями:
        - need_clarification: bool - нужны ли уточнения
        - clarification_question: str - вопрос для уточнения (если нужно)
        - reason: str - причина необходимости уточнения
    """
    # Формируем краткую историю
    history_text = ""
    if history_messages and len(history_messages) > 1:
        for msg in history_messages[-6:-1]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                history_text += f"User: {content}\n"
            elif role == "assistant":
                history_text += f"Assistant: {content}\n"
    
    # Явно помечаем секции для LLM
    history_section = (
        f"\n## ИСТОРИЯ_ДИАЛОГА (текущий тред)\n{history_text}" if history_text else "\n## ИСТОРИЯ_ДИАЛОГА\n—"
    )
    kb_section = "\n## КОНТЕКСТ_БАЗЫ_ЗНАНИЙ\n—"
    if kb_context:
        # Ограничим размер для промпта, чтобы избежать переполнения контекста
        kb_snippet = kb_context
        print(f"вот контекст: {kb_snippet[:100]}")
        kb_section = f"\n## КОНТЕКСТ_БАЗЫ_ЗНАНИЙ (фрагменты)\n{kb_snippet}"
    
    messages = [
        {
            "role": "system",
            "content": """# РОЛЬ
Ты — классификатор необходимости уточнений.

# ЦЕЛЬ
Определить, нужны ли уточняющие вопросы для текущего запроса пользователя.

# ИНСТРУКЦИИ
- Учитывай ИСТОРИЮ_ДИАЛОГА, чтобы снять неоднозначность (местоимения, ссылки на ранее сказанное).
- Для фактов опирайся на КОНТЕКСТ_БАЗЫ_ЗНАНИЙ; если он отсутствует, принимай решение без него.
- Оцени, достаточно ли информации, чтобы выбрать один однозначный ответ без уточнений.

# КРИТЕРИИ ДЛЯ УТОЧНЕНИЙ
- Запрос неоднозначен или слишком общий.
- Отсутствуют критичные параметры (даты, имена, типы и т.д.).
- В КОНТЕКСТЕ_БАЗЫ_ЗНАНИЙ встречаются 2+ разных категории/аспекта, между которыми нужен выбор.

# КРИТЕРИИ, КОГДА УТОЧНЕНИЯ НЕ НУЖНЫ
- Запрос конкретный и понятный.
- История текущего треда уже снимает неоднозначность.
- Можно выполнить осмысленный поиск и выбрать однозначный ответ.

# ФОРМАТ ОТВЕТА (JSON)
{
  "need_clarification": true/false,
  "clarification_question": "краткий вопрос для уточнения, основанный на КОНТЕКСТЕ_БАЗЫ_ЗНАНИЙ (если need_clarification=true)",
  "reason": "краткая причина"
}
"""
        },
        {
            "role": "user",
            "content": f"""# ВХОДНЫЕ ДАННЫЕ
## ВОПРОС
{question}
{history_section}
{kb_section}

Верни ТОЛЬКО валидный JSON без дополнительного текста."""
        }
    ]
    
    response = _call_llm_chat(messages, temperature=0.1, max_tokens=1000)
    try:
        logger.info(f"🤖 LLM raw clarification response: {response}")
    except Exception:
        pass
    
    # Парсим JSON
    import json
    try:
        # Извлекаем JSON из ответа (на случай если LLM добавил текст)
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
            
            need = bool(result.get("need_clarification", False))
            clarification = result.get("clarification_question", "").strip()
            reason = result.get("reason", "").strip()
            
            try:
                clarification_preview = clarification.replace("\n", " ")[:160]
                logger.info(
                    f"🤖 LLM decision (in _check_clarification_needed): "
                    f"need_clarification={need}; reason={reason}; "
                    f"question={clarification_preview}"
                )
            except Exception:
                pass
            
            return {
                "need_clarification": need,
                "clarification_question": clarification if need else "",
                "reason": reason
            }
    except Exception as e:
        logger.warning(f"Failed to parse clarification check response: {e}")
    
    # Fail-safe: не требуем уточнения при ошибке парсинга
    return {
        "need_clarification": False,
        "clarification_question": "",
        "reason": "Parse error, proceeding without clarification"
    }


def _format_docs_with_citations(docs: List[Document]) -> tuple[str, List[Dict]]:
    """
    Форматирует документы для контекста с нумерацией для цитирования.
    
    Args:
        docs: Список документов LangChain
    
    Returns:
        Tuple из:
        - context: Строка с пронумерованным контекстом
        - citations: Список метаданных источников с ID
    """
    context_parts = []
    citations = []
    
    for i, doc in enumerate(docs, 1):
        meta = getattr(doc, 'metadata', {}) or {}
        content = getattr(doc, 'page_content', '').strip()
        
        if not content:
            continue
        
        # Извлекаем имя документа
        raw_path = (
            meta.get('path')
            or meta.get('source_path')
            or meta.get('source_uri')
            or meta.get('source')
            or ''
        )
        try:
            document_name = os.path.basename(str(raw_path)) if raw_path else ''
        except Exception:
            document_name = ''
        
        if not document_name:
            document_name = str(meta.get('doc_id') or f'source_{i}')
        
        # Ограничиваем длину чанка для контекста (первые 600 символов)
        snippet = content[:600] + "..." if len(content) > 600 else content
        
        # Добавляем в контекст с номером для цитирования
        context_parts.append(f"[#{i}] Источник: {document_name}\n{snippet}")
        
        # Сохраняем метаданные для ответа
        citations.append({
            "id": i,
            "document": document_name,
            "path": raw_path,
            "text": content,  # Полный текст чанка
            "page": meta.get("page"),
            "chunk_id": meta.get("chunk_id")
        })
    
    context = "\n\n".join(context_parts)
    return context, citations


class ConversationalRequest(BaseModel):
    """Запрос для конверсационного RAG пайплайна."""
    query: str
    session_name: Optional[str] = None
    k: int = 5  # Количество документов для retrieval
    max_clarification_rounds: Optional[int] = None
    use_reranking: bool = True
    rerank_top_k: Optional[int] = None
    skip_condensation: bool = False  # Пропустить переформулировку вопроса
    skip_clarification_check: bool = False  # Пропустить проверку на уточнения


class ConversationalResponse(BaseModel):
    """Ответ конверсационного RAG пайплайна."""
    session_id: int
    session_name: Optional[str] = None
    state: str  # "awaiting_clarification" или "completed"
    
    # Если нужны уточнения
    clarification_question: Optional[str] = None
    clarification_reason: Optional[str] = None
    
    # Если ответ готов
    answer: Optional[str] = None
    citations: Optional[List[Dict]] = None  # Список источников с ID для цитирования
    
    # Метаданные
    condensed_query: Optional[str] = None  # Переформулированный запрос
    conversation_history: List[Dict] = []
    metadata: Optional[Dict] = None  # Техническая информация о процессе


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
    answer, thoughts = generate_response(prompt, max_tokens=1200)

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


@app.post("/chat")
def chat_endpoint(req: ChatRequest, pretty: bool = Query(True)):
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
        try:
            logger.info(
                f"🤖 Решение LLM: уточнения нужны={needs_clarification} "
                f"(confidence={confidence:.2f}); reason={reason}"
            )
        except Exception:
            pass
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
            try:
                logger.info(
                    "📝 Уточняющие вопросы: " + " | ".join([q.replace("\n", " ") for q in questions])
                )
            except Exception:
                pass
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
        if pretty:
            # Старый формат оставлен в коде для отладки
            # return PlainTextResponse(_format_plain("Извините, не удалось найти релевантную информацию.", []))
            return PlainTextResponse(_format_qna(req.query, "Извините, не удалось найти релевантную информацию.", []))
        return {
            "answer": "Извините, не удалось найти релевантную информацию.",
            "sources": [],
            "sourses": [],
            "conversation_history": [m for m in session.messages],
            "metadata": metadata
        }
    
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
    
    # Собираем источники в требуемом формате (полный текст чанка)
    sourses = []
    for d in docs:
        meta = getattr(d, 'metadata', {}) or {}
        text_content = getattr(d, 'page_content', '')
        # Prefer a path-like field to extract basename
        raw_path = (
            meta.get('path')
            or meta.get('source_path')
            or meta.get('source_uri')
            or meta.get('source')
            or ''
        )
        try:
            document_name = os.path.basename(str(raw_path)) if raw_path else ''
        except Exception:
            document_name = ''
        if not document_name:
            # Fallback to doc_id or unknown
            document_name = str(meta.get('doc_id') or 'unknown')
        source_path = raw_path or ''
        sourses.append({
            'text': text_content,
            'document': document_name,
            'path': source_path
        })

    # Сформируем детальные источники (metadata + text)
    sources = []
    for d in docs:
        meta = getattr(d, 'metadata', {}) or {}
        text_content = getattr(d, 'page_content', '')
        sources.append({**meta, 'text': text_content})

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
    if pretty:
        # Старый формат оставлен в коде для отладки
        # return PlainTextResponse(_format_plain(answer, sourses))
        return PlainTextResponse(_format_qna(req.query, answer, sourses))
    return ChatResponse(
        session_id=session.session_id,
        state="completed",
        answer=answer,
        sources=sources,
        conversation_history=[m for m in session.messages],
        metadata=metadata
    )


@app.post("/chat/conversational", response_model=ConversationalResponse)
async def conversational_rag_endpoint(req: ConversationalRequest, pretty: bool = Query(True)):
    """
    Полноценный конверсационный RAG-пайплайн с историей, query condensation и автоматическими уточнениями.
    
    Особенности:
    - Query condensation: переформулировка вопроса с учетом истории диалога
    - Автоматическое определение необходимости уточнений через LLM
    - Гибридный поиск (если доступен) + реранкинг
    - Ответ с пронумерованными цитатами источников [#1], [#2], ...
    - Полная история диалога в рамках сессии
    
    Поток работы:
    1. Получить/создать сессию
    2. Проверка на необходимость уточнений (если не пропущено)
    3. Переформулировка вопроса с учетом истории (query condensation)
    4. RAG: гибридный поиск + реранкинг
    5. Генерация ответа с цитатами
    
    Args:
        req: ConversationalRequest с вопросом и параметрами
    
    Returns:
        ConversationalResponse с ответом или вопросом для уточнения
    """
    import time
    start_time = time.time()
    
    metadata = {
        "pipeline": "conversational_rag",
        "steps": {},
        "timing": {}
    }
    
    # Инициализируем embeddings
    embeddings = TEIEmbeddings()
    
    # Получаем или создаем сессию по имени
    session = await session_manager.get_or_create_session_by_name(req.session_name)
    # Установим лимит раундов если передан
    if req.max_clarification_rounds is not None:
        try:
            await session.set_max_rounds(int(req.max_clarification_rounds))
        except Exception:
            pass
    
    # Определяем/создаем ветку для этого вопроса
    thread_id = await session.resolve_thread_id_for_new_user_question()
    await session.add_message("user", req.query, metadata={"stage": "user_question", "thread_id": thread_id, "round": 0})
    
    # Получаем историю для обработки
    history_messages = await session.get_thread_messages(thread_id)
    # Логирование истории ветки для контроля учета контекста
    try:
        logger.info(
            f"🧵 История ветки (thread={thread_id}): {len(history_messages)} сообщений; "
            f"state={session.state}"
        )
        if history_messages:
            preview = []
            for m in history_messages[-5:]:
                role = m.get('role', 'user')
                content = (m.get('content') or '')[:120].replace('\n', ' ')
                preview.append(f"{role}: {content}")
            logger.info("📝 История (последние): " + " | ".join(preview))
    except Exception:
        pass
    
    # Подготовим список вспомогательных суммаризаций прошлых веток (до 3 шт.)
    aux_summaries: List[str] = []
    try:
        aux_summaries = await session.get_last_thread_summaries(limit=3, exclude_thread_id=thread_id)
    except Exception:
        aux_summaries = []

    # Подготовим вопрос для проверки (учитывая историю диалога), чтобы не делать двойную переформулировку
    query_for_check = req.query
    if not req.skip_clarification_check and not req.skip_condensation and len(history_messages) > 1:
        query_for_check = _condense_question_with_history(req.query, history_messages, aux_summaries)

    # ===== ШАГ 1: Проверка на необходимость уточнений =====
    # Посчитаем текущий раунд уточнений по ветке
    try:
        current_round = await session.get_thread_round_count(thread_id)
        max_rounds = await session.get_max_rounds()
    except Exception:
        current_round, max_rounds = 0, 2

    if (not req.skip_clarification_check
        and current_round < max_rounds):
        # Лёгкий предварительный поиск для генерации уточнения на основании KB
        kb_docs = []
        kb_context = ""
        kb_search_start = time.time()
        try:
            pre_k = 5
            if HYBRID_SEARCH_AVAILABLE:
                hr = get_global_hybrid_retriever(embeddings)
                hr.update_k(pre_k)
                kb_docs = hr.get_relevant_documents(query_for_check)
            else:
                vs_pre = CustomQdrant.from_existing_collection(
                    embedding=embeddings,
                    collection_name=QDRANT_COLLECTION,
                    url=QDRANT_URL,
                    prefer_grpc=False,
                    path=None,
                )
                kb_docs = vs_pre.similarity_search(query_for_check, k=pre_k)
            parts = []
            for i, d in enumerate(kb_docs[:pre_k], 1):
                txt = getattr(d, 'page_content', '')
                if txt:
                    parts.append(f"Документ {i}:\n{txt}")
            kb_context = "\n\n".join(parts)
        except Exception:
            kb_context = ""
        kb_time = time.time() - kb_search_start
        clarification_start = time.time()
        clarification_result = _check_clarification_needed(query_for_check, history_messages, kb_context)
        clarification_time = time.time() - clarification_start
        
        metadata["steps"]["clarification_check"] = {
            "performed": True,
            "need_clarification": clarification_result["need_clarification"],
            "reason": clarification_result["reason"],
            "time_ms": round(clarification_time * 1000, 2),
            "kb_used": bool(kb_context),
            "kb_search_ms": round(kb_time * 1000, 2) if kb_context else 0,
            "history_len": len(history_messages),
            "round": current_round,
            "max_rounds": max_rounds
        }
        
        if clarification_result["need_clarification"]:
            clarification_q = clarification_result["clarification_question"]
            await session.add_message("assistant", clarification_q, metadata={"stage": "clarification_request", "thread_id": thread_id, "round": current_round + 1})
            await session.set_pending_questions([clarification_q])
            session.state = "awaiting_clarification"
            
            metadata["timing"]["total_ms"] = round((time.time() - start_time) * 1000, 2)
            
            # Красивый вывод для уточняющих вопросов: вернем только текст вопроса
            if pretty:
                return PlainTextResponse(clarification_q)
            
            return ConversationalResponse(
                session_id=session.session_id,
                session_name=session.session_name,
                state="awaiting_clarification",
                clarification_question=clarification_q,
                clarification_reason=clarification_result["reason"],
                conversation_history=await session.get_thread_messages(thread_id),
                metadata=metadata
            )
    else:
        metadata["steps"]["clarification_check"] = {
            "performed": False,
            "reason": "Skipped by request" if req.skip_clarification_check else "Not first message"
        }
    
    # Обновляем состояние сессии
    if session.state != "active":
        session.state = "active"
    
    # ===== ШАГ 2: Query Condensation (переформулировка с учетом истории) =====
    # Используем уже конденсированный для проверки вариант, чтобы избежать двойной переформулировки
    query_for_search = query_for_check
    already_condensed = (query_for_search != req.query)
    if not req.skip_condensation and len(history_messages) > 1 and not already_condensed:
        condensation_start = time.time()
        query_for_search = _condense_question_with_history(req.query, history_messages, aux_summaries)
        condensation_time = time.time() - condensation_start
        
        metadata["steps"]["query_condensation"] = {
            "performed": True,
            "original_query": req.query,
            "condensed_query": query_for_search,
            "time_ms": round(condensation_time * 1000, 2),
            "history_len": len(history_messages),
            "thread_id": thread_id
        }
        try:
            logger.debug(f"🔄 Переформулировка запроса (thread={thread_id}): '{req.query}' → '{query_for_search}'")
        except Exception:
            pass
    else:
        # Либо уже переформулировали ранее для проверки, либо нет смысла
        metadata["steps"]["query_condensation"] = {
            "performed": already_condensed,
            "reason": "Already condensed for check" if already_condensed else ("Skipped by request" if req.skip_condensation else "No history")
        }
    
    # ===== ШАГ 3: RAG Retrieval (гибридный поиск если доступен) =====
    search_start = time.time()
    
    # Определяем количество кандидатов для начального поиска
    initial_k = req.k * 3 if (req.use_reranking and RERANKING_AVAILABLE) else req.k
    
    # Гибридный поиск (если доступен)
    if HYBRID_SEARCH_AVAILABLE:
        try:
            logger.info("🔍 Using hybrid search for conversational RAG")
            hybrid_retriever = get_global_hybrid_retriever(embeddings)
            hybrid_retriever.update_k(initial_k)
            try:
                logger.info(f"🔎 Поисковый запрос (thread={thread_id}): '{query_for_search}' (orig='{req.query}') k={initial_k}")
            except Exception:
                pass
            docs = hybrid_retriever.get_relevant_documents(query_for_search)
            search_method = "hybrid"
        except Exception as e:
            logger.error(f"❌ Hybrid search failed, fallback to vector search: {e}")
            vs = CustomQdrant.from_existing_collection(
                embedding=embeddings,
                collection_name=QDRANT_COLLECTION,
                url=QDRANT_URL,
                prefer_grpc=False,
                path=None,
            )
            docs = vs.similarity_search(query_for_search, k=initial_k)
            search_method = "vector_fallback"
    else:
        # Обычный векторный поиск
        vs = CustomQdrant.from_existing_collection(
            embedding=embeddings,
            collection_name=QDRANT_COLLECTION,
            url=QDRANT_URL,
            prefer_grpc=False,
            path=None,
        )
        try:
            logger.info(f"🔎 Поисковый запрос (thread={thread_id}): '{query_for_search}' (orig='{req.query}') k={initial_k}")
        except Exception:
            pass
        docs = vs.similarity_search(query_for_search, k=initial_k)
        search_method = "vector"
    
    search_time = time.time() - search_start
    
    # Фильтруем валидные документы
    docs_before_filter = len(docs)
    docs = [doc for doc in docs if doc and doc.page_content and doc.page_content.strip()]
    
    metadata["steps"]["retrieval"] = {
        "method": search_method,
        "query_used": query_for_search,
        "initial_k": initial_k,
        "documents_found": docs_before_filter,
        "documents_after_filter": len(docs),
        "time_ms": round(search_time * 1000, 2)
    }
    
    if not docs:
        error_msg = "Извините, не удалось найти релевантную информацию для вашего запроса."
        await session.add_message("assistant", error_msg)
        metadata["timing"]["total_ms"] = round((time.time() - start_time) * 1000, 2)
        
        return ConversationalResponse(
            session_id=session.session_id,
            session_name=session.session_name,
            state="completed",
            answer=error_msg,
            citations=[],
            condensed_query=query_for_search if not req.skip_condensation else None,
            conversation_history=await session.get_thread_messages(thread_id),
            metadata=metadata
        )
    
    # ===== ШАГ 4: Reranking (если включен) =====
    if req.use_reranking and RERANKING_AVAILABLE and docs:
        try:
            rerank_start = time.time()
            
            # Конвертируем docs в формат для реранкера
            candidates = []
            for d in docs:
                meta = getattr(d, 'metadata', {}) or {}
                if hasattr(d, 'page_content') and d.page_content:
                    meta = {**meta, 'text': d.page_content}
                candidates.append(meta)
            
            rerank_k = req.rerank_top_k or req.k
            logger.info(f"🔄 Reranking {len(candidates)} candidates...")
            reranked_candidates = rerank_candidates(query_for_search, candidates, top_k=rerank_k)
            
            rerank_time = time.time() - rerank_start
            docs_before_rerank = len(docs)
            
            if reranked_candidates:
                docs = docs[:len(reranked_candidates)]
                rerank_method = reranked_candidates[0].get('rerank_method', 'unknown')
                rerank_model = reranked_candidates[0].get('rerank_model', 'unknown')
                
                metadata["steps"]["reranking"] = {
                    "performed": True,
                    "method": rerank_method,
                    "model": rerank_model,
                    "candidates_before": docs_before_rerank,
                    "candidates_after": len(docs),
                    "target_k": rerank_k,
                    "time_ms": round(rerank_time * 1000, 2)
                }
            else:
                docs = docs[:req.k]
                metadata["steps"]["reranking"] = {
                    "performed": True,
                    "success": False,
                    "fallback": "Using original order"
                }
        
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            docs = docs[:req.k]
            metadata["steps"]["reranking"] = {
                "performed": True,
                "error": str(e),
                "fallback": "Using original order"
            }
    else:
        docs = docs[:req.k]
        metadata["steps"]["reranking"] = {
            "performed": False,
            "reason": "Disabled by user" if not req.use_reranking else "Not available"
        }
    
    # ===== ШАГ 5: Форматирование контекста с цитатами =====
    context, citations = _format_docs_with_citations(docs)
    
    if not context:
        error_msg = "Извините, не удалось сформировать контекст для ответа."
        await session.add_message("assistant", error_msg)
        metadata["timing"]["total_ms"] = round((time.time() - start_time) * 1000, 2)
        
        return ConversationalResponse(
            session_id=session.session_id,
            session_name=session.session_name,
            state="completed",
            answer=error_msg,
            citations=[],
            condensed_query=query_for_search if not req.skip_condensation else None,
            conversation_history=await session.get_thread_messages(thread_id),
            metadata=metadata
        )
    
    # ===== ШАГ 6: Генерация ответа с учетом истории =====
    generation_start = time.time()
    
    # Формируем историю для промпта (последние 6 сообщений)
    history_for_prompt = []
    for msg in history_messages[-7:-1]:  # Исключаем текущий вопрос
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            history_for_prompt.append({"role": "user", "content": content})
        elif role == "assistant":
            history_for_prompt.append({"role": "assistant", "content": content})
    
    # Системный промпт
    system_message = {
        "role": "system",
        "content": """Ты — эксперт-ассистент для RAG-системы. Твоя задача — предоставить точный и развернутый ответ на основе контекста из базы знаний.

Правила:
1. Отвечай ТОЛЬКО на основе предоставленного контекста
2. Если информации недостаточно — четко укажи это
3. Используй цитаты источников в формате [#N] где N — номер источника
4. Будь конкретным и структурированным
5. Не выдумывай информацию, которой нет в контексте
Не используй теги <think> и не выводи внутренние размышления. Пиши только финальный ответ.
6. В конце ответа добавь краткий список использованных источников в формате:

Источники: [#1] название_документа, [#2] название_документа..."""
    }
    
    # Формируем финальные сообщения
    messages = [system_message] + history_for_prompt + [
        {
            "role": "user",
            "content": f"""Контекст из базы знаний (с номерами для цитирования):

{context}

Вопрос пользователя: {req.query}

Предоставь развернутый ответ, используя контекст. Обязательно цитируй источники в формате [#N]."""
        }
    ]
    
    # Вызываем LLM
    answer = _call_llm_chat(messages, temperature=0.2, max_tokens=2000)
    generation_time = time.time() - generation_start
    
    await session.add_message("assistant", answer, metadata={"stage": "rag_answer", "thread_id": thread_id})
    # Сохраняем суммаризацию для текущей ветки
    try:
        thread_msgs = await session.get_thread_messages(thread_id)
        thread_summary = _summarize_thread(thread_msgs, answer)
        if thread_summary:
            await session.save_thread_summary(thread_id, thread_summary)
            metadata["steps"]["thread_summary"] = {"saved": True, "length": len(thread_summary)}
        else:
            metadata["steps"]["thread_summary"] = {"saved": False, "reason": "empty"}
    except Exception as e:
        logger.warning(f"Failed to save thread summary (thread={thread_id}): {e}")
        metadata["steps"]["thread_summary"] = {"saved": False, "error": str(e)}
    
    metadata["steps"]["generation"] = {
        "model": LLM_MODEL,
        "max_tokens": 1000,
        "temperature": 0.2,
        "context_length": len(context),
        "history_messages_used": len(history_for_prompt),
        "time_ms": round(generation_time * 1000, 2),
        "thread_id": thread_id
    }
    try:
        examples = []
        for m in history_for_prompt[-2:]:
            role = m.get('role')
            content_preview = (m.get('content') or '').replace('\n', ' ')[:80]
            examples.append((role, content_preview))
        logger.info(
            f"🧾 Генерация (thread={thread_id}): history_used={len(history_for_prompt)}; examples={examples}"
        )
    except Exception:
        pass
    
    metadata["timing"]["total_ms"] = round((time.time() - start_time) * 1000, 2)
    metadata["timing"]["breakdown"] = {
        "clarification_check": metadata["steps"]["clarification_check"].get("time_ms", 0),
        "query_condensation": metadata["steps"]["query_condensation"].get("time_ms", 0),
        "retrieval": metadata["steps"]["retrieval"]["time_ms"],
        "reranking": metadata["steps"]["reranking"].get("time_ms", 0),
        "generation": metadata["steps"]["generation"]["time_ms"]
    }

    if pretty:
        # Для красивого вывода: использовать тот же сгенерированный ответ, только убрать <think>…</think>
        try:
            cleaned_answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.S).strip()
        except Exception:
            cleaned_answer = answer
        return PlainTextResponse(cleaned_answer)
    
    return ConversationalResponse(
        session_id=session.session_id,
        session_name=session.session_name,
        state="completed",
        answer=answer,
        citations=citations,
        condensed_query=query_for_search if not req.skip_condensation else None,
        conversation_history=await session.get_thread_messages(thread_id),
        metadata=metadata
    )


@app.delete("/chat/{session_id}")
async def delete_chat_session(session_id: int):
    """
    Удалить сессию диалога.
    
    Зачем: Позволяет пользователю явно завершить диалог и очистить историю.
    
    Args:
        session_id: ID сессии для удаления
    
    Returns:
        Статус удаления
    """
    await session_manager.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.get("/chat/{session_id}")
async def get_chat_session(session_id: int):
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
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@app.get("/chat/stats")
async def get_chat_stats():
    """
    Получить статистику по всем сессиям.
    
    Зачем: Для мониторинга и отладки системы.
    
    Returns:
        Статистика сессий
    """
    return await session_manager.get_stats()


@app.post("/chat/cleanup")
async def cleanup_expired_sessions():
    """
    Очистить истекшие сессии.
    
    Зачем: Периодическая очистка памяти от неактивных сессий.
    Рекомендуется вызывать по расписанию (например, каждые 10 минут).
    
    Returns:
        Результат очистки
    """
    await session_manager.cleanup_expired()
    stats = await session_manager.get_stats()
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


# ==================== Web UI Endpoints ====================

@app.get("/", response_class=FileResponse)
async def serve_ui():
    """
    Serve the web UI for chatting with the RAG system.
    
    Returns:
        HTML file with the chat interface
    """
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    index_path = os.path.join(static_dir, "index.html")
    
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    else:
        # Fallback message if UI is not found
        return PlainTextResponse(
            "RAG Chat UI not found. "
            "Please ensure static/index.html exists in the services/online-rag directory.",
            status_code=404
        )
