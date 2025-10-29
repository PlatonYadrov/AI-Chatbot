"""PostgreSQL-based async session manager for conversational RAG.

Зачем: Persistent хранение истории диалогов в PostgreSQL с async/await.
Позволяет сохранять сессии между перезапусками сервиса и масштабироваться.
Использует asyncpg для неблокирующих операций с БД.
"""

import os
import json
import uuid
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from sqlalchemy import Column, String, Text, DateTime, BigInteger, ForeignKey, Index, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import JSONB
from loguru import logger

# Database connection configuration (asyncpg)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://rag_user:rag_password@postgres:5432/rag_chatbot"
)

SESSION_TTL = timedelta(hours=24)

Base = declarative_base()


class ChatSessionDB(Base):
    """SQLAlchemy модель для хранения сессий чата (numeric id + name)."""
    
    __tablename__ = "chat_sessions"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)
    state = Column(String(50), nullable=False, default="initial")
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    pending_questions = Column(JSONB, default=[])
    clarifications = Column(JSONB, default={})
    current_thread_id = Column(BigInteger, nullable=False, default=0)
    max_clarification_rounds = Column(BigInteger, nullable=False, default=2)
    
    messages = relationship("ChatMessageDB", back_populates="session", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_chat_sessions_updated_at', 'updated_at'),
        Index('idx_chat_sessions_state', 'state'),
    )


class ChatMessageDB(Base):
    """SQLAlchemy модель для хранения сообщений чата."""
    
    __tablename__ = "chat_messages"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(BigInteger, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    message_metadata = Column('metadata', JSONB, default={})
    stage = Column(String(50), nullable=False, default='user_question')
    thread_id = Column(BigInteger, nullable=False, default=0)
    round = Column(BigInteger, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    session = relationship("ChatSessionDB", back_populates="messages")
    
    __table_args__ = (
        Index('idx_chat_messages_session_id', 'session_id'),
        Index('idx_chat_messages_created_at', 'created_at'),
    )


class DatabaseSessionManager:
    """Async менеджер подключений к БД с connection pooling."""
    
    def __init__(self, database_url: str = DATABASE_URL):
        """
        Инициализация менеджера подключений.
        
        Args:
            database_url: URL подключения к PostgreSQL (asyncpg)
        """
        self.engine = create_async_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
        self.async_session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        self._initialized = False
    
    async def initialize(self):
        """Инициализация таблиц БД (вызывается один раз при старте)."""
        if self._initialized:
            return
        
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("✅ Database tables created/verified successfully")
            self._initialized = True
        except Exception as e:
            logger.error(f"❌ Failed to create database tables: {e}")
            raise
    
    def get_session(self) -> AsyncSession:
        """Получить новую async сессию БД."""
        return self.async_session_factory()


_db_manager: Optional[DatabaseSessionManager] = None


def get_db_manager() -> DatabaseSessionManager:
    """Получить глобальный экземпляр менеджера БД (singleton)."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseSessionManager()
    return _db_manager


class ChatSession:
    """
    Представляет одну сессию диалога с пользователем (PostgreSQL version).
    
    Зачем: Хранит всю историю диалога, состояние уточнений и контекст в БД.
    """
    
    def __init__(
        self,
        session_id: int = None,
        session_name: str = None,
        db_session: AsyncSession = None,
        state: str = "initial",
        created_at: datetime = None,
        updated_at: datetime = None,
        pending_questions: List[str] = None,
        clarifications: Dict[str, str] = None
    ):
        """
        Инициализация сессии.
        
        Args:
            session_id: UUID сессии
            db_session: SQLAlchemy session для работы с БД
            state: Состояние сессии
            context: Дополнительный контекст
            created_at: Время создания
            updated_at: Время обновления
            pending_questions: Ожидающие уточнения
            clarifications: Ответы на уточнения
        """
        self.session_id = session_id
        self.session_name = session_name
        self.state = state
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
        self.pending_questions = pending_questions or []
        self.clarifications = clarifications or {}
        self._db_session = db_session
        self._messages_cache: Optional[List[Dict]] = None
    
    async def get_messages(self) -> List[Dict]:
        """
        Получить все сообщения сессии.
        
        Returns:
            Список сообщений
        """
        if self._messages_cache is None:
            await self._load_messages()
        return self._messages_cache
    
    @property
    def messages(self) -> List[Dict]:
        """
        Sync property для обратной совместимости.
        WARNING: Может вернуть пустой список если сообщения не загружены!
        Используйте await get_messages() для гарантированной загрузки.
        """
        return self._messages_cache or []
    
    async def _load_messages(self):
        """Загрузить сообщения из БД."""
        if self._db_session is None:
            self._messages_cache = []
            return
        
        try:
            result = await self._db_session.execute(
                select(ChatMessageDB)
                .filter(ChatMessageDB.session_id == self.session_id)
                .order_by(ChatMessageDB.created_at)
            )
            messages_db = result.scalars().all()
            
            self._messages_cache = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.created_at.isoformat(),
                    "metadata": msg.message_metadata or {}
                }
                for msg in messages_db
            ]
        except Exception as e:
            logger.error(f"Failed to load messages for session {self.session_id}: {e}")
            self._messages_cache = []
    
    async def add_message(self, role: str, content: str, metadata: Dict = None):
        """
        Добавить сообщение в историю диалога.
        
        Args:
            role: Роль отправителя ("user", "assistant", "clarification")
            content: Текст сообщения
            metadata: Дополнительные метаданные
        """
        if self._db_session is None:
            logger.warning(f"Cannot add message: no database session for {self.session_id}")
            return
        
        try:
            meta = metadata or {}
            message = ChatMessageDB(
                session_id=self.session_id,
                role=role,
                content=content,
                message_metadata=meta,
                stage=str(meta.get('stage') or 'user_question'),
                thread_id=int(meta.get('thread_id') or 0),
                round=int(meta.get('round') or 0),
                created_at=datetime.utcnow()
            )
            self._db_session.add(message)
            await self._db_session.commit()
            
            # Invalidate cache
            self._messages_cache = None
            
            # Update session timestamp
            self.updated_at = datetime.utcnow()
            await self._save_session()
            
        except Exception as e:
            logger.error(f"Failed to add message to session {self.session_id}: {e}")
            await self._db_session.rollback()
    
    async def set_pending_questions(self, questions: List[str]):
        """
        Установить уточняющие вопросы.
        
        Args:
            questions: Список уточняющих вопросов
        """
        self.pending_questions = questions
        self.state = "awaiting_clarification"
        self.updated_at = datetime.utcnow()
        await self._save_session()
    
    async def add_clarification(self, question: str, answer: str):
        """
        Добавить ответ на уточняющий вопрос.
        
        Args:
            question: Уточняющий вопрос
            answer: Ответ пользователя
        """
        self.clarifications[question] = answer
        self.updated_at = datetime.utcnow()
        
        if len(self.clarifications) >= len(self.pending_questions):
            self.state = "active"
        
        await self._save_session()
    
    async def _save_session(self):
        """Сохранить изменения сессии в БД."""
        if self._db_session is None:
            return
        
        try:
            result = await self._db_session.execute(
                select(ChatSessionDB).filter(ChatSessionDB.id == self.session_id)
            )
            session_db = result.scalar_one_or_none()
            
            if session_db:
                session_db.state = self.state
                session_db.updated_at = self.updated_at
                session_db.pending_questions = self.pending_questions
                session_db.clarifications = self.clarifications
            else:
                # Создание сессии без имени через saver не поддерживается
                raise RuntimeError("ChatSession save called for missing session id")
                self._db_session.add(session_db)
            
            await self._db_session.commit()
            
        except Exception as e:
            logger.error(f"Failed to save session {self.session_id}: {e}")
            await self._db_session.rollback()
    
    def get_conversation_context(self, last_n: int = 5) -> str:
        """
        Получить отформатированную историю диалога.
        
        Args:
            last_n: Количество последних сообщений
        
        Returns:
            Отформатированная строка с историей
        """
        messages = self.messages[-last_n:]
        context_parts = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == "user":
                context_parts.append(f"ПОЛЬЗОВАТЕЛЬ: {content}")
            elif role == "assistant":
                context_parts.append(f"АССИСТЕНТ: {content}")
            elif role == "clarification":
                context_parts.append(f"УТОЧНЕНИЕ: {content}")
        
        return "\n".join(context_parts)
    
    def get_clarifications_text(self) -> str:
        """
        Получить текстовое представление уточнений.
        
        Returns:
            Отформатированная строка с уточнениями
        """
        if not self.clarifications:
            return ""
        
        parts = [f"- {q}: {a}" for q, a in self.clarifications.items()]
        return "\n".join(parts)
    
    def to_dict(self) -> Dict:
        """
        Сериализовать сессию в словарь.
        
        Returns:
            Словарь с данными сессии
        """
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "state": self.state,
            "pending_questions": self.pending_questions,
            "clarifications": self.clarifications
        }

    # ===== Thread helpers =====
    async def get_max_rounds(self) -> int:
        if self._db_session is None:
            return 2
        result = await self._db_session.execute(
            select(ChatSessionDB.max_clarification_rounds).filter(ChatSessionDB.id == self.session_id)
        )
        val = result.scalar_one_or_none()
        return int(val or 2)

    async def set_max_rounds(self, n: int) -> None:
        if self._db_session is None:
            return
        async with self._db_session.begin():
            result = await self._db_session.execute(
                select(ChatSessionDB).filter(ChatSessionDB.id == self.session_id)
            )
            row = result.scalar_one_or_none()
            if row:
                row.max_clarification_rounds = int(max(0, n))

    async def resolve_thread_id_for_new_user_question(self) -> int:
        if self._db_session is None:
            return 0
        # Получим последнюю ассистентскую стадию
        last_assistant = await self._db_session.execute(
            select(ChatMessageDB).filter(
                ChatMessageDB.session_id == self.session_id
            ).order_by(ChatMessageDB.created_at.desc()).limit(1)
        )
        last = last_assistant.scalar_one_or_none()
        # Если последний ответ был финальным RAG, начинаем новую ветку
        start_new = bool(last and last.stage == 'rag_answer')
        # Получим/обновим current_thread_id
        result = await self._db_session.execute(
            select(ChatSessionDB).filter(ChatSessionDB.id == self.session_id)
        )
        sess = result.scalar_one_or_none()
        if not sess:
            return 0
        if start_new:
            sess.current_thread_id = int(sess.current_thread_id or 0) + 1
            await self._db_session.commit()
        return int(sess.current_thread_id or 0)

    async def get_thread_messages(self, thread_id: int) -> List[Dict]:
        if self._db_session is None:
            return []
        res = await self._db_session.execute(
            select(ChatMessageDB).filter(
                ChatMessageDB.session_id == self.session_id,
                ChatMessageDB.thread_id == int(thread_id)
            ).order_by(ChatMessageDB.created_at)
        )
        rows = res.scalars().all()
        out = []
        for r in rows:
            out.append({
                "role": r.role,
                "content": r.content,
                "timestamp": r.created_at.isoformat(),
                "metadata": r.message_metadata or {}
            })
        return out

    async def get_thread_round_count(self, thread_id: int) -> int:
        if self._db_session is None:
            return 0
        res = await self._db_session.execute(
            select(ChatMessageDB).filter(
                ChatMessageDB.session_id == self.session_id,
                ChatMessageDB.thread_id == int(thread_id),
                ChatMessageDB.stage == 'clarification_answer'
            )
        )
        cnt = len(res.scalars().all())
        return cnt


class PostgresSessionManager:
    """
    Async PostgreSQL-based менеджер сессий для диалогового RAG.
    
    Зачем: Persistent хранение сессий в БД с неблокирующими операциями.
    """
    
    def __init__(self):
        """Инициализация менеджера."""
        self.db_manager = get_db_manager()
        self._initialized = False
    
    async def initialize(self):
        """Инициализация БД (вызывается при старте приложения)."""
        if not self._initialized:
            await self.db_manager.initialize()
            self._initialized = True
    
    async def create_session(self, session_name: str) -> ChatSession:
        """
        Создать новую сессию.
        
        Returns:
            Новый объект ChatSession
        """
        async with self.db_manager.get_session() as db_session:
            
            try:
                session_db = ChatSessionDB(
                    name=session_name,
                    state="initial",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    pending_questions=[],
                    clarifications={}
                )
                db_session.add(session_db)
                await db_session.commit()
                
                # Create new session for ongoing operations
                new_db_session = self.db_manager.get_session()
                
                chat_session = ChatSession(
                    session_id=session_db.id,
                    session_name=session_name,
                    db_session=new_db_session,
                    state="initial"
                )
                
                logger.info(f"✅ Created new session: id={session_db.id}, name={session_name}")
                return chat_session
                
            except Exception as e:
                logger.error(f"❌ Failed to create session: {e}")
                await db_session.rollback()
                raise
    
    async def get_session(self, session_id: int) -> Optional[ChatSession]:
        """
        Получить существующую сессию по ID.
        
        Args:
            session_id: ID сессии
        
        Returns:
            ChatSession или None
        """
        async with self.db_manager.get_session() as db_session:
            try:
                result = await db_session.execute(
                    select(ChatSessionDB).filter(ChatSessionDB.id == session_id)
                )
                session_db = result.scalar_one_or_none()
                
                if not session_db:
                    return None
                
                # Проверка TTL
                if (datetime.utcnow() - session_db.updated_at) > SESSION_TTL:
                    logger.info(f"Session {session_id} expired, deleting")
                    await db_session.delete(session_db)
                    await db_session.commit()
                    return None
                
                # Create new session for ongoing operations
                new_db_session = self.db_manager.get_session()
                
                chat_session = ChatSession(
                    session_id=session_db.id,
                    session_name=session_db.name,
                    db_session=new_db_session,
                    state=session_db.state,
                    created_at=session_db.created_at,
                    updated_at=session_db.updated_at,
                    pending_questions=session_db.pending_questions or [],
                    clarifications=session_db.clarifications or {}
                )
                # Сохраним текущую конфигурацию лимита раундов в metadata для использования в API
                # (получать через отдельные методы ниже)
                
                # Preload messages
                await chat_session._load_messages()
                
                return chat_session
                
            except Exception as e:
                logger.error(f"❌ Failed to get session {session_id}: {e}")
                return None
    
    async def get_or_create_session_by_name(self, session_name: Optional[str]) -> ChatSession:
        """
        Получить существующую сессию или создать новую.
        
        Args:
            session_id: ID сессии (опционально)
        
        Returns:
            ChatSession
        """
        name = (session_name or "").strip()
        if not name:
            raise ValueError("session_name is required")
        
        # Ищем по имени
        async with self.db_manager.get_session() as db_session:
            result = await db_session.execute(
                select(ChatSessionDB).filter(ChatSessionDB.name == name)
            )
            session_db = result.scalar_one_or_none()
            
            if session_db:
                # Возвращаем существующую
                new_db_session = self.db_manager.get_session()
                chat_session = ChatSession(
                    session_id=session_db.id,
                    session_name=session_db.name,
                    db_session=new_db_session,
                    state=session_db.state,
                    created_at=session_db.created_at,
                    updated_at=session_db.updated_at,
                    pending_questions=session_db.pending_questions or [],
                    clarifications=session_db.clarifications or {}
                )
                await chat_session._load_messages()
                return chat_session
        
        # Нет — создаем новую по имени
        return await self.create_session(name)
    
    async def delete_session(self, session_id: str):
        """
        Удалить сессию.
        
        Args:
            session_id: ID сессии для удаления
        """
        async with self.db_manager.get_session() as db_session:
            try:
                result = await db_session.execute(
                    select(ChatSessionDB).filter(ChatSessionDB.session_id == session_id)
                )
                session_db = result.scalar_one_or_none()
                
                if session_db:
                    await db_session.delete(session_db)
                    await db_session.commit()
                    logger.info(f"✅ Deleted session: {session_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to delete session {session_id}: {e}")
                await db_session.rollback()
    
    async def cleanup_expired(self):
        """Удалить все истекшие сессии."""
        async with self.db_manager.get_session() as db_session:
            try:
                cutoff_time = datetime.utcnow() - SESSION_TTL
                
                result = await db_session.execute(
                    select(ChatSessionDB).filter(ChatSessionDB.updated_at < cutoff_time)
                )
                expired = result.scalars().all()
                
                count = len(expired)
                
                for session in expired:
                    await db_session.delete(session)
                
                await db_session.commit()
                
                if count > 0:
                    logger.info(f"✅ Cleaned up {count} expired sessions")
                
            except Exception as e:
                logger.error(f"❌ Failed to cleanup expired sessions: {e}")
                await db_session.rollback()
    
    async def get_stats(self) -> Dict:
        """
        Получить статистику по сессиям.
        
        Returns:
            Словарь со статистикой
        """
        async with self.db_manager.get_session() as db_session:
            try:
                # Total sessions
                total_result = await db_session.execute(select(ChatSessionDB))
                total = len(total_result.scalars().all())
                
                # Active sessions (last 10 minutes)
                active_cutoff = datetime.utcnow() - timedelta(minutes=10)
                active_result = await db_session.execute(
                    select(ChatSessionDB).filter(ChatSessionDB.updated_at >= active_cutoff)
                )
                active = len(active_result.scalars().all())
                
                # Awaiting clarification
                awaiting_result = await db_session.execute(
                    select(ChatSessionDB).filter(ChatSessionDB.state == "awaiting_clarification")
                )
                awaiting = len(awaiting_result.scalars().all())
                
                return {
                    "total_sessions": total,
                    "active_sessions": active,
                    "awaiting_clarification": awaiting
                }
                
            except Exception as e:
                logger.error(f"❌ Failed to get stats: {e}")
                return {
                    "total_sessions": 0,
                    "active_sessions": 0,
                    "awaiting_clarification": 0,
                    "error": str(e)
                }


