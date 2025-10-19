"""Менеджер сессий для диалогового RAG чатбота.

Зачем: Хранение истории диалогов, состояния уточняющих вопросов и контекста беседы.
Каждая сессия имеет уникальный ID и хранит все сообщения пользователя и ассистента.
"""

import json
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import uuid


# Простое in-memory хранилище (можно заменить на Redis для production)
# Зачем: Для development достаточно хранить в памяти, для production нужен Redis
SESSIONS = {}
SESSION_TTL = timedelta(hours=1)  # Сессия живет 1 час без активности


class ChatSession:
    """
    Представляет одну сессию диалога с пользователем.
    
    Зачем: Хранит всю историю диалога, состояние уточнений и контекст.
    Позволяет LLM учитывать предыдущие сообщения при генерации ответа.
    """
    
    def __init__(self, session_id: str = None):
        """
        Инициализация новой сессии.
        
        Args:
            session_id: Уникальный ID сессии (генерируется автоматически если не указан)
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.messages: List[Dict] = []  # История всех сообщений
        self.context: Dict = {}  # Дополнительный контекст (метаданные)
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Состояния сессии:
        # - "initial": новая сессия, еще не было взаимодействия
        # - "awaiting_clarification": ждем ответов на уточняющие вопросы
        # - "ready": готовы к генерации финального ответа
        self.state = "initial"
        
        self.pending_questions: List[str] = []  # Уточняющие вопросы, ожидающие ответа
        self.clarifications: Dict[str, str] = {}  # Пары "вопрос: ответ"
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """
        Добавить сообщение в историю диалога.
        
        Зачем: Сохраняет все сообщения для контекста. LLM может использовать
        предыдущие сообщения для лучшего понимания текущего запроса.
        
        Args:
            role: Роль отправителя ("user", "assistant", "clarification")
            content: Текст сообщения
            metadata: Дополнительные метаданные (опционально)
        """
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()
    
    def set_pending_questions(self, questions: List[str]):
        """
        Установить уточняющие вопросы, ожидающие ответа.
        
        Зачем: Переводит сессию в режим ожидания уточнений.
        Пользователь должен ответить на эти вопросы перед получением финального ответа.
        
        Args:
            questions: Список уточняющих вопросов
        """
        self.pending_questions = questions
        self.state = "awaiting_clarification"
        self.updated_at = datetime.now()
    
    def add_clarification(self, question: str, answer: str):
        """
        Добавить ответ на уточняющий вопрос.
        
        Зачем: Собирает ответы пользователя на уточняющие вопросы.
        Когда все вопросы отвечены, сессия готова к финальному ответу.
        
        Args:
            question: Уточняющий вопрос
            answer: Ответ пользователя
        """
        self.clarifications[question] = answer
        self.updated_at = datetime.now()
        
        # Проверяем, все ли вопросы отвечены
        if len(self.clarifications) >= len(self.pending_questions):
            self.state = "ready"
    
    def get_conversation_context(self, last_n: int = 5) -> str:
        """
        Получить отформатированную историю диалога.
        
        Зачем: Для передачи в LLM как контекст. LLM видит предыдущие
        сообщения и может давать более релевантные ответы.
        
        Args:
            last_n: Количество последних сообщений для включения
        
        Returns:
            Отформатированная строка с историей диалога
        """
        context_parts = []
        for msg in self.messages[-last_n:]:  # Берем последние N сообщений
            role = msg['role']
            content = msg['content']
            
            # Форматируем в читаемый вид
            if role == "user":
                context_parts.append(f"ПОЛЬЗОВАТЕЛЬ: {content}")
            elif role == "assistant":
                context_parts.append(f"АССИСТЕНТ: {content}")
            elif role == "clarification":
                context_parts.append(f"УТОЧНЕНИЕ: {content}")
        
        return "\n".join(context_parts)
    
    def get_clarifications_text(self) -> str:
        """
        Получить текстовое представление всех уточнений.
        
        Зачем: Для добавления в промпт LLM. Уточнения помогают
        сузить запрос и дать более точный ответ.
        
        Returns:
            Отформатированная строка с уточнениями
        """
        if not self.clarifications:
            return ""
        
        parts = []
        for question, answer in self.clarifications.items():
            parts.append(f"- {question}: {answer}")
        
        return "\n".join(parts)
    
    def to_dict(self) -> Dict:
        """
        Сериализовать сессию в словарь.
        
        Зачем: Для отправки клиенту или сохранения в БД.
        
        Returns:
            Словарь с данными сессии
        """
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "state": self.state,
            "pending_questions": self.pending_questions,
            "clarifications": self.clarifications
        }


class SessionManager:
    """
    Менеджер для управления всеми сессиями чатбота.
    
    Зачем: Централизованное управление сессиями - создание, получение,
    удаление и очистка устаревших сессий.
    """
    
    def __init__(self):
        """Инициализация менеджера сессий."""
        self.sessions = SESSIONS
    
    def create_session(self) -> ChatSession:
        """
        Создать новую сессию.
        
        Зачем: Каждый новый диалог начинается с создания сессии.
        
        Returns:
            Новый объект ChatSession
        """
        session = ChatSession()
        self.sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Получить существующую сессию по ID.
        
        Зачем: Для продолжения существующего диалога.
        Проверяет, не истекла ли сессия (TTL).
        
        Args:
            session_id: ID сессии
        
        Returns:
            Объект ChatSession или None если сессия не найдена/истекла
        """
        session = self.sessions.get(session_id)
        
        # Проверяем, не истекла ли сессия
        if session and (datetime.now() - session.updated_at) > SESSION_TTL:
            # Сессия истекла, удаляем её
            del self.sessions[session_id]
            return None
        
        return session
    
    def get_or_create_session(self, session_id: Optional[str]) -> ChatSession:
        """
        Получить существующую сессию или создать новую.
        
        Зачем: Универсальный метод для работы с сессиями.
        Если ID не указан или сессия не найдена - создается новая.
        
        Args:
            session_id: ID сессии (опционально)
        
        Returns:
            Объект ChatSession (существующий или новый)
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        # Если сессия не найдена или ID не указан - создаем новую
        return self.create_session()
    
    def delete_session(self, session_id: str):
        """
        Удалить сессию.
        
        Зачем: Для явного завершения диалога или очистки памяти.
        
        Args:
            session_id: ID сессии для удаления
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def cleanup_expired(self):
        """
        Удалить все истекшие сессии.
        
        Зачем: Периодическая очистка памяти от неактивных сессий.
        Рекомендуется вызывать периодически (например, каждые 10 минут).
        """
        expired = []
        for sid, session in self.sessions.items():
            if (datetime.now() - session.updated_at) > SESSION_TTL:
                expired.append(sid)
        
        for sid in expired:
            del self.sessions[sid]
        
        if expired:
            print(f"Удалено истекших сессий: {len(expired)}")
    
    def get_stats(self) -> Dict:
        """
        Получить статистику по сессиям.
        
        Зачем: Для мониторинга и отладки.
        
        Returns:
            Словарь со статистикой
        """
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": sum(1 for s in self.sessions.values() 
                                  if (datetime.now() - s.updated_at) < timedelta(minutes=10)),
            "awaiting_clarification": sum(1 for s in self.sessions.values() 
                                         if s.state == "awaiting_clarification")
        }

