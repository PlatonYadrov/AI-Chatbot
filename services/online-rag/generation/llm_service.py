"""Клиент для локального vLLM OpenAI-совместимого эндпоинта с поддержкой JSON Schema."""

import os
from typing import Optional, List, Tuple, Dict, Any
import json
import requests
import re
import logging


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация vLLM сервера
VLLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct-AWQ")
VLLM_API_KEY = os.getenv("LLM_API_KEY", "dummy")  # vLLM может работать без аутентификации


def _post(path: str, payload: dict) -> dict:
    """
    Выполнить POST запрос к vLLM API.
    
    Args:
        path: Путь API (например, "/chat/completions")
        payload: JSON данные для отправки
    
    Returns:
        Ответ от vLLM в формате dict
    """
    url = f"{VLLM_BASE_URL}{path}"
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}", "Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка запроса к vLLM: {e}")
        raise


def generate_response(
    prompt: str, 
    max_tokens: int = 512, 
    temperature: float = 0.2, 
    top_p: float = 0.95,
    stop: Optional[List[str]] = None
) -> Tuple[str, str]:
    """
    Генерирует текстовый ответ (обратная совместимость со старым кодом).
    
    Зачем: Основная функция для генерации текста в RAG системе.
    Используется в gateway.py для создания финального ответа пользователю.
    
    Args:
        prompt: Промпт пользователя с контекстом
        max_tokens: Максимальное количество токенов для генерации
        temperature: Температура сэмплирования (0.0 = детерминированно, 1.0 = креативно)
        top_p: Nucleus sampling параметр
        stop: Стоп-последовательности для остановки генерации
    
    Returns:
        Кортеж (ответ, размышления) где размышления извлекаются из <think> тегов
    """
    messages = [
        {"role": "system", "content": "Ты полезный и лаконичный ассистент."},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if stop:
        payload["stop"] = stop

    data = _post("/chat/completions", payload)
    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    
    # Извлекаем внутренние размышления модели (если есть)
    thoughts = ""
    if "<think>" in content:
        m = re.search(r"<think>(.*?)</think>", content, flags=re.S)
        if m:
            thoughts = m.group(1).strip()
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.S).strip()
    
    return content, thoughts


def generate_json_response(
    prompt: str,
    json_schema: Optional[Dict[str, Any]] = None,
    max_tokens: int = 800,
    temperature: float = 0.3,
    top_p: float = 0.95,
    use_guided_decoding: bool = True
) -> Dict[str, Any]:
    """
    Генерирует структурированный JSON ответ с автоматической валидацией через vLLM.
    
    Зачем: Для диалогового чатбота нужны структурированные ответы:
    - Определение нужны ли уточнения (boolean)
    - Список уточняющих вопросов (array)
    - Метаданные ответа (confidence, sources_used и т.д.)
    
    vLLM guided decoding гарантирует 100% валидный JSON по схеме!
    
    Args:
        prompt: Промпт пользователя
        json_schema: JSON Schema для валидации (vLLM гарантирует соответствие)
        max_tokens: Максимальное количество токенов
        temperature: Температура сэмплирования
        top_p: Nucleus sampling параметр
        use_guided_decoding: Использовать guided decoding vLLM (рекомендуется)
    
    Returns:
        Распарсенный JSON dict (гарантированно валидный при use_guided_decoding=True)
    
    Example:
        schema = {
            "type": "object",
            "properties": {
                "needs_clarification": {"type": "boolean"},
                "reason": {"type": "string"}
            },
            "required": ["needs_clarification", "reason"]
        }
        
        result = generate_json_response(
            "Проанализируй запрос: медицинские услуги",
            json_schema=schema
        )
        # Гарантированно вернет валидный JSON: {"needs_clarification": true, "reason": "..."}
    """
    
    # Системный промпт для JSON генерации
    system_content = """Ты полезный ассистент, который отвечает в формате JSON.
Всегда возвращай валидный JSON, который можно распарсить.
Не включай markdown блоки кода или текст вне JSON объекта.
Все строковые значения должны быть на русском языке."""
    
    # Добавляем описание схемы в промпт для лучшего понимания
    if json_schema:
        schema_str = json.dumps(json_schema, indent=2, ensure_ascii=False)
        system_content += f"\n\nОжидаемая структура JSON:\n{schema_str}"
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    
    # 🎯 КЛЮЧЕВАЯ ФИЧА: vLLM Guided Decoding
    # Зачем: vLLM на каждом шаге генерации проверяет валидность токена для JSON схемы
    # Результат: 100% гарантия валидного JSON без необходимости повторных попыток
    if use_guided_decoding and json_schema:
        logger.info("Используем vLLM guided_json для гарантии валидности")
        payload["guided_json"] = json_schema
    elif use_guided_decoding and not json_schema:
        # Базовый JSON mode без строгой схемы (просто валидный JSON)
        logger.info("Используем базовый JSON mode")
        payload["response_format"] = {"type": "json_object"}
    
    data = _post("/chat/completions", payload)
    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    
    # Парсинг JSON
    try:
        # Убираем markdown блоки если модель их добавила
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        content = content.strip()
        
        # Парсим JSON (если использовали guided_json, это гарантированно валидный JSON)
        result = json.loads(content)
        logger.info(f"JSON успешно распарсен: {list(result.keys())}")
        return result
        
    except json.JSONDecodeError as e:
        # Это не должно происходить при use_guided_decoding=True
        # Но добавляем fallback для надежности
        logger.warning(f"Ошибка парсинга JSON: {e}")
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        # Если всё провалилось, возвращаем структуру с ошибкой
        return {
            "error": "Не удалось распарсить JSON",
            "raw_content": content,
            "parse_error": str(e)
        }


def generate_with_choice(
    prompt: str,
    choices: List[str],
    temperature: float = 0.3
) -> str:
    """
    Генерирует ответ, ограниченный списком вариантов.
    
    Зачем: Для простых бинарных решений (да/нет, нужны уточнения/не нужны)
    vLLM гарантирует, что ответ будет одним из предложенных вариантов.
    
    Args:
        prompt: Промпт пользователя
        choices: Список допустимых вариантов ответа
        temperature: Температура сэмплирования
    
    Returns:
        Один из вариантов из списка choices
    
    Example:
        result = generate_with_choice(
            "Нужны ли уточнения для этого запроса?",
            choices=["да", "нет", "не уверен"]
        )
        # Гарантированно вернет "да", "нет" или "не уверен"
    """
    
    messages = [
        {"role": "system", "content": "Ты полезный ассистент."},
        {"role": "user", "content": prompt},
    ]
    
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "max_tokens": 50,
        "temperature": temperature,
        # 🎯 vLLM Choice Guided Decoding
        "guided_choice": choices
    }
    
    data = _post("/chat/completions", payload)
    choice = data.get("choices", [{}])[0]
    return choice.get("message", {}).get("content", "")


# 🎯 Готовые JSON схемы для диалогового RAG чатбота
# Зачем: Переиспользуемые схемы для типовых задач диалога

CLARIFICATION_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_clarification": {
            "type": "boolean",
            "description": "Нужны ли уточняющие вопросы для ответа на запрос"
        },
        "reason": {
            "type": "string",
            "description": "Причина почему нужны или не нужны уточнения"
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Уверенность в решении от 0 до 1"
        }
    },
    "required": ["needs_clarification", "reason", "confidence"]
}

CLARIFICATION_QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 3,
            "description": "Список из 2-3 уточняющих вопросов на русском языке"
        },
        "priority": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["high", "medium", "low"]
            },
            "description": "Приоритет каждого вопроса (соответствует порядку questions)"
        }
    },
    "required": ["questions", "priority"]
}

RAG_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Основной ответ на вопрос пользователя"
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Уверенность в ответе от 0 до 1"
        },
        "sources_used": {
            "type": "integer",
            "minimum": 0,
            "description": "Количество использованных источников"
        },
        "follow_up_questions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3,
            "description": "Предложения для продолжения диалога (опционально)"
        }
    },
    "required": ["answer", "confidence", "sources_used"]
}
