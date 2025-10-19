"""Сервис для генерации и обработки уточняющих вопросов.

Зачем: Определяет, нужны ли уточнения для запроса пользователя,
и генерирует 2-3 уточняющих вопроса с использованием LLM.
Использует JSON схемы для гарантии структурированных ответов.
"""

from typing import List, Dict, Tuple
import logging

# Импортируем функции генерации с JSON схемами
from generation.llm_service import (
    generate_json_response,
    CLARIFICATION_CHECK_SCHEMA,
    CLARIFICATION_QUESTIONS_SCHEMA
)


logger = logging.getLogger(__name__)


class ClarificationService:
    """
    Сервис для работы с уточняющими вопросами в диалоговом RAG.
    
    Зачем: Улучшает качество ответов RAG системы, запрашивая у пользователя
    дополнительную информацию когда запрос слишком расплывчатый или неоднозначный.
    """
    
    def should_clarify(self, query: str, context: str = "") -> Tuple[bool, str, float]:
        """
        Определить, нужны ли уточняющие вопросы для запроса.
        
        Зачем: Анализирует запрос пользователя и решает, достаточно ли информации
        для ответа или нужны дополнительные уточнения.
        
        Использует vLLM guided_json для гарантии структурированного ответа:
        - needs_clarification: boolean
        - reason: string
        - confidence: float (0-1)
        
        Args:
            query: Запрос пользователя
            context: Предыдущий контекст диалога (опционально)
        
        Returns:
            Кортеж (нужны_уточнения, причина, уверенность)
        
        Example:
            needs, reason, conf = service.should_clarify("медицинские услуги")
            # (True, "Запрос слишком общий, нужны детали", 0.9)
        """
        
        # Формируем промпт для анализа
        prompt = f"""Проанализируй запрос пользователя и определи, нужны ли уточняющие вопросы.

Запрос пользователя: "{query}"

Предыдущий контекст: {context if context else "Нет предыдущего контекста"}

Определи, является ли запрос:
1. Слишком расплывчатым или общим
2. Неоднозначным (может иметь несколько интерпретаций)
3. Недостаточно конкретным для качественного ответа

Если запрос достаточно конкретный и понятный - уточнения не нужны.
Если запрос слишком общий или неоднозначный - нужны уточнения.

Примеры когда нужны уточнения:
- "медицинские услуги" (слишком общо)
- "страховка" (неоднозначно)
- "как получить" (непонятно что именно)

Примеры когда уточнения НЕ нужны:
- "когда не оплачивают мед услуги при переломе руки" (конкретно)
- "какие документы нужны для оформления полиса ОМС" (понятно)
"""
        
        try:
            # Используем vLLM guided_json для гарантии структуры
            # Зачем: Гарантирует что получим именно boolean, string и number
            result = generate_json_response(
                prompt, 
                json_schema=CLARIFICATION_CHECK_SCHEMA,
                temperature=0.3  # Низкая температура для стабильности
            )
            
            needs_clarification = result.get("needs_clarification", False)
            reason = result.get("reason", "")
            confidence = result.get("confidence", 0.5)
            
            logger.info(f"Анализ запроса: needs_clarification={needs_clarification}, confidence={confidence}")
            
            return needs_clarification, reason, confidence
            
        except Exception as e:
            logger.error(f"Ошибка при анализе необходимости уточнений: {e}")
            # Fallback: по умолчанию не требуем уточнений при ошибке
            return False, f"Ошибка анализа: {str(e)}", 0.0
    
    def generate_clarification_questions(
        self, 
        query: str, 
        context: str = "",
        num_questions: int = 3
    ) -> Tuple[List[str], List[str]]:
        """
        Сгенерировать уточняющие вопросы для запроса пользователя.
        
        Зачем: Создает 2-3 конкретных вопроса, которые помогут уточнить
        намерения пользователя и дать более точный ответ.
        
        Использует vLLM guided_json для гарантии структуры:
        - questions: array of strings (2-3 вопроса)
        - priority: array of "high"/"medium"/"low"
        
        Args:
            query: Исходный запрос пользователя
            context: Предыдущий контекст диалога
            num_questions: Количество вопросов для генерации (2-3)
        
        Returns:
            Кортеж (список_вопросов, список_приоритетов)
        
        Example:
            questions, priorities = service.generate_clarification_questions(
                "медицинские услуги"
            )
            # (["Какой тип услуг?", "Про оплату или доступность?"], ["high", "high"])
        """
        
        # Формируем промпт для генерации вопросов
        prompt = f"""Сгенерируй {num_questions} уточняющих вопроса для запроса пользователя.

Исходный запрос: "{query}"

Предыдущий контекст: {context if context else "Нет предыдущего контекста"}

Сгенерируй вопросы, которые помогут:
1. Сузить тему запроса (какой конкретно аспект интересует)
2. Понять специфические потребности пользователя
3. Получить важные недостающие детали

Требования к вопросам:
- Вопросы должны быть конкретными и понятными
- На русском языке
- Короткие (не более 15 слов каждый)
- Напрямую связаны с исходным запросом
- Помогают сузить область поиска

Примеры хороших вопросов:
- "Какой конкретно тип медицинских услуг вас интересует?"
- "Это касается государственных или частных клиник?"
- "Вы хотите узнать о стоимости или о правилах оплаты?"

Определи приоритет каждого вопроса:
- high: критически важный для понимания запроса
- medium: полезный но не обязательный
- low: дополнительный контекст
"""
        
        try:
            # Используем vLLM guided_json для гарантии структуры
            # Зачем: Гарантирует что получим массив из 2-3 строк и приоритеты
            result = generate_json_response(
                prompt,
                json_schema=CLARIFICATION_QUESTIONS_SCHEMA,
                temperature=0.7,  # Выше температура для креативности
                max_tokens=500
            )
            
            questions = result.get("questions", [])
            priorities = result.get("priority", [])
            
            # Обрезаем до нужного количества
            questions = questions[:num_questions]
            priorities = priorities[:num_questions]
            
            # Дополняем приоритеты если их меньше чем вопросов
            while len(priorities) < len(questions):
                priorities.append("medium")
            
            logger.info(f"Сгенерировано {len(questions)} уточняющих вопросов")
            
            return questions, priorities
            
        except Exception as e:
            logger.error(f"Ошибка при генерации уточняющих вопросов: {e}")
            # Fallback: возвращаем базовые вопросы
            fallback_questions = [
                "Уточните, пожалуйста, какой именно аспект вас интересует?",
                "Можете предоставить больше деталей о вашем запросе?"
            ]
            return fallback_questions, ["high", "medium"]
    
    def enhance_query_with_clarifications(
        self,
        original_query: str,
        clarifications: Dict[str, str]
    ) -> str:
        """
        Объединить исходный запрос с ответами на уточняющие вопросы.
        
        Зачем: Создает расширенный запрос, который включает дополнительную
        информацию от пользователя. Этот расширенный запрос используется
        для более точного поиска в RAG системе.
        
        Args:
            original_query: Исходный запрос пользователя
            clarifications: Словарь {вопрос: ответ}
        
        Returns:
            Расширенный запрос с уточнениями
        
        Example:
            enhanced = service.enhance_query_with_clarifications(
                "медицинские услуги",
                {
                    "Какой тип услуг?": "стоматология",
                    "Про оплату?": "да, про правила оплаты"
                }
            )
            # "медицинские услуги\n\nДополнительная информация:\n- стоматология\n- правила оплаты"
        """
        
        if not clarifications:
            return original_query
        
        # Собираем все ответы пользователя
        clarification_parts = []
        for question, answer in clarifications.items():
            # Добавляем только ответы (вопросы уже понятны из контекста)
            clarification_parts.append(f"- {answer}")
        
        # Формируем расширенный запрос
        enhanced_query = f"""{original_query} Дополнительная информация: {chr(10).join(clarification_parts)}"""
        
        logger.info(f"Расширенный запрос создан с {len(clarifications)} уточнениями")
        
        return enhanced_query
    
    def validate_clarification_answers(
        self,
        questions: List[str],
        answers: Dict[str, str]
    ) -> Tuple[bool, List[str]]:
        """
        Проверить, что пользователь ответил на все уточняющие вопросы.
        
        Зачем: Валидация перед генерацией финального ответа.
        Гарантирует что у нас есть все необходимые уточнения.
        
        Args:
            questions: Список заданных вопросов
            answers: Словарь с ответами пользователя
        
        Returns:
            Кортеж (все_отвечены, список_неотвеченных_вопросов)
        """
        unanswered = []
        
        for question in questions:
            if question not in answers or not answers[question].strip():
                unanswered.append(question)
        
        all_answered = len(unanswered) == 0
        
        if all_answered:
            logger.info("Все уточняющие вопросы отвечены")
        else:
            logger.warning(f"Неотвеченные вопросы: {len(unanswered)}")
        
        return all_answered, unanswered

