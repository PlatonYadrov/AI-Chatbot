# Интеграция Docling - Сводка изменений

## ✅ Выполненные задачи

Успешно интегрирована библиотека **Docling** от IBM Research для унифицированной обработки документов всех форматов.

---

## 📁 Созданные файлы

### 1. **services/ingestion/parsers/docling_parser.py** (новый)
Унифицированный парсер на базе Docling:
- Поддержка PDF, DOCX, PPTX, XLSX, изображений, HTML, Markdown
- Встроенный OCR высокого качества
- Автоматическое извлечение таблиц
- Сохранение структуры документа (заголовки, параграфы, списки)
- Богатые метаданные для каждого блока

**Основные возможности:**
```python
parser = DoclingParser(
    extract_tables=True,
    extract_images=True,
    ocr_enabled=True,
    preserve_structure=True,
)
blocks = list(parser.parse("document.pdf"))
```

### 2. **tests/test_docling_integration.py** (новый)
Полный набор интеграционных тестов:
- Тест парсинга PDF
- Тест парсинга DOCX
- Тест интеграции с chunker
- Тест сохранения метаданных

### 3. **tests/test_docling_minimal.py** (новый)
Быстрый тест для проверки установки и базовой функциональности.

### 4. **services/ingestion/DOCLING_MIGRATION.md** (новый)
Подробное руководство по миграции:
- Сравнение старого и нового подходов
- Примеры использования
- Решение типичных проблем
- Информация о производительности

### 5. **services/ingestion/README_DOCLING.md** (новый)
Краткое руководство по быстрому старту:
- Установка
- Примеры использования
- Таблица поддерживаемых форматов
- Советы и рекомендации

---

## 🔧 Изменённые файлы

### 1. **services/ingestion/api.py**

**Изменения:**
- Удалены импорты старых парсеров (PdfParser, DocxParser, PptxParser, XlsxParser, EbookParser, ScormParser)
- Удалён импорт `normalize_text` (Docling сам нормализует текст)
- Добавлен импорт `DoclingParser`
- Упрощена логика выбора парсера: теперь используется единый `DoclingParser` для всех форматов
- Убрана дополнительная нормализация текста (Docling возвращает уже очищенный текст)

**Было (40+ строк):**
```python
if suffix == "pdf":
    parser_blocks = list(PdfParser().parse(...))
elif suffix == "docx":
    parser_blocks = list(DocxParser().parse(...))
elif suffix == "pptx":
    parser_blocks = list(PptxParser().parse(...))
# ... и т.д. для каждого формата

for block in parser_blocks:
    normalized = normalize_text(block.text)  # Доп. нормализация
```

**Стало (15 строк):**
```python
parser = DoclingParser(
    extract_tables=True,
    extract_images=True,
    ocr_enabled=True,
    preserve_structure=True,
)
parser_blocks = list(parser.parse(str(tmp_path), doc_id=doc_id))

for block in parser_blocks:
    # Текст уже очищен Docling
    if not block.text or not block.text.strip():
        continue
```

### 2. **services/ingestion/pipline.py**

**Изменения:**
- Добавлен импорт `DoclingParser`
- Удалён импорт `normalise_text` (не нужен)
- Изменён `__init__` для поддержки автоматического использования Docling
- Добавлен параметр `use_docling=True` (по умолчанию)
- Убрана нормализация текста в `ingest_path`
- Парсеры теперь опциональны (если не указаны, используется Docling)

**Было:**
```python
def __init__(self, *, parsers: Mapping[str, BaseParser], ...):
    if not parsers:
        raise ValueError("At least one parser must be provided")
    self.parsers = {...}

for raw_block in parser.parse(str(path)):
    normalised = normalise_text(raw_block.text)  # Доп. нормализация
```

**Стало:**
```python
def __init__(self, *, parsers: Optional[Mapping[str, BaseParser]] = None, 
             use_docling: bool = True, ...):
    if parsers:
        self.parsers = {...}
    elif use_docling:
        docling_parser = DoclingParser(...)
        self.parsers = {ext: docling_parser for ext in supported_exts}

for raw_block in parser.parse(str(path)):
    # Docling уже вернул чистый текст
    if not raw_block.text or not raw_block.text.strip():
        continue
```

### 3. **services/ingestion/requirements.txt**

**Изменения:**
- Добавлена зависимость `docling>=2.0.0`
- Добавлены комментарии о legacy парсерах
- Старые парсеры помечены как "kept for backward compatibility"

**Добавлено:**
```txt
# Docling - unified document processing library
docling>=2.0.0

# Legacy parsers (kept for backward compatibility, but Docling is preferred)
```

### 4. **services/ingestion/processors/chunker.py**

**Изменения:**
- Обновлена документация функции `chunk_text`
- Добавлено упоминание совместимости с Docling

**Изменено:**
```python
"""
Chunk text with token-aware sentence boundaries and metadata.

Compatible with Docling-parsed text (already cleaned and normalized).

Args:
    normalised_text: cleaned text (from Docling or other parsers)
    doc_metadata: document metadata (doc_id, source_uri, label, headings, etc.)
    ...
"""
```

---

## 🎯 Преимущества изменений

### 1. **Унификация**
- Один парсер вместо 6+ разных
- Единый API для всех форматов
- Меньше кода, меньше поддержки

### 2. **Качество**
- Профессиональный OCR от IBM Research
- Лучшее извлечение таблиц
- Сохранение структуры документа
- Автоматическая очистка текста

### 3. **Метаданные**
- Иерархия заголовков
- Типы контента (title, paragraph, table, list)
- Координаты на странице (bbox)
- Номера страниц

### 4. **Упрощение кода**
- Убрана избыточная нормализация
- Меньше условных операторов
- Чище архитектура

---

## 📊 Сравнение: До vs После

| Аспект | До (старые парсеры) | После (Docling) |
|--------|---------------------|-----------------|
| **Количество парсеров** | 6+ (PDF, DOCX, PPTX, XLSX, Ebook, SCORM) | 1 (DoclingParser) |
| **Строк кода в api.py** | ~40 для выбора парсера | ~15 |
| **Нормализация текста** | Требуется вручную | Автоматическая |
| **OCR** | Внешний сервис | Встроенный |
| **Таблицы** | Pdfplumber (только PDF) | Все форматы |
| **Структура** | Частично | Полная |
| **Метаданные** | Базовые | Богатые |

---

## 🚀 Как использовать

### Минимальный пример

```python
from services.ingestion.parsers.docling_parser import DoclingParser
from services.ingestion.processors.chunker import chunk_text

# Парсинг
parser = DoclingParser()
blocks = list(parser.parse("document.pdf"))

# Чанкинг
for block in blocks:
    chunks = chunk_text(block.text, doc_metadata=block.meta)
    for chunk in chunks:
        print(chunk.text[:100])
```

### Через API

```bash
# Установка
pip install -r services/ingestion/requirements.txt

# Запуск
cd services/ingestion
uvicorn api:app --host 0.0.0.0 --port 8001

# Загрузка документа
curl -F "file=@document.pdf" http://localhost:8001/ingest
```

### Тестирование

```bash
# Быстрая проверка
python tests/test_docling_minimal.py

# Полное тестирование
pytest tests/test_docling_integration.py -v
```

---

## 📚 Документация

1. **README_DOCLING.md** - Краткое руководство и примеры
2. **DOCLING_MIGRATION.md** - Подробное руководство по миграции
3. **test_docling_minimal.py** - Проверка установки
4. **test_docling_integration.py** - Интеграционные тесты

---

## ⚠️ Обратная совместимость

Старые парсеры **НЕ удалены** из requirements.txt, но помечены как "legacy". Они будут работать, если вы явно их импортируете:

```python
from parsers.pdf_parser import PdfParser  # Все еще работает

parser = PdfParser()
blocks = list(parser.parse("document.pdf"))
```

Однако рекомендуется **переходить на Docling**.

---

## 🔄 Миграция существующего кода

Если у вас есть код, использующий старые парсеры:

**Шаг 1:** Замените импорты
```python
# Было
from parsers.pdf_parser import PdfParser
from processors.normalizer import normalize_text

# Стало
from parsers.docling_parser import DoclingParser
```

**Шаг 2:** Замените парсер
```python
# Было
parser = PdfParser()

# Стало
parser = DoclingParser()
```

**Шаг 3:** Уберите нормализацию
```python
# Было
for block in blocks:
    text = normalize_text(block.text)

# Стало
for block in blocks:
    text = block.text  # Уже нормализован!
```

---

## 🐛 Известные проблемы

### 1. Первый запуск медленный
**Причина:** Docling загружает модели  
**Решение:** Нормально, последующие запуски быстрые

### 2. Большой размер зависимостей
**Причина:** Docling включает ML-модели  
**Решение:** ~2GB при установке, это ожидаемо

### 3. Требуется интернет при первом запуске
**Причина:** Загрузка моделей  
**Решение:** После загрузки работает офлайн

---

## ✅ Checklist внедрения

- [x] Создан DoclingParser
- [x] Обновлен api.py
- [x] Обновлен pipline.py
- [x] Обновлен requirements.txt
- [x] Обновлен chunker (документация)
- [x] Созданы тесты
- [x] Написана документация
- [ ] Запущены тесты на реальных данных (требует установки)
- [ ] Обновлён Dockerfile (если нужно)
- [ ] Обновлён docker-compose.yml (если нужно)

---

## 📞 Дополнительная помощь

- Смотрите **README_DOCLING.md** для быстрого старта
- Смотрите **DOCLING_MIGRATION.md** для подробной миграции
- Запустите **test_docling_minimal.py** для проверки установки
- Официальная документация: https://github.com/DS4SD/docling

---

**Готово!** 🎉 Теперь ваш ingestion сервис использует современный unified parser от IBM Research.

