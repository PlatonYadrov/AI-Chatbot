# Docling Integration - Quick Start Guide

## 🚀 Что такое Docling?

**Docling** - это мощная библиотека от IBM Research для работы с документами любых форматов. Мы интегрировали её как единый парсер для всех типов документов в нашем RAG пайплайне.

## 📦 Быстрый старт

### 1. Установка

```bash
# Из корня репозитория
cd services/ingestion
pip install -r requirements.txt
```

Это установит Docling и все необходимые зависимости.

### 2. Простой пример использования

```python
from services.ingestion.parsers.docling_parser import DoclingParser
from services.ingestion.processors.chunker import chunk_text

# Создаём парсер
parser = DoclingParser()

# Парсим любой документ (PDF, DOCX, PPTX, изображения...)
blocks = list(parser.parse("path/to/document.pdf", doc_id="my_doc"))

print(f"Извлечено блоков: {len(blocks)}")

# Разбиваем на чанки
for block in blocks:
    chunks = chunk_text(block.text, doc_metadata=block.meta)
    print(f"Блок -> {len(chunks)} чанков")
```

### 3. Запуск API с Docling

API автоматически использует Docling:

```bash
# Запуск локально без Docker
cd services/ingestion
uvicorn api:app --host 0.0.0.0 --port 8001

# В другом терминале
curl -F "file=@../../tests/pdf/Knoleges.pdf" http://localhost:8001/ingest
```

## 🧪 Тестирование

### Минимальный тест (проверка установки)

```bash
cd tests
python test_docling_minimal.py
```

Вывод:
```
============================================================
Docling Installation & Integration Test
============================================================

1. Testing Docling import...
✓ Docling imported successfully (version: 2.x.x)

2. Testing DoclingParser import...
✓ DoclingParser imported and instantiated successfully

3. Testing simple document conversion...
Converting: tests/pdf/Knoleges.pdf
✓ Successfully converted document with 45 items

First 3 items:
  1. [title] Introduction to Machine Learning...
  2. [paragraph] This document provides an overview...
  3. [section_header] Chapter 1: Fundamentals...

============================================================
✓ All checks passed! Docling is ready to use.
============================================================
```

### Полное тестирование

```bash
pytest tests/test_docling_integration.py -v -s
```

## 📋 Поддерживаемые форматы

| Формат | Расширения | OCR | Таблицы | Структура |
|--------|-----------|-----|---------|-----------|
| PDF | `.pdf` | ✅ | ✅ | ✅ |
| Word | `.docx`, `.doc` | ✅ | ✅ | ✅ |
| PowerPoint | `.pptx`, `.ppt` | ✅ | ✅ | ✅ |
| Excel | `.xlsx`, `.xls` | ✅ | ✅ | ✅ |
| Изображения | `.png`, `.jpg`, `.tiff` | ✅ | ⚠️ | ⚠️ |
| HTML | `.html`, `.htm` | - | ✅ | ✅ |
| Markdown | `.md` | - | ✅ | ✅ |
| Текст | `.txt` | - | - | - |

## ⚙️ Конфигурация

```python
parser = DoclingParser(
    extract_tables=True,      # Извлекать таблицы
    extract_images=True,      # Обрабатывать изображения
    ocr_enabled=True,         # Включить OCR для сканов
    preserve_structure=True,  # Сохранять структуру (заголовки, параграфы)
)
```

## 🔍 Метаданные

Каждый блок содержит богатые метаданные:

```python
for block in blocks:
    print(block.meta)
    # {
    #   'doc_id': 'my_doc',
    #   'type': 'docling_paragraph',
    #   'label': 'paragraph',
    #   'level': 1,
    #   'page': 3,
    #   'headings': ['Chapter 1', 'Section 1.1'],
    #   'bbox': (72.0, 100.0, 500.0, 200.0),
    #   'path': '/path/to/document.pdf'
    # }
```

## 🔄 Миграция со старых парсеров

См. [DOCLING_MIGRATION.md](./DOCLING_MIGRATION.md) для подробной инструкции по миграции.

Краткая версия:

**Было:**
```python
from parsers.pdf_parser import PdfParser
from processors.normalizer import normalize_text

parser = PdfParser()
blocks = list(parser.parse(path))
for block in blocks:
    text = normalize_text(block.text)  # Нужна доп. нормализация
```

**Стало:**
```python
from parsers.docling_parser import DoclingParser

parser = DoclingParser()
blocks = list(parser.parse(path))
for block in blocks:
    text = block.text  # Уже очищен!
```

## 🎯 Примеры использования

### Пример 1: Парсинг и вывод структуры

```python
from services.ingestion.parsers.docling_parser import DoclingParser

parser = DoclingParser(preserve_structure=True)
blocks = list(parser.parse("document.pdf"))

# Группировка по типам
from collections import Counter
types = Counter(b.meta.get('label') for b in blocks)
print("Структура документа:", dict(types))
# {'title': 1, 'paragraph': 45, 'section_header': 8, 'table': 3, 'list_item': 12}
```

### Пример 2: Извлечение только таблиц

```python
parser = DoclingParser(extract_tables=True)
blocks = list(parser.parse("report.pdf"))

tables = [b for b in blocks if b.meta.get('label') == 'table']
print(f"Найдено таблиц: {len(tables)}")

for i, table in enumerate(tables):
    print(f"\nТаблица {i+1} (страница {table.meta.get('page')}):")
    print(table.text[:200])
```

### Пример 3: Обработка с контекстом заголовков

```python
parser = DoclingParser(preserve_structure=True)
blocks = list(parser.parse("book.pdf"))

for block in blocks:
    headings = block.meta.get('headings', [])
    context = " > ".join(headings) if headings else "Root"
    print(f"[{context}] {block.text[:80]}...")
```

### Пример 4: Батч-обработка директории

```python
from pathlib import Path
from services.ingestion.pipline import LocalIngestionPipeline

# Настройка пайплайна
pipeline = LocalIngestionPipeline(
    embedder=my_embedder,
    use_docling=True,
)

# Обработка всех документов
docs_dir = Path("./documents")
records = pipeline.ingest_directory(docs_dir)

print(f"Обработано документов: {len(set(r.payload['doc_id'] for r in records))}")
print(f"Создано чанков: {len(records)}")
```

## 🐛 Устранение проблем

### Проблема: Долгая загрузка при первом запуске

**Решение:** Docling загружает модели при первом использовании. Это нормально.

### Проблема: Ошибка памяти на больших PDF

**Решение:** 
```python
# Обрабатывайте документ постранично
parser = DoclingParser()
for block in parser.parse(path):  # Ленивая итерация
    process_block(block)
    # Не загружаем весь документ в память
```

### Проблема: OCR слишком медленный

**Решение:**
```python
# Отключите OCR для документов с текстовым слоем
parser = DoclingParser(ocr_enabled=False)
```

### Проблема: Неправильное распознавание таблиц

**Решение:**
```python
# Docling использует AI для таблиц, но можно вернуться к pdfplumber
from parsers.pdf_parser import PdfParser  # Legacy
parser = PdfParser(extract_tables=True)
```

## 📚 Дополнительные ресурсы

- **Документация проекта:** [docs/architecture.md](../../docs/architecture.md)
- **Полная миграция:** [DOCLING_MIGRATION.md](./DOCLING_MIGRATION.md)
- **Docling GitHub:** https://github.com/DS4SD/docling
- **Примеры Docling:** https://github.com/DS4SD/docling/tree/main/examples

## 💡 Советы и рекомендации

1. **Используйте preserve_structure=True** для академических статей и книг
2. **Отключайте OCR** для современных PDF с текстовым слоем
3. **Проверяйте метаданные** для отладки: `print(block.meta)`
4. **Используйте headings** для контекстного поиска
5. **Батч-обработка** эффективнее последовательной

## 🤝 Вклад

Если вы нашли проблему или хотите улучшить интеграцию:
1. Создайте issue с описанием
2. Приложите пример документа (если возможно)
3. Укажите версию Docling: `pip show docling`

---

**Готово!** 🎉 Теперь ваш RAG пайплайн использует современный unified parser от IBM Research.

