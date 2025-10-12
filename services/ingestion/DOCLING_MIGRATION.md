# Миграция на Docling для разбиения документов

## Обзор

Мы заменили все индивидуальные парсеры (PDF, DOCX, PPTX, XLSX) на единый **Docling** парсер от IBM Research. 

### Преимущества Docling:

✅ **Унифицированный API** для всех форматов документов  
✅ **Встроенный OCR** высокого качества  
✅ **Извлечение таблиц** с сохранением структуры  
✅ **Сохранение семантики** (заголовки, параграфы, списки)  
✅ **Автоматическая очистка** текста  
✅ **Layout analysis** для сложных документов  
✅ **Поддержка изображений** (PNG, JPG, TIFF)

## Установка

```bash
# Основная установка
pip install docling

# Или из requirements.txt
pip install -r services/ingestion/requirements.txt
```

## Что изменилось

### До (старый подход)

```python
from parsers.pdf_parser import PdfParser
from parsers.docx_parser import DocxParser
from parsers.pptx_parser import PptxParser
from processors.normalizer import normalize_text

# Разные парсеры для разных форматов
if suffix == "pdf":
    parser = PdfParser()
elif suffix == "docx":
    parser = DocxParser()
# ... и т.д.

blocks = list(parser.parse(path))

# Требуется дополнительная нормализация
for block in blocks:
    text = normalize_text(block.text)
```

### После (с Docling)

```python
from parsers.docling_parser import DoclingParser

# Один парсер для всех форматов
parser = DoclingParser(
    extract_tables=True,
    extract_images=True,
    ocr_enabled=True,
    preserve_structure=True,
)

blocks = list(parser.parse(path))

# Текст уже очищен и нормализован
for block in blocks:
    text = block.text  # Готов к использованию!
```

## Использование

### Базовый пример

```python
from services.ingestion.parsers.docling_parser import DoclingParser
from services.ingestion.processors.chunker import chunk_text

# Инициализация парсера
parser = DoclingParser(
    extract_tables=True,
    extract_images=True,
    ocr_enabled=True,
    preserve_structure=True,
)

# Парсинг документа
blocks = list(parser.parse("document.pdf", doc_id="my_doc"))

# Разбиение на чанки
for block in blocks:
    chunks = chunk_text(
        block.text,
        doc_metadata=block.meta,
        chunk_size_tokens=300,
        overlap_tokens=75,
    )
    
    for chunk in chunks:
        print(f"Chunk: {chunk.text[:100]}...")
        print(f"Metadata: {chunk.metadata}")
```

### Полный пайплайн с эмбеддингами

```python
from services.ingestion.pipline import LocalIngestionPipeline

# Создание пайплайна (автоматически использует Docling)
pipeline = LocalIngestionPipeline(
    embedder=my_embedder,
    indexer=my_indexer,
    use_docling=True,  # По умолчанию True
)

# Обработка файла
records = pipeline.ingest_path("document.pdf")

# Или целой директории
records = pipeline.ingest_directory("./documents/")
```

### API использование

API автоматически использует Docling для всех поддерживаемых форматов:

```bash
# Загрузка PDF
curl -F "file=@document.pdf" http://localhost:8001/ingest

# Загрузка DOCX
curl -F "file=@document.docx" http://localhost:8001/ingest

# Загрузка изображения (с OCR)
curl -F "file=@scan.png" http://localhost:8001/ingest
```

## Поддерживаемые форматы

Docling поддерживает:

- **Документы**: PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS
- **Изображения**: PNG, JPG, JPEG, TIFF, TIF, BMP, GIF
- **Разметка**: HTML, HTM, Markdown (MD), TXT

## Метаданные

Docling обогащает каждый блок дополнительными метаданными:

```python
{
    "doc_id": "unique_doc_id",
    "type": "docling_paragraph",  # или docling_title, docling_table, etc.
    "label": "paragraph",  # тип контента
    "level": 1,  # уровень вложенности
    "page": 5,  # номер страницы
    "headings": ["Chapter 1", "Section 1.1"],  # иерархия заголовков
    "bbox": (x0, y0, x1, y1),  # координаты на странице
    "path": "/path/to/document.pdf",
}
```

## Тестирование

### Быстрая проверка установки

```bash
cd tests
python test_docling_minimal.py
```

### Полное тестирование интеграции

```bash
pytest tests/test_docling_integration.py -v
```

### Ручной тест

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from services.ingestion.parsers.docling_parser import DoclingParser
from services.ingestion.processors.chunker import chunk_text

# Парсинг
parser = DoclingParser()
blocks = list(parser.parse("tests/pdf/Knoleges.pdf"))

print(f"Блоков: {len(blocks)}")

# Чанкинг
text = "\n\n".join(b.text for b in blocks)
chunks = chunk_text(text, doc_metadata={"doc_id": "test"})

print(f"Чанков: {len(chunks)}")
print(f"Первый чанк: {chunks[0].text[:200]}...")
```

## Производительность

Docling оптимизирован для работы с большими документами:

- **PDF (10 страниц)**: ~5-10 секунд
- **DOCX (50 страниц)**: ~3-7 секунд  
- **Изображение с OCR**: ~2-5 секунд

Для ускорения можно:
- Отключить `ocr_enabled=False` для документов с текстовым слоем
- Отключить `extract_tables=False` если таблицы не нужны
- Использовать батч-обработку для множества документов

## Устранение проблем

### Ошибка: "ModuleNotFoundError: No module named 'docling'"

```bash
pip install docling
```

### Ошибка: "Model download failed"

Docling при первом запуске загружает модели. Убедитесь в наличии интернета.

### Медленная обработка

1. Отключите OCR для документов с текстом: `ocr_enabled=False`
2. Уменьшите качество: настройте параметры через переменные окружения
3. Используйте GPU если доступен

### Проблемы с кодировкой

Docling автоматически определяет кодировку. Если возникают проблемы, сообщите тип документа в issue.

## Обратная совместимость

Старые парсеры сохранены в `requirements.txt` с пометкой "legacy". Они будут удалены в будущих версиях.

Если необходимо использовать старый парсер:

```python
from parsers.pdf_parser import PdfParser  # Устаревший

parser = PdfParser()
blocks = list(parser.parse("document.pdf"))
```

## Дополнительная информация

- [Docling на GitHub](https://github.com/DS4SD/docling)
- [Документация Docling](https://ds4sd.github.io/docling/)
- [Примеры использования](https://github.com/DS4SD/docling/tree/main/examples)

## Поддержка

При возникновении проблем:
1. Проверьте версию: `pip show docling`
2. Запустите тесты: `python tests/test_docling_minimal.py`
3. Проверьте логи API: `docker logs ingestion-service`

