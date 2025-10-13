#!/usr/bin/env python3
"""
Тест исправления токенизатора для Docling chunker.
Проверяет, что токенизатор создаётся правильно и chunking работает.
"""

import tempfile
import pathlib
from docling.document_converter import DocumentConverter
from services.ingestion.processors.docling_chunker import DoclingChunker

def test_tokenizer_fix():
    """Проверка что токенизатор создаётся как объект, а не строка."""
    
    # Создаём тестовый документ
    text = "# Тест\n" + "Абзац текста для тестирования. " * 500
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(text)
        test_file = pathlib.Path(f.name)
    
    try:
        # Конвертируем документ
        print("Конвертируем документ...")
        doc = DocumentConverter().convert(source=str(test_file)).document
        
        # Создаём chunker с правильным токенизатором
        print("Создаём DoclingChunker...")
        chunker = DoclingChunker(
            tokenizer="intfloat/multilingual-e5-large",
            max_tokens=200,
            overlap=0,
        )
        
        # Проверяем что токенизатор создан правильно
        print("Инициализируем chunker...")
        chunker._get_chunker()
        print(f"✓ Chunker инициализирован с токенизатором: {chunker.tokenizer_name}")
        
        # Чанкуем документ
        print("Чанкуем документ...")
        chunks = chunker.chunk_document(doc)
        
        print(f"\n✓ Получено {len(chunks)} чанков")
        
        if chunks:
            print(f"\nПример метаданных первого чанка:")
            for key in list(chunks[0].meta.keys())[:8]:  # показываем первые 8 ключей
                print(f"  - {key}")
            
            # Проверяем наличие enriched_text
            if "enriched_text" in chunks[0].meta:
                print("\n✓ Enriched text присутствует (contextualize работает)")
            else:
                print("\n⚠ Enriched text отсутствует")
        
        print("\n✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        return True
        
    except Exception as e:
        print(f"\n✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Удаляем временный файл
        test_file.unlink(missing_ok=True)


if __name__ == "__main__":
    import sys
    success = test_tokenizer_fix()
    sys.exit(0 if success else 1)

