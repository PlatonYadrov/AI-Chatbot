#!/usr/bin/env python3
"""Скрипт для очистки Qdrant от пустых документов."""

import os
import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

def clean_empty_documents():
    """Удаляет документы с пустым page_content из Qdrant."""
    print("🧹 Очистка Qdrant от пустых документов...")
    
    # Подключение к Qdrant
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION", "rag_chunks")
    
    try:
        client = QdrantClient(url=qdrant_url)
        
        # Проверяем коллекцию
        collections = client.get_collections()
        if collection_name not in [c.name for c in collections.collections]:
            print(f"❌ Коллекция '{collection_name}' не найдена")
            return False
        
        # Получаем информацию о коллекции
        collection_info = client.get_collection(collection_name)
        total_points = collection_info.points_count
        print(f"📊 Всего точек в коллекции: {total_points}")
        
        if total_points == 0:
            print("✅ Коллекция пуста, очистка не нужна")
            return True
        
        # Ищем точки с пустым текстом
        print("🔍 Поиск пустых документов...")
        
        # Получаем все точки порциями
        batch_size = 100
        empty_points = []
        
        for offset in range(0, total_points, batch_size):
            # Получаем точки
            points = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )[0]
            
            for point in points:
                payload = point.payload
                text = payload.get('text', '') if payload else ''
                
                # Проверяем, пустой ли текст
                if not text or not text.strip():
                    empty_points.append(point.id)
                    print(f"   Найден пустой документ: {point.id}")
        
        print(f"📋 Найдено пустых документов: {len(empty_points)}")
        
        if empty_points:
            # Удаляем пустые точки
            print("🗑️ Удаление пустых документов...")
            
            # Удаляем порциями по 50
            delete_batch_size = 50
            for i in range(0, len(empty_points), delete_batch_size):
                batch = empty_points[i:i + delete_batch_size]
                client.delete(
                    collection_name=collection_name,
                    points_selector=batch
                )
                print(f"   Удалено: {len(batch)} документов")
            
            print(f"✅ Удалено {len(empty_points)} пустых документов")
        else:
            print("✅ Пустых документов не найдено")
        
        # Проверяем результат
        new_collection_info = client.get_collection(collection_name)
        new_total_points = new_collection_info.points_count
        print(f"📊 Текущее количество точек: {new_total_points}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при очистке: {e}")
        return False

def test_collection():
    """Тестирует коллекцию после очистки."""
    print("\n🧪 Тестирование коллекции...")
    
    try:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        collection_name = os.getenv("QDRANT_COLLECTION", "rag_chunks")
        
        client = QdrantClient(url=qdrant_url)
        
        # Получаем несколько точек для проверки
        points = client.scroll(
            collection_name=collection_name,
            limit=5,
            with_payload=True,
            with_vectors=False
        )[0]
        
        print(f"📋 Проверка {len(points)} документов:")
        
        valid_count = 0
        for i, point in enumerate(points):
            payload = point.payload
            text = payload.get('text', '') if payload else ''
            
            if text and text.strip():
                valid_count += 1
                print(f"   {i+1}. ✅ Валидный: {text[:50]}...")
            else:
                print(f"   {i+1}. ❌ Пустой: {point.id}")
        
        print(f"📊 Валидных документов: {valid_count}/{len(points)}")
        
        if valid_count == len(points):
            print("✅ Все проверенные документы валидны")
        else:
            print("⚠️ Найдены проблемы с документами")
        
        return valid_count == len(points)
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        return False

def main():
    """Основная функция."""
    print("🚀 Очистка и тестирование Qdrant коллекции")
    print("=" * 50)
    
    # Очистка
    clean_success = clean_empty_documents()
    
    if clean_success:
        # Тестирование
        test_success = test_collection()
        
        if test_success:
            print("\n🎉 Коллекция готова к использованию!")
            print("🧪 Теперь можно тестировать: python test_one_request.py")
        else:
            print("\n⚠️ Коллекция очищена, но есть проблемы")
    else:
        print("\n❌ Не удалось очистить коллекцию")
    
    return clean_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
