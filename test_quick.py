#!/usr/bin/env python3
"""Быстрый тест RAG системы."""

import requests
import json
import time

def quick_test():
    """Быстрый тест основных функций."""
    print("🚀 Быстрый тест RAG системы")
    print("=" * 40)
    
    # 1. Проверка здоровья сервисов
    print("1️⃣ Проверка сервисов...")
    try:
        response = requests.get("http://localhost:7000/docs", timeout=5)
        if response.status_code == 200:
            print("✅ Online-RAG сервис работает")
        else:
            print("❌ Online-RAG сервис недоступен")
            return False
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False
    
    # 2. Проверка коллекции Qdrant
    print("\n2️⃣ Проверка данных в Qdrant...")
    try:
        response = requests.get("http://localhost:6333/collections/rag_chunks")
        if response.status_code == 200:
            collection_info = response.json()
            points_count = collection_info.get("result", {}).get("points_count", 0)
            print(f"✅ Коллекция найдена, точек: {points_count}")
            
            if points_count == 0:
                print("⚠️ Коллекция пуста - нужно загрузить документы")
                print("   Запустите: python test_full_rag_system.py")
                return False
        else:
            print("❌ Коллекция не найдена")
            return False
    except Exception as e:
        print(f"❌ Ошибка Qdrant: {e}")
        return False
    
    # 3. Тест поиска без реранкинга
    print("\n3️⃣ Тест поиска без реранкинга...")
    try:
        payload = {
            "query": "Что такое машинное обучение?",
            "k": 2,
            "use_reranking": False
        }
        
        start_time = time.time()
        response = requests.post("http://localhost:7000/query", json=payload, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            sources = result.get("sources", [])
            print(f"✅ Поиск работает, найдено: {len(sources)} источников")
            print(f"   Время ответа: {(end_time - start_time)*1000:.0f}ms")
            
            if sources:
                print(f"   Первый результат: {sources[0].get('text', '')[:60]}...")
        else:
            print(f"❌ Ошибка поиска: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ошибка поиска: {e}")
        return False
    
    # 4. Тест поиска с реранкингом
    print("\n4️⃣ Тест поиска с реранкингом...")
    try:
        payload = {
            "query": "Что такое машинное обучение?",
            "k": 2,
            "use_reranking": True,
            "rerank_top_k": 2
        }
        
        start_time = time.time()
        response = requests.post("http://localhost:7000/query", json=payload, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            sources = result.get("sources", [])
            print(f"✅ Реранкинг работает, найдено: {len(sources)} источников")
            print(f"   Время ответа: {(end_time - start_time)*1000:.0f}ms")
            
            if sources:
                first_source = sources[0]
                rerank_score = first_source.get("rerank_score", "N/A")
                rerank_method = first_source.get("rerank_method", "N/A")
                print(f"   Первый результат: {first_source.get('text', '')[:60]}...")
                print(f"   Реранкинг: {rerank_method}, Score: {rerank_score}")
        else:
            print(f"❌ Ошибка реранкинга: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ошибка реранкинга: {e}")
        return False
    
    print("\n🎉 Все тесты прошли успешно!")
    return True

if __name__ == "__main__":
    success = quick_test()
    if not success:
        print("\n❌ Тесты провалены. Проверьте:")
        print("   1. Запущены ли все сервисы: docker-compose up -d")
        print("   2. Загружены ли документы в систему")
        print("   3. Работает ли Qdrant: http://localhost:6333")
        print("   4. Работает ли Online-RAG: http://localhost:7000/docs")
