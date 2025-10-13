#!/usr/bin/env python3
"""Полный тест RAG системы - от ingestion до reranking."""

import os
import sys
import time
import requests
import json
from pathlib import Path
from typing import Dict, Any, List

# Добавляем пути для импорта
sys.path.append('services/online-rag')
sys.path.append('services/ingestion')

def test_system_health():
    """Проверка здоровья всех сервисов."""
    print("🔍 Проверка здоровья сервисов...")
    
    services = {
        "ingestion": "http://localhost:6000/docs",
        "online-rag": "http://localhost:7000/docs", 
        "qdrant": "http://localhost:6333/collections",
        "embeddings": "http://localhost:8080/health",
        "llm": "http://localhost:8000/health",
        "ocr": "http://localhost:9000/health"
    }
    
    healthy_services = []
    failed_services = []
    
    for service_name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code in [200, 404]:  # 404 тоже ОК для некоторых endpoints
                print(f"✅ {service_name}: OK")
                healthy_services.append(service_name)
            else:
                print(f"❌ {service_name}: HTTP {response.status_code}")
                failed_services.append(service_name)
        except Exception as e:
            print(f"❌ {service_name}: {str(e)[:50]}...")
            failed_services.append(service_name)
    
    print(f"\n📊 Результат: {len(healthy_services)}/{len(services)} сервисов работают")
    return len(failed_services) == 0


def test_ingestion_pipeline():
    """Тест ingestion пайплайна."""
    print("\n📥 Тестирование ingestion пайплайна...")
    
    # Создаем тестовый документ
    test_content = """
    Машинное обучение - это подраздел искусственного интеллекта, который фокусируется на алгоритмах.
    
    Глубокое обучение использует нейронные сети для решения сложных задач.
    
    Искусственный интеллект применяется в различных областях: медицина, транспорт, финансы.
    """
    
    test_file_path = "test_document.txt"
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    try:
        # Загружаем документ
        with open(test_file_path, "rb") as f:
            files = {"file": ("test_document.txt", f, "text/plain")}
            response = requests.post("http://localhost:6000/ingest", files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Документ загружен успешно")
            print(f"   - Чанков создано: {result.get('chunks_created', 'N/A')}")
            print(f"   - Время обработки: {result.get('processing_time_ms', 'N/A')}ms")
            return True
        else:
            print(f"❌ Ошибка загрузки: HTTP {response.status_code}")
            print(f"   Ответ: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка ingestion: {e}")
        return False
    finally:
        # Удаляем тестовый файл
        if os.path.exists(test_file_path):
            os.remove(test_file_path)


def test_qdrant_collection():
    """Проверка коллекции в Qdrant."""
    print("\n🗄️ Проверка коллекции Qdrant...")
    
    try:
        # Проверяем коллекцию
        response = requests.get("http://localhost:6333/collections/rag_chunks")
        
        if response.status_code == 200:
            collection_info = response.json()
            points_count = collection_info.get("result", {}).get("points_count", 0)
            print(f"✅ Коллекция 'rag_chunks' найдена")
            print(f"   - Количество точек: {points_count}")
            
            if points_count > 0:
                # Получаем несколько точек для проверки
                search_response = requests.post(
                    "http://localhost:6333/collections/rag_chunks/points/scroll",
                    json={"limit": 3}
                )
                
                if search_response.status_code == 200:
                    points_data = search_response.json()
                    points = points_data.get("result", {}).get("points", [])
                    
                    print(f"   - Примеры точек:")
                    for i, point in enumerate(points[:2]):
                        payload = point.get("payload", {})
                        text = payload.get("text", "")[:50] + "..." if payload.get("text") else "N/A"
                        print(f"     {i+1}. ID: {point.get('id', 'N/A')[:20]}...")
                        print(f"        Текст: {text}")
                
                return True
            else:
                print("⚠️ Коллекция пуста - нужно загрузить документы")
                return False
        else:
            print(f"❌ Коллекция не найдена: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка проверки Qdrant: {e}")
        return False


def test_search_without_reranking():
    """Тест поиска без реранкинга."""
    print("\n🔍 Тестирование поиска без реранкинга...")
    
    test_queries = [
        "Что такое машинное обучение?",
        "Как работают нейронные сети?",
        "Применение искусственного интеллекта"
    ]
    
    results = []
    
    for query in test_queries:
        try:
            payload = {
                "query": query,
                "k": 3,
                "use_reranking": False,
                "include_vectors": False
            }
            
            start_time = time.time()
            response = requests.post("http://localhost:7000/query", json=payload, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                sources = result.get("sources", [])
                
                print(f"✅ Запрос: '{query}'")
                print(f"   - Время ответа: {(end_time - start_time)*1000:.0f}ms")
                print(f"   - Найдено источников: {len(sources)}")
                
                for i, source in enumerate(sources[:2]):
                    text = source.get("text", "")[:80] + "..." if source.get("text") else "N/A"
                    print(f"     {i+1}. {text}")
                
                results.append(True)
            else:
                print(f"❌ Ошибка запроса: HTTP {response.status_code}")
                print(f"   Ответ: {response.text[:200]}...")
                results.append(False)
                
        except Exception as e:
            print(f"❌ Ошибка поиска: {e}")
            results.append(False)
    
    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 Успешность поиска без реранкинга: {success_rate:.0f}%")
    return success_rate > 50


def test_search_with_reranking():
    """Тест поиска с реранкингом."""
    print("\n🔄 Тестирование поиска с реранкингом...")
    
    test_queries = [
        "Что такое машинное обучение?",
        "Как работают нейронные сети?",
        "Применение искусственного интеллекта"
    ]
    
    results = []
    
    for query in test_queries:
        try:
            payload = {
                "query": query,
                "k": 3,
                "use_reranking": True,
                "rerank_top_k": 3,
                "include_vectors": False
            }
            
            start_time = time.time()
            response = requests.post("http://localhost:7000/query", json=payload, timeout=60)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                sources = result.get("sources", [])
                
                print(f"✅ Запрос: '{query}'")
                print(f"   - Время ответа: {(end_time - start_time)*1000:.0f}ms")
                print(f"   - Найдено источников: {len(sources)}")
                
                for i, source in enumerate(sources[:2]):
                    text = source.get("text", "")[:80] + "..." if source.get("text") else "N/A"
                    rerank_score = source.get("rerank_score", "N/A")
                    rerank_method = source.get("rerank_method", "N/A")
                    print(f"     {i+1}. {text}")
                    print(f"        Реранкинг: {rerank_method}, Score: {rerank_score}")
                
                results.append(True)
            else:
                print(f"❌ Ошибка запроса: HTTP {response.status_code}")
                print(f"   Ответ: {response.text[:200]}...")
                results.append(False)
                
        except Exception as e:
            print(f"❌ Ошибка поиска: {e}")
            results.append(False)
    
    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 Успешность поиска с реранкингом: {success_rate:.0f}%")
    return success_rate > 50


def test_reranking_comparison():
    """Сравнение результатов с реранкингом и без."""
    print("\n⚖️ Сравнение результатов с реранкингом и без...")
    
    query = "Что такое машинное обучение?"
    
    try:
        # Тест без реранкинга
        payload_no_rerank = {
            "query": query,
            "k": 3,
            "use_reranking": False
        }
        
        response_no_rerank = requests.post("http://localhost:7000/query", json=payload_no_rerank, timeout=30)
        
        # Тест с реранкингом
        payload_with_rerank = {
            "query": query,
            "k": 3,
            "use_reranking": True,
            "rerank_top_k": 3
        }
        
        response_with_rerank = requests.post("http://localhost:7000/query", json=payload_with_rerank, timeout=60)
        
        if response_no_rerank.status_code == 200 and response_with_rerank.status_code == 200:
            result_no_rerank = response_no_rerank.json()
            result_with_rerank = response_with_rerank.json()
            
            sources_no_rerank = result_no_rerank.get("sources", [])
            sources_with_rerank = result_with_rerank.get("sources", [])
            
            print(f"📝 Запрос: '{query}'")
            print(f"\n🔍 Без реранкинга:")
            for i, source in enumerate(sources_no_rerank):
                text = source.get("text", "")[:60] + "..." if source.get("text") else "N/A"
                print(f"   {i+1}. {text}")
            
            print(f"\n🔄 С реранкингом:")
            for i, source in enumerate(sources_with_rerank):
                text = source.get("text", "")[:60] + "..." if source.get("text") else "N/A"
                rerank_score = source.get("rerank_score", "N/A")
                rerank_method = source.get("rerank_method", "N/A")
                print(f"   {i+1}. {text}")
                print(f"      Score: {rerank_score}, Method: {rerank_method}")
            
            # Проверяем, изменился ли порядок
            if len(sources_no_rerank) > 0 and len(sources_with_rerank) > 0:
                first_no_rerank = sources_no_rerank[0].get("text", "")[:50]
                first_with_rerank = sources_with_rerank[0].get("text", "")[:50]
                
                if first_no_rerank != first_with_rerank:
                    print(f"\n✅ Реранкинг изменил порядок результатов!")
                else:
                    print(f"\n⚠️ Реранкинг не изменил порядок результатов")
            
            return True
        else:
            print(f"❌ Ошибка сравнения: {response_no_rerank.status_code}, {response_with_rerank.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка сравнения: {e}")
        return False


def test_performance():
    """Тест производительности."""
    print("\n⚡ Тестирование производительности...")
    
    queries = [
        "Что такое машинное обучение?",
        "Как работают нейронные сети?",
        "Применение ИИ в медицине",
        "Глубокое обучение и нейросети",
        "Алгоритмы машинного обучения"
    ]
    
    times_no_rerank = []
    times_with_rerank = []
    
    for query in queries:
        try:
            # Тест без реранкинга
            payload = {"query": query, "k": 3, "use_reranking": False}
            start = time.time()
            response = requests.post("http://localhost:7000/query", json=payload, timeout=30)
            end = time.time()
            
            if response.status_code == 200:
                times_no_rerank.append((end - start) * 1000)
            
            # Тест с реранкингом
            payload = {"query": query, "k": 3, "use_reranking": True}
            start = time.time()
            response = requests.post("http://localhost:7000/query", json=payload, timeout=60)
            end = time.time()
            
            if response.status_code == 200:
                times_with_rerank.append((end - start) * 1000)
                
        except Exception as e:
            print(f"⚠️ Ошибка теста производительности для '{query}': {e}")
    
    if times_no_rerank and times_with_rerank:
        avg_no_rerank = sum(times_no_rerank) / len(times_no_rerank)
        avg_with_rerank = sum(times_with_rerank) / len(times_with_rerank)
        
        print(f"📊 Среднее время ответа:")
        print(f"   - Без реранкинга: {avg_no_rerank:.0f}ms")
        print(f"   - С реранкингом: {avg_with_rerank:.0f}ms")
        print(f"   - Разница: +{avg_with_rerank - avg_no_rerank:.0f}ms ({((avg_with_rerank/avg_no_rerank - 1) * 100):.0f}%)")
        
        return True
    else:
        print("❌ Не удалось получить данные о производительности")
        return False


def test_error_handling():
    """Тест обработки ошибок."""
    print("\n🚨 Тестирование обработки ошибок...")
    
    error_tests = [
        {"query": "", "k": 3},  # Пустой запрос
        {"query": "test", "k": 0},  # Нулевое количество
        {"query": "test", "k": 100},  # Слишком много результатов
    ]
    
    error_count = 0
    
    for test_case in error_tests:
        try:
            response = requests.post("http://localhost:7000/query", json=test_case, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ Тест прошел: {test_case}")
            else:
                print(f"⚠️ Ожидаемая ошибка: {test_case} → HTTP {response.status_code}")
                error_count += 1
                
        except Exception as e:
            print(f"⚠️ Ожидаемая ошибка: {test_case} → {str(e)[:50]}...")
            error_count += 1
    
    print(f"📊 Обработано ошибок: {error_count}/{len(error_tests)}")
    return error_count > 0


def main():
    """Запуск всех тестов."""
    print("🚀 Запуск полного тестирования RAG системы")
    print("=" * 60)
    
    tests = [
        ("Проверка здоровья сервисов", test_system_health),
        ("Тест ingestion пайплайна", test_ingestion_pipeline),
        ("Проверка коллекции Qdrant", test_qdrant_collection),
        ("Поиск без реранкинга", test_search_without_reranking),
        ("Поиск с реранкингом", test_search_with_reranking),
        ("Сравнение реранкинга", test_reranking_comparison),
        ("Тест производительности", test_performance),
        ("Обработка ошибок", test_error_handling),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Критическая ошибка в тесте '{test_name}': {e}")
            results.append((test_name, False))
    
    # Итоговый отчет
    print(f"\n{'='*60}")
    print("📋 ИТОГОВЫЙ ОТЧЕТ")
    print(f"{'='*60}")
    
    passed = 0
    for test_name, success in results:
        status = "✅ ПРОШЕЛ" if success else "❌ ПРОВАЛЕН"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    success_rate = (passed / len(results)) * 100
    print(f"\n🎯 Общий результат: {passed}/{len(results)} тестов прошли ({success_rate:.0f}%)")
    
    if success_rate >= 80:
        print("🎉 Система работает отлично!")
    elif success_rate >= 60:
        print("⚠️ Система работает с проблемами")
    else:
        print("❌ Система требует серьезного исправления")
    
    return success_rate >= 60


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
