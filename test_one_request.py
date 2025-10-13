#!/usr/bin/env python3
"""Один запрос для проверки всей RAG системы."""

import requests
import json
import time

def one_request_test():
    """Проверка всей системы одним запросом."""
    print("🚀 Проверка RAG системы одним запросом")
    print("=" * 50)
    
    # Тестовый запрос
    query = "Что такое машинное обучение?"
    
    print(f"📝 Запрос: '{query}'")
    print("⏳ Выполняю запрос с реранкингом...")
    
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
            
            # Извлекаем данные
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            processing_time = (end_time - start_time) * 1000
            
            print(f"✅ Запрос выполнен успешно!")
            print(f"⏱️ Время ответа: {processing_time:.0f}ms")
            print(f"📚 Найдено источников: {len(sources)}")
            
            # Ответ системы
            if answer:
                print(f"\n🤖 Ответ системы:")
                print(f"   {answer[:200]}{'...' if len(answer) > 200 else ''}")
            
            # Источники
            if sources:
                print(f"\n📖 Источники:")
                for i, source in enumerate(sources):
                    text = source.get("text", "")[:80] + "..." if source.get("text") else "N/A"
                    rerank_score = source.get("rerank_score", "N/A")
                    rerank_method = source.get("rerank_method", "N/A")
                    
                    print(f"   {i+1}. {text}")
                    print(f"      Реранкинг: {rerank_method}, Score: {rerank_score}")
            
            # Проверка компонентов
            print(f"\n🔍 Проверка компонентов:")
            
            # Проверка реранкинга
            if any(source.get("rerank_method") for source in sources):
                print("   ✅ Реранкинг работает")
            else:
                print("   ⚠️ Реранкинг не применен")
            
            # Проверка качества ответа
            if answer and len(answer) > 50:
                print("   ✅ LLM генерирует ответы")
            else:
                print("   ⚠️ LLM не генерирует ответы")
            
            # Проверка источников
            if sources and len(sources) > 0:
                print("   ✅ Поиск в Qdrant работает")
            else:
                print("   ⚠️ Поиск в Qdrant не работает")
            
            # Проверка времени
            if processing_time < 1000:
                print("   ✅ Производительность хорошая")
            elif processing_time < 2000:
                print("   ⚠️ Производительность приемлемая")
            else:
                print("   ❌ Производительность низкая")
            
            print(f"\n🎉 Система работает!")
            return True
            
        else:
            print(f"❌ Ошибка запроса: HTTP {response.status_code}")
            print(f"   Ответ: {response.text[:200]}...")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Не удается подключиться к сервису")
        print("   Проверьте: docker-compose up -d")
        return False
    except requests.exceptions.Timeout:
        print("❌ Превышено время ожидания")
        print("   Сервис может быть перегружен")
        return False
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        return False

def check_services():
    """Быстрая проверка сервисов."""
    print("\n🔍 Проверка сервисов...")
    
    services = {
        "Online-RAG": "http://localhost:7000/docs",
        "Qdrant": "http://localhost:6333/collections",
        "Embeddings": "http://localhost:8080/health"
    }
    
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=3)
            if response.status_code in [200, 404]:
                print(f"   ✅ {name}: OK")
            else:
                print(f"   ❌ {name}: HTTP {response.status_code}")
        except Exception:
            print(f"   ❌ {name}: недоступен")

if __name__ == "__main__":
    # Проверяем сервисы
    check_services()
    
    # Выполняем основной тест
    success = one_request_test()
    
    if not success:
        print(f"\n❌ Тест провален. Проверьте:")
        print(f"   1. Запущены ли сервисы: docker-compose up -d")
        print(f"   2. Есть ли данные в Qdrant")
        print(f"   3. Работает ли Online-RAG: http://localhost:7000/docs")
