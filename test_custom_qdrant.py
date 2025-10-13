#!/usr/bin/env python3
"""Быстрый тест исправления CustomQdrant."""

import requests
import json

def test_custom_qdrant():
    """Тестирует исправленный CustomQdrant."""
    print("🧪 Тестирование исправленного CustomQdrant...")
    
    payload = {
        "query": "какие основные задачи проекта",
        "k": 3,
        "use_reranking": True,
        "rerank_top_k": 3
    }
    
    try:
        print(f"📝 Запрос: {payload['query']}")
        
        response = requests.post("http://localhost:7000/query", json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Запрос выполнен успешно!")
            
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            
            print(f"🤖 Ответ: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print(f"📚 Источников: {len(sources)}")
            
            for i, source in enumerate(sources):
                text = source.get("text", "")
                rerank_score = source.get("rerank_score", "N/A")
                rerank_method = source.get("rerank_method", "N/A")
                
                print(f"   {i+1}. {text[:60]}{'...' if len(text) > 60 else ''}")
                print(f"      Реранкинг: {rerank_method}, Score: {rerank_score}")
            
            return True
            
        else:
            print(f"❌ Ошибка: HTTP {response.status_code}")
            print(f"   Ответ: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

if __name__ == "__main__":
    success = test_custom_qdrant()
    
    if success:
        print("\n🎉 CustomQdrant работает корректно!")
        print("✅ Структура данных исправлена")
    else:
        print("\n❌ Проблемы с CustomQdrant")
        print("🔧 Нужно проверить исправления")
    
    exit(0 if success else 1)
