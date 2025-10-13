#!/usr/bin/env python3
"""Минимальная проверка RAG системы."""

import requests
import json

# Один запрос для проверки всей системы
def test():
    try:
        response = requests.post("http://localhost:7000/query", 
            json={"query": "test", "k": 1, "use_reranking": True}, 
            timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            sources = data.get("sources", [])
            answer = data.get("answer", "")
            
            if sources and answer:
                print("✅ RAG система работает!")
                print(f"   Ответ: {answer[:50]}...")
                print(f"   Источников: {len(sources)}")
                return True
            else:
                print("⚠️ Система отвечает, но нет данных")
                return False
        else:
            print(f"❌ Ошибка: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False

if __name__ == "__main__":
    test()
