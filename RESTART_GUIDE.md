# 🔄 Перезапуск RAG системы с реранкингом

## 🚨 Почему нужен перезапуск?

После добавления Cross-Encoder реранкинга нужно перезапустить контейнеры, потому что:

1. **📦 Новые зависимости** - добавлены LangChain компоненты
2. **🔧 Изменения в коде** - обновлен `online-rag` сервис
3. **⚙️ Новая конфигурация** - изменены переменные окружения
4. **🔄 Обновленные импорты** - новые модули реранкинга

## 🛠️ Способы перезапуска

### 1. Автоматический перезапуск (рекомендуется)

#### Windows:
```cmd
restart_with_reranking.bat
```

#### Linux/Mac:
```bash
chmod +x restart_with_reranking.sh
./restart_with_reranking.sh
```

### 2. Ручной перезапуск

#### Шаг 1: Остановка
```bash
docker-compose down
```

#### Шаг 2: Пересборка (важно!)
```bash
# Пересборка только online-rag с новыми изменениями
docker-compose build --no-cache online-rag

# Или пересборка всех сервисов
docker-compose build --no-cache
```

#### Шаг 3: Запуск
```bash
docker-compose up -d
```

#### Шаг 4: Ожидание
```bash
# Подождать 30 секунд для полного запуска
sleep 30
```

#### Шаг 5: Проверка
```bash
docker-compose ps
```

### 3. Быстрый перезапуск (если нет изменений в коде)

```bash
# Просто перезапуск без пересборки
docker-compose restart online-rag
```

## 🔍 Что происходит при перезапуске

### 1. Остановка сервисов
- ✅ Все контейнеры останавливаются
- ✅ Сети и volumes сохраняются
- ✅ Данные в Qdrant остаются

### 2. Пересборка контейнеров
- ✅ Новый код загружается в контейнер
- ✅ Новые зависимости устанавливаются
- ✅ Обновленная конфигурация применяется

### 3. Запуск сервисов
- ✅ Контейнеры запускаются в правильном порядке
- ✅ Зависимости проверяются
- ✅ Сервисы инициализируются

## ⏱️ Время перезапуска

### Полный перезапуск:
- **Остановка**: ~10 секунд
- **Пересборка**: ~2-5 минут
- **Запуск**: ~30 секунд
- **Итого**: ~3-6 минут

### Быстрый перезапуск:
- **Перезапуск**: ~30 секунд
- **Итого**: ~30 секунд

## 🔧 Проверка после перезапуска

### 1. Статус контейнеров
```bash
docker-compose ps
```

Ожидаемый результат:
```
Name                     Command               State           Ports         
-----------------------------------------------------------------------------
ai-chatbot-ingestion-1   python api.py         Up      0.0.0.0:6000->6000/tcp
ai-chatbot-online-rag-1  python -m uvicorn ... Up      0.0.0.0:7000->7000/tcp
ai-chatbot-qdrant-1      /bin/sh -c ./qdrant   Up      0.0.0.0:6333->6333/tcp
ai-chatbot-embeddings-1  text-embeddings-...   Up      0.0.0.0:8080->80/tcp
ai-chatbot-llm-1         python -m vllm.entry Up      0.0.0.0:8000->8000/tcp
```

### 2. Логи Online-RAG
```bash
docker-compose logs --tail=20 online-rag
```

Ожидаемые сообщения:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:7000
```

### 3. Тест реранкинга
```bash
python test_one_request.py
```

Ожидаемый результат:
```
✅ RAG система работает!
   Ответ: Машинное обучение - это подраздел...
   Источников: 3
```

## 🚨 Решение проблем

### Проблема: "Container failed to start"
```bash
# Решение: проверить логи
docker-compose logs online-rag

# Частые причины:
# - Ошибки в коде
# - Недоступные зависимости
# - Конфликты портов
```

### Проблема: "Module not found"
```bash
# Решение: пересобрать контейнер
docker-compose build --no-cache online-rag
docker-compose up -d online-rag
```

### Проблема: "Service not responding"
```bash
# Решение: проверить зависимости
docker-compose logs online-rag
docker-compose logs qdrant
docker-compose logs embeddings
```

### Проблема: "Reranking not working"
```bash
# Решение: проверить импорты
docker-compose exec online-rag python -c "from reranker.cross_encoder_service import rerank_candidates; print('OK')"
```

## 📊 Мониторинг перезапуска

### Проверка ресурсов
```bash
# Использование CPU/памяти
docker stats

# Использование диска
docker system df
```

### Проверка сетей
```bash
# Сети Docker
docker network ls

# Подключения
netstat -tulpn | grep -E "(6000|7000|6333|8080|8000)"
```

## 🎯 Оптимизация перезапуска

### 1. Использование .dockerignore
```dockerfile
# В services/online-rag/.dockerignore
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
```

### 2. Многоэтапная сборка
```dockerfile
# В services/online-rag/Dockerfile
FROM python:3.9-slim

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Запуск
CMD ["python", "-m", "uvicorn", "api.gateway:app", "--host", "0.0.0.0", "--port", "7000"]
```

### 3. Кэширование слоев
```bash
# Использование кэша при пересборке
docker-compose build online-rag
```

## 🔄 Автоматизация перезапуска

### Скрипт для CI/CD
```bash
#!/bin/bash
# deploy.sh

echo "Deploying RAG system with reranking..."

# Остановка
docker-compose down

# Пересборка
docker-compose build --no-cache online-rag

# Запуск
docker-compose up -d

# Ожидание
sleep 30

# Тест
python test_one_request.py

if [ $? -eq 0 ]; then
    echo "✅ Deployment successful!"
else
    echo "❌ Deployment failed!"
    exit 1
fi
```

### Мониторинг здоровья
```bash
#!/bin/bash
# health_check.sh

while true; do
    if curl -s http://localhost:7000/docs > /dev/null; then
        echo "✅ Service is healthy"
    else
        echo "❌ Service is down, restarting..."
        docker-compose restart online-rag
    fi
    sleep 60
done
```

---

**Перезапуск необходим для применения изменений реранкинга!** 🔄✅
