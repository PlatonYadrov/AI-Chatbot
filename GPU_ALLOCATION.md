# GPU Allocation для 2x NVIDIA L4 (24GB each)

## Конфигурация оборудования

- **GPU 0:** NVIDIA L4 24GB
- **GPU 1:** NVIDIA L4 24GB
- **Total VRAM:** 48GB

---

## Распределение VRAM

### Текущая конфигурация (оптимизированная для vLLM)

```
┌─────────────────────────────────────────────────────────────┐
│ GPU 0 + GPU 1 (Tensor Parallel)                            │
│                                                             │
│ vLLM: Qwen3-30B-A3B-AWQ                                    │
│ ├─ Model weights: ~18-20GB                                 │
│ ├─ KV cache: ~8-10GB                                       │
│ ├─ Activations: ~6-8GB                                     │
│ └─ Total: ~36-38GB (на обе GPU)                           │
│                                                             │
│ Utilization: 92% * 48GB = 44.16GB                         │
│ Used by vLLM: ~36-38GB                                     │
│ Free: ~6-8GB                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ GPU 1 only                                                  │
│                                                             │
│ Embeddings: e5-base                                        │
│ └─ VRAM: ~1.5-2GB                                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ CPU (No GPU)                                               │
│                                                             │
│ Ingestion (Docling)                                        │
│ ├─ Layout Detection (DocLayNet)                           │
│ ├─ TableFormer                                             │
│ ├─ Tesseract OCR                                           │
│ ├─ Formula Detection                                       │
│ └─ Code Detection                                          │
│                                                             │
│ VLM: DISABLED (экономим VRAM для vLLM)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Расчет для Qwen3-30B-A3B-AWQ

### AWQ квантизация (4-bit):

```
Model size: 30B parameters
Bits per parameter: 4 (AWQ)
Memory = (30B * 4 bits) / 8 bits/byte = 15GB

+ KV cache (max_model_len=8000): ~8-10GB
+ Activations + overhead: ~6-8GB
+ Safety margin: ~5GB

Total: ~36-38GB для tensor_parallel_size=2
```

### С 2x L4 (48GB total):

```
Available: 48GB * 0.92 = 44.16GB ✅
Required: ~36-38GB
Spare: ~6-8GB (для embeddings + буфер)
```

---

## Почему VLM отключен?

### Если включить VLM:

```
GPU 0: vLLM (50%) = ~18-19GB
GPU 1: vLLM (50%) + Embeddings (2GB) + VLM (4GB) = ~24-25GB ❌ ПЕРЕПОЛНЕНИЕ!
```

**L4 имеет только 24GB!** VLM не поместится.

### Альтернативы:

1. **VLM на CPU** (медленно, 5-10 сек на изображение)
2. **Использовать меньшую LLM** (например, Qwen2-7B)
3. **Добавить 3-ю GPU** для VLM

---

## Оптимизация vLLM для L4

### Параметры в docker-compose.yml:

```yaml
--gpu-memory-utilization 0.92  # Максимум для L4
--max-model-len 8000           # Ограничение контекста
--max-num-seqs 64              # Батч-размер
--max-num-batched-tokens 8192  # Оптимизация throughput
--tensor-parallel-size 2       # Используем обе GPU
```

### Почему 92% а не 95%?

L4 требует больше запаса для CUDA kernels и temporary buffers.

---

## Производительность

### С текущей конфигурацией:

| Сервис | GPU | VRAM | Throughput |
|--------|-----|------|-----------|
| vLLM (Qwen3-30B) | 0+1 | ~38GB | 40-60 tokens/sec |
| Embeddings (e5-base) | 1 | ~2GB | 100-200 docs/sec |
| Ingestion (CPU) | - | - | 2-5 pages/sec |

**OCR, TableFormer, Layout работают на CPU** - не требуют GPU!

---

## Мониторинг VRAM

### Проверка использования:

```bash
# Реальное время
watch -n 1 nvidia-smi

# Логи vLLM
docker logs -f ai-chatbot_llm_1 | grep memory

# Проверка перед запуском
nvidia-smi --query-gpu=memory.total,memory.free --format=csv
```

### Ожидаемый output:

```
+-----------------------------------------------------------------------------+
| GPU 0 | vLLM | ~19-20GB / 22GB (92% util) |
| GPU 1 | vLLM + Emb | ~21-22GB / 22GB (92% util) |
+-----------------------------------------------------------------------------+
```

---

## Если не хватает VRAM

### Вариант 1: Уменьшить контекст

```yaml
--max-model-len 6000  # Вместо 8000
```

Освободит: ~2-3GB

### Вариант 2: Уменьшить батч

```yaml
--max-num-seqs 32  # Вместо 64
```

Освободит: ~1-2GB (но снизит throughput)

### Вариант 3: Использовать меньшую модель

```yaml
--model Qwen/Qwen2.5-14B-Instruct-AWQ  # ~16GB вместо ~38GB
```

Освободит место для VLM!

---

## Как включить VLM (если появится больше VRAM)

### 1. В `docker-compose.yml`:

```yaml
ingestion:
  environment:
    - CUDA_VISIBLE_DEVICES=1
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['1']
            capabilities: [gpu]
```

### 2. В `configs/dev.env`:

```bash
DOCLING_USE_VLM=true
```

### 3. Пересобрать:

```bash
make build
make download-models-with-vlm
make up
```

---

## Рекомендации для Production

### Текущая конфигурация (без VLM):

✅ **Стабильно**  
✅ **Оптимально для 2x L4**  
✅ **Все модели Docling на CPU работают отлично**  
✅ **vLLM использует максимум GPU**  

### Если нужен VLM:

Рассмотрите:
1. **Upgrade до 2x L40 (48GB)** или **2x A100 (40/80GB)**
2. **Добавить 3-ю GPU** только для VLM
3. **Использовать меньшую LLM** (14B вместо 30B)

---

**Итог:** Текущая конфигурация оптимальна для ваших 2x L4! 🚀

