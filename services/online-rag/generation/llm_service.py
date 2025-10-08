"""Simple client for local vLLM OpenAI-compatible endpoint."""

import os
from typing import Optional, List

import requests


VLLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("LLM_MODEL_NAME", "cognitivecomputations/Qwen3-30B-A3B-AWQ")
VLLM_API_KEY = os.getenv("LLM_API_KEY", "dummy")  # vLLM can run without auth


def _post(path: str, payload: dict) -> dict:
    url = f"{VLLM_BASE_URL}{path}"
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}", "Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    return response.json()


def generate_response(prompt: str, max_tokens: int = 512, temperature: float = 0.2, top_p: float = 0.95,
                      stop: Optional[List[str]] = None) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful, concise assistant."},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if stop:
        payload["stop"] = stop

    data = _post("/chat/completions", payload)
    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    return content
