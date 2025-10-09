"""Simple client for local vLLM OpenAI-compatible endpoint."""

import os
from typing import Optional, List, Tuple

import requests
import re


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
                      stop: Optional[List[str]] = None) -> Tuple[str, str]:
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
    thoughts = ""
    if "<think>" in content:
        # capture inner think text and remove it from final answer
        m = re.search(r"<think>(.*?)</think>", content, flags=re.S)
        if m:
            thoughts = m.group(1).strip()
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.S).strip()
    return content, thoughts
