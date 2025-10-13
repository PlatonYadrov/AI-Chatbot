"""LLM client for LangChain integration."""

import os
import logging
from typing import List, Optional, Dict, Any
import requests

logger = logging.getLogger(__name__)


class VLLMClient:
    """LangChain-compatible client for vLLM service."""
    
    def __init__(self, base_url: str = None):
        """Initialize vLLM client.
        
        Args:
            base_url: Base URL for vLLM service
        """
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://llm:8000/v1")
        self.api_key = os.getenv("LLM_API_KEY", "dummy")
        self.model_name = os.getenv("LLM_MODEL_NAME", "cognitivecomputations/Qwen3-30B-A3B-AWQ")
        
        logger.info(f"Initialized VLLMClient with base_url={self.base_url}")
    
    def _post(self, path: str, payload: dict) -> dict:
        """Make POST request to vLLM service."""
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"vLLM request failed: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.1) -> str:
        """Generate text using vLLM service."""
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
        }
        
        try:
            data = self._post("/chat/completions", payload)
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            return content.strip()
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return ""
    
    def is_relevant(self, query: str, document: str) -> bool:
        """Check if document is relevant to query using LLM."""
        prompt = f"""You are a helpful assistant that determines if a document is relevant to a query.

Query: {query}

Document: {document}

Is this document relevant to the query? Answer only "YES" or "NO"."""
        
        response = self.generate(prompt, max_tokens=10, temperature=0.0)
        return response.upper().strip() == "YES"


# Global instance
_vllm_client: Optional[VLLMClient] = None


def get_llm_client() -> VLLMClient:
    """Get global vLLM client instance."""
    global _vllm_client
    if _vllm_client is None:
        _vllm_client = VLLMClient()
    return _vllm_client


def get_reranking_info() -> Dict[str, Any]:
    """Get information about reranking capabilities."""
    try:
        client = get_llm_client()
        return {
            "llm_available": True,
            "llm_url": client.base_url,
            "llm_model": client.model_name
        }
    except Exception as e:
        logger.warning(f"LLM client not available: {e}")
        return {
            "llm_available": False,
            "error": str(e)
        }
