import httpx
import logging
from typing import List, Optional

from ..config.settings import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL, OLLAMA_CHAT_MODEL

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, 
                 base_url: str = OLLAMA_BASE_URL,
                 embedding_model: str = OLLAMA_EMBEDDING_MODEL,
                 chat_model: str = OLLAMA_CHAT_MODEL):
        self.base_url = base_url.rstrip('/')
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        logger.info(f"Initialized OllamaClient with base_url: {self.base_url}")
        logger.info(f"Using embedding model: {self.embedding_model}")
        logger.info(f"Using chat model: {self.chat_model}")

    async def get_embedding(self, text: str) -> List[float]:
        logger.debug(f"Getting embedding for text: {text[:100]}...")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
                timeout=30.0
            )
            if response.status_code == 200:
                response_data = response.json()
                return response_data.get("embedding", [])
            else:
                error_msg = f"Error from Ollama API: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)

    async def chat(self, message: str, context: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
        prompt = ""
        if system_prompt:
            prompt += f"System: {system_prompt}\n\n"
        if context:
            prompt += f"Context:\n{context}\n\n"
        prompt += f"User: {message}\nAssistant:"
        logger.debug(f"Sending request to Ollama with prompt: {prompt}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.chat_model, "prompt": prompt, "stream": False},
                timeout=30.0
            )
            if response.status_code == 200:
                response_data = response.json()
                return response_data.get("response", "").strip()
            else:
                error_msg = f"Error from Ollama API: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)