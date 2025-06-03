import httpx
import numpy as np
from typing import List
from ..config.settings import settings

class OllamaEmbedder:
    def __init__(self, model_name: str = settings.EMBED_MODEL_NAME):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = model_name

    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings for given text using Ollama's embedding model."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            
            if response.status_code == 200:
                embedding = np.array(response.json()["embedding"])
                # Normalize the embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding
            else:
                raise Exception(f"Embedding generation failed: {response.text}")

    async def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        return [await self.generate_embedding(text) for text in texts]