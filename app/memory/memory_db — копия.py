import os
import json
import faiss
import numpy as np
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..chat.ollama_client import OllamaClient
from ..config.settings import settings

logger = logging.getLogger(__name__)

class MemoryDB:
    def __init__(self, db_name: str = "chat_memory", dimension: int = None, model: str = "mxbai-embed-large:latest"):
        self.db_name = db_name
        self.memory_dir = settings.MEMORY_PATH
        self.model = model
        self.ollama_client = OllamaClient(embedding_model=self.model)
        self.dimension = dimension
        self.index = None
        self.memories: Dict[str, Dict] = {}
        logger.info(f"Initializing MemoryDB with model: {self.model}")
        os.makedirs(self.memory_dir, exist_ok=True)

    @classmethod
    async def create(cls, db_name: str = "chat_memory", dimension: int = None, model: str = "mxbai-embed-large:latest") -> 'MemoryDB':
        instance = cls(db_name, dimension, model)
        await instance.initialize()
        return instance

    async def initialize(self):
        self.load_memories()
        logger.info(f"Loaded {len(self.memories)} memories from disk")
        if self.dimension is None:
            if self.memories:
                first_memory = next(iter(self.memories.values()))
                if 'vector' in first_memory:
                    self.dimension = len(first_memory['vector'])
                    logger.info(f"Using dimension {self.dimension} from stored memory")
                else:
                    await self._initialize_dimension()
            else:
                await self._initialize_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        logger.info(f"Initialized FAISS index with dimension {self.dimension}")
        if self.memories:
            vectors = []
            for key, memory in self.memories.items():
                if 'vector' in memory:
                    vector = np.array(memory['vector']).astype('float32')
                    vector = vector / np.linalg.norm(vector)
                    vectors.append(vector)
                    logger.debug(f"Added normalized vector for memory {key}")
            if vectors:
                vectors_np = np.array(vectors).astype('float32')
                self.index.add(vectors_np)
                logger.info(f"Added {len(vectors)} vectors to FAISS index")

    async def _initialize_dimension(self):
        try:
            test_embedding = await self.ollama_client.get_embedding("test")
            self.dimension = len(test_embedding)
            logger.info(f"Initialized dimension to {self.dimension} from model")
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {str(e)}")
            raise

    def load_memories(self):
        try:
            memory_file = os.path.join(self.memory_dir, f"{self.db_name}.json")
            if os.path.exists(memory_file):
                with open(memory_file, 'r') as f:
                    self.memories = json.load(f)
                logger.debug(f"Loaded {len(self.memories)} memories from {memory_file}")
            else:
                logger.debug(f"No existing memory file found at {memory_file}")
                self.memories = {}
        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}")
            self.memories = {}

    def save_memories(self):
        try:
            memory_file = os.path.join(self.memory_dir, f"{self.db_name}.json")
            logger.info(f"Saving memories to: {memory_file}")
            with open(memory_file, 'w') as f:
                json.dump(self.memories, f)
            logger.info(f"Successfully saved {len(self.memories)} memories")
        except Exception as e:
            logger.error(f"Error saving memories: {str(e)}")
            raise

    async def add_memory(self, text: str, metadata: Optional[Dict] = None) -> str:
        try:
            vector = await self.ollama_client.get_embedding(text)
            vector = np.array(vector).astype('float32')
            vector = vector / np.linalg.norm(vector)
            vector_np = np.array([vector]).astype('float32')
            key = str(uuid.uuid4())
            memory_entry = {
                'text': text,
                'vector': vector.tolist(),
                'metadata': metadata or {},
                'created_at': datetime.utcnow().isoformat()
            }
            self.index.add(vector_np)
            self.memories[key] = memory_entry
            self.save_memories()
            return key
        except Exception as e:
            logger.error(f"Error adding memory: {str(e)}")
            raise

    async def query(self, query: str, k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        try:
            query_vector = await self.ollama_client.get_embedding(query)
            query_vector = np.array(query_vector).astype('float32')
            query_vector = query_vector / np.linalg.norm(query_vector)
            query_vector_np = np.array([query_vector]).astype('float32')
            logger.info(f"Searching index with {self.index.ntotal} vectors")
            if self.index.ntotal == 0:
                logger.warning("No vectors in index!")
                return []
            S, I = self.index.search(query_vector_np, min(k, self.index.ntotal))
            logger.info(f"Search returned similarities: {S[0]}")
            results = []
            for i, (similarity, idx) in enumerate(zip(S[0], I[0])):
                if similarity >= threshold and idx < len(self.memories):
                    memory_key = list(self.memories.keys())[idx]
                    memory = self.memories[memory_key]
                    results.append({
                        'key': memory_key,
                        'text': memory['text'],
                        'metadata': memory['metadata'],
                        'similarity': float(similarity),
                        'distance': 1 - similarity
                    })
            results = sorted(results, key=lambda x: x['similarity'], reverse=True)
            return results
        except Exception as e:
            logger.error(f"Error querying memories: {str(e)}")
            raise