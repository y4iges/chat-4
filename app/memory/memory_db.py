import os
import json
import faiss
import numpy as np
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from ..chat.ollama_client import OllamaClient
from ..config.settings import settings

logger = logging.getLogger(__name__)

class MemoryDB:
    def __init__(self, 
                 db_name: str = "chat_memory", 
                 session_name: Optional[str] = None):
        """
        If session_name is provided, the memory file is stored in the sessions directory as {session_name}_memory.json.
        The memory records will now include an embedding 'vector' along with text and metadata.
        """
        self.session_name = session_name
        if self.session_name:
            self.memory_dir = settings.SESSIONS_PATH
            self.db_filename = f"{self.session_name}_memory.json"
        else:
            self.memory_dir = settings.MEMORY_PATH
            self.db_filename = f"{db_name}.json"
        os.makedirs(self.memory_dir, exist_ok=True)        
        self.db_fullpath = os.path.join(self.memory_dir, self.db_filename)
        self.memories: Dict[str, Dict] = {}
        self.dimension: Optional[int] = None
        self.index = None  # FAISS index for similarity search
        self.ollama_client = OllamaClient()  # Ensure your client supports get_embedding
        logger.info(f"Initializing MemoryDB for {self.db_fullpath}")

    @classmethod
    async def create(cls, 
                     db_name: str = "chat_memory", 
                     session_name: Optional[str] = None) -> 'MemoryDB':
        instance = cls(db_name, session_name)
        await instance.initialize()
        return instance

    async def initialize(self):
        self.load_memories()
        logger.info(f"Loaded {len(self.memories)} memories from disk at {self.db_fullpath}")
        # Determine dimension from an existing record if available, or initialize using a test message embedding.
        if self.memories:
            first_record = next(iter(self.memories.values()))
            if 'vector' in first_record:
                self.dimension = len(first_record['vector'])
        if not self.dimension:
            test_embedding = await self.ollama_client.get_embedding("test")
            self.dimension = len(test_embedding)
            logger.info(f"Initialized embedding dimension to {self.dimension}")

        # Create FAISS index to conduct similarity searches.
        self.index = faiss.IndexFlatIP(self.dimension)
        # Add existing vectors to the FAISS index.
        if self.memories:
            vectors = []
            for key, memory in self.memories.items():
                if 'vector' in memory:
                    vector = np.array(memory['vector']).astype('float32')
                    # Normalize the vector
                    vector = vector / np.linalg.norm(vector)
                    vectors.append(vector)
            if vectors:
                vectors_np = np.array(vectors).astype('float32')
                self.index.add(vectors_np)
                logger.info(f"Added {len(vectors)} vectors to FAISS index")

    def load_memories(self):
        try:
            if os.path.exists(self.db_fullpath):
                with open(self.db_fullpath, 'r') as f:
                    self.memories = json.load(f)
                logger.debug(f"Loaded {len(self.memories)} memories from {self.db_fullpath}")
            else:
                logger.debug(f"No existing memory file found at {self.db_fullpath}")
                self.memories = {}
        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}")
            self.memories = {}

    def save_memories(self):
        try:
            logger.info(f"Saving memories to: {self.db_fullpath}")
            with open(self.db_fullpath, 'w') as f:
                json.dump(self.memories, f)
            logger.info(f"Successfully saved {len(self.memories)} memories")
        except Exception as e:
            logger.error(f"Error saving memories: {str(e)}")
            raise

    async def add_memory(self, text: str, metadata: Optional[Dict] = None) -> str:
        """
        Compute an embedding for the text, store the vector with the memory record,
        update the FAISS index, and save the record to disk.
        """
        try:
            # Get the embedding for the text.
            vector = await self.ollama_client.get_embedding(text)
            vector = np.array(vector).astype('float32')
            # Normalize the vector.
            norm = np.linalg.norm(vector)
            if norm:
                vector = vector / norm
            else:
                logger.warning("Received zero vector for embedding.")
            key = str(uuid.uuid4())
            memory_entry = {
                'text': text,
                'vector': vector.tolist(),
                'metadata': metadata or {},
                'created_at': datetime.utcnow().isoformat()
            }
            # Add to FAISS index.
            self.index.add(np.array([vector]).astype('float32'))
            self.memories[key] = memory_entry
            self.save_memories()
            return key
        except Exception as e:
            logger.error(f"Error adding memory: {str(e)}")
            raise

    async def query(self, query_text: str, k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Generate an embedding for the query, perform a similarity search using FAISS,
        and return only those memories that meet the specified similarity threshold.
        """
        try:
            query_vector = await self.ollama_client.get_embedding(query_text)
            query_vector = np.array(query_vector).astype('float32')
            query_vector = query_vector / np.linalg.norm(query_vector)
            query_vector_np = np.array([query_vector]).astype('float32')
            if self.index.ntotal == 0:
                logger.warning("No vectors in FAISS index!")
                return []
            scores, indices = self.index.search(query_vector_np, min(k, self.index.ntotal))
            results = []
            all_keys = list(self.memories.keys())
            for similarity, idx in zip(scores[0], indices[0]):
                if similarity >= threshold and idx < len(all_keys):
                    memory_key = all_keys[idx]
                    memory = self.memories[memory_key]
                    results.append({
                        'key': memory_key,
                        'text': memory['text'],
                        'similarity': float(similarity),
                        'metadata': memory.get('metadata', {}),
                        'created_at': memory.get('created_at')
                    })
            results = sorted(results, key=lambda x: x['similarity'], reverse=True)
            return results
        except Exception as e:
            logger.error(f"Error querying memories: {str(e)}")
            raise