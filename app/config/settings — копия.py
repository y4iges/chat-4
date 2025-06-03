import os
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Memory settings
MEMORY_PATH = os.path.join(BASE_DIR, "data", "memory")
os.makedirs(MEMORY_PATH, exist_ok=True)

# Session settings (for saving chat sessions)
SESSIONS_PATH = os.path.join(BASE_DIR, "data", "sessions")
os.makedirs(SESSIONS_PATH, exist_ok=True)

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large:latest"  # For embeddings
OLLAMA_CHAT_MODEL = "gemma-2-2b-it.q8_0:latest"  # For chat responses

# Memory DB settings
MEMORY_SIMILARITY_THRESHOLD = 0.3
MEMORY_MAX_RESULTS = 5

# Chat settings
SYSTEM_PROMPT = f"""You are a helpful AI assistant. Current time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC.
Use the context from relevant memories if provided to give more informed answers.
If no relevant context is found, you can still provide general responses based on your knowledge."""

settings = SimpleNamespace(
    BASE_DIR=BASE_DIR,
    MEMORY_PATH=MEMORY_PATH,
    SESSIONS_PATH=SESSIONS_PATH,
    OLLAMA_BASE_URL=OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL=OLLAMA_EMBEDDING_MODEL,
    OLLAMA_CHAT_MODEL=OLLAMA_CHAT_MODEL,
    MEMORY_SIMILARITY_THRESHOLD=MEMORY_SIMILARITY_THRESHOLD,
    MEMORY_MAX_RESULTS=MEMORY_MAX_RESULTS,
    SYSTEM_PROMPT=SYSTEM_PROMPT,
)