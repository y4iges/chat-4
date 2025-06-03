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
OLLAMA_CHAT_MODEL = "phi-4-Q5_K_Munsloth:latest"  # For chat responses

# Memory DB settings
MEMORY_SIMILARITY_THRESHOLD = 0.3
MEMORY_MAX_RESULTS = 5

# LLM settings
LLM_TEMPERATURE = 0.7  # Controls randomness: 0.0 = deterministic, 1.0 = more random
LLM_MAX_TOKENS = 128  # Maximum tokens in a single response
LLM_TOP_P = 0.95       # Nucleus sampling (probabilities sum to top_p)
LLM_FREQUENCY_PENALTY = 0.0  # Discourage repeating words/phrases
LLM_PRESENCE_PENALTY = 0.0   # Encourage introducing new topics
LLM_REPETITION_PENALTY = 1.2 # Penalty for repeating tokens/phrases
MAX_CONTEXT_TOKENS = 8000  # Reduced value for testing memory recall

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
    # LLM settings
    LLM_TEMPERATURE=LLM_TEMPERATURE,
    LLM_MAX_TOKENS=LLM_MAX_TOKENS,
    LLM_TOP_P=LLM_TOP_P,
    LLM_FREQUENCY_PENALTY=LLM_FREQUENCY_PENALTY,
    LLM_PRESENCE_PENALTY=LLM_PRESENCE_PENALTY,
    LLM_REPETITION_PENALTY=LLM_REPETITION_PENALTY,
    SYSTEM_PROMPT=SYSTEM_PROMPT,
)