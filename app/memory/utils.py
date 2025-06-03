import uuid
from typing import List, Dict, Any

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks of approximately equal size."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for space
        if current_size + word_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def generate_memory_key() -> str:
    """Generate a unique key for a memory entry."""
    return str(uuid.uuid4())

def format_chat_history(history: List[Dict[str, Any]]) -> str:
    """Format chat history into a string."""
    formatted = []
    for entry in history:
        formatted.append(f"User: {entry.get('user', '')}")
        formatted.append(f"Assistant: {entry.get('assistant', '')}")
    return "\n".join(formatted)