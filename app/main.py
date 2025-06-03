import logging
import uuid
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.memory.memory_db import MemoryDB
from app.chat.ollama_client import OllamaClient
from app.memory import session_manager

app = FastAPI()
logger = logging.getLogger("app.main")

# Mount static files; index.html is served separately.
app.mount("/static", StaticFiles(directory="app/static", html=True), name="static")

# Global dictionary to hold session-related MemoryDB instances.
session_memory_dbs = {}

@app.get("/")
async def root():
    index_path = Path("app/static/index.html")
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html not found")

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        session_name = data.get("session", "").strip()  # session name provided in request
        system_prompt = data.get("system_prompt", "").strip()
        selected_model = data.get("model", None)
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message not provided")
        if not session_name:
            return JSONResponse(content={"detail": "Session name is required."}, status_code=400)

        logger.info(f"Received chat request: {user_message}")

        # Use the session-specific MemoryDB instance.
        if session_name not in session_memory_dbs:
            session_memory_dbs[session_name] = await MemoryDB.create(
                db_name="chat_memory",
                session_name=session_name
            )
        memory_db = session_memory_dbs[session_name]

        try:
            memories = await memory_db.query(user_message)
        except Exception as e:
            logger.error(f"Error querying memories: {str(e)}")
            memories = {}

        # Merge system prompt with context, memories, and user message.
        final_prompt = ""
        if system_prompt:
            final_prompt += system_prompt + "\n\n"
        # Optionally include memories or other context if needed.
        if memories:
            # Here you could format the memories to add extra context.
            final_prompt += "Relevant Memories:\n"
            for mem in memories.values():
                final_prompt += mem.get('text', '') + "\n"
            final_prompt += "\n"
        final_prompt += "User: " + user_message + "\nAssistant:"

        # Use selected model if provided; defaults to Gemma for chat.
        if selected_model:
            ollama_client = OllamaClient(chat_model=selected_model)
        else:
            ollama_client = OllamaClient()

        response = await ollama_client.chat(final_prompt)

        return {"response": response, "memories": memories}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))