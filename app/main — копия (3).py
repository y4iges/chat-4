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
        # We no longer call add_memory here.
        memory_db = session_memory_dbs[session_name]

        try:
            memories = await memory_db.query(user_message)
        except Exception as e:
            logger.error(f"Error querying memories: {str(e)}")
            memories = {}

        # Use selected model if provided; defaults to Gemma for chat.
        if selected_model:
            ollama_client = OllamaClient(chat_model=selected_model)
        else:
            ollama_client = OllamaClient()
        response = await ollama_client.chat(user_message)

        return {"response": response, "memories": memories}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Renamed /summarize to /memorize.
@app.post("/memorize")
async def memorize_endpoint(request: Request):
    """
    Memorize endpoint: summarizes the chosen messages from the chat
    (provided by the user) and stores the summary into the session's memory file.
    """
    try:
        data = await request.json()
        messages = data.get("messages", [])
        session_name = data.get("session", "").strip()
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided for memorization.")
        if not session_name:
            raise HTTPException(status_code=400, detail="Session name is required for memorization.")

        # Ensure session-specific MemoryDB exists.
        if session_name not in session_memory_dbs:
            session_memory_dbs[session_name] = await MemoryDB.create(
                db_name="chat_memory",
                session_name=session_name
            )
        memory_db = session_memory_dbs[session_name]

        conversation_text = "\n".join(messages)
        prompt = (
            "Please summarize the following conversation concisely, focusing on key points and important details.\n\n"
            f"{conversation_text}\n\nSummary:"
        )
        # Generate summary using OllamaClient.
        ollama_client = OllamaClient()
        summary = await ollama_client.chat(prompt)
        # Save the summary into the session's memory file.
        summary_metadata = {"memorized": True}
        await memory_db.add_memory(summary, metadata=summary_metadata)
        return JSONResponse(content={"detail": "Memorized and stored summary."})
    except Exception as e:
        logger.error(f"Error in memorize endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/save")
async def save_session_endpoint(request: Request):
    try:
        data = await request.json()
        session_name = data.get("session_name", "").strip()
        chat_history = data.get("chat_history", [])
        if not session_name:
            raise HTTPException(status_code=400, detail="Session name must be provided.")
        session_manager.save_session(session_name, chat_history)
        return JSONResponse(content={"session_name": session_name, "detail": "Session saved."})
    except Exception as e:
        logger.error(f"Error in session saving endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/list")
async def list_session_endpoint():
    try:
        sessions = session_manager.list_sessions()
        return JSONResponse(content={"sessions": sessions})
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/load")
async def load_session_endpoint(session_name: str = Query(..., description="The session file name to load (without .json extension)")):
    try:
        session_data = session_manager.load_session(session_name)
        return JSONResponse(content=session_data)
    except Exception as e:
        logger.error(f"Error in session loading endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))