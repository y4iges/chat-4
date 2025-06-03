import asyncio
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

app.mount("/static", StaticFiles(directory="app/static", html=True), name="static")

memory_db = None

@app.on_event("startup")
async def startup_event():
    global memory_db
    try:
        memory_db = await MemoryDB.create(db_name="chat_memory")
        logger.info("MemoryDB initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MemoryDB: {str(e)}")
        raise e

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
        session_id = data.get("session_id", None)
        selected_model = data.get("model", None)
        if not user_message:
            raise HTTPException(status_code=400, detail="Message not provided")

        logger.info(f"Received chat request: {user_message}")

        try:
            memories = await memory_db.query(user_message)
        except Exception as e:
            logger.error(f"Error querying memories: {str(e)}")
            memories = []

        # Create a new OllamaClient with the selected model (if provided)
        if selected_model:
            ollama_client = OllamaClient(chat_model=selected_model)
        else:
            ollama_client = OllamaClient()
        response = await ollama_client.chat(user_message)

        try:
            await memory_db.add_memory(user_message)
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")

        return {"response": response, "memories": memories}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_endpoint(request: Request):
    try:
        data = await request.json()
        messages = data.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided for summarization.")
        conversation_text = "\n".join(messages)
        prompt = (
            "Please summarize the following conversation concisely, focusing on key points and important details. "
            "Ignore any previous summary lines.\n\n"
            f"{conversation_text}\n\nSummary:"
        )
        ollama_client = OllamaClient()
        summary = await ollama_client.chat(prompt)
        summary_metadata = {"summary": True}
        try:
            await memory_db.add_memory(summary, metadata=summary_metadata)
        except Exception as e:
            logger.error(f"Error storing summary memory: {str(e)}")
        return JSONResponse(content={"detail": "Session summarized and stored."})
    except Exception as e:
        logger.error(f"Error in summarization endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/save")
async def save_session_endpoint(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id", None)
        chat_history = data.get("chat_history", [])
        if not session_id:
            session_id = "session_" + str(uuid.uuid4())
        session_manager.save_session(session_id, chat_history)
        return JSONResponse(content={"session_id": session_id, "detail": "Session saved."})
    except Exception as e:
        logger.error(f"Error in session saving endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/load")
async def load_session_endpoint(session_id: str = Query(..., description="The ID of the session to load")):
    try:
        session_data = session_manager.load_session(session_id)
        return JSONResponse(content=session_data)
    except Exception as e:
        logger.error(f"Error in session loading endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))