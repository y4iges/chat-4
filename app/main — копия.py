import asyncio
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.memory.memory_db import MemoryDB
from app.chat.ollama_client import OllamaClient

app = FastAPI()
logger = logging.getLogger("app.main")

# Mount static files for other assets if needed.
app.mount("/static", StaticFiles(directory="app/static", html=True), name="static")

# Global variable to store the MemoryDB instance
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

# Serve index.html for the root route
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
        if not user_message:
            raise HTTPException(status_code=400, detail="Message not provided")

        logger.info(f"Received chat request: {user_message}")

        try:
            memories = await memory_db.query(user_message)
        except Exception as e:
            logger.error(f"Error querying memories: {str(e)}")
            memories = []

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

# New endpoint for summarization
@app.post("/summarize")
async def summarize_endpoint(request: Request):
    """
    Summarize important conversation parts that the user has selected.
    The frontend should send an array of message texts or let the backend choose recent conversation items.
    """
    try:
        data = await request.json()
        # This could be an array of messages or an instruction of how much to summarize
        messages = data.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided for summarization.")

        # Combine messages into a single string context; you could also access MemoryDB here.
        conversation_text = "\n".join(messages)

        # Create a summarization prompt for your model.
        prompt = (f"Please summarize the following conversation concisely, focusing on the key points and important details:\n\n"
                  f"{conversation_text}\n\n"
                  "Summary:")

        ollama_client = OllamaClient()
        summary = await ollama_client.chat(prompt)

        # Optionally, store the summary back into MemoryDB with metadata indicating it's a summary.
        summary_metadata = {"summary": True}
        try:
            await memory_db.add_memory(summary, metadata=summary_metadata)
        except Exception as e:
            logger.error(f"Error storing summary memory: {str(e)}")

        return JSONResponse(content={"summary": summary})
    except Exception as e:
        logger.error(f"Error in summarization endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))