import os
import json
from datetime import datetime
from ..config.settings import settings

SESSION_FILE_SUFFIX = ".json"

def get_session_filepath(session_name: str) -> str:
    filename = f"{session_name}{SESSION_FILE_SUFFIX}"
    return os.path.join(settings.SESSIONS_PATH, filename)

def save_session(session_name: str, chat_history: list) -> None:
    session_data = {
        "session_name": session_name,
        "chat_history": chat_history,
        "saved_at": datetime.utcnow().isoformat()
    }
    filepath = get_session_filepath(session_name)
    with open(filepath, 'w') as f:
        json.dump(session_data, f)

def load_session(session_name: str) -> dict:
    filepath = get_session_filepath(session_name)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Session file {filepath} not found.")
    with open(filepath, 'r') as f:
        session_data = json.load(f)
    return session_data

def list_sessions() -> list:
    files = os.listdir(settings.SESSIONS_PATH)
    sessions = [f for f in files if f.endswith(SESSION_FILE_SUFFIX)]
    return [f[:-len(SESSION_FILE_SUFFIX)] for f in sessions]