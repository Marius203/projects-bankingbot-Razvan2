import os
import sqlite3
from typing import List, Dict
import config

DB_FILE = os.path.join(config.BASE_DIR, "chat_history.db")

def init_db(path: str = DB_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def save_message(session_id: str, role: str, content: str, path: str = DB_FILE):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()

# Retrieve chat history for a session
def get_history(session_id: str, limit: int = 200, path: str = DB_FILE) -> List[Dict]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
        (session_id, limit)
    )
    rows = cur.fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in rows]

def list_sessions(path: str = DB_FILE, limit: int = 100):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT session_id, COUNT(*) AS messages, MAX(created_at) AS last_at "
        "FROM messages GROUP BY session_id ORDER BY last_at DESC LIMIT ?",
        (limit,)
    )
    rows = cur.fetchall()
    conn.close()
    return [{"session_id": r["session_id"], "messages": r["messages"], "last_at": r["last_at"]} for r in rows]

def delete_session(session_id: str, path: str = DB_FILE):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()