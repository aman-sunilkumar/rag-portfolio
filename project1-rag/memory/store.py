import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from .schemas import UserProfile, SessionMessage, FeedbackEvent

DB_PATH = "./memory/memory.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    Path("./memory").mkdir(exist_ok=True)
    with get_conn() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at TEXT NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
            query TEXT NOT NULL, answer TEXT NOT NULL,
            sources TEXT NOT NULL, timestamp TEXT NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL, doc_id TEXT NOT NULL,
            positive INTEGER NOT NULL, timestamp TEXT NOT NULL)""")
        conn.commit()


def save_profile(profile: UserProfile):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute("""INSERT INTO user_profiles (user_id, data, updated_at) VALUES (?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at""",
            (profile.user_id, profile.model_dump_json(), now))
        conn.commit()


def get_profile(user_id: str) -> Optional[UserProfile]:
    with get_conn() as conn:
        row = conn.execute("SELECT data FROM user_profiles WHERE user_id=?", (user_id,)).fetchone()
        if row:
            return UserProfile.model_validate_json(row["data"])
    return None


def add_session(msg: SessionMessage):
    with get_conn() as conn:
        conn.execute("""INSERT INTO sessions (user_id, query, answer, sources, timestamp)
            VALUES (?,?,?,?,?)""",
            (msg.user_id, msg.query, msg.answer,
             json.dumps(msg.sources), datetime.utcnow().isoformat()))
        conn.commit()


def get_recent_queries(user_id: str, n: int = 5) -> List[str]:
    with get_conn() as conn:
        rows = conn.execute("""SELECT query FROM sessions WHERE user_id=?
            ORDER BY timestamp DESC LIMIT ?""", (user_id, n)).fetchall()
        return [r["query"] for r in rows]


def get_liked_docs(user_id: str) -> List[str]:
    with get_conn() as conn:
        rows = conn.execute("""SELECT DISTINCT doc_id FROM feedback
            WHERE user_id=? AND positive=1""", (user_id,)).fetchall()
        return [r["doc_id"] for r in rows]


def add_feedback(event: FeedbackEvent):
    with get_conn() as conn:
        conn.execute("""INSERT INTO feedback (user_id,chunk_id,doc_id,positive,timestamp)
            VALUES (?,?,?,?,?)""",
            (event.user_id, event.chunk_id, event.doc_id,
             1 if event.positive else 0, datetime.utcnow().isoformat()))
        conn.commit()
