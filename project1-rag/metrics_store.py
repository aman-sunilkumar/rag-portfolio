import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "metrics.db"


def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS request_metrics (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                REAL,
                question          TEXT,
                ret_ms            REAL,
                gen_ms            REAL,
                total_ms          REAL,
                prompt_tokens     INTEGER,
                completion_tokens INTEGER,
                grounded          INTEGER,
                success           INTEGER
            )
        """)


def log(question, ret_ms, gen_ms, prompt_tokens, completion_tokens, grounded, success):
    with _conn() as c:
        c.execute("""
            INSERT INTO request_metrics
            (ts, question, ret_ms, gen_ms, total_ms, prompt_tokens,
             completion_tokens, grounded, success)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(), question, ret_ms, gen_ms, round(ret_ms + gen_ms, 1),
            prompt_tokens, completion_tokens, int(grounded), int(success)
        ))


init_db()
