import sqlite3
from typing import List

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)

conn = sqlite3.connect(
    "chat_memory1.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    role TEXT,
    content TEXT
)
""")

conn.commit()


def save_message(session_id: str, role: str, content: str):

    cursor.execute("""
    INSERT INTO chat_history (session_id, role, content)
    VALUES (?, ?, ?)
    """, (session_id, role, content))

    conn.commit()


def load_chat_history(session_id: str) -> List[BaseMessage]:

    cursor.execute("""
    SELECT role, content
    FROM chat_history
    WHERE session_id = ?
    ORDER BY id
    """, (session_id,))

    rows = cursor.fetchall()

    history: List[BaseMessage] = []

    for role, content in rows:

        if role == "human":
            history.append(HumanMessage(content=content))

        elif role == "ai":
            history.append(AIMessage(content=content))

        elif role == "system":
            history.append(SystemMessage(content=content))

    return history


def get_all_sessions() -> List[str]:

    cursor.execute("""
    SELECT DISTINCT session_id
    FROM chat_history
    ORDER BY id DESC
    """)

    return [row[0] for row in cursor.fetchall()]