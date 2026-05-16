from typing import List
import os
import sqlite3

from src.config import CHAT_DB_PATH

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)

## Database setup for storing chat history, with functions to save messages and load chat history based on session IDs. Also includes a function to get a preview of the chat based on the first human message in the session.

os.makedirs(
    os.path.dirname(CHAT_DB_PATH),
    exist_ok=True
)

conn = sqlite3.connect(
    CHAT_DB_PATH,
    check_same_thread=False
)

cursor = conn.cursor()

## Create the chat_history table if it doesn't exist, with columns for id, session_id, role, and content. The id is an auto-incrementing primary key, session_id is a text field to group messages by chat session, role indicates
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    role TEXT,
    content TEXT
)
""")

conn.commit()


def save_message(session_id: str, role: str, content: str): ## This function saves a message to the chat history database, associating it with a specific session ID and role (human, ai, or system).

    cursor.execute("""
    INSERT INTO chat_history (session_id, role, content)
    VALUES (?, ?, ?)
    """, (session_id, role, content))

    conn.commit()


def load_chat_history(session_id: str) -> List[BaseMessage]: ## This function loads the chat history for a given session ID from the database, retrieves the messages in order, and returns them as a list of BaseMessage objects with the appropriate roles.

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


def get_all_sessions() -> List[str]: ## This function retrieves a list of all unique session IDs from the chat history database, ordered by the most recent message, and returns them as a list of strings.

    cursor.execute("""
    SELECT DISTINCT session_id
    FROM chat_history
    ORDER BY id DESC
    """)

    return [row[0] for row in cursor.fetchall()]

def get_chat_preview(session_id: str) -> str: ## This function retrieves the first human message for a given session ID from the chat history database, extracts a preview of the content (up to 30 characters), and returns it as a string. If there are no messages, it returns "New Chat".
    cursor.execute(
        '''
        SELECT content
        FROM chat_history
        WHERE session_id = ?
        AND role = 'human'
        ORDER BY rowid ASC
        LIMIT 1
        ''',
        (session_id,)
    )
    row = cursor.fetchone()

    if row:
        preview = row[0][:30]
        if len(row[0]) > 30:
            preview += "..."
        return preview
    return "New Chat"