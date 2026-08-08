import os
from contextlib import contextmanager
from threading import Lock

import mysql.connector
from dotenv import load_dotenv


load_dotenv()

_schema_lock = Lock()
_schema_ready = False


def _connection_config() -> dict:
    return {
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "assignment8"),
        "password": os.getenv("MYSQL_PASSWORD", "assignment8"),
        "database": os.getenv("MYSQL_DATABASE", "assignment8_chat"),
    }


@contextmanager
def database_connection():
    connection = mysql.connector.connect(**_connection_config())
    try:
        yield connection
    finally:
        connection.close()


def ensure_schema() -> None:
    global _schema_ready

    if _schema_ready:
        return

    with _schema_lock:
        if _schema_ready:
            return

        with database_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                    external_id VARCHAR(255) NOT NULL UNIQUE,
                    model VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                    conversation_id BIGINT UNSIGNED NOT NULL,
                    external_id VARCHAR(255) NOT NULL,
                    role ENUM('user', 'assistant', 'system') NOT NULL,
                    content LONGTEXT NOT NULL,
                    message_type VARCHAR(50) NOT NULL DEFAULT 'chat',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_message_external (conversation_id, external_id),
                    CONSTRAINT fk_messages_conversation
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    conversation_id BIGINT UNSIGNED PRIMARY KEY,
                    summary LONGTEXT NOT NULL,
                    through_message_id BIGINT UNSIGNED NOT NULL,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,
                    CONSTRAINT fk_summary_conversation
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            connection.commit()
            cursor.close()

        _schema_ready = True


def save_message(
    conversation_external_id: str,
    message_external_id: str,
    model: str,
    role: str,
    content: str,
    message_type: str = "chat",
) -> None:
    ensure_schema()

    with database_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO conversations (external_id, model)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE model = VALUES(model)
            """,
            (conversation_external_id, model),
        )
        cursor.execute(
            "SELECT id FROM conversations WHERE external_id = %s",
            (conversation_external_id,),
        )
        conversation_id = cursor.fetchone()[0]
        cursor.execute(
            """
            INSERT INTO messages
                (conversation_id, external_id, role, content, message_type)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                role = VALUES(role),
                content = VALUES(content),
                message_type = VALUES(message_type)
            """,
            (
                conversation_id,
                message_external_id,
                role,
                content,
                message_type,
            ),
        )
        connection.commit()
        cursor.close()


def get_conversation_history(conversation_external_id: str) -> list[dict]:
    ensure_schema()

    with database_connection() as connection:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT m.id, m.external_id, m.role, m.content, m.message_type,
                   m.created_at
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE c.external_id = %s
            ORDER BY m.id
            """,
            (conversation_external_id,),
        )
        history = cursor.fetchall()
        cursor.close()
        return history


def get_conversation_summary(conversation_external_id: str) -> dict | None:
    ensure_schema()

    with database_connection() as connection:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT s.summary, s.through_message_id, s.updated_at
            FROM conversation_summaries AS s
            JOIN conversations AS c ON c.id = s.conversation_id
            WHERE c.external_id = %s
            """,
            (conversation_external_id,),
        )
        summary = cursor.fetchone()
        cursor.close()
        return summary


def save_conversation_summary(
    conversation_external_id: str,
    summary: str,
    through_message_id: int,
) -> None:
    ensure_schema()

    with database_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT id FROM conversations WHERE external_id = %s",
            (conversation_external_id,),
        )
        row = cursor.fetchone()
        if row is None:
            cursor.close()
            raise ValueError("Cannot summarize a conversation that does not exist")

        cursor.execute(
            """
            INSERT INTO conversation_summaries
                (conversation_id, summary, through_message_id)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
                summary = VALUES(summary),
                through_message_id = VALUES(through_message_id)
            """,
            (row[0], summary, through_message_id),
        )
        connection.commit()
        cursor.close()


def find_conversation_by_assistant_content(content: str) -> str | None:
    """Find the most recent conversation containing an exact assistant reply."""
    ensure_schema()

    with database_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT c.external_id
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.role = 'assistant' AND m.content = %s
            ORDER BY m.id DESC
            LIMIT 1
            """,
            (content,),
        )
        row = cursor.fetchone()
        cursor.close()
        return row[0] if row else None
