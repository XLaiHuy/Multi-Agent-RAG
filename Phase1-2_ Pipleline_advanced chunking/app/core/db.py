"""
SQLite Database Engine for RAG Enterprise:
- Lưu trữ người dùng (Users) với mật khẩu bcrypt & phân quyền Role
- Lưu trữ các phiên hội thoại (Conversations) và tin nhắn (Messages)
- Lưu trữ bộ nhớ đệm ngữ nghĩa (Semantic Cache) với vector similarity
"""
import os
import json
import sqlite3
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import bcrypt

DB_PATH = Path("data/app.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(plain: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def init_database():
    """Khởi tạo schema database và seed các tài khoản ban đầu nếu chưa có."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Bảng Users
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        full_name TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. Bảng Conversations
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        username TEXT NOT NULL,
        title TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
    );
    """)

    # 3. Bảng Messages
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conv_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        sources TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conv_id) REFERENCES conversations(id) ON DELETE CASCADE
    );
    """)

    # 4. Bảng Semantic Cache
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS semantic_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT NOT NULL,
        query_embedding TEXT NOT NULL,
        answer TEXT NOT NULL,
        sources TEXT,
        hit_count INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Seed Default Users nếu bảng trống
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        default_users = [
            ("admin", hash_password("admin"), "System Administrator", "admin"),
            ("hr01", hash_password("hr"), "HR Manager", "HR"),
            ("ketoan01", hash_password("ketoan"), "Finance Officer", "Finance"),
            ("user01", hash_password("user123"), "Standard Student/User", "user"),
        ]
        cursor.executemany(
            "INSERT INTO users (username, hashed_password, full_name, role) VALUES (?, ?, ?, ?)",
            default_users
        )
        print("[Database] Initialized default accounts: admin, hr01, ketoan01, user01")

    conn.commit()
    conn.close()


# --- User Management Operations ---

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# --- Conversation Operations ---

def list_user_conversations(username: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM conversations WHERE username = ? ORDER BY updated_at DESC",
        (username,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_conversation_messages(conv_id: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM messages WHERE conv_id = ? ORDER BY id ASC",
        (conv_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    messages = []
    for r in rows:
        d = dict(r)
        if d.get("sources"):
            try:
                d["sources"] = json.loads(d["sources"])
            except Exception:
                d["sources"] = []
        else:
            d["sources"] = []
        messages.append(d)
    return messages


def save_message(conv_id: str, username: str, role: str, content: str, title: Optional[str] = None, sources: Optional[List] = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.datetime.now().isoformat()

    # Tạo hoặc update conversation
    cursor.execute("SELECT id FROM conversations WHERE id = ?", (conv_id,))
    if not cursor.fetchone():
        conv_title = title or content[:40] + ("..." if len(content) > 40 else "")
        cursor.execute(
            "INSERT INTO conversations (id, username, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (conv_id, username, conv_title, now, now)
        )
    else:
        cursor.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conv_id)
        )

    sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
    cursor.execute(
        "INSERT INTO messages (conv_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
        (conv_id, role, content, sources_json, now)
    )
    conn.commit()
    conn.close()


def delete_conversation(conv_id: str, username: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE id = ? AND username = ?", (conv_id, username))
    conn.commit()
    conn.close()


# --- Semantic Cache Operations ---

def lookup_semantic_cache(query_embedding: list[float], similarity_threshold: float = 0.95) -> Optional[Dict[str, Any]]:
    """
    Tìm kiếm câu trả lời đã lưu trong Semantic Cache nếu độ tương đồng cosine >= similarity_threshold.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, query, query_embedding, answer, sources, hit_count FROM semantic_cache")
    rows = cursor.fetchall()
    
    if not rows:
        conn.close()
        return None

    q_vec = np.array(query_embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        conn.close()
        return None
    q_vec = q_vec / q_norm

    best_match = None
    best_sim = -1.0

    for r in rows:
        cached_emb = np.array(json.loads(r["query_embedding"]), dtype=np.float32)
        c_norm = np.linalg.norm(cached_emb)
        if c_norm == 0:
            continue
        cached_emb = cached_emb / c_norm
        
        sim = float(np.dot(q_vec, cached_emb))
        if sim > best_sim and sim >= similarity_threshold:
            best_sim = sim
            best_match = r

    if best_match:
        # Tăng hit_count
        cursor.execute(
            "UPDATE semantic_cache SET hit_count = hit_count + 1, last_accessed_at = ? WHERE id = ?",
            (datetime.datetime.now().isoformat(), best_match["id"])
        )
        conn.commit()
        conn.close()
        
        sources_list = []
        if best_match["sources"]:
            try:
                sources_list = json.loads(best_match["sources"])
            except Exception:
                pass

        return {
            "cached_query": best_match["query"],
            "similarity": best_sim,
            "answer": best_match["answer"],
            "sources": sources_list,
            "hit_count": best_match["hit_count"] + 1,
        }

    conn.close()
    return None


def store_semantic_cache(query: str, query_embedding: list[float], answer: str, sources: Optional[List] = None):
    """Lưu trữ câu hỏi và câu trả lời vào Semantic Cache."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.datetime.now().isoformat()
    emb_json = json.dumps(query_embedding)
    src_json = json.dumps(sources, ensure_ascii=False) if sources else None

    cursor.execute(
        "INSERT INTO semantic_cache (query, query_embedding, answer, sources, created_at, last_accessed_at) VALUES (?, ?, ?, ?, ?, ?)",
        (query, emb_json, answer, src_json, now, now)
    )
    conn.commit()
    conn.close()


# Khởi tạo DB khi module được load
init_database()
