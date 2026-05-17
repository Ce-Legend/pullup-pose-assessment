from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_schema(conn: sqlite3.Connection, schema_path: Path) -> None:
    schema_sql = schema_path.read_text(encoding="utf-8")
    conn.executescript(schema_sql)
    conn.commit()
    # 执行数据库迁移
    migrate(conn)


def migrate(conn: sqlite3.Connection) -> None:
    """执行数据库迁移，添加新表和新列"""
    cursor = conn.cursor()
    
    # 检查 standard_videos 表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='standard_videos'")
    if not cursor.fetchone():
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS standard_videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                view_angle TEXT NOT NULL CHECK (view_angle IN ('front', 'side', 'angle')),
                action TEXT NOT NULL DEFAULT 'pullup',
                file_path TEXT NOT NULL,
                version TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 0,
                uploaded_by INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (uploaded_by) REFERENCES users(id)
            )
        """)
        # 创建唯一索引
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_standard_active 
            ON standard_videos(view_angle, action) WHERE is_active = 1
        """)
        conn.commit()
        print("[DB] Created standard_videos table")
    
    # 检查 analyses 表是否有新增列
    cursor.execute("PRAGMA table_info(analyses)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if "standard_video_id" not in columns:
        cursor.execute("ALTER TABLE analyses ADD COLUMN standard_video_id INTEGER")
        conn.commit()
        print("[DB] Added standard_video_id column to analyses")
    
    if "compare_mode" not in columns:
        cursor.execute('ALTER TABLE analyses ADD COLUMN compare_mode TEXT DEFAULT "standard"')
        conn.commit()
        print("[DB] Added compare_mode column to analyses")
    
    if "compare_analysis_id" not in columns:
        cursor.execute("ALTER TABLE analyses ADD COLUMN compare_analysis_id TEXT")
        conn.commit()
        print("[DB] Added compare_analysis_id column to analyses")


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}

