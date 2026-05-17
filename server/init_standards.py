"""
标准视频初始化脚本

用于预设标准视频，或从备份恢复。

使用方法：
1. 将标准视频文件放入 data/standard/ 目录，命名为：
   - front_standard.mp4  (正面标准)
   - side_standard.mp4   (侧面标准)
   - angle_standard.mp4  (斜侧面标准，可选)

2. 运行脚本：
   python -m server.init_standards

脚本会自动：
- 计算视频版本号
- 写入数据库
- 设置为激活状态
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from server.db import connect, init_schema
from server.settings import load_settings
from server.util import utc_now_iso


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def init_standard_videos():
    settings = load_settings()
    conn = connect(settings.db_path)
    init_schema(conn, settings.repo_root / "server" / "schema.sql")
    
    # 预设的标准视频配置
    presets = [
        {
            "view_angle": "front",
            "name": "正面标准视频",
            "filename": "front_standard.mp4",
        },
        {
            "view_angle": "side", 
            "name": "侧面标准视频",
            "filename": "side_standard.mp4",
        },
        {
            "view_angle": "angle",
            "name": "斜侧面标准视频",
            "filename": "angle_standard.mp4",
        },
    ]
    
    standard_dir = settings.standard_videos_dir
    standard_dir.mkdir(parents=True, exist_ok=True)
    
    for preset in presets:
        file_path = standard_dir / preset["filename"]
        
        # 也检查旧的命名格式（兼容已有文件）
        if not file_path.exists():
            # 尝试查找 view_angle_*.mp4 格式的文件
            pattern = f"{preset['view_angle']}_*.mp4"
            matches = list(standard_dir.glob(pattern))
            if matches:
                file_path = matches[0]
        
        if not file_path.exists():
            print(f"[SKIP] {preset['view_angle']}: 文件不存在 {file_path}")
            continue
        
        version = sha256_file(file_path)[:12]
        
        # 检查是否已存在相同版本
        existing = conn.execute(
            "SELECT id FROM standard_videos WHERE version=?", (version,)
        ).fetchone()
        
        if existing:
            print(f"[SKIP] {preset['view_angle']}: 版本 {version} 已存在")
            # 确保是激活状态
            conn.execute(
                "UPDATE standard_videos SET is_active=0 WHERE view_angle=? AND id!=?",
                (preset["view_angle"], existing["id"])
            )
            conn.execute(
                "UPDATE standard_videos SET is_active=1 WHERE id=?",
                (existing["id"],)
            )
            conn.commit()
            continue
        
        # 先取消该视角的所有激活
        conn.execute(
            "UPDATE standard_videos SET is_active=0 WHERE view_angle=?",
            (preset["view_angle"],)
        )
        
        # 插入新记录
        conn.execute("""
            INSERT INTO standard_videos (name, view_angle, action, file_path, version, is_active, uploaded_by, created_at)
            VALUES (?, ?, ?, ?, ?, 1, NULL, ?)
        """, (
            preset["name"],
            preset["view_angle"],
            "pullup",
            str(file_path),
            version,
            utc_now_iso(),
        ))
        conn.commit()
        
        print(f"[OK] {preset['view_angle']}: {file_path.name} (版本: {version}) 已激活")
    
    # 显示当前激活的标准视频
    print("\n当前激活的标准视频：")
    rows = conn.execute("SELECT * FROM standard_videos WHERE is_active=1").fetchall()
    for row in rows:
        print(f"  - {row['view_angle']}: {row['name']} ({row['version']})")
    
    if not rows:
        print("  (无)")
        print("\n提示：请将标准视频放入 data/standard/ 目录，命名为：")
        print("  - front_standard.mp4  (正面)")
        print("  - side_standard.mp4   (侧面)")
        print("  - angle_standard.mp4  (斜侧面)")


if __name__ == "__main__":
    init_standard_videos()
