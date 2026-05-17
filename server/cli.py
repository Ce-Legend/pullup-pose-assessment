from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from server.auth import hash_password
from server.db import connect, init_schema
from server.settings import load_settings
from server.util import sha256_file, utc_now_iso


def cmd_init_db(_: argparse.Namespace) -> None:
    settings = load_settings()
    conn = connect(settings.db_path)
    init_schema(conn, settings.repo_root / "server" / "schema.sql")
    print(f"OK: 已初始化数据库 {settings.db_path}")


def cmd_create_user(args: argparse.Namespace) -> None:
    settings = load_settings()
    conn = connect(settings.db_path)
    init_schema(conn, settings.repo_root / "server" / "schema.sql")

    role = args.role.strip().lower()
    if role not in {"student", "teacher"}:
        raise SystemExit("role 仅支持 student/teacher")

    password_hash = hash_password(args.password)
    existing = conn.execute("SELECT id FROM users WHERE username=?", (args.username,)).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO users (username, display_name, password_hash, role, created_at) VALUES (?,?,?,?,?)",
            (args.username, args.display_name, password_hash, role, utc_now_iso()),
        )
        conn.commit()
        print(f"OK: 已创建用户 {args.username}（{role}）")
        return

    if not args.update:
        raise SystemExit(f"用户名已存在：{args.username}（如需覆盖请加 --update）")

    conn.execute(
        "UPDATE users SET display_name=?, password_hash=?, role=?, is_active=1 WHERE username=?",
        (args.display_name, password_hash, role, args.username),
    )
    conn.commit()
    print(f"OK: 已更新用户 {args.username}（{role}）")


def cmd_set_standard(args: argparse.Namespace) -> None:
    settings = load_settings()
    src = Path(args.file).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"文件不存在：{src}")

    dst = settings.standard_video_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)

    version = sha256_file(dst)[:12]
    print(f"OK: 已更新标准视频：{dst}")
    print(f"standard_version: {version}")
    print("提示：若有标准缓存，建议重启 worker 或删除 data/cache/standard 下旧缓存后再跑新分析。")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="server.cli", description="一期系统管理命令")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-db", help="初始化数据库")
    p_init.set_defaults(func=cmd_init_db)

    p_user = sub.add_parser("create-user", help="创建用户（象征性登录）")
    p_user.add_argument("--username", required=True)
    p_user.add_argument("--password", required=True)
    p_user.add_argument("--role", required=True, choices=["student", "teacher"])
    p_user.add_argument("--display-name", required=True)
    p_user.add_argument("--update", action="store_true", help="若用户名已存在，则覆盖更新（显示名/密码/角色）")
    p_user.set_defaults(func=cmd_create_user)

    p_std = sub.add_parser("set-standard", help="替换标准视频（无后台UI）")
    p_std.add_argument("--file", required=True, help="标准视频文件路径（MP4）")
    p_std.set_defaults(func=cmd_set_standard)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
