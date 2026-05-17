from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import Flask, abort, g, jsonify, redirect, render_template, request, send_file, session, url_for
from flask_cors import CORS

from server.auth import User, hash_password, login_user, logout_user, require_login_api, require_login_page, verify_password
from server.db import connect, init_schema, row_to_dict
from server.settings import load_settings
from server.util import ensure_parent_dir, utc_now_iso


def sha256_file(path: Path) -> str:
    """计算文件的 SHA256 哈希值"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def require_teacher_api(fn):
    """要求教师权限的 API 装饰器"""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = session.get("user")
        if user is None:
            return jsonify({"error_code": "UNAUTHORIZED", "error_message": "请先登录"}), 401
        if user.get("role") != "teacher":
            return jsonify({"error_code": "FORBIDDEN", "error_message": "只有教师可以执行此操作"}), 403
        return fn(*args, **kwargs)
    return wrapper


def require_teacher_page(fn):
    """要求教师权限的页面装饰器"""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = session.get("user")
        if user is None:
            return redirect(url_for("login"))
        if user.get("role") != "teacher":
            abort(403)
        return fn(*args, **kwargs)
    return wrapper


def create_app() -> Flask:
    settings = load_settings()

    templates_dir = settings.repo_root / "server" / "templates"
    static_dir = settings.repo_root / "server" / "static"

    app = Flask(__name__, template_folder=str(templates_dir), static_folder=str(static_dir))
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-me")
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB
    CORS(app, supports_credentials=True)

    conn = connect(settings.db_path)
    init_schema(conn, settings.repo_root / "server" / "schema.sql")

    @app.before_request
    def _attach_context() -> None:
        g.db = conn
        g.settings = settings

    @app.get("/")
    def index():
        if session.get("user"):
            return redirect(url_for("upload"))
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            return render_template("login.html")

        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        if not username or not password:
            return render_template("login.html", error="请输入账号和密码")

        row = conn.execute("SELECT * FROM users WHERE username=? AND is_active=1", (username,)).fetchone()
        if row is None or not verify_password(row["password_hash"], password):
            return render_template("login.html", error="账号或密码错误")

        user = User(id=row["id"], username=row["username"], display_name=row["display_name"], role=row["role"])
        login_user(user)
        return redirect(url_for("upload"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        """学生自助注册"""
        if request.method == "GET":
            return render_template("register.html")

        username = (request.form.get("username") or "").strip()
        display_name = (request.form.get("display_name") or "").strip()
        password = request.form.get("password") or ""
        password_confirm = request.form.get("password_confirm") or ""

        # 验证
        if not username or not display_name or not password:
            return render_template("register.html", error="请填写所有字段")
        
        if len(username) < 3 or len(username) > 20:
            return render_template("register.html", error="账号长度需要3-20个字符")
        
        if not username.replace("_", "").isalnum():
            return render_template("register.html", error="账号只能包含字母、数字和下划线")
        
        if len(display_name) < 2 or len(display_name) > 20:
            return render_template("register.html", error="姓名长度需要2-20个字符")
        
        if len(password) < 6:
            return render_template("register.html", error="密码至少6个字符")
        
        if password != password_confirm:
            return render_template("register.html", error="两次输入的密码不一致")

        # 检查用户名是否已存在
        existing = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if existing:
            return render_template("register.html", error="该账号已被注册")

        # 创建学生账号
        password_hash = hash_password(password)
        conn.execute(
            "INSERT INTO users (username, display_name, password_hash, role, created_at) VALUES (?,?,?,?,?)",
            (username, display_name, password_hash, "student", utc_now_iso()),
        )
        conn.commit()

        return render_template("register.html", success=f"注册成功！账号：{username}，请返回登录")

    @app.post("/logout")
    @require_login_page
    def logout():
        logout_user()
        return redirect(url_for("login"))

    @app.get("/upload")
    @require_login_page
    def upload():
        return render_template("upload.html", user=session.get("user"), view_default=settings.view_default)

    @app.get("/history")
    @require_login_page
    def history():
        user = session["user"]
        if user["role"] == "teacher":
            rows = conn.execute(
                "SELECT analyses.*, users.display_name FROM analyses JOIN users ON users.id=analyses.user_id ORDER BY analyses.created_at DESC LIMIT 200"
            ).fetchall()
            items = [row_to_dict(r) for r in rows]
        else:
            rows = conn.execute("SELECT * FROM analyses WHERE user_id=? ORDER BY created_at DESC LIMIT 200", (user["id"],)).fetchall()
            items = [row_to_dict(r) for r in rows]
        return render_template("history.html", user=user, items=items)

    @app.get("/result/<analysis_id>")
    @require_login_page
    def result_page(analysis_id: str):
        return render_template("result.html", user=session.get("user"), analysis_id=analysis_id)

    # --- API ---
    def _get_analysis_or_404(analysis_id: str) -> dict:
        row = conn.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
        if row is None:
            abort(404)
        record = row_to_dict(row) or {}
        user = session.get("user")
        if user is None:
            abort(401)
        if user["role"] != "teacher" and int(record["user_id"]) != int(user["id"]):
            abort(403)
        return record

    @app.get("/api/my-analyses")
    @require_login_api
    def api_my_analyses():
        """获取当前用户的分析历史列表（用于历史对比选择）"""
        user = session["user"]
        status_filter = request.args.get("status", "").strip()
        
        if status_filter:
            rows = conn.execute(
                "SELECT id, created_at, view, score_total, diff_joint FROM analyses WHERE user_id=? AND status=? ORDER BY created_at DESC LIMIT 50",
                (user["id"], status_filter)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, created_at, view, score_total, diff_joint FROM analyses WHERE user_id=? ORDER BY created_at DESC LIMIT 50",
                (user["id"],)
            ).fetchall()
        
        items = [row_to_dict(r) for r in rows]
        return jsonify({"items": items})

    @app.post("/api/analyses")
    @require_login_api
    def api_create_analysis():
        user = session["user"]
        view = (request.form.get("view") or settings.view_default).strip().lower()
        # 验证视角是否有效
        valid_views = ("front", "side", "angle")
        if view not in valid_views:
            return jsonify({"error_code": "INVALID_VIEW", "error_message": "视角类型无效，请选择正面/侧面/斜侧面"}), 400

        # 解析对比模式
        compare_mode = (request.form.get("compare_mode") or "standard").strip().lower()
        if compare_mode not in ("standard", "history"):
            compare_mode = "standard"
        
        compare_analysis_id = None
        if compare_mode == "history":
            compare_analysis_id = (request.form.get("compare_analysis_id") or "").strip()
            if not compare_analysis_id:
                return jsonify({"error_code": "INVALID_REQUEST", "error_message": "历史对比模式需要选择一个历史动作"}), 400
            # 验证历史分析存在且属于当前用户
            history_row = conn.execute(
                "SELECT id, user_id, view, status FROM analyses WHERE id=?", 
                (compare_analysis_id,)
            ).fetchone()
            if history_row is None:
                return jsonify({"error_code": "NOT_FOUND", "error_message": "选择的历史记录不存在"}), 404
            if int(history_row["user_id"]) != int(user["id"]):
                return jsonify({"error_code": "FORBIDDEN", "error_message": "不能与他人的动作对比"}), 403
            if history_row["status"] != "succeeded":
                return jsonify({"error_code": "INVALID_REQUEST", "error_message": "只能与已完成的分析进行对比"}), 400
            if history_row["view"] != view:
                return jsonify({"error_code": "INVALID_REQUEST", "error_message": "视角不匹配，请选择相同视角的历史记录"}), 400

        file = request.files.get("file")
        if file is None or not file.filename:
            return jsonify({"error_code": "INVALID_MEDIA", "error_message": "请选择 MP4 视频文件"}), 400

        filename = Path(file.filename).name
        if not filename.lower().endswith(".mp4"):
            return jsonify({"error_code": "INVALID_MEDIA", "error_message": "仅支持 MP4 格式"}), 400

        analysis_id = uuid.uuid4().hex

        day_dir = settings.uploads_dir / datetime.now().strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        upload_path = day_dir / f"{analysis_id}.mp4"
        ensure_parent_dir(upload_path)
        file.save(upload_path)

        conn.execute(
            """
            INSERT INTO analyses (
              id, user_id, created_at, status, progress, action, view,
              standard_version, compare_mode, compare_analysis_id, upload_filename, upload_path
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                analysis_id,
                int(user["id"]),
                utc_now_iso(),
                "queued",
                0,
                settings.action_default,
                view,
                "pending",
                compare_mode,
                compare_analysis_id,
                filename,
                str(upload_path),
            ),
        )
        conn.commit()
        return jsonify({"analysis_id": analysis_id, "status": "queued"})

    @app.post("/api/auth/login")
    def api_auth_login():
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""
        if not username or not password:
            return jsonify({"error_code": "INVALID_REQUEST", "error_message": "请输入账号和密码"}), 400

        row = conn.execute("SELECT * FROM users WHERE username=? AND is_active=1", (username,)).fetchone()
        if row is None or not verify_password(row["password_hash"], password):
            return jsonify({"error_code": "INVALID_CREDENTIALS", "error_message": "账号或密码错误"}), 400

        user = User(id=row["id"], username=row["username"], display_name=row["display_name"], role=row["role"])
        login_user(user)
        return jsonify({"user": {"id": user.id, "username": user.username, "display_name": user.display_name, "role": user.role}})

    @app.post("/api/auth/logout")
    @require_login_api
    def api_auth_logout():
        logout_user()
        return jsonify({"ok": True})

    @app.get("/api/me")
    @require_login_api
    def api_me():
        user = session["user"]
        return jsonify({"id": user["id"], "username": user["username"], "display_name": user["display_name"], "role": user["role"]})

    @app.get("/api/analyses")
    @require_login_api
    def api_list_analyses():
        user = session["user"]
        if user["role"] == "teacher":
            rows = conn.execute(
                "SELECT analyses.*, users.display_name FROM analyses JOIN users ON users.id=analyses.user_id ORDER BY analyses.created_at DESC LIMIT 200"
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM analyses WHERE user_id=? ORDER BY created_at DESC LIMIT 200", (user["id"],)).fetchall()
        return jsonify([row_to_dict(r) for r in rows])

    @app.get("/api/analyses/<analysis_id>")
    @require_login_api
    def api_get_analysis(analysis_id: str):
        record = _get_analysis_or_404(analysis_id)
        payload: dict = {
            "analysis_id": record["id"],
            "status": record["status"],
            "progress": record["progress"],
            "standard_version": record.get("standard_version"),
        }
        if record["status"] == "failed":
            payload.update(
                {
                    "error_code": record.get("error_code"),
                    "error_message": record.get("error_message") or "分析失败，请按拍摄规范重新录制后上传",
                }
            )
            return jsonify(payload)

        if record["status"] == "succeeded":
            tips: list[str] = []
            diff_top: list[dict] = []
            keyframes: list[dict] = []
            result_path = record.get("result_json_path")
            evaluation: dict = {}
            view_angle: str = "front"
            view_label: str = "正面"
            compare_mode: str = "standard"
            compare_label: str = "标准动作"
            if result_path and Path(result_path).exists():
                try:
                    result_obj = json.loads(Path(result_path).read_text(encoding="utf-8"))
                    tips = list(result_obj.get("tips") or [])
                    diff_top = list(result_obj.get("diff_top") or [])
                    keyframes = list(result_obj.get("keyframes") or [])
                    evaluation = dict(result_obj.get("evaluation") or {})
                    view_angle = str(result_obj.get("view_angle") or "front")
                    view_label = str(result_obj.get("view_label") or "正面")
                    compare_mode = str(result_obj.get("compare_mode") or "standard")
                    compare_label = str(result_obj.get("compare_label") or "标准动作")
                except Exception:
                    tips = []
                    diff_top = []
                    keyframes = []
                    evaluation = {}
                    view_angle = "front"
                    view_label = "正面"
                    compare_mode = "standard"
                    compare_label = "标准动作"

            if not keyframes:
                keyframes = [
                    {
                        "key": "diff",
                        "label": "最大差异",
                        "time_ms": record.get("diff_time_ms") or 0,
                        "joint": record.get("diff_joint"),
                    }
                ]

            keyframes_payload: list[dict] = []
            for kf in keyframes:
                key = str(kf.get("key") or "")
                label = str(kf.get("label") or key or "关键帧")
                joint = kf.get("joint")
                keyframes_payload.append(
                    {
                        "key": key,
                        "label": label,
                        "time_ms": kf.get("time_ms"),
                        "joint": joint,
                        "joint_label": _joint_label(str(joint)) if joint else None,
                        "images": {
                            "standard": url_for("api_get_keyframe_image", analysis_id=analysis_id, kind="standard", frame=key),
                            "student": url_for("api_get_keyframe_image", analysis_id=analysis_id, kind="student", frame=key),
                        },
                    }
                )

            diff_top_payload: list[dict] = []
            for it in diff_top:
                joint = it.get("joint")
                diff_top_payload.append(
                    {
                        "joint": joint,
                        "joint_label": _joint_label(str(joint)) if joint else None,
                        "time_ms": it.get("time_ms"),
                        "max_diff_deg": it.get("max_diff_deg"),
                        "severity": it.get("severity"),
                        "phase": it.get("phase"),
                    }
                )
            payload.update(
                {
                    "score_total": record.get("score_total"),
                    "diff_joint": record.get("diff_joint"),
                    "diff_joint_label": _joint_label(record.get("diff_joint")),
                    "diff_time_ms": record.get("diff_time_ms"),
                    "evaluation": evaluation,
                    "view_angle": view_angle,
                    "view_label": view_label,
                    "compare_mode": compare_mode,
                    "compare_label": compare_label,
                    "tips": tips,
                    "diff_top": diff_top_payload,
                    "keyframes": keyframes_payload,
                    "videos": {
                        "standard": url_for("api_get_video", analysis_id=analysis_id, kind="standard"),
                        "student": url_for("api_get_video", analysis_id=analysis_id, kind="student"),
                    },
                }
            )
        return jsonify(payload)

    _keyframe_re = re.compile(r"^[a-z0-9_]{1,24}$")

    @app.get("/api/analyses/<analysis_id>/images/<kind>")
    @require_login_api
    def api_get_image(analysis_id: str, kind: str):
        record = _get_analysis_or_404(analysis_id)
        if record["status"] != "succeeded":
            abort(404)
        if kind == "standard":
            path = record.get("image_standard_path")
        elif kind == "student":
            path = record.get("image_student_path")
        else:
            abort(404)
        if not path:
            abort(404)
        file_path = Path(path)
        if not file_path.exists():
            abort(404)
        return send_file(file_path, mimetype="image/png")

    @app.get("/api/analyses/<analysis_id>/images/<kind>/<frame>")
    @require_login_api
    def api_get_keyframe_image(analysis_id: str, kind: str, frame: str):
        record = _get_analysis_or_404(analysis_id)
        if record["status"] != "succeeded":
            abort(404)
        if kind not in {"standard", "student"}:
            abort(404)
        frame_key = (frame or "").strip().lower()
        if not _keyframe_re.match(frame_key):
            abort(404)

        if frame_key == "diff":
            path = record.get("image_standard_path") if kind == "standard" else record.get("image_student_path")
            if not path:
                abort(404)
            file_path = Path(str(path))
        else:
            file_path = settings.results_dir / analysis_id / f"{kind}_{frame_key}.png"

        if not file_path.exists():
            abort(404)
        return send_file(file_path, mimetype="image/png")

    @app.get("/api/analyses/<analysis_id>/videos/<kind>")
    @require_login_api
    def api_get_video(analysis_id: str, kind: str):
        record = _get_analysis_or_404(analysis_id)
        if kind == "standard":
            # 优先使用分析记录中关联的标准视频
            std_video_id = record.get("standard_video_id")
            if std_video_id:
                std_row = conn.execute("SELECT file_path FROM standard_videos WHERE id=?", (std_video_id,)).fetchone()
                if std_row:
                    file_path = Path(std_row["file_path"])
                else:
                    file_path = settings.standard_video_path
            else:
                file_path = settings.standard_video_path
        elif kind == "student":
            file_path = Path(str(record.get("upload_path") or ""))
        else:
            abort(404)
        if not file_path.exists():
            abort(404)
        return send_file(file_path, mimetype="video/mp4", conditional=True)

    # ========== 标准视频管理 ==========
    
    @app.get("/standards")
    @require_teacher_page
    def standards_page():
        """标准视频管理页面"""
        return render_template("standards.html", user=session.get("user"))
    
    @app.get("/api/standards")
    @require_login_api
    def api_list_standards():
        """获取所有标准视频列表"""
        rows = conn.execute("""
            SELECT sv.*, u.display_name as uploader_name 
            FROM standard_videos sv 
            LEFT JOIN users u ON sv.uploaded_by = u.id 
            ORDER BY sv.view_angle, sv.created_at DESC
        """).fetchall()
        items = []
        for row in rows:
            item = row_to_dict(row)
            item["view_label"] = {"front": "正面", "side": "侧面", "angle": "斜侧面"}.get(item["view_angle"], item["view_angle"])
            items.append(item)
        return jsonify(items)
    
    @app.get("/api/standards/active")
    @require_login_api
    def api_get_active_standards():
        """获取当前激活的标准视频（按视角分类）"""
        rows = conn.execute("SELECT * FROM standard_videos WHERE is_active=1").fetchall()
        result = {}
        for row in rows:
            item = row_to_dict(row)
            result[item["view_angle"]] = item
        return jsonify(result)
    
    @app.post("/api/standards")
    @require_teacher_api
    def api_upload_standard():
        """上传新标准视频"""
        user = session["user"]
        
        # 获取视角类型
        view_angle = (request.form.get("view_angle") or "front").strip().lower()
        if view_angle not in ("front", "side", "angle"):
            return jsonify({"error_code": "INVALID_VIEW", "error_message": "视角类型无效，请选择 front/side/angle"}), 400
        
        # 获取视频名称
        name = (request.form.get("name") or "").strip()
        if not name:
            view_labels = {"front": "正面", "side": "侧面", "angle": "斜侧面"}
            name = f"{view_labels.get(view_angle, view_angle)}标准视频"
        
        # 检查文件
        file = request.files.get("file")
        if file is None or not file.filename:
            return jsonify({"error_code": "INVALID_MEDIA", "error_message": "请选择 MP4 视频文件"}), 400
        
        filename = Path(file.filename).name
        if not filename.lower().endswith(".mp4"):
            return jsonify({"error_code": "INVALID_MEDIA", "error_message": "仅支持 MP4 格式"}), 400
        
        # 保存文件
        settings.standard_videos_dir.mkdir(parents=True, exist_ok=True)
        file_id = uuid.uuid4().hex[:12]
        save_path = settings.standard_videos_dir / f"{view_angle}_{file_id}.mp4"
        file.save(save_path)
        
        # 计算版本哈希
        version = sha256_file(save_path)[:12]
        
        # 插入数据库
        conn.execute("""
            INSERT INTO standard_videos (name, view_angle, action, file_path, version, is_active, uploaded_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, view_angle, settings.action_default, str(save_path), version, 0, user["id"], utc_now_iso()))
        conn.commit()
        
        # 获取新插入的记录
        row = conn.execute("SELECT * FROM standard_videos WHERE version=? ORDER BY id DESC LIMIT 1", (version,)).fetchone()
        item = row_to_dict(row)
        item["view_label"] = {"front": "正面", "side": "侧面", "angle": "斜侧面"}.get(view_angle, view_angle)
        
        return jsonify({"ok": True, "standard": item})
    
    @app.post("/api/standards/<int:standard_id>/activate")
    @require_teacher_api
    def api_activate_standard(standard_id: int):
        """激活指定标准视频（作为该视角的当前标准）"""
        # 检查标准视频是否存在
        row = conn.execute("SELECT * FROM standard_videos WHERE id=?", (standard_id,)).fetchone()
        if not row:
            return jsonify({"error_code": "NOT_FOUND", "error_message": "标准视频不存在"}), 404
        
        view_angle = row["view_angle"]
        action = row["action"]
        
        # 先取消该视角+动作的所有激活
        conn.execute("UPDATE standard_videos SET is_active=0 WHERE view_angle=? AND action=?", (view_angle, action))
        # 激活指定的标准视频
        conn.execute("UPDATE standard_videos SET is_active=1 WHERE id=?", (standard_id,))
        conn.commit()
        
        view_labels = {"front": "正面", "side": "侧面", "angle": "斜侧面"}
        return jsonify({"ok": True, "message": f"已激活为{view_labels.get(view_angle, view_angle)}标准视频"})
    
    @app.delete("/api/standards/<int:standard_id>")
    @require_teacher_api
    def api_delete_standard(standard_id: int):
        """删除标准视频"""
        row = conn.execute("SELECT * FROM standard_videos WHERE id=?", (standard_id,)).fetchone()
        if not row:
            return jsonify({"error_code": "NOT_FOUND", "error_message": "标准视频不存在"}), 404
        
        # 解除分析记录对该标准视频的引用（允许删除被使用过的标准视频）
        conn.execute("UPDATE analyses SET standard_video_id=NULL WHERE standard_video_id=?", (standard_id,))
        
        # 删除文件
        file_path = Path(row["file_path"])
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass  # 文件删除失败不影响数据库删除
        
        # 删除缓存
        cache_dir = settings.standard_cache_dir / row["version"]
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
            except Exception:
                pass
        
        # 删除数据库记录
        conn.execute("DELETE FROM standard_videos WHERE id=?", (standard_id,))
        conn.commit()
        
        return jsonify({"ok": True})
    
    @app.get("/api/standards/<int:standard_id>/video")
    @require_login_api
    def api_get_standard_video(standard_id: int):
        """获取标准视频文件"""
        row = conn.execute("SELECT file_path FROM standard_videos WHERE id=?", (standard_id,)).fetchone()
        if not row:
            abort(404)
        file_path = Path(row["file_path"])
        if not file_path.exists():
            abort(404)
        return send_file(file_path, mimetype="video/mp4", conditional=True)

    return app


def _joint_label(joint: str | None) -> str | None:
    if not joint:
        return None
    mapping = {
        # 正面视角关节
        "left_elbow": "左肘",
        "right_elbow": "右肘",
        "left_shoulder": "左肩",
        "right_shoulder": "右肩",
        "left_hip": "左髋",
        "right_hip": "右髋",
        "left_knee": "左膝",
        "right_knee": "右膝",
        "torso": "躯干稳定性",
        # 侧面视角关节
        "elbow": "肘关节",
        "shoulder": "肩关节",
        "hip": "髋关节",
        "knee": "膝关节",
        "torso_forward": "躯干前后倾斜",
        "body_swing": "身体摆动",
        # 可见侧关节
        "visible_elbow": "可见肘",
        "visible_shoulder": "可见肩",
        "visible_hip": "可见髋",
        "visible_knee": "可见膝",
    }
    return mapping.get(joint, joint)


def main() -> None:
    app = create_app()
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")), debug=True)


if __name__ == "__main__":
    main()
