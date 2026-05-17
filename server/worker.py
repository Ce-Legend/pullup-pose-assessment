from __future__ import annotations

import os
import time
from pathlib import Path

from server.analysis.errors import (
    INTERNAL_ERROR,
    INVALID_DURATION,
    INVALID_MEDIA,
    AnalysisError,
    user_message_for,
)
from server.analysis.mp4 import Mp4ParseError, get_mp4_duration_ms
from server.analysis.pipeline import run_mock, run_real
from server.db import connect, init_schema, row_to_dict
from server.settings import load_settings
from server.util import sha256_file, utc_now_iso


def claim_one(conn) -> dict | None:
    row = conn.execute("SELECT * FROM analyses WHERE status='queued' ORDER BY created_at ASC LIMIT 1").fetchone()
    if row is None:
        return None
    analysis_id = row["id"]
    cur = conn.execute("UPDATE analyses SET status='running', progress=1 WHERE id=? AND status='queued'", (analysis_id,))
    conn.commit()
    if cur.rowcount != 1:
        return None
    return row_to_dict(row)


def update_progress(conn, analysis_id: str, progress: int) -> None:
    conn.execute("UPDATE analyses SET progress=? WHERE id=?", (int(progress), analysis_id))
    conn.commit()


def fail(conn, analysis_id: str, code: str, message: str) -> None:
    conn.execute(
        "UPDATE analyses SET status='failed', progress=100, finished_at=?, error_code=?, error_message=? WHERE id=?",
        (utc_now_iso(), code, message, analysis_id),
    )
    conn.commit()


def succeed(conn, analysis_id: str, *, duration_ms: int, standard_version: str, standard_video_id: int | None, artifacts) -> None:
    conn.execute(
        """
        UPDATE analyses SET
          status='succeeded',
          progress=100,
          finished_at=?,
          duration_ms=?,
          standard_version=?,
          standard_video_id=?,
          score_total=?,
          diff_joint=?,
          diff_time_ms=?,
          result_json_path=?,
          image_standard_path=?,
          image_student_path=?
        WHERE id=?
        """,
        (
            utc_now_iso(),
            int(duration_ms),
            standard_version,
            standard_video_id,
            int(artifacts.score_total),
            str(artifacts.diff_joint),
            int(artifacts.diff_time_ms),
            str(artifacts.result_json_path),
            str(artifacts.image_standard_path),
            str(artifacts.image_student_path),
            analysis_id,
        ),
    )
    conn.commit()


def get_standard_video_for_view(conn, settings, view_angle: str, action: str = "pullup") -> tuple[Path, str, int | None]:
    """
    根据视角获取对应的标准视频
    返回: (视频路径, 版本hash, 数据库ID)
    
    优先级：
    1. 数据库中该视角激活的标准视频
    2. 默认标准视频路径（settings.standard_video_path）
    """
    # 查找数据库中该视角激活的标准视频
    row = conn.execute(
        "SELECT * FROM standard_videos WHERE view_angle=? AND action=? AND is_active=1",
        (view_angle, action)
    ).fetchone()
    
    if row:
        file_path = Path(row["file_path"])
        if file_path.exists():
            return file_path, row["version"], row["id"]
    
    # 如果没有找到对应视角的标准视频，尝试正面的
    if view_angle != "front":
        row = conn.execute(
            "SELECT * FROM standard_videos WHERE view_angle='front' AND action=? AND is_active=1",
            (action,)
        ).fetchone()
        if row:
            file_path = Path(row["file_path"])
            if file_path.exists():
                print(f"[WARN] No standard video for {view_angle}, using front standard")
                return file_path, row["version"], row["id"]
    
    # 最后回退到默认配置
    if settings.standard_video_path.exists():
        version = sha256_file(settings.standard_video_path)[:12]
        return settings.standard_video_path, version, None
    
    return None, "", None


def process_one(conn, settings, job: dict) -> None:
    analysis_id = str(job["id"])
    upload_path = Path(str(job["upload_path"]))
    # 使用用户选择的视角（不再自动检测）
    selected_view = str(job.get("view") or "front")
    # 对比模式和历史分析ID
    compare_mode = str(job.get("compare_mode") or "standard")
    compare_analysis_id = job.get("compare_analysis_id")
    
    try:
        update_progress(conn, analysis_id, 5)

        print(f"[INFO] Processing {analysis_id} with view: {selected_view}, compare_mode: {compare_mode}")
        
        update_progress(conn, analysis_id, 10)

        # 根据对比模式获取对比视频
        if compare_mode == "history" and compare_analysis_id:
            # 历史对比模式：从历史分析记录获取视频
            history_row = conn.execute(
                "SELECT upload_path, created_at FROM analyses WHERE id=? AND status='succeeded'",
                (compare_analysis_id,)
            ).fetchone()
            if history_row is None:
                raise AnalysisError(INTERNAL_ERROR, "历史分析记录不存在或未完成")
            
            compare_video_path = Path(str(history_row["upload_path"]))
            if not compare_video_path.exists():
                raise AnalysisError(INTERNAL_ERROR, "历史视频文件已丢失")
            
            # 用历史分析的时间作为版本标识
            standard_version = f"history_{compare_analysis_id[:8]}"
            standard_video_id = None
            print(f"[INFO] Using history video: {compare_video_path} (history_id={compare_analysis_id})")
        else:
            # 标准对比模式：使用标准视频
            compare_video_path, standard_version, standard_video_id = get_standard_video_for_view(
                conn, settings, selected_view, settings.action_default
            )
            
            if compare_video_path is None or not compare_video_path.exists():
                view_labels = {"front": "正面", "side": "侧面", "angle": "斜侧面"}
                raise AnalysisError(
                    INTERNAL_ERROR,
                    f"未配置{view_labels.get(selected_view, selected_view)}标准视频，请在标准视频管理中上传",
                )
            print(f"[INFO] Using standard video: {compare_video_path} (version={standard_version}, id={standard_video_id})")
        
        conn.execute("UPDATE analyses SET standard_version=?, standard_video_id=? WHERE id=?", 
                     (standard_version, standard_video_id, analysis_id))
        conn.commit()

        update_progress(conn, analysis_id, 15)

        try:
            duration_ms = get_mp4_duration_ms(upload_path)
        except Mp4ParseError as e:
            raise AnalysisError(INVALID_MEDIA, user_message_for(INVALID_MEDIA)) from e

        if duration_ms < settings.video_min_sec * 1000 or duration_ms > settings.video_max_sec * 1000:
            raise AnalysisError(INVALID_DURATION, user_message_for(INVALID_DURATION))

        conn.execute("UPDATE analyses SET duration_ms=? WHERE id=?", (int(duration_ms), analysis_id))
        conn.commit()

        update_progress(conn, analysis_id, 20)

        out_dir = settings.results_dir / analysis_id
        out_dir.mkdir(parents=True, exist_ok=True)

        mock_mode = os.environ.get("MOCK_ANALYSIS", "0") == "1"
        if mock_mode:
            artifacts = run_mock(out_dir)
        else:
            # 对于历史对比，传入 compare_mode 以便 pipeline 调整输出标签
            artifacts = run_real(
                upload_path, compare_video_path, settings, out_dir, 
                view_angle=selected_view, compare_mode=compare_mode
            )

        update_progress(conn, analysis_id, 95)
        succeed(conn, analysis_id, duration_ms=duration_ms, standard_version=standard_version, 
                standard_video_id=standard_video_id, artifacts=artifacts)
        print(f"[OK] {analysis_id} view={selected_view} mode={compare_mode} score={artifacts.score_total} diff={artifacts.diff_joint}")

    except AnalysisError as e:
        fail(conn, analysis_id, e.error_code, e.error_message)
        print(f"[FAIL] {analysis_id} {e.error_code} {e.error_message}")
    except Exception as e:  # pragma: no cover
        fail(conn, analysis_id, INTERNAL_ERROR, user_message_for(INTERNAL_ERROR))
        print(f"[ERROR] {analysis_id} INTERNAL_ERROR {e}")


def main() -> None:
    settings = load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)

    conn = connect(settings.db_path)
    init_schema(conn, settings.repo_root / "server" / "schema.sql")

    poll_sec = float(os.environ.get("WORKER_POLL_SEC", "1.0"))
    print("worker started")
    print(f"db: {settings.db_path}")
    print(f"standard: {settings.standard_video_path}")
    print(f"mock: {os.environ.get('MOCK_ANALYSIS', '0')}")

    while True:
        job = claim_one(conn)
        if job is None:
            time.sleep(poll_sec)
            continue
        process_one(conn, settings, job)


if __name__ == "__main__":
    main()
