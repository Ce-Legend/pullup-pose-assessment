from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    data_dir: Path
    db_path: Path

    # 标准视频相关（支持多视角）
    standard_video_path: Path          # 默认标准视频路径（向后兼容）
    standard_videos_dir: Path          # 标准视频存储目录
    standard_cache_dir: Path

    uploads_dir: Path
    results_dir: Path

    action_default: str = "pullup"
    view_default: str = "front"

    video_min_sec: int = 1      # 最短1秒（支持快速完成的动作）
    video_max_sec: int = 300    # 最长5分钟（基本不限制）

    fps_sample: int = 12

    # 置信度阈值（适当放宽，提高真实场景成功率）
    pose_conf_mean_min: float = 0.25
    pose_low_ratio_max: float = 0.55


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = Path(os.environ.get("DATA_DIR", repo_root / "data")).resolve()
    db_path = Path(os.environ.get("DB_PATH", data_dir / "app.db")).resolve()

    # 标准视频配置
    standard_video_path = Path(os.environ.get("STANDARD_VIDEO", data_dir / "standard" / "standard.mp4")).resolve()
    standard_videos_dir = Path(os.environ.get("STANDARD_VIDEOS_DIR", data_dir / "standard")).resolve()
    standard_cache_dir = Path(os.environ.get("STANDARD_CACHE_DIR", data_dir / "cache" / "standard")).resolve()

    uploads_dir = Path(os.environ.get("UPLOADS_DIR", data_dir / "uploads")).resolve()
    results_dir = Path(os.environ.get("RESULTS_DIR", data_dir / "results")).resolve()

    return Settings(
        repo_root=repo_root,
        data_dir=data_dir,
        db_path=db_path,
        standard_video_path=standard_video_path,
        standard_videos_dir=standard_videos_dir,
        standard_cache_dir=standard_cache_dir,
        uploads_dir=uploads_dir,
        results_dir=results_dir,
    )

