from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from server.analysis.errors import AnalysisError, INTERNAL_ERROR, POSE_LOW_CONFIDENCE
from server.settings import Settings
from server.util import ensure_parent_dir, sha256_file


FEATURE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class AnalysisArtifacts:
    score_total: int
    diff_joint: str
    diff_time_ms: int
    tips: list[str]
    result_json_path: Path
    image_standard_path: Path
    image_student_path: Path


def run_mock(out_dir: Path) -> AnalysisArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)
    standard_png = out_dir / "standard.png"
    student_png = out_dir / "student.png"
    standard_start_png = out_dir / "standard_start.png"
    student_start_png = out_dir / "student_start.png"
    standard_peak_png = out_dir / "standard_peak.png"
    student_peak_png = out_dir / "student_peak.png"
    standard_end_png = out_dir / "standard_end.png"
    student_end_png = out_dir / "student_end.png"

    _write_placeholder_image(standard_png, "标准动作（最大差异示例）")
    _write_placeholder_image(student_png, "学员动作（最大差异示例）")
    _write_placeholder_image(standard_start_png, "标准动作（起始示例）")
    _write_placeholder_image(student_start_png, "学员动作（起始示例）")
    _write_placeholder_image(standard_peak_png, "标准动作（顶点示例）")
    _write_placeholder_image(student_peak_png, "学员动作（顶点示例）")
    _write_placeholder_image(standard_end_png, "标准动作（结束示例）")
    _write_placeholder_image(student_end_png, "学员动作（结束示例）")

    score_total = 86
    diff_joint = "left_elbow"
    diff_time_ms = 8200
    tips = [
        "整体完成不错，动作比较接近标准。",
        "上拉阶段屈肘还可以更充分，尝试把肘向下向后收紧。",
        "保持身体更稳定，减少摆动借力。",
    ]

    result_json_path = out_dir / "result.json"
    result_json_path.write_text(
        json.dumps(
            {
                "score_total": score_total,
                "diff_joint": diff_joint,
                "diff_time_ms": diff_time_ms,
                "diff_top": [
                    {"joint": diff_joint, "time_ms": diff_time_ms, "max_diff_deg": 18.0},
                    {"joint": "right_elbow", "time_ms": 7600, "max_diff_deg": 12.0},
                    {"joint": "left_shoulder", "time_ms": 8400, "max_diff_deg": 9.0},
                ],
                "keyframes": [
                    {"key": "start", "label": "起始（底部）", "time_ms": 0},
                    {"key": "peak", "label": "顶点（最高）", "time_ms": 5600},
                    {"key": "end", "label": "结束（回到底部）", "time_ms": 9800},
                    {"key": "diff", "label": "最大差异", "time_ms": diff_time_ms, "joint": diff_joint},
                ],
                "tips": tips,
                "mode": "mock",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return AnalysisArtifacts(
        score_total=score_total,
        diff_joint=diff_joint,
        diff_time_ms=diff_time_ms,
        tips=tips,
        result_json_path=result_json_path,
        image_standard_path=standard_png,
        image_student_path=student_png,
    )


def detect_video_view_angle(video_path: Path, settings: Settings) -> str:
    """
    快速检测视频的拍摄视角（只检测前几帧）
    返回: "front", "side", "angle"
    用于在分析前选择正确的标准视频
    """
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode  # type: ignore
        from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions  # type: ignore
    except Exception:
        return "front"  # 导入失败时默认正面
    
    model_path = _ensure_pose_model(settings)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "front"
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 只采样5帧进行检测
    sample_count = min(5, total_frames)
    sample_interval = max(1, total_frames // (sample_count + 1))
    
    view_votes = {"front": 0, "side": 0, "angle": 0}
    
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    
    try:
        landmarker = PoseLandmarker.create_from_options(options)
        
        for i in range(sample_count):
            frame_idx = (i + 1) * sample_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)
            
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lms = result.pose_landmarks[0]
                # 转换为字典格式
                lm_dict = {}
                landmark_names = ["nose", "left_eye_inner", "left_eye", "left_eye_outer",
                                  "right_eye_inner", "right_eye", "right_eye_outer",
                                  "left_ear", "right_ear", "mouth_left", "mouth_right",
                                  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                                  "left_wrist", "right_wrist", "left_pinky", "right_pinky",
                                  "left_index", "right_index", "left_thumb", "right_thumb",
                                  "left_hip", "right_hip", "left_knee", "right_knee",
                                  "left_ankle", "right_ankle", "left_heel", "right_heel",
                                  "left_foot_index", "right_foot_index"]
                for j, name in enumerate(landmark_names):
                    if j < len(lms):
                        lm_dict[name] = (float(lms[j].x * frame.shape[1]), float(lms[j].y * frame.shape[0]), float(lms[j].z))
                
                view = _detect_view_angle(lm_dict)
                view_votes[view] += 1
        
        landmarker.close()
    except Exception:
        pass
    finally:
        cap.release()
    
    # 返回得票最多的视角
    if sum(view_votes.values()) == 0:
        return "front"
    return max(view_votes, key=view_votes.get)


def run_real(student_video: Path, standard_video: Path, settings: Settings, out_dir: Path, view_angle: str = "front", compare_mode: str = "standard") -> AnalysisArtifacts:
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (  # type: ignore
            VisionTaskRunningMode,
        )
        from mediapipe.tasks.python.vision.pose_landmarker import (  # type: ignore
            PoseLandmarker,
            PoseLandmarkerOptions,
        )
    except Exception as e:  # pragma: no cover
        raise AnalysisError(POSE_LOW_CONFIDENCE, "姿态关键点检测失败或整体置信度过低，请按拍摄规范重新录制后上传") from e

    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = _ensure_pose_model(settings)

    standard_version = sha256_file(standard_video)[:12]
    cache_dir = settings.standard_cache_dir / standard_version
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_json = cache_dir / "standard_features.json"

    standard_data: dict[str, Any] = {}
    if cache_json.exists():
        standard_data = json.loads(cache_json.read_text(encoding="utf-8"))
        if int(standard_data.get("schema_version") or 0) != FEATURE_SCHEMA_VERSION:
            standard_data = {}

    if not standard_data:
        standard_data = _extract_features_to_json(
            mp=mp,
            BaseOptions=BaseOptions,
            VisionTaskRunningMode=VisionTaskRunningMode,
            PoseLandmarker=PoseLandmarker,
            PoseLandmarkerOptions=PoseLandmarkerOptions,
            cv2=cv2,
            model_path=model_path,
            video_path=standard_video,
            fps_sample=settings.fps_sample,
        )
        cache_json.write_text(json.dumps(standard_data, ensure_ascii=False), encoding="utf-8")

    student_data = _extract_features_to_json(
        mp=mp,
        BaseOptions=BaseOptions,
        VisionTaskRunningMode=VisionTaskRunningMode,
        PoseLandmarker=PoseLandmarker,
        PoseLandmarkerOptions=PoseLandmarkerOptions,
        cv2=cv2,
        model_path=model_path,
        video_path=student_video,
        fps_sample=settings.fps_sample,
    )

    _quality_check(student_data, settings)
    _quality_check(standard_data, settings)

    diff_joint, diff_time_ms, std_frame_index, stu_frame_index, diff_top, _ = _compare_and_locate(
        standard_data=standard_data,
        student_data=student_data,
    )
    # 使用用户选择的视角（不再依赖自动检测）
    print(f"[INFO] Using view angle: {view_angle}")

    # 获取评分和详细评估数据（传入视角）
    score_result = _score_total(standard_data=standard_data, student_data=student_data, view_angle=view_angle)
    score_total = score_result["score"]
    
    # 关键帧渲染（起始/顶点/结束/最大差异）
    standard_png = out_dir / "standard.png"
    student_png = out_dir / "student.png"
    standard_start_png = out_dir / "standard_start.png"
    student_start_png = out_dir / "student_start.png"
    standard_peak_png = out_dir / "standard_peak.png"
    student_peak_png = out_dir / "student_peak.png"
    standard_end_png = out_dir / "standard_end.png"
    student_end_png = out_dir / "student_end.png"

    std_frames_raw: list[dict[str, Any]] = list(standard_data.get("frames") or [])
    stu_frames_raw: list[dict[str, Any]] = list(student_data.get("frames") or [])
    
    # 裁剪掉"手不在杆上"的帧（松手/落地/跳起抓杠前的帧）
    std_frames = _trim_non_hanging_frames(std_frames_raw)
    stu_frames = _trim_non_hanging_frames(stu_frames_raw)
    
    std_peak_i = _peak_index(std_frames)
    stu_peak_i = _peak_index(stu_frames)

    # 辅助函数：获取帧的平均肘角
    def get_avg_elbow(f: dict[str, Any]) -> float:
        feats = f.get("features") or {}
        left = feats.get("left_elbow_angle")
        right = feats.get("right_elbow_angle")
        if left is not None and right is not None:
            return (float(left) + float(right)) / 2
        if left is not None:
            return float(left)
        if right is not None:
            return float(right)
        return 150.0

    # 辅助函数：基于肘角找到匹配帧
    def find_matching_frame(target_frame: dict[str, Any], candidates: list[dict[str, Any]]) -> int:
        target_elbow = get_avg_elbow(target_frame)
        best_i = 0
        best_diff = float("inf")
        for i, f in enumerate(candidates):
            diff = abs(get_avg_elbow(f) - target_elbow)
            if diff < best_diff:
                best_diff = diff
                best_i = i
        return best_i

    # 先确定学员的关键帧索引
    stu_start_i = _bottom_index(stu_frames[: stu_peak_i + 1])
    stu_end_i = stu_peak_i + _bottom_index(stu_frames[stu_peak_i:])
    
    # 然后基于学员帧的肘角，在标准视频中找匹配帧
    # 起始帧：在标准视频的上拉前段找匹配
    stu_start_frame = stu_frames[stu_start_i]
    std_start_i = find_matching_frame(stu_start_frame, std_frames[: std_peak_i + 1])
    
    # 结束帧：在标准视频的下放段找匹配
    stu_end_frame = stu_frames[stu_end_i]
    std_end_offset = find_matching_frame(stu_end_frame, std_frames[std_peak_i:])
    std_end_i = std_peak_i + std_end_offset
    
    # 下巴过杠状态从评分结果获取
    chin_over_bar = score_result.get("chin_over_bar", False)
    
    # 检测动作问题（替代原来的"差异对比"，传入视角）
    action_issues = _detect_action_issues(student_data, view_angle)
    
    has_kipping = score_result.get("has_kipping", False)
    tips = _make_tips(score_total=score_total, diff_joint=diff_joint, diff_top=None, chin_over_bar=chin_over_bar, view_angle=view_angle, has_kipping=has_kipping)

    std_start_frame = std_frames[std_start_i]
    std_peak_frame = std_frames[std_peak_i]
    stu_peak_frame = stu_frames[stu_peak_i]
    std_end_frame = std_frames[std_end_i]

    std_frame = _get_frame_near_index(standard_data, std_frame_index)
    stu_frame = _get_frame_near_index(student_data, stu_frame_index)

    # 根据对比模式决定标签
    compare_label = "历史动作" if compare_mode == "history" else "标准动作"
    current_label = "当前动作" if compare_mode == "history" else "学员动作"

    _render_keyframe(
        cv2=cv2,
        video_path=standard_video,
        frame_index=int(std_start_frame.get("frame_index") or 0),
        landmarks=_frame_landmarks_from_frame(std_start_frame),
        highlight_joint=diff_joint,
        out_path=standard_start_png,
        title=f"{compare_label}｜起始",
    )
    _render_keyframe(
        cv2=cv2,
        video_path=student_video,
        frame_index=int(stu_start_frame.get("frame_index") or 0),
        landmarks=_frame_landmarks_from_frame(stu_start_frame),
        highlight_joint=diff_joint,
        out_path=student_start_png,
        title=f"{current_label}｜起始",
    )

    _render_keyframe(
        cv2=cv2,
        video_path=standard_video,
        frame_index=int(std_peak_frame.get("frame_index") or 0),
        landmarks=_frame_landmarks_from_frame(std_peak_frame),
        highlight_joint=diff_joint,
        out_path=standard_peak_png,
        title=f"{compare_label}｜顶点",
    )
    _render_keyframe(
        cv2=cv2,
        video_path=student_video,
        frame_index=int(stu_peak_frame.get("frame_index") or 0),
        landmarks=_frame_landmarks_from_frame(stu_peak_frame),
        highlight_joint=diff_joint,
        out_path=student_peak_png,
        title=f"{current_label}｜顶点",
    )

    _render_keyframe(
        cv2=cv2,
        video_path=standard_video,
        frame_index=int(std_end_frame.get("frame_index") or 0),
        landmarks=_frame_landmarks_from_frame(std_end_frame),
        highlight_joint=diff_joint,
        out_path=standard_end_png,
        title=f"{compare_label}｜结束",
    )
    _render_keyframe(
        cv2=cv2,
        video_path=student_video,
        frame_index=int(stu_end_frame.get("frame_index") or 0),
        landmarks=_frame_landmarks_from_frame(stu_end_frame),
        highlight_joint=diff_joint,
        out_path=student_end_png,
        title=f"{current_label}｜结束",
    )

    _render_keyframe(
        cv2=cv2,
        video_path=standard_video,
        frame_index=int(std_frame.get("frame_index") or 0),
        landmarks=_frame_landmarks_from_frame(std_frame),
        highlight_joint=diff_joint,
        out_path=standard_png,
        title=compare_label,
    )
    _render_keyframe(
        cv2=cv2,
        video_path=student_video,
        frame_index=int(stu_frame.get("frame_index") or 0),
        landmarks=_frame_landmarks_from_frame(stu_frame),
        highlight_joint=diff_joint,
        out_path=student_png,
        title=current_label,
    )

    # 视角标签映射
    view_label = {
        "front": "正面",
        "side": "侧面",
        "angle": "斜侧面(45°)",
    }.get(view_angle, "正面")
    
    # 构建完整的评估数据（包含新增的多动作和分项评分字段）
    evaluation_data = {
        # 多动作评分
        "action_count": score_result.get("action_count", 1),
        "action_scores": score_result.get("action_scores", []),
        "score_breakdown": score_result.get("score_breakdown", {}),
        # 下巴过杠
        "chin_over_bar": score_result.get("chin_over_bar", False),
        "chin_gap_px": score_result.get("chin_gap_px", 0),
        "chin_detail": score_result.get("chin_detail", ""),
        # 手臂伸直
        "arm_straight": score_result.get("arm_straight", False),
        "elbow_angle": score_result.get("elbow_angle", 0),
        "arm_detail": score_result.get("arm_detail", ""),
        # 腿部并拢
        "legs_together": score_result.get("legs_together", False),
        "leg_distance_px": score_result.get("leg_distance_px", 0),
        "legs_detail": score_result.get("legs_detail", ""),
        # 身体稳定
        "body_stable": score_result.get("body_stable", False),
        "swing_amplitude": score_result.get("swing_amplitude", 0),
        "body_detail": score_result.get("body_detail", ""),
        # 摆浪蹬腿（新增）
        "has_kipping": score_result.get("has_kipping", False),
        "kipping_count": score_result.get("kipping_count", 0),
        "kipping_detail": score_result.get("kipping_detail", ""),
        # 评分说明
        "score_reason": score_result.get("score_reason", ""),
        "deductions": score_result.get("deductions", []),
    }
    
    result_json_path = out_dir / "result.json"
    result_json_path.write_text(
        json.dumps(
            {
                "score_total": score_total,
                "action_issues": action_issues,  # 动作问题分析（替代原来的diff_top）
                "evaluation": evaluation_data,
                "view_angle": view_angle,  # 视角类型：front/side/angle
                "view_label": view_label,      # 视角中文标签
                "compare_mode": compare_mode,  # 对比模式：standard/history
                "compare_label": compare_label,  # 对比标签：标准动作/历史动作
                "keyframes": [
                    {"key": "start", "label": "起始（底部）", "time_ms": int(stu_start_frame.get("t_ms") or 0)},
                    {"key": "peak", "label": "顶点（最高）", "time_ms": int(stu_peak_frame.get("t_ms") or 0)},
                    {"key": "end", "label": "结束（回到底部）", "time_ms": int(stu_end_frame.get("t_ms") or 0)},
                ],
                "tips": tips,
                "mode": "real",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return AnalysisArtifacts(
        score_total=score_total,
        diff_joint=diff_joint,
        diff_time_ms=diff_time_ms,
        tips=tips,
        result_json_path=result_json_path,
        image_standard_path=standard_png,
        image_student_path=student_png,
    )


def _write_placeholder_image(path: Path, title: str) -> None:
    img = Image.new("RGB", (960, 540), (18, 26, 51))
    draw = ImageDraw.Draw(img)
    draw.text((40, 40), title, fill=(231, 236, 255))
    draw.text((40, 90), "（MOCK 模式：未运行真实姿态分析）", fill=(154, 166, 209))
    img.save(path, format="PNG")


def _extract_features_to_json(
    *,
    mp: Any,
    BaseOptions: Any,
    VisionTaskRunningMode: Any,
    PoseLandmarker: Any,
    PoseLandmarkerOptions: Any,
    cv2: Any,
    model_path: Path,
    video_path: Path,
    fps_sample: int,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise AnalysisError(POSE_LOW_CONFIDENCE, "姿态关键点检测失败或整体置信度过低，请按拍摄规范重新录制后上传")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / max(1, fps_sample))))

    landmarker = _create_pose_landmarker(
        BaseOptions=BaseOptions,
        VisionTaskRunningMode=VisionTaskRunningMode,
        PoseLandmarker=PoseLandmarker,
        PoseLandmarkerOptions=PoseLandmarkerOptions,
        model_path=model_path,
    )

    frames: list[dict[str, Any]] = []
    frame_index = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_index % step != 0:
            frame_index += 1
            continue

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        t_ms = int((frame_index / fps) * 1000)
        res = landmarker.detect_for_video(mp_image, t_ms)

        frame: dict[str, Any] = {
            "frame_index": frame_index,
            "t_ms": t_ms,
            "conf": 0.0,
            "landmarks": {},
            "features": {},
        }

        if res.pose_landmarks:
            lms = res.pose_landmarks[0]
            lm = _landmark_map(lms, w=w, h=h)
            core = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_hip", "right_hip"]
            core_keys = [k for k in core if k in lm]
            conf_core = float(np.mean([lm[k][2] for k in core_keys])) if core_keys else 0.0
            conf_full = float(np.mean([lm[k][2] for k in lm.keys()])) if lm else 0.0
            frame["conf"] = conf_core
            frame["conf_full"] = conf_full
            frame["landmarks"] = {k: {"x": lm[k][0], "y": lm[k][1], "v": lm[k][2]} for k in lm.keys()}

            feats = _compute_features(lm)
            frame["features"] = feats
        frames.append(frame)
        frame_index += 1

    cap.release()
    landmarker.close()

    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "video": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "step": step,
        "frames": frames,
    }


def _landmark_map(lms: Any, *, w: int, h: int) -> dict[str, tuple[float, float, float]]:
    # MediaPipe PoseLandmarker（BlazePose 33点）索引
    idx = {
        "nose": 0,
        "left_eye": 2,
        "right_eye": 5,
        "left_ear": 7,
        "right_ear": 8,
        "mouth_left": 9,
        "mouth_right": 10,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot_index": 31,
        "right_foot_index": 32,
    }

    def get(i: int) -> tuple[float, float, float]:
        p = lms[int(i)]
        x = float(p.x) * w
        y = float(p.y) * h
        v = float(getattr(p, "presence", 0.0))
        return (x, y, v)

    return {k: get(i) for k, i in idx.items()}


def _ensure_pose_model(settings: Settings) -> Path:
    """
    MediaPipe Tasks 版 PoseLandmarker 需要 .task 模型文件。
    注意：MediaPipe 无法处理包含中文的路径，所以必须使用纯英文路径。
    优先从 AppData 目录加载，如不存在则从项目目录复制过去或从网络下载。
    """
    import shutil

    # 目标路径：必须是纯英文路径（AppData 或 TEMP）
    default_root = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or "C:\\ProgramData"
    target_path = Path(default_root) / "pullup_pose" / "models" / "pose_landmarker_lite.task"
    
    # 如果环境变量指定了路径，使用环境变量
    if os.environ.get("POSE_MODEL_PATH"):
        target_path = Path(os.environ["POSE_MODEL_PATH"]).resolve()

    # 1. 如果目标位置已有模型，直接返回
    if target_path.exists():
        return target_path

    # 2. 尝试从项目自带的模型复制过去
    bundled_model = settings.data_dir / "models" / "pose_landmarker_lite.task"
    if bundled_model.exists():
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundled_model, target_path)
            return target_path
        except Exception:
            pass  # 复制失败，继续尝试下载

    # 3. 尝试从网络下载
    url = os.environ.get(
        "POSE_MODEL_URL",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    )
    if os.environ.get("DISABLE_MODEL_DOWNLOAD", "0") == "1":
        raise AnalysisError(INTERNAL_ERROR, f"缺少姿态模型文件：{target_path}（请手动下载并放置，或开启自动下载）")

    try:
        import urllib.request

        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        with urllib.request.urlopen(url, timeout=30) as resp:  # nosec - URL 固定且可配置
            data = resp.read()
        tmp_path.write_bytes(data)
        tmp_path.replace(target_path)
        return target_path
    except Exception as e:  # pragma: no cover
        raise AnalysisError(
            INTERNAL_ERROR,
            f"下载姿态模型失败，请手动下载并放置到：{target_path}",
        ) from e


def _create_pose_landmarker(*, BaseOptions: Any, VisionTaskRunningMode: Any, PoseLandmarker: Any, PoseLandmarkerOptions: Any, model_path: Path) -> Any:
    base_options = BaseOptions(model_asset_path=str(model_path))
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def _angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    # 角 b，返回 0~180
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba == 0.0 or nbc == 0.0:
        return float("nan")
    cosang = float(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


# ========== 视角检测相关 ==========

def _detect_view_angle(lm: dict[str, tuple[float, float, float]]) -> str:
    """
    自动检测拍摄视角：front（正面）、side（侧面）、angle（斜侧面45°左右）
    
    核心原理：
    - 正面视角：左右肩膀的X坐标距离较大（肩宽明显）
    - 侧面视角：左右肩膀的X坐标距离很小（几乎重叠）
    - 45°斜侧面：介于两者之间
    
    同时检测是左侧还是右侧（用于侧面时选择哪边的关节）
    """
    ls = lm.get("left_shoulder", (0, 0, 0))
    rs = lm.get("right_shoulder", (0, 0, 0))
    lh = lm.get("left_hip", (0, 0, 0))
    rh = lm.get("right_hip", (0, 0, 0))
    
    # 计算肩膀宽度（X方向距离）
    shoulder_width = abs(ls[0] - rs[0])
    # 计算躯干高度（Y方向距离，从肩到髋）
    shoulder_center_y = (ls[1] + rs[1]) / 2
    hip_center_y = (lh[1] + rh[1]) / 2
    torso_height = abs(hip_center_y - shoulder_center_y)
    
    if torso_height < 10:  # 避免除零
        return "front"
    
    # 肩宽与躯干高度的比例
    # 正面：比例约 0.8-1.2（肩宽约等于躯干高度的80%-120%）
    # 侧面：比例约 0-0.2（肩宽很小，几乎是点）
    # 45°：比例约 0.3-0.6
    ratio = shoulder_width / torso_height
    
    if ratio >= 0.5:
        return "front"
    elif ratio <= 0.2:
        return "side"
    else:
        return "angle"  # 45°斜侧面


def _detect_side_direction(lm: dict[str, tuple[float, float, float]]) -> str:
    """
    检测侧面方向：left（从左侧拍摄，看到右侧身体）或 right（从右侧拍摄，看到左侧身体）
    
    原理：侧面时，靠近镜头的那一侧关节置信度更高，Z坐标更小
    """
    ls = lm.get("left_shoulder", (0, 0, 0))
    rs = lm.get("right_shoulder", (0, 0, 0))
    
    # Z坐标：越小表示越靠近镜头
    # 如果右肩Z更小，说明从右侧拍摄（看到的主要是左半边身体）
    # 如果左肩Z更小，说明从左侧拍摄（看到的主要是右半边身体）
    if rs[2] < ls[2]:
        return "right"  # 从右侧拍摄
    else:
        return "left"  # 从左侧拍摄


def _get_visible_side(lm: dict[str, tuple[float, float, float]], view: str) -> str:
    """
    获取可见的身体侧面（用于侧面视角时选择使用哪边的关节）
    返回 "left" 或 "right"
    """
    if view == "front":
        return "both"  # 正面两边都可见
    
    direction = _detect_side_direction(lm)
    # 从右侧拍摄看到左半边身体，从左侧拍摄看到右半边身体
    return "left" if direction == "right" else "right"


def _compute_features(lm: dict[str, tuple[float, float, float]]) -> dict[str, float]:
    """
    计算帧的特征，包括正面和侧面视角的特征
    """
    def pt(name: str) -> tuple[float, float]:
        return (lm[name][0], lm[name][1])
    
    def pt3(name: str) -> tuple[float, float, float]:
        return lm.get(name, (0, 0, 0))

    # ========== 基础角度特征（正面和侧面通用） ==========
    left_elbow = _angle(pt("left_shoulder"), pt("left_elbow"), pt("left_wrist"))
    right_elbow = _angle(pt("right_shoulder"), pt("right_elbow"), pt("right_wrist"))
    left_shoulder = _angle(pt("left_hip"), pt("left_shoulder"), pt("left_elbow"))
    right_shoulder = _angle(pt("right_hip"), pt("right_shoulder"), pt("right_elbow"))
    left_hip = _angle(pt("left_shoulder"), pt("left_hip"), pt("left_knee"))
    right_hip = _angle(pt("right_shoulder"), pt("right_hip"), pt("right_knee"))
    left_knee = _angle(pt("left_hip"), pt("left_knee"), pt("left_ankle"))
    right_knee = _angle(pt("right_hip"), pt("right_knee"), pt("right_ankle"))

    shoulder_center = ((pt("left_shoulder")[0] + pt("right_shoulder")[0]) / 2, (pt("left_shoulder")[1] + pt("right_shoulder")[1]) / 2)
    hip_center = ((pt("left_hip")[0] + pt("right_hip")[0]) / 2, (pt("left_hip")[1] + pt("right_hip")[1]) / 2)

    vx = shoulder_center[0] - hip_center[0]
    vy = shoulder_center[1] - hip_center[1]
    # 躯干左右倾斜（正面视角主要指标）
    torso_lean = float(abs(math.degrees(math.atan2(vx, -vy)))) if (vx != 0 or vy != 0) else 0.0

    nose_y = float(pt("nose")[1])
    
    # 多点融合的面部最高点（侧面时鼻子可能被手遮挡，用耳朵/眼睛补充）
    face_y_candidates = [nose_y]
    for fkey in ("left_ear", "right_ear", "left_eye", "right_eye"):
        if fkey in lm:
            fy = float(lm[fkey][1])
            fv = float(lm[fkey][2])
            if fy > 0 and fv > 0.2:
                face_y_candidates.append(fy)
    face_top_y = min(face_y_candidates)  # y最小 = 最高位置
    
    # ========== 视角检测 ==========
    view_angle = _detect_view_angle(lm)
    visible_side = _get_visible_side(lm, view_angle)
    
    # ========== 侧面视角特有特征 ==========
    
    # 1. 躯干前后倾斜角度（侧面核心指标）
    # 从侧面看，躯干应该保持垂直，前倾或后仰都是问题
    # 使用肩膀中心和髋部中心的连线与垂直线的夹角
    # 侧面时，X方向的偏移代表前后倾斜
    torso_forward_lean = 0.0
    if view_angle in ("side", "angle"):
        # 侧面时，躯干前后倾斜 = 肩膀X相对于髋部X的偏移
        torso_height = abs(vy) if vy != 0 else 1
        forward_offset = vx  # 正值=前倾，负值=后仰
        torso_forward_lean = float(math.degrees(math.atan2(forward_offset, torso_height)))
    
    # 2. 身体摆动幅度（通过肩膀和髋部的相对位置判断）
    # 侧面时，如果身体前后摆动，肩膀X会明显偏离髋部X
    body_swing = abs(torso_forward_lean)  # 摆动幅度就是前后倾斜的绝对值
    
    # 3. 可见侧肘角（侧面时应该用可见的那一侧）
    if visible_side == "left":
        visible_elbow_angle = left_elbow
        visible_shoulder_angle = left_shoulder
        visible_hip_angle = left_hip
        visible_knee_angle = left_knee
    elif visible_side == "right":
        visible_elbow_angle = right_elbow
        visible_shoulder_angle = right_shoulder
        visible_hip_angle = right_hip
        visible_knee_angle = right_knee
    else:  # both
        visible_elbow_angle = (left_elbow + right_elbow) / 2 if not (math.isnan(left_elbow) or math.isnan(right_elbow)) else left_elbow if not math.isnan(left_elbow) else right_elbow
        visible_shoulder_angle = (left_shoulder + right_shoulder) / 2 if not (math.isnan(left_shoulder) or math.isnan(right_shoulder)) else left_shoulder
        visible_hip_angle = (left_hip + right_hip) / 2 if not (math.isnan(left_hip) or math.isnan(right_hip)) else left_hip
        visible_knee_angle = (left_knee + right_knee) / 2 if not (math.isnan(left_knee) or math.isnan(right_knee)) else left_knee
    
    # 4. 肘部前后位置（侧面时肘部不应该过于前伸或后摆）
    elbow_forward_offset = 0.0
    if view_angle in ("side", "angle"):
        # 使用可见侧的肘部
        if visible_side == "left":
            elbow_x = pt("left_elbow")[0]
            shoulder_x = pt("left_shoulder")[0]
        elif visible_side == "right":
            elbow_x = pt("right_elbow")[0]
            shoulder_x = pt("right_shoulder")[0]
        else:
            elbow_x = (pt("left_elbow")[0] + pt("right_elbow")[0]) / 2
            shoulder_x = shoulder_center[0]
        elbow_forward_offset = elbow_x - shoulder_x  # 正值=肘向前，负值=肘向后

    return {
        # 基础角度（双侧）
        "left_elbow_angle": float(left_elbow),
        "right_elbow_angle": float(right_elbow),
        "left_shoulder_angle": float(left_shoulder),
        "right_shoulder_angle": float(right_shoulder),
        "left_hip_angle": float(left_hip),
        "right_hip_angle": float(right_hip),
        "left_knee_angle": float(left_knee),
        "right_knee_angle": float(right_knee),
        # 躯干相关
        "torso_lean": torso_lean,  # 左右倾斜（正面）
        "torso_forward_lean": torso_forward_lean,  # 前后倾斜（侧面）
        "body_swing": body_swing,  # 身体摆动幅度
        # 其他
        "nose_y": nose_y,
        "face_top_y": face_top_y,  # 多点融合的面部最高y（侧面时更可靠）
        # 视角信息
        "view_angle": 0 if view_angle == "front" else (1 if view_angle == "side" else 0.5),  # 编码为数值
        # 可见侧特征（侧面时使用）
        "visible_elbow_angle": float(visible_elbow_angle) if not math.isnan(visible_elbow_angle) else 150.0,
        "visible_shoulder_angle": float(visible_shoulder_angle) if not math.isnan(visible_shoulder_angle) else 90.0,
        "visible_hip_angle": float(visible_hip_angle) if not math.isnan(visible_hip_angle) else 170.0,
        "visible_knee_angle": float(visible_knee_angle) if not math.isnan(visible_knee_angle) else 170.0,
        # 肘部前后偏移（侧面）
        "elbow_forward_offset": elbow_forward_offset,
    }


def _quality_check(data: dict[str, Any], settings: Settings) -> None:
    frames = data.get("frames") or []
    if len(frames) < 10:
        raise AnalysisError(POSE_LOW_CONFIDENCE, "姿态关键点检测失败或整体置信度过低，请按拍摄规范重新录制后上传")
    confs = np.array([float(f.get("conf") or 0.0) for f in frames], dtype=float)
    mean_conf = float(np.mean(confs))
    low_ratio = float(np.mean(confs < 0.2))
    if mean_conf < settings.pose_conf_mean_min or low_ratio > settings.pose_low_ratio_max:
        raise AnalysisError(POSE_LOW_CONFIDENCE, "姿态关键点检测失败或整体置信度过低，请按拍摄规范重新录制后上传")


def _get_wrist_y(frame: dict[str, Any]) -> float:
    """获取帧的平均手腕y坐标"""
    lms = frame.get("landmarks") or {}
    lw = lms.get("left_wrist", {})
    rw = lms.get("right_wrist", {})
    lw_y = float(lw.get("y", 9999))
    rw_y = float(rw.get("y", 9999))
    if lw_y > 9000 and rw_y > 9000:
        return 9999.0
    if lw_y > 9000:
        return rw_y
    if rw_y > 9000:
        return lw_y
    return (lw_y + rw_y) / 2


def _get_wrist_position(frame: dict[str, Any]) -> tuple[float, float]:
    """获取帧的平均手腕位置 (x, y)"""
    lms = frame.get("landmarks") or {}
    lw = lms.get("left_wrist", {})
    rw = lms.get("right_wrist", {})
    lw_x = float(lw.get("x", 9999))
    lw_y = float(lw.get("y", 9999))
    rw_x = float(rw.get("x", 9999))
    rw_y = float(rw.get("y", 9999))
    
    # 计算有效的平均值
    valid_x, valid_y = [], []
    if lw_x < 9000:
        valid_x.append(lw_x)
        valid_y.append(lw_y)
    if rw_x < 9000:
        valid_x.append(rw_x)
        valid_y.append(rw_y)
    
    if not valid_x:
        return 9999.0, 9999.0
    return sum(valid_x) / len(valid_x), sum(valid_y) / len(valid_y)


def _is_hanging(frame: dict[str, Any], max_ankle_y: float = 0) -> bool:
    """
    判断该帧是否为悬垂状态（手在杆上）
    
    核心判断：悬垂时脚是离地悬空的，站立时脚在地面上
    最可靠的特征：踝关节的y坐标！
    - 悬垂时：踝关节不在画面最底部
    - 站立时：踝关节在地面上，y坐标很大（接近画面底部）
    
    max_ankle_y: 用于判断"脚在地面"的阈值，由外部传入
    """
    lms = frame.get("landmarks") or {}
    
    # 获取关键点
    lw = lms.get("left_wrist", {})
    rw = lms.get("right_wrist", {})
    la = lms.get("left_ankle", {})
    ra = lms.get("right_ankle", {})
    ls = lms.get("left_shoulder", {})
    rs = lms.get("right_shoulder", {})
    
    # 计算平均位置
    wrist_y = (float(lw.get("y", 0)) + float(rw.get("y", 0))) / 2
    ankle_y = (float(la.get("y", 0)) + float(ra.get("y", 0))) / 2
    shoulder_y = (float(ls.get("y", 0)) + float(rs.get("y", 0))) / 2
    
    if wrist_y <= 0 or ankle_y <= 0:
        return True  # 数据不完整，默认认为是悬垂
    
    # 关键判断1：手腕是否在肩膀上方（悬垂时手在杠上）
    wrist_above_shoulder = wrist_y < shoulder_y + 20  # 允许一点容差
    
    # 简化判断：只检查手腕是否在肩膀上方
    # 
    # 原因：之前的 "foot_on_ground" 判断有 bug：
    # 如果视频全程都是悬垂的，max_ankle_y 就是底部悬垂时的位置，
    # 导致底部帧（手臂伸直）被误判为 "脚在地面"
    # 
    # 新策略：只用 "手腕在肩膀上方" 判断。
    # "站立举手" 的短暂时刻会被 "最长连续区间" 过滤掉。
    # 
    # 悬垂 = 手腕在肩膀上方（允许一些容差）
    return wrist_above_shoulder


def _trim_non_hanging_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    裁剪掉"手不在杆上"的帧，只保留引体向上动作部分
    
    核心逻辑：用 _is_hanging() 判断每一帧是否为悬垂状态，
    只保留悬垂状态的帧。关键判断是脚是否在地面上。
    """
    if len(frames) < 10:
        return frames
    
    # 首先计算所有帧的踝关节y坐标，找到最大值（地面位置）
    ankle_ys = []
    for f in frames:
        lms = f.get("landmarks") or {}
        la = lms.get("left_ankle", {})
        ra = lms.get("right_ankle", {})
        la_y = float(la.get("y", 0))
        ra_y = float(ra.get("y", 0))
        if la_y > 0 and ra_y > 0:
            ankle_ys.append((la_y + ra_y) / 2)
        elif la_y > 0:
            ankle_ys.append(la_y)
        elif ra_y > 0:
            ankle_ys.append(ra_y)
    
    if not ankle_ys:
        return frames
    
    # 最大踝关节y坐标 = 地面位置
    max_ankle_y = max(ankle_ys)
    print(f"[DEBUG] max_ankle_y = {max_ankle_y:.1f}")
    
    # 判断每一帧是否为悬垂状态
    hanging_flags = [_is_hanging(f, max_ankle_y) for f in frames]
    
    # 打印调试信息
    hanging_count = sum(hanging_flags)
    print(f"[DEBUG] hanging frames: {hanging_count} / {len(frames)}")
    
    if hanging_count < 10:
        # 悬垂帧太少，可能是判断太严格，放宽条件
        print(f"[WARN] Only {hanging_count} hanging frames detected, using all frames")
        return frames
    
    # ========== 第一步：找出所有连续悬垂区间 ==========
    segments = []  # [(start, end), ...]
    current_start = -1
    for i, is_hang in enumerate(hanging_flags):
        if is_hang:
            if current_start < 0:
                current_start = i
        else:
            if current_start >= 0:
                segments.append((current_start, i))
                current_start = -1
    if current_start >= 0:
        segments.append((current_start, len(frames)))
    
    if not segments:
        print(f"[WARN] No hanging segments found, using all frames")
        return frames
    
    print(f"[DEBUG] Found {len(segments)} raw hanging segments: {segments}")
    
    # ========== 第二步：合并间隙 ≤ GAP_TOLERANCE 帧的相邻区间 ==========
    # 引体过程中偶尔1-2帧误判为"非悬垂"（头倾斜、遮挡等），不应打断区间
    GAP_TOLERANCE = 5  # 允许的最大间隙帧数
    
    merged = [segments[0]]
    for seg_start, seg_end in segments[1:]:
        prev_start, prev_end = merged[-1]
        gap = seg_start - prev_end
        if gap <= GAP_TOLERANCE:
            # 间隙很小，合并（填充中间的"非悬垂"帧）
            merged[-1] = (prev_start, seg_end)
            print(f"[DEBUG] Merged segments: gap={gap} frames, now ({prev_start}, {seg_end})")
        else:
            merged.append((seg_start, seg_end))
    
    print(f"[DEBUG] After merging: {len(merged)} segments: {merged}")
    
    # ========== 第三步：选择最长的合并区间 ==========
    best_start, best_end = max(merged, key=lambda s: s[1] - s[0])
    best_len = best_end - best_start
    
    # 确保至少保留10帧
    if best_len < 10:
        print(f"[WARN] Longest merged hanging segment too small ({best_len}), using all frames")
        return frames
    
    print(f"[INFO] Trimmed frames: {best_start} to {best_end} (len={best_len}, original: 0 to {len(frames)})")
    return frames[best_start:best_end]


def _peak_index(frames: list[dict[str, Any]]) -> int:
    """
    找到顶点帧（面部最高 = y坐标最小）
    优先使用多点融合的 face_top_y（侧面时鼻子可能被手遮挡）
    注意：调用前应先用 _trim_non_hanging_frames 裁剪掉非悬垂帧
    """
    nose_y = []
    for f in frames:
        feats = f.get("features") or {}
        ny = feats.get("face_top_y") or feats.get("nose_y")
        nose_y.append(float(ny) if ny is not None else float("nan"))
    arr = np.array(nose_y, dtype=float)
    
    if not np.isfinite(arr).any():
        raise AnalysisError(POSE_LOW_CONFIDENCE, "姿态关键点检测失败或整体置信度过低，请按拍摄规范重新录制后上传")
    
    return int(np.nanargmin(arr))


def _bottom_index(frames: list[dict[str, Any]]) -> int:
    """
    找到底部帧（悬垂状态，手臂完全伸直）
    核心判断：肘角最大的帧（手臂最直，接近180°）
    """
    # 用肘角判断（手臂最直 = 肘角最大 = 悬垂底部）
    elbow_angles = []
    for f in frames:
        feats = f.get("features") or {}
        left = feats.get("left_elbow_angle")
        right = feats.get("right_elbow_angle")
        
        # 取两边肘角的最大值（更严格地找手臂伸直的帧）
        angles = []
        if left is not None:
            angles.append(float(left))
        if right is not None:
            angles.append(float(right))
        
        if angles:
            elbow_angles.append(sum(angles) / len(angles))
        else:
            elbow_angles.append(float("nan"))
    
    arr = np.array(elbow_angles, dtype=float)
    if np.isfinite(arr).any():
        # 肘角最大 = 手臂最直 = 底部
        best_idx = int(np.nanargmax(arr))
        print(f"[DEBUG] _bottom_index: max_elbow={arr[best_idx]:.1f}° at frame {best_idx}, range=[{np.nanmin(arr):.1f}°, {np.nanmax(arr):.1f}°]")
        return best_idx
    
    # 备用：用鼻子最低
    nose_y = []
    for f in frames:
        feats = f.get("features") or {}
        ny = feats.get("nose_y")
        nose_y.append(float(ny) if ny is not None else float("nan"))
    arr = np.array(nose_y, dtype=float)
    
    if not np.isfinite(arr).any():
        return 0
    
    return int(np.nanargmax(arr))


def _detect_all_peaks(frames: list[dict[str, Any]], min_distance: int = 10) -> list[int]:
    """
    检测所有顶点帧（鼻子位置的局部最小值）
    
    使用双信号融合策略提高检测准确率：
    1. 主信号：鼻子y坐标（顶点=最高=y最小）
    2. 辅助信号：平均肘角（顶点=弯曲最大=肘角最小）
    3. 用 scipy.signal.find_peaks 做鲁棒的峰值检测
    4. 两个信号互相验证，减少漏检和误检
    
    min_distance: 两个顶点之间的最小帧数间隔
    返回：顶点帧索引列表
    """
    from scipy.signal import find_peaks
    from scipy.ndimage import uniform_filter1d
    
    if len(frames) < 10:
        return [_peak_index(frames)]
    
    # ========== 提取双信号 ==========
    nose_y_raw = []
    elbow_raw = []
    for f in frames:
        feats = f.get("features") or {}
        # 面部最高y（优先使用多点融合值，侧面时鼻子可能被手遮挡）
        ny = feats.get("face_top_y") or feats.get("nose_y")
        nose_y_raw.append(float(ny) if ny is not None else float("nan"))
        # 肘角（左右平均）
        le = feats.get("left_elbow_angle")
        re = feats.get("right_elbow_angle")
        angles = []
        if le is not None and not math.isnan(float(le)):
            angles.append(float(le))
        if re is not None and not math.isnan(float(re)):
            angles.append(float(re))
        elbow_raw.append(sum(angles) / len(angles) if angles else float("nan"))
    
    nose_arr = np.array(nose_y_raw, dtype=float)
    elbow_arr = np.array(elbow_raw, dtype=float)
    
    # ========== 插值填充 NaN ==========
    x = np.arange(len(nose_arr))
    
    nose_mask = np.isfinite(nose_arr)
    if nose_mask.sum() < 5:
        return [_peak_index(frames)]
    nose_filled = np.interp(x, x[nose_mask], nose_arr[nose_mask])
    
    elbow_mask = np.isfinite(elbow_arr)
    if elbow_mask.sum() >= 3:
        elbow_filled = np.interp(x, x[elbow_mask], elbow_arr[elbow_mask])
    else:
        elbow_filled = None
    
    # ========== 平滑（小窗口，保留细节） ==========
    nose_smooth = uniform_filter1d(nose_filled, size=3)  # 缩小平滑窗口：5→3
    if elbow_filled is not None:
        elbow_smooth = uniform_filter1d(elbow_filled, size=3)
    else:
        elbow_smooth = None
    
    # ========== 信号1：鼻子y最小值检测 ==========
    # 取反（find_peaks找最大值，我们要找最小值）
    nose_inv = -nose_smooth
    
    # 动态计算 prominence（基于信号整体波动幅度）
    nose_range = np.max(nose_smooth) - np.min(nose_smooth)
    # prominence 设为总幅度的 10%，最低8像素（比原来的20px更宽松）
    nose_prominence = max(8, nose_range * 0.10)
    
    nose_peaks, nose_props = find_peaks(
        nose_inv,
        distance=max(5, min_distance - 2),  # 允许稍近的峰（比 min_distance 宽松一点）
        prominence=nose_prominence,
    )
    
    print(f"[DEBUG] nose signal: range={nose_range:.1f}px, prominence_threshold={nose_prominence:.1f}px, "
          f"found {len(nose_peaks)} peaks at {nose_peaks.tolist()}")
    
    # ========== 信号2：肘角最小值检测（辅助） ==========
    elbow_peaks_set = set()
    if elbow_smooth is not None:
        elbow_inv = -elbow_smooth  # 肘角最小 = 弯曲最大 = 顶点
        elbow_range = np.max(elbow_smooth) - np.min(elbow_smooth)
        # 肘角的 prominence 设为总幅度的 15%，最低10°
        elbow_prominence = max(10, elbow_range * 0.15)
        
        elbow_peaks_arr, elbow_props = find_peaks(
            elbow_inv,
            distance=max(5, min_distance - 2),
            prominence=elbow_prominence,
        )
        elbow_peaks_set = set(elbow_peaks_arr.tolist())
        
        print(f"[DEBUG] elbow signal: range={elbow_range:.1f}°, prominence_threshold={elbow_prominence:.1f}°, "
              f"found {len(elbow_peaks_arr)} peaks at {elbow_peaks_arr.tolist()}")
    
    # ========== 双信号融合 ==========
    # 策略：以鼻子信号为主，肘角信号辅助补充
    # 1. 鼻子信号检测到的 → 直接采纳
    # 2. 肘角信号检测到、但鼻子信号没检测到的 → 验证后补充
    
    confirmed_peaks = list(nose_peaks)
    
    # 补充：肘角检测到但鼻子没检测到的峰
    if elbow_smooth is not None:
        for ep in elbow_peaks_set:
            # 检查这个肘角峰是否已被鼻子峰覆盖（±min_distance范围内）
            already_covered = any(abs(ep - np) <= min_distance for np in confirmed_peaks)
            if not already_covered:
                # 验证：在鼻子信号中，这个位置附近是否确实有下降趋势（哪怕不够显著）
                # 放宽条件：只要鼻子y在这附近低于前后各3帧的均值就算
                window = max(3, min_distance // 2)
                before_mean = np.mean(nose_smooth[max(0, ep - window):ep]) if ep > 0 else nose_smooth[ep]
                after_mean = np.mean(nose_smooth[ep + 1:min(len(nose_smooth), ep + window + 1)]) if ep < len(nose_smooth) - 1 else nose_smooth[ep]
                if nose_smooth[ep] < before_mean and nose_smooth[ep] < after_mean:
                    confirmed_peaks.append(ep)
                    print(f"[DEBUG] Elbow-assisted peak added at frame {ep} (nose_y={nose_smooth[ep]:.1f}, "
                          f"before_avg={before_mean:.1f}, after_avg={after_mean:.1f})")
    
    # 排序
    confirmed_peaks.sort()
    
    # ========== 合并太近的顶点 ==========
    merged_peaks = []
    for p in confirmed_peaks:
        if not merged_peaks or p - merged_peaks[-1] >= min_distance:
            merged_peaks.append(p)
        else:
            # 取鼻子位置更高（y更小）的那个
            if nose_smooth[p] < nose_smooth[merged_peaks[-1]]:
                merged_peaks[-1] = p
    
    # ========== 边界检查：首尾是否遗漏动作 ==========
    
    # --- 情况A：信号起点本身就在高位（人已经在顶点/正在下落） ---
    # 判定条件：开头几帧的nose_y明显低于后续的底部值
    if len(nose_smooth) > min_distance * 2:
        # 开头5帧的平均值
        head_avg = np.mean(nose_smooth[:min(5, len(nose_smooth))])
        # 开头到第一个已检测峰之间的最大值（底部）
        first_peak = merged_peaks[0] if merged_peaks else len(nose_smooth) // 2
        bottom_region = nose_smooth[:first_peak]
        if len(bottom_region) > 5:
            bottom_max = np.max(bottom_region)
            drop = bottom_max - head_avg  # 从开头到底部的下降幅度
            # 如果开头比底部高至少 nose_prominence * 0.5 像素，说明开头就是一个峰
            if drop > nose_prominence * 0.5:
                # 在前几帧中找到nose_y最小的（最高位置）作为峰值
                head_search = min(first_peak, min_distance + 3)
                head_peak_idx = int(np.argmin(nose_smooth[:head_search]))
                # 确保和已有的峰不太近
                if not merged_peaks or (merged_peaks[0] - head_peak_idx >= min_distance):
                    merged_peaks.insert(0, head_peak_idx)
                    print(f"[DEBUG] Start-boundary peak: frame {head_peak_idx} "
                          f"(nose_y={nose_smooth[head_peak_idx]:.1f}, drop_to_bottom={drop:.1f}px)")
    
    # --- 情况B：在已检测的第一个峰之前，用 find_peaks 找更多峰 ---
    if merged_peaks and merged_peaks[0] > min_distance + 3:
        head_segment = nose_inv[:merged_peaks[0]]
        if len(head_segment) > 3:
            head_peaks, _ = find_peaks(head_segment, prominence=nose_prominence * 0.5)  # 更宽松
            for hp in sorted(head_peaks):
                hp_int = int(hp)
                if not merged_peaks or (merged_peaks[0] - hp_int >= min_distance):
                    merged_peaks.insert(0, hp_int)
                    print(f"[DEBUG] Head find_peaks peak at frame {hp_int}")
    
    # --- 情况C：在最后一个峰之后，用 find_peaks 找更多峰 ---
    if merged_peaks and merged_peaks[-1] < len(nose_smooth) - min_distance - 3:
        tail_start = merged_peaks[-1] + 1
        tail_segment = nose_inv[tail_start:]
        if len(tail_segment) > 3:
            tail_peaks, _ = find_peaks(tail_segment, prominence=nose_prominence * 0.5)
            for tp in sorted(tail_peaks):
                abs_idx = int(tail_start + tp)
                if abs_idx - merged_peaks[-1] >= min_distance:
                    merged_peaks.append(abs_idx)
                    print(f"[DEBUG] Tail find_peaks peak at frame {abs_idx}")
    
    # --- 情况D：信号尾部本身就在高位 ---
    if len(nose_smooth) > min_distance * 2:
        tail_avg = np.mean(nose_smooth[-min(5, len(nose_smooth)):])
        last_peak = merged_peaks[-1] if merged_peaks else len(nose_smooth) // 2
        bottom_after = nose_smooth[last_peak:]
        if len(bottom_after) > 5:
            bottom_max_tail = np.max(bottom_after)
            drop_tail = bottom_max_tail - tail_avg
            if drop_tail > nose_prominence * 0.5:
                tail_search_start = max(last_peak, len(nose_smooth) - min_distance - 3)
                tail_peak_idx = int(np.argmin(nose_smooth[tail_search_start:])) + tail_search_start
                if tail_peak_idx - merged_peaks[-1] >= min_distance:
                    merged_peaks.append(tail_peak_idx)
                    print(f"[DEBUG] End-boundary peak: frame {tail_peak_idx} "
                          f"(nose_y={nose_smooth[tail_peak_idx]:.1f}, drop={drop_tail:.1f}px)")
    
    # 如果还是空的，用全局最小值
    if not merged_peaks:
        merged_peaks = [int(np.argmin(nose_smooth))]
    
    print(f"[DEBUG] Final detected {len(merged_peaks)} peaks at frames: {merged_peaks}")
    return merged_peaks


def _detect_action_cycles(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    检测所有完整的动作周期
    一个完整动作 = 底部(手臂伸直) → 顶点(下巴过杠) → 底部(手臂伸直)
    
    返回：动作列表，每个动作包含 {
        "start_idx": 起始帧索引,
        "peak_idx": 顶点帧索引,
        "end_idx": 结束帧索引,
        "start_frame": 起始帧数据,
        "peak_frame": 顶点帧数据,
        "end_frame": 结束帧数据,
    }
    """
    if len(frames) < 15:
        # 帧数太少，当作单个动作处理
        peak_i = _peak_index(frames)
        return [{
            "start_idx": 0,
            "peak_idx": peak_i,
            "end_idx": len(frames) - 1,
            "start_frame": frames[0],
            "peak_frame": frames[peak_i],
            "end_frame": frames[-1],
            "action_num": 1,
        }]
    
    # 获取肘角数组（用于找底部）
    elbow_angles = []
    for f in frames:
        feats = f.get("features") or {}
        left = feats.get("left_elbow_angle")
        right = feats.get("right_elbow_angle")
        angles = []
        if left is not None and not math.isnan(float(left)):
            angles.append(float(left))
        if right is not None and not math.isnan(float(right)):
            angles.append(float(right))
        elbow_angles.append(sum(angles) / len(angles) if angles else 150.0)
    
    elbow_arr = np.array(elbow_angles, dtype=float)
    
    # 检测所有顶点（min_distance=6: 在12fps下约0.5秒，对快节奏也能检测）
    peaks = _detect_all_peaks(frames, min_distance=6)
    
    if len(peaks) == 0:
        return []
    
    actions = []
    
    for i, peak_i in enumerate(peaks):
        # 找这个顶点前的底部（上拉起始）
        if peak_i > 5:
            search_start = peaks[i - 1] if i > 0 else 0
            segment = elbow_arr[search_start:peak_i]
            if len(segment) > 0:
                start_offset = int(np.argmax(segment))
                start_i = search_start + start_offset
            else:
                start_i = max(0, peak_i - 5)
        else:
            start_i = 0
        
        # 找这个顶点后的底部（下放结束）
        if peak_i < len(frames) - 5:
            search_end = peaks[i + 1] if i < len(peaks) - 1 else len(frames)
            segment = elbow_arr[peak_i:search_end]
            if len(segment) > 0:
                end_offset = int(np.argmax(segment))
                end_i = peak_i + end_offset
            else:
                end_i = min(len(frames) - 1, peak_i + 5)
        else:
            end_i = len(frames) - 1
        
        # 确保索引有效
        start_i = max(0, min(start_i, len(frames) - 1))
        end_i = max(0, min(end_i, len(frames) - 1))
        
        actions.append({
            "start_idx": start_i,
            "peak_idx": peak_i,
            "end_idx": end_i,
            "start_frame": frames[start_i],
            "peak_frame": frames[peak_i],
            "end_frame": frames[end_i],
            "action_num": i + 1,
        })
    
    print(f"[DEBUG] Detected {len(actions)} complete actions")
    return actions


def _score_single_action(action: dict[str, Any], all_frames: list[dict[str, Any]], view_angle: str = "front") -> dict[str, Any]:
    """
    对单个动作进行评分 - 按照老师的档位标准（视角自适应）
    
    老师的评分标准（档位制）：
    - 90分以上：手臂放直 + 下巴过杠 + 腿并拢 + 无大幅晃动
    - 80分：手臂放直 + 下巴过杠 + 腿没完全并拢（但身体稳定）
    - 70分：手臂放直 + 下巴过杠 + 腿没完全并拢但身体晃动大 或 腿并拢但晃动大
    - 60分：手臂伸直 + 下巴过杠 + 又晃动腿又分开
    - 60以下：手臂弯曲 或 下巴不过杠
    
    各指标阈值按视角校准（基于老师三个方向标准视频提取的数据）。
    """
    peak_frame = action["peak_frame"]
    start_frame = action["start_frame"]
    end_frame = action["end_frame"]
    
    start_idx = action["start_idx"]
    end_idx = action["end_idx"]
    action_frames = all_frames[start_idx:end_idx + 1] if end_idx > start_idx else [peak_frame]
    
    # ========== 检测四个关键指标（视角自适应） ==========
    
    # 1. 下巴是否过杠（顶点帧检测，视角自适应）
    chin_over, chin_gap = _check_chin_over_bar(peak_frame, view_angle)
    
    # 2. 手臂是否伸直（底部帧检测）
    if view_angle in ("side", "angle"):
        # 侧面/斜面：使用可见侧肘角（避免被遮挡侧的错误数据拉低平均值）
        # 老师侧面标准 visible_elbow max=179.9°
        start_feats = start_frame.get("features") or {}
        end_feats = end_frame.get("features") or {}
        start_elbow = float(start_feats.get("visible_elbow_angle", 0) or 0)
        end_elbow = float(end_feats.get("visible_elbow_angle", 0) or 0)
        # 备选：如果 visible 为0，用左右平均
        if start_elbow < 10:
            sl = float(start_feats.get("left_elbow_angle", 0) or 0)
            sr = float(start_feats.get("right_elbow_angle", 0) or 0)
            start_elbow = (sl + sr) / 2 if sl > 0 and sr > 0 else max(sl, sr)
        if end_elbow < 10:
            el = float(end_feats.get("left_elbow_angle", 0) or 0)
            er = float(end_feats.get("right_elbow_angle", 0) or 0)
            end_elbow = (el + er) / 2 if el > 0 and er > 0 else max(el, er)
    else:
        # 正面：左右平均
        start_feats = start_frame.get("features") or {}
        end_feats = end_frame.get("features") or {}
        
        start_left = float(start_feats.get("left_elbow_angle", 0) or 0)
        start_right = float(start_feats.get("right_elbow_angle", 0) or 0)
        start_elbow = (start_left + start_right) / 2 if start_left > 0 and start_right > 0 else max(start_left, start_right)
        
        end_left = float(end_feats.get("left_elbow_angle", 0) or 0)
        end_right = float(end_feats.get("right_elbow_angle", 0) or 0)
        end_elbow = (end_left + end_right) / 2 if end_left > 0 and end_right > 0 else max(end_left, end_right)
    
    bottom_elbow = max(start_elbow, end_elbow)
    arm_straight = bottom_elbow >= 165
    
    # 3. 腿是否并拢（全程检测）- 视角自适应阈值
    _, leg_dist = _check_legs_together(action_frames, view_angle)
    # 阈值已在 _check_legs_together 内部按视角设定
    LEGS_THRESHOLDS = {"front": 65, "side": 45, "angle": 45}
    legs_together = leg_dist < LEGS_THRESHOLDS.get(view_angle, 65)
    
    # 4. 身体是否稳定（全程检测）- 视角自适应阈值
    _, swing = _check_body_swing(action_frames, view_angle)
    SWING_THRESHOLDS = {"front": 4, "side": 30, "angle": 22}
    body_stable = swing < SWING_THRESHOLDS.get(view_angle, 4)
    
    # 5. 摆浪蹬腿借力检测（新增，优先级最高）
    is_kipping, hip_swing_ratio, knee_std = _check_kipping(action_frames, view_angle)
    
    # ========== 按老师档位标准评分 + 档位内客观数据细分 ==========
    deductions = []
    bonus_details = []  # 记录加分项
    penalty_details = []  # 记录减分项
    
    # ================================================================
    # 评分档位（按教学场景确认的标准）：
    #   90+   : 5项全部达标
    #   80-90 : 腿或身体违反其中一项（仅此两项）
    #   70-80 : 腿+身体两项都违反
    #   60-70 : 有蹬腿摆浪（不管腿是否并拢）
    #   <60   : 下巴过杠/手臂伸直 任一不达标
    # 原则：违反指标越多，评分越低
    # ================================================================
    
    # ===== 档位1：摆浪蹬腿 → 60-70分 =====
    # 只要有摆浪，不管腿是否并拢，都是60-70档（需下巴过杠+手臂伸直）
    if is_kipping and chin_over and arm_straight:
        score = 63  # 基础分
        deductions.append("身体大幅摆浪蹬腿借力（动作依靠惯性完成，非肌肉主导发力）")
        
        # 摆浪严重程度（越轻微越高）
        HIP_THRESHOLDS = {"front": 0.35, "side": 0.80, "angle": 0.35}
        ht = HIP_THRESHOLDS.get(view_angle, 0.35)
        if hip_swing_ratio < ht * 1.2 and knee_std < 12:
            score += 3
            bonus_details.append("摆浪较轻+3")
        elif hip_swing_ratio < ht * 1.5 and knee_std < 18:
            score += 1
            bonus_details.append("摆浪中等+1")
        elif hip_swing_ratio >= ht * 2.0:
            score -= 2
            penalty_details.append("摆浪严重-2")
        
        # 其他指标加减分（违反越多越低）
        if legs_together:
            score += 1
            bonus_details.append("腿并拢+1")
        else:
            deductions.append(f"双腿未完全并拢（间距 {leg_dist:.0f}px）")
        if body_stable:
            score += 1
            bonus_details.append("身体稳定+1")
        else:
            deductions.append(f"身体晃动（晃动 {swing:.1f}°）")
        if not legs_together and not body_stable:
            score -= 1
            penalty_details.append("多项不达标-1")
        
        score = max(60, min(70, score))
        score_reason = f"摆浪蹬腿借力（{score}分）"
    
    # ===== 档位2：基础条件不满足 → <60分 =====
    elif not chin_over or not arm_straight:
        base_score = 40
        
        if not chin_over:
            deductions.append(f"下巴未过杠（低于杠位 {chin_gap:.0f}px）")
            if chin_gap < 10:
                base_score += 6
                bonus_details.append(f"接近过杠+6")
            elif chin_gap < 25:
                base_score += 4
                bonus_details.append(f"差距较小+4")
            elif chin_gap < 40:
                base_score += 2
                bonus_details.append(f"差距中等+2")
        else:
            base_score += 8
            bonus_details.append(f"下巴过杠+8")
        
        if not arm_straight:
            deductions.append(f"手臂未完全伸直（肘角 {bottom_elbow:.0f}°，标准≥165°）")
            if bottom_elbow >= 160:
                base_score += 6
                bonus_details.append(f"手臂接近伸直+6")
            elif bottom_elbow >= 150:
                base_score += 4
                bonus_details.append(f"手臂略弯+4")
            elif bottom_elbow >= 140:
                base_score += 2
                bonus_details.append(f"手臂弯曲+2")
        else:
            base_score += 8
            bonus_details.append(f"手臂伸直+8")
        
        if is_kipping:
            deductions.append("身体摆浪蹬腿借力")
            base_score -= 3
        
        score = min(58, base_score)
        score_reason = f"基础条件不满足（{score}分）"
    
    # ===== 档位3：全部达标 → 90-100分 =====
    elif legs_together and body_stable:
        score = 90
        
        # 手臂伸直程度（最多+3）
        if bottom_elbow >= 178:
            score += 3
            bonus_details.append(f"手臂完美{bottom_elbow:.0f}°+3")
        elif bottom_elbow >= 175:
            score += 2
            bonus_details.append(f"手臂很直{bottom_elbow:.0f}°+2")
        elif bottom_elbow >= 170:
            score += 1
            bonus_details.append(f"手臂较直{bottom_elbow:.0f}°+1")
        
        # 身体稳定程度（最多+3）
        if swing < 1.5:
            score += 3
            bonus_details.append(f"极稳{swing:.1f}°+3")
        elif swing < 2.5:
            score += 2
            bonus_details.append(f"很稳{swing:.1f}°+2")
        elif swing < 3.5:
            score += 1
            bonus_details.append(f"稳定{swing:.1f}°+1")
        
        # 腿部并拢程度（最多+2）
        if leg_dist < 35:
            score += 2
            bonus_details.append(f"腿紧并{leg_dist:.0f}px+2")
        elif leg_dist < 50:
            score += 1
            bonus_details.append(f"腿并拢{leg_dist:.0f}px+1")
        
        # 下巴过杠幅度（最多+2）
        if chin_gap < -50:
            score += 2
            bonus_details.append(f"大幅过杠{abs(chin_gap):.0f}px+2")
        elif chin_gap < -25:
            score += 1
            bonus_details.append(f"明显过杠{abs(chin_gap):.0f}px+1")
        
        score = min(100, score)
        score_reason = f"动作标准（90+{score-90}分）"
    
    # ===== 档位4：腿或身体违反其中一项 → 80-90分 =====
    elif legs_together != body_stable:
        # 恰好一项不达标（legs_together XOR body_stable）
        score = 80
        
        if not legs_together:
            deductions.append(f"双腿未完全并拢（间距 {leg_dist:.0f}px，标准<65px）")
            # 腿部分开程度
            if leg_dist < 75:
                score += 4
                bonus_details.append(f"腿接近并拢{leg_dist:.0f}px+4")
            elif leg_dist < 90:
                score += 2
                bonus_details.append(f"腿略分开{leg_dist:.0f}px+2")
            elif leg_dist >= 120:
                score -= 2
                penalty_details.append(f"腿分开较大{leg_dist:.0f}px-2")
        
        if not body_stable:
            deductions.append(f"身体晃动（晃动 {swing:.1f}°）")
            # 晃动程度
            if swing < 6:
                score += 4
                bonus_details.append(f"晃动较小{swing:.1f}°+4")
            elif swing < 9:
                score += 2
                bonus_details.append(f"晃动中等{swing:.1f}°+2")
            elif swing >= 15:
                score -= 2
                penalty_details.append(f"晃动很大{swing:.1f}°-2")
        
        # 手臂伸直程度
        if bottom_elbow >= 175:
            score += 2
            bonus_details.append(f"手臂很直{bottom_elbow:.0f}°+2")
        elif bottom_elbow >= 170:
            score += 1
            bonus_details.append(f"手臂较直{bottom_elbow:.0f}°+1")
        
        # 达标项的质量加分
        if legs_together and leg_dist < 40:
            score += 1
            bonus_details.append(f"腿很并拢+1")
        if body_stable and swing < 2:
            score += 1
            bonus_details.append(f"很稳定+1")
        
        score = max(80, min(90, score))
        failed_item = "腿部" if not legs_together else "稳定性"
        score_reason = f"{failed_item}需改进（{score}分）"
    
    # ===== 档位5：腿+身体两项都违反 → 70-80分 =====
    else:
        score = 70
        deductions.append(f"双腿未完全并拢（间距 {leg_dist:.0f}px，标准<65px）")
        deductions.append(f"身体晃动（晃动 {swing:.1f}°）")
        
        # 腿部接近程度
        if leg_dist < 80:
            score += 3
            bonus_details.append(f"腿接近并拢{leg_dist:.0f}px+3")
        elif leg_dist < 100:
            score += 1
            bonus_details.append(f"腿略分开{leg_dist:.0f}px+1")
        elif leg_dist >= 130:
            score -= 2
            penalty_details.append(f"腿分开很大{leg_dist:.0f}px-2")
        
        # 晃动程度
        if swing < 6:
            score += 2
            bonus_details.append(f"晃动较小{swing:.1f}°+2")
        elif swing < 10:
            score += 1
            bonus_details.append(f"晃动中等{swing:.1f}°+1")
        elif swing >= 15:
            score -= 2
            penalty_details.append(f"晃动很大{swing:.1f}°-2")
        
        # 手臂伸直加分
        if bottom_elbow >= 175:
            score += 2
            bonus_details.append(f"手臂很直{bottom_elbow:.0f}°+2")
        elif bottom_elbow >= 170:
            score += 1
            bonus_details.append(f"手臂较直{bottom_elbow:.0f}°+1")
        
        # 下巴过杠加分
        if chin_gap < -30:
            score += 1
            bonus_details.append(f"过杠明显+1")
        
        score = max(70, min(80, score))
        score_reason = f"腿部和稳定性需改进（{score}分）"
    
    # ========== 构建返回数据 ==========
    # 获取顶点屈肘角度（用于展示）
    peak_feats = peak_frame.get("features") or {}
    peak_left = peak_feats.get("left_elbow_angle", 90)
    peak_right = peak_feats.get("right_elbow_angle", 90)
    peak_elbow = (float(peak_left or 90) + float(peak_right or 90)) / 2
    elbow_range = bottom_elbow - peak_elbow
    
    # 生成详情描述
    chin_detail = f"过杠（高于杠位 {abs(chin_gap):.0f}px）" if chin_gap < 0 else f"未过杠（低于杠位 {chin_gap:.0f}px）"
    arm_detail = f"伸直（肘角 {bottom_elbow:.0f}°）" if arm_straight else f"弯曲（肘角 {bottom_elbow:.0f}°）"
    leg_detail = f"并拢（间距 {leg_dist:.0f}px）" if legs_together else f"分开（间距 {leg_dist:.0f}px）"
    swing_detail = f"稳定（晃动 {swing:.1f}°）" if body_stable else f"晃动（晃动 {swing:.1f}°）"
    
    # 构建评分细节说明
    score_detail_parts = []
    if bonus_details:
        score_detail_parts.append("加分：" + "、".join(bonus_details))
    if penalty_details:
        score_detail_parts.append("减分：" + "、".join(penalty_details))
    score_detail = "；".join(score_detail_parts) if score_detail_parts else ""
    
    # 摆浪蹬腿详情
    kipping_detail = ""
    if is_kipping:
        parts = []
        if hip_swing_ratio > 0.2:
            parts.append(f"身体摆幅 {hip_swing_ratio:.2f}")
        if knee_std > 8:
            parts.append(f"蹬腿幅度 {knee_std:.1f}°")
        kipping_detail = "摆浪蹬腿借力（" + "、".join(parts) + "）" if parts else "摆浪蹬腿借力"
    else:
        kipping_detail = "无摆浪蹬腿"
    
    return {
        "action_num": action["action_num"],
        "total": score,
        "score_reason": score_reason,
        "score_detail": score_detail,
        "deductions": deductions,
        # 五个关键指标（新增摆浪蹬腿）
        "chin_over": chin_over,
        "arm_straight": arm_straight,
        "legs_together": legs_together,
        "body_stable": body_stable,
        "is_kipping": is_kipping,
        # 详细数据（客观数值）
        "chin_gap": round(chin_gap, 0),
        "chin_detail": chin_detail,
        "bottom_elbow": round(bottom_elbow, 0),
        "arm_detail": arm_detail,
        "leg_dist": round(leg_dist, 0),
        "leg_detail": leg_detail,
        "swing": round(swing, 1),
        "swing_detail": swing_detail,
        "hip_swing_ratio": round(hip_swing_ratio, 3),
        "knee_angle_std": round(knee_std, 1),
        "kipping_detail": kipping_detail,
        "peak_elbow": round(peak_elbow, 0),
        "elbow_range": round(elbow_range, 0),
        # 分项详情（用于前端展示）
        "details": {
            "chin": {"passed": chin_over, "detail": chin_detail},
            "arm": {"passed": arm_straight, "detail": arm_detail},
            "legs": {"passed": legs_together, "detail": leg_detail},
            "swing": {"passed": body_stable, "detail": swing_detail},
            "kipping": {"passed": not is_kipping, "detail": kipping_detail},
        }
    }


def _get_feature_array(frames: list[dict[str, Any]], name: str) -> np.ndarray:
    vals = []
    for f in frames:
        feats = f.get("features") or {}
        v = feats.get(name)
        vals.append(float(v) if v is not None else float("nan"))
    return np.array(vals, dtype=float)


def _fill_and_resample(values: np.ndarray, out_len: int) -> np.ndarray | None:
    x = np.arange(len(values), dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return None
    filled = np.interp(x, x[mask], values[mask])
    out_x = np.linspace(0, len(values) - 1, out_len, dtype=float)
    return np.interp(out_x, x, filled)


def _compare_and_locate(
    *, standard_data: dict[str, Any], student_data: dict[str, Any]
) -> tuple[str, int, int, int, list[dict[str, Any]], str]:
    """
    对比标准视频和学员视频，定位最大差异点
    返回: (最大差异关节, 差异时间ms, 标准帧索引, 学员帧索引, Top3差异列表, 视角类型)
    """
    std_frames_raw: list[dict[str, Any]] = list(standard_data.get("frames") or [])
    stu_frames_raw: list[dict[str, Any]] = list(student_data.get("frames") or [])

    # 裁剪掉"手不在杆上"的帧
    std_frames = _trim_non_hanging_frames(std_frames_raw)
    stu_frames = _trim_non_hanging_frames(stu_frames_raw)
    
    # 检测视角
    stu_view = _detect_video_view(stu_frames)
    print(f"[DEBUG] _compare_and_locate detected view: {stu_view}")
    
    std_peak = _peak_index(std_frames)
    stu_peak = _peak_index(stu_frames)

    std_pre = std_frames[: std_peak + 1]
    std_post = std_frames[std_peak:]
    stu_pre = stu_frames[: stu_peak + 1]
    stu_post = stu_frames[stu_peak:]

    if min(len(std_pre), len(std_post), len(stu_pre), len(stu_post)) < 5:
        raise AnalysisError(POSE_LOW_CONFIDENCE, "姿态关键点检测失败或整体置信度过低，请按拍摄规范重新录制后上传")

    out_len = 50
    
    # 根据视角选择要比较的关节指标
    if stu_view == "front":
        # 正面视角：双侧关节
        joints = {
            "left_elbow": "left_elbow_angle",
            "right_elbow": "right_elbow_angle",
            "left_shoulder": "left_shoulder_angle",
            "right_shoulder": "right_shoulder_angle",
            "left_hip": "left_hip_angle",
            "right_hip": "right_hip_angle",
            "left_knee": "left_knee_angle",
            "right_knee": "right_knee_angle",
        }
        extra_features = {
            "torso": "torso_lean",
        }
    elif stu_view == "side":
        # 纯侧面视角：可见侧关节 + 前后稳定性
        joints = {
            "elbow": "visible_elbow_angle",
            "shoulder": "visible_shoulder_angle",
            "hip": "visible_hip_angle",
            "knee": "visible_knee_angle",
        }
        extra_features = {
            "torso_forward": "torso_forward_lean",  # 前后倾斜
            "body_swing": "body_swing",              # 身体摆动
        }
    else:  # angle (45°)
        # 斜侧面：混合
        joints = {
            "elbow": "visible_elbow_angle",
            "left_elbow": "left_elbow_angle",
            "right_elbow": "right_elbow_angle",
            "shoulder": "visible_shoulder_angle",
        }
        extra_features = {
            "torso": "torso_lean",
            "torso_forward": "torso_forward_lean",
        }

    # 拼接成 100 点（上拉 50 + 下放 50）
    diffs: dict[str, np.ndarray] = {}
    
    # 处理关节角度
    for joint, feat in joints.items():
        std_pre_r = _fill_and_resample(_get_feature_array(std_pre, feat), out_len)
        stu_pre_r = _fill_and_resample(_get_feature_array(stu_pre, feat), out_len)
        std_post_r = _fill_and_resample(_get_feature_array(std_post, feat), out_len)
        stu_post_r = _fill_and_resample(_get_feature_array(stu_post, feat), out_len)
        if std_pre_r is None or stu_pre_r is None or std_post_r is None or stu_post_r is None:
            continue

        diffs[joint] = np.concatenate([np.abs(stu_pre_r - std_pre_r), np.abs(stu_post_r - std_post_r)])

    # 处理额外指标（躯干稳定性等）
    for name, feat in extra_features.items():
        std_pre_r = _fill_and_resample(_get_feature_array(std_pre, feat), out_len)
        stu_pre_r = _fill_and_resample(_get_feature_array(stu_pre, feat), out_len)
        std_post_r = _fill_and_resample(_get_feature_array(std_post, feat), out_len)
        stu_post_r = _fill_and_resample(_get_feature_array(stu_post, feat), out_len)
        if std_pre_r is None or stu_pre_r is None or std_post_r is None or stu_post_r is None:
            continue
        diffs[name] = np.concatenate([np.abs(stu_pre_r - std_pre_r), np.abs(stu_post_r - std_post_r)])

    if not diffs:
        raise AnalysisError(POSE_LOW_CONFIDENCE, "姿态关键点检测失败或整体置信度过低，请按拍摄规范重新录制后上传")

    # 获取肘角数组用于动作匹配
    def get_avg_elbow(f: dict[str, Any]) -> float:
        feats = f.get("features") or {}
        left = feats.get("left_elbow_angle")
        right = feats.get("right_elbow_angle")
        if left is not None and right is not None:
            return (float(left) + float(right)) / 2
        if left is not None:
            return float(left)
        if right is not None:
            return float(right)
        return 150.0  # 默认值

    def find_matching_std_frame(stu_frame: dict[str, Any], std_frames_list: list[dict[str, Any]]) -> int:
        """基于肘角找到标准视频中最匹配的帧"""
        stu_elbow = get_avg_elbow(stu_frame)
        best_i = 0
        best_diff = float("inf")
        for i, std_f in enumerate(std_frames_list):
            std_elbow = get_avg_elbow(std_f)
            diff = abs(std_elbow - stu_elbow)
            if diff < best_diff:
                best_diff = diff
                best_i = i
        return best_i

    # 存储重采样后的数据（用于获取学员角度和标准角度）
    # 同时存储 joints 和 extra_features 的数据
    resampled_data: dict[str, dict[str, np.ndarray]] = {}
    all_features = {**joints, **extra_features}  # 合并 joints 和 extra_features
    for joint, feat in all_features.items():
        std_pre_r = _fill_and_resample(_get_feature_array(std_pre, feat), out_len)
        stu_pre_r = _fill_and_resample(_get_feature_array(stu_pre, feat), out_len)
        std_post_r = _fill_and_resample(_get_feature_array(std_post, feat), out_len)
        stu_post_r = _fill_and_resample(_get_feature_array(stu_post, feat), out_len)
        if std_pre_r is not None and stu_pre_r is not None and std_post_r is not None and stu_post_r is not None:
            resampled_data[joint] = {
                "std": np.concatenate([std_pre_r, std_post_r]),
                "stu": np.concatenate([stu_pre_r, stu_post_r]),
            }

    # 找到最大差异（joint + time index）
    def map_idx(idx: int) -> tuple[int, int, int]:
        # 映射到原始帧（学员/标准）
        # 改进：使用肘角匹配而不是简单的线性插值
        if idx < out_len:
            t = idx / max(1, out_len - 1)
            stu_i = int(round(t * (len(stu_pre) - 1)))
            stu_frame = stu_pre[stu_i]
            # 在标准视频的同阶段（上拉）中找匹配帧
            std_i = find_matching_std_frame(stu_frame, std_pre)
            std_frame_index = int(std_pre[std_i]["frame_index"])
            stu_frame_index = int(stu_frame["frame_index"])
            diff_time_ms = int(stu_frame["t_ms"])
        else:
            t = (idx - out_len) / max(1, out_len - 1)
            stu_i = int(round(t * (len(stu_post) - 1)))
            stu_frame = stu_post[stu_i]
            # 在标准视频的同阶段（下放）中找匹配帧
            std_i = find_matching_std_frame(stu_frame, std_post)
            std_frame_index = int(std_post[std_i]["frame_index"])
            stu_frame_index = int(stu_frame["frame_index"])
            diff_time_ms = int(stu_frame["t_ms"])
        return std_frame_index, stu_frame_index, diff_time_ms

    def severity(deg: float) -> str:
        """差异严重程度判断"""
        if deg >= 25:
            return "高"
        if deg >= 15:
            return "中"
        return "低"

    def get_joint_category(joint: str) -> str:
        """获取关节类别（用于合并左右同类关节）"""
        # 移除 left_/right_ 前缀
        if joint.startswith("left_"):
            return joint[5:]
        if joint.startswith("right_"):
            return joint[6:]
        return joint

    # 收集所有差异数据
    all_items: list[dict[str, Any]] = []
    best_joint = "left_elbow"
    best_val = -1.0
    best_idx = 0
    
    for joint, arr in diffs.items():
        idx = int(np.argmax(arr))
        val = float(arr[idx])
        
        # 只记录差异 > 10° 的关节（过滤小差异）
        if val < 10:
            continue
        
        std_frame_index, stu_frame_index, diff_time_ms = map_idx(idx)
        
        # 获取学员角度和标准角度
        student_angle = None
        standard_angle = None
        if joint in resampled_data:
            student_angle = round(float(resampled_data[joint]["stu"][idx]), 0)
            standard_angle = round(float(resampled_data[joint]["std"][idx]), 0)
        
        item = {
            "joint": joint,
            "joint_category": get_joint_category(joint),
            "time_ms": diff_time_ms,
            "max_diff_deg": round(val, 1),
            "severity": severity(val),
            "std_frame_index": std_frame_index,
            "stu_frame_index": stu_frame_index,
            "phase": "up" if idx < out_len else "down",
            "student_angle": student_angle,
            "standard_angle": standard_angle,
        }
        all_items.append(item)
        
        if val > best_val:
            best_val = val
            best_joint = joint
            best_idx = idx
    
    # ========== 增加差异关节多样性 ==========
    # 1. 合并左右同类关节，取差异大的那个
    # 2. 每个类别只保留一个，确保展示不同部位
    
    category_best: dict[str, dict[str, Any]] = {}
    for item in all_items:
        cat = item["joint_category"]
        if cat not in category_best or item["max_diff_deg"] > category_best[cat]["max_diff_deg"]:
            category_best[cat] = item
    
    # 按差异大小排序
    unique_items = sorted(category_best.values(), key=lambda x: float(x.get("max_diff_deg") or 0.0), reverse=True)
    
    # 取 Top3 不同部位
    diff_top = unique_items[:3]
    
    # 如果不够3个，从原始列表补充（允许同类别）
    if len(diff_top) < 3 and len(all_items) > len(diff_top):
        existing_joints = {item["joint"] for item in diff_top}
        for item in sorted(all_items, key=lambda x: float(x.get("max_diff_deg") or 0.0), reverse=True):
            if item["joint"] not in existing_joints:
                diff_top.append(item)
                existing_joints.add(item["joint"])
                if len(diff_top) >= 3:
                    break

    std_frame_index, stu_frame_index, diff_time_ms = map_idx(best_idx)

    return best_joint, diff_time_ms, std_frame_index, stu_frame_index, diff_top, stu_view


def _detect_video_view(frames: list[dict[str, Any]]) -> str:
    """
    检测整个视频的视角类型
    通过采样多帧的视角检测结果来确定主要视角
    返回: "front", "side", "angle"
    """
    if not frames:
        return "front"
    
    # 采样帧进行检测
    sample_indices = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1]
    sample_indices = [i for i in sample_indices if i < len(frames)]
    
    view_counts = {"front": 0, "side": 0, "angle": 0}
    for i in sample_indices:
        lm = frames[i].get("landmarks") or {}
        if not lm:
            continue
        # 转换格式
        lm_tuple = {}
        for name, data in lm.items():
            if isinstance(data, dict):
                lm_tuple[name] = (float(data.get("x", 0)), float(data.get("y", 0)), float(data.get("z", 0)))
            else:
                lm_tuple[name] = data
        
        view = _detect_view_angle(lm_tuple)
        view_counts[view] += 1
    
    # 返回出现最多的视角
    return max(view_counts, key=view_counts.get)


def _score_total(*, standard_data: dict[str, Any], student_data: dict[str, Any], view_angle: str = "front") -> dict[str, Any]:
    """
    计算总分（0-100），按照老师的档位标准评分（视角自适应）
    
    老师的评分标准（档位制）：
    - 90分以上：手臂放直 + 下巴过杠 + 腿并拢 + 无大幅晃动
    - 80分：手臂放直 + 下巴过杠 + 腿没完全并拢（但身体稳定）
    - 70分：手臂放直 + 下巴过杠 + 腿没完全并拢但身体晃动大 或 腿并拢但晃动大
    - 60分：手臂伸直 + 下巴过杠 + 又晃动腿又分开
    - 60以下：手臂弯曲 或 下巴不过杠
    
    各指标阈值按视角校准（基于老师三个方向标准视频的数据）。
    
    多动作处理：
    - 自动检测视频中所有完整动作
    - 每个动作按老师标准独立评分
    - 总分 = 所有动作分数的平均值
    """
    stu_frames_raw: list[dict[str, Any]] = list(student_data.get("frames") or [])

    stu_frames = _trim_non_hanging_frames(stu_frames_raw)
    
    if len(stu_frames) < 5:
        raise AnalysisError(POSE_LOW_CONFIDENCE, "有效动作帧太少，请确保视频包含完整的引体向上动作")
    
    # ========== 确定视角 ==========
    # 始终以用户选择的视角为准（用户在上传时已选择正面/侧面/斜面）
    # 仅在视角明显异常时打印警告（不覆盖用户选择）
    detected_view = _detect_video_view(stu_frames)
    effective_view = view_angle  # 信任用户选择
    if detected_view != view_angle:
        print(f"[SCORE] NOTE: User selected '{view_angle}' but auto-detect suggests '{detected_view}'. Using user selection.")
    print(f"[SCORE] Using view angle: {effective_view} (requested={view_angle}, detected={detected_view})")
    
    # ========== 检测所有动作周期 ==========
    actions = _detect_action_cycles(stu_frames)
    
    if not actions:
        raise AnalysisError(POSE_LOW_CONFIDENCE, "未检测到完整的引体向上动作")
    
    print(f"[SCORE] Detected {len(actions)} actions")
    
    # ========== 对每个动作按老师标准评分（传入视角） ==========
    action_scores = []
    for action in actions:
        scores = _score_single_action(action, stu_frames, effective_view)
        action_scores.append(scores)
        print(f"[SCORE] Action {scores['action_num']}: {scores['total']}/100 - {scores['score_reason']}")
    
    # ========== 计算平均分 ==========
    total_scores = [s["total"] for s in action_scores]
    avg_score = int(round(sum(total_scores) / len(total_scores)))
    
    # ========== 汇总评估数据 ==========
    # 统计四项指标的达标情况
    chin_pass_count = sum(1 for s in action_scores if s["chin_over"])
    arm_pass_count = sum(1 for s in action_scores if s["arm_straight"])
    legs_pass_count = sum(1 for s in action_scores if s["legs_together"])
    body_pass_count = sum(1 for s in action_scores if s["body_stable"])
    
    total_actions = len(action_scores)
    
    # 下巴过杠（取最好的数据展示）
    chin_gaps = [s["chin_gap"] for s in action_scores]
    best_chin_gap = min(chin_gaps)
    chin_over_bar = chin_pass_count > 0
    chin_detail = f"{chin_pass_count}/{total_actions}次过杠" + (f"（最佳：高于杠位 {abs(best_chin_gap):.0f}px）" if best_chin_gap < 0 else f"（最佳差距 {best_chin_gap:.0f}px）")
    
    # 手臂伸直（取最好的数据展示）
    bottom_elbows = [s["bottom_elbow"] for s in action_scores]
    best_elbow = max(bottom_elbows)
    arm_straight = arm_pass_count > 0
    arm_detail = f"{arm_pass_count}/{total_actions}次伸直（最佳肘角 {best_elbow:.0f}°）"
    
    # 腿部并拢（取平均）
    leg_dists = [s["leg_dist"] for s in action_scores]
    avg_leg_dist = sum(leg_dists) / len(leg_dists)
    legs_together = legs_pass_count > total_actions / 2  # 超过半数达标
    legs_detail = f"{legs_pass_count}/{total_actions}次并拢（平均间距 {avg_leg_dist:.0f}px）"
    
    # 身体稳定（取平均）
    swings = [s["swing"] for s in action_scores]
    avg_swing = sum(swings) / len(swings)
    body_stable = body_pass_count > total_actions / 2  # 超过半数达标
    body_detail = f"{body_pass_count}/{total_actions}次稳定（平均晃动 {avg_swing:.1f}°）"
    
    # 摆浪蹬腿（新增第5项指标）
    kipping_count = sum(1 for s in action_scores if s.get("is_kipping", False))
    has_kipping = kipping_count > 0
    hip_ratios = [s.get("hip_swing_ratio", 0) for s in action_scores]
    knee_stds = [s.get("knee_angle_std", 0) for s in action_scores]
    avg_hip_ratio = sum(hip_ratios) / len(hip_ratios) if hip_ratios else 0
    avg_knee_std = sum(knee_stds) / len(knee_stds) if knee_stds else 0
    if has_kipping:
        kipping_detail = f"{kipping_count}/{total_actions}次存在摆浪蹬腿（平均摆幅 {avg_hip_ratio:.2f}，蹬腿 {avg_knee_std:.1f}°）"
    else:
        kipping_detail = "无摆浪蹬腿"
    
    # ========== 生成评分说明 ==========
    all_deductions = []
    for s in action_scores:
        all_deductions.extend(s.get("deductions", []))
    # 去重
    deductions = list(dict.fromkeys(all_deductions))
    
    # 评分说明
    if avg_score >= 90:
        score_reason = f"动作标准！检测到 {total_actions} 个动作，平均分 {avg_score} 分"
    elif avg_score >= 80:
        score_reason = f"动作良好。检测到 {total_actions} 个动作，平均分 {avg_score} 分"
    elif avg_score >= 70:
        score_reason = f"动作一般。检测到 {total_actions} 个动作，平均分 {avg_score} 分"
    elif avg_score >= 60:
        score_reason = f"动作需改进。检测到 {total_actions} 个动作，平均分 {avg_score} 分"
    else:
        score_reason = f"动作不标准。检测到 {total_actions} 个动作，平均分 {avg_score} 分"
    
    print(f"[SCORE] Final: {avg_score}/100 (avg of {total_actions} actions)")
    
    return {
        "score": avg_score,
        "action_count": total_actions,
        "action_scores": action_scores,
        # 汇总数据（用于顶部指标展示）
        "chin_over_bar": chin_over_bar,
        "chin_gap_px": round(best_chin_gap, 1),
        "chin_detail": chin_detail,
        "arm_straight": arm_straight,
        "elbow_angle": round(best_elbow, 1),
        "arm_detail": arm_detail,
        "legs_together": legs_together,
        "leg_distance_px": round(avg_leg_dist, 1),
        "legs_detail": legs_detail,
        "body_stable": body_stable,
        "swing_amplitude": round(avg_swing, 1),
        "body_detail": body_detail,
        # 摆浪蹬腿（新增）
        "has_kipping": has_kipping,
        "kipping_count": kipping_count,
        "kipping_detail": kipping_detail,
        # 评分说明
        "score_reason": score_reason,
        "deductions": deductions,
    }


def _detect_action_issues(student_data: dict[str, Any], view_angle: str = "front") -> list[dict[str, Any]]:
    """
    检测学员动作中的具体问题（视角自适应，基于关键帧分析，不依赖标准视频对比）
    
    返回问题列表，每个问题包含：
    - issue: 问题名称
    - phase: 检测阶段（顶点/底部/全程）
    - status: "good" / "warning" / "bad"
    - detail: 详细描述
    - value: 实际数值
    - standard: 标准值
    """
    stu_frames_raw: list[dict[str, Any]] = list(student_data.get("frames") or [])
    stu_frames = _trim_non_hanging_frames(stu_frames_raw)
    
    if len(stu_frames) < 5:
        return []
    
    # 检测视角
    detected_view = _detect_video_view(stu_frames)
    effective_view = view_angle if view_angle != "front" else detected_view
    
    stu_peak = _peak_index(stu_frames)
    stu_bottom = _bottom_index(stu_frames)
    
    peak_frame = stu_frames[stu_peak]
    bottom_frame = stu_frames[stu_bottom]
    
    issues: list[dict[str, Any]] = []
    
    # ========== 1. 底部帧检测：手臂是否伸直 ==========
    # 侧面/斜面使用可见侧肘角
    if effective_view in ("side", "angle"):
        bottom_feats = bottom_frame.get("features") or {}
        ve = float(bottom_feats.get("visible_elbow_angle", 0) or 0)
        if ve > 10:
            elbow_angle = ve
            arm_straight = ve >= 165
        else:
            arm_straight, elbow_angle = _check_arm_straight(bottom_frame)
    else:
        arm_straight, elbow_angle = _check_arm_straight(bottom_frame)
    if elbow_angle >= 175:
        issues.append({
            "issue": "底部手臂伸直度",
            "phase": "底部（下放最低点）",
            "status": "good",
            "detail": f"手臂完全伸直（肘角 {elbow_angle:.0f}°，接近180°）",
            "value": round(elbow_angle, 0),
            "standard": "≥170°",
            "icon": "✓",
        })
    elif elbow_angle >= 170:
        issues.append({
            "issue": "底部手臂伸直度",
            "phase": "底部（下放最低点）",
            "status": "good",
            "detail": f"手臂基本伸直（肘角 {elbow_angle:.0f}°）",
            "value": round(elbow_angle, 0),
            "standard": "≥170°",
            "icon": "✓",
        })
    elif elbow_angle >= 165:
        issues.append({
            "issue": "底部手臂伸直度",
            "phase": "底部（下放最低点）",
            "status": "warning",
            "detail": f"手臂略微弯曲（肘角 {elbow_angle:.0f}°，标准≥170°）",
            "value": round(elbow_angle, 0),
            "standard": "≥170°",
            "icon": "⚠",
        })
    else:
        issues.append({
            "issue": "底部手臂伸直度",
            "phase": "底部（下放最低点）",
            "status": "bad",
            "detail": f"手臂弯曲明显（肘角 {elbow_angle:.0f}°，标准≥170°）",
            "value": round(elbow_angle, 0),
            "standard": "≥170°",
            "icon": "✗",
        })
    
    # ========== 2. 顶点帧检测：下巴是否过杠（视角自适应） ==========
    chin_over, chin_gap = _check_chin_over_bar(peak_frame, effective_view)
    if chin_gap < -30:
        issues.append({
            "issue": "顶点下巴过杠",
            "phase": "顶点（上拉最高点）",
            "status": "good",
            "detail": f"下巴明显过杠（高于杠位 {abs(chin_gap):.0f}px）",
            "value": round(chin_gap, 0),
            "standard": "高于杠位",
            "icon": "✓",
        })
    elif chin_gap < 0:
        issues.append({
            "issue": "顶点下巴过杠",
            "phase": "顶点（上拉最高点）",
            "status": "good",
            "detail": f"下巴刚好过杠（高于杠位 {abs(chin_gap):.0f}px）",
            "value": round(chin_gap, 0),
            "standard": "高于杠位",
            "icon": "✓",
        })
    elif chin_gap < 20:
        issues.append({
            "issue": "顶点下巴过杠",
            "phase": "顶点（上拉最高点）",
            "status": "warning",
            "detail": f"下巴接近过杠（低于杠位 {chin_gap:.0f}px，再高一点）",
            "value": round(chin_gap, 0),
            "standard": "高于杠位",
            "icon": "⚠",
        })
    else:
        issues.append({
            "issue": "顶点下巴过杠",
            "phase": "顶点（上拉最高点）",
            "status": "bad",
            "detail": f"下巴未过杠（低于杠位 {chin_gap:.0f}px）",
            "value": round(chin_gap, 0),
            "standard": "高于杠位",
            "icon": "✗",
        })
    
    # ========== 3. 顶点帧检测：肘角（屈肘程度） ==========
    peak_feats = peak_frame.get("features") or {}
    peak_left_elbow = peak_feats.get("left_elbow_angle", 90)
    peak_right_elbow = peak_feats.get("right_elbow_angle", 90)
    peak_elbow = (float(peak_left_elbow) + float(peak_right_elbow)) / 2 if peak_left_elbow and peak_right_elbow else 90
    
    if peak_elbow <= 50:
        issues.append({
            "issue": "顶点屈肘程度",
            "phase": "顶点（上拉最高点）",
            "status": "good",
            "detail": f"屈肘充分（肘角 {peak_elbow:.0f}°，很标准）",
            "value": round(peak_elbow, 0),
            "standard": "≤60°",
            "icon": "✓",
        })
    elif peak_elbow <= 70:
        issues.append({
            "issue": "顶点屈肘程度",
            "phase": "顶点（上拉最高点）",
            "status": "good",
            "detail": f"屈肘良好（肘角 {peak_elbow:.0f}°）",
            "value": round(peak_elbow, 0),
            "standard": "≤60°",
            "icon": "✓",
        })
    elif peak_elbow <= 90:
        issues.append({
            "issue": "顶点屈肘程度",
            "phase": "顶点（上拉最高点）",
            "status": "warning",
            "detail": f"屈肘不够充分（肘角 {peak_elbow:.0f}°，可以再拉高）",
            "value": round(peak_elbow, 0),
            "standard": "≤60°",
            "icon": "⚠",
        })
    else:
        issues.append({
            "issue": "顶点屈肘程度",
            "phase": "顶点（上拉最高点）",
            "status": "bad",
            "detail": f"上拉高度不够（肘角 {peak_elbow:.0f}°，需要拉更高）",
            "value": round(peak_elbow, 0),
            "standard": "≤60°",
            "icon": "✗",
        })
    
    # ========== 4. 全程检测：腿部并拢（视角自适应） ==========
    LEGS_THRESHOLDS = {"front": 65, "side": 45, "angle": 45}
    legs_threshold = LEGS_THRESHOLDS.get(effective_view, 65)
    legs_together, leg_dist = _check_legs_together(stu_frames, effective_view)
    # 按视角调整"好/警告/差"的分界线
    good_threshold = legs_threshold * 0.6
    warn_threshold = legs_threshold * 1.4
    if leg_dist < good_threshold:
        issues.append({
            "issue": "全程腿部并拢",
            "phase": "全程",
            "status": "good",
            "detail": f"双腿紧密并拢（双脚间距 {leg_dist:.0f}px，很标准）",
            "value": round(leg_dist, 0),
            "standard": f"<{legs_threshold}px",
            "icon": "✓",
        })
    elif leg_dist < legs_threshold:
        issues.append({
            "issue": "全程腿部并拢",
            "phase": "全程",
            "status": "good",
            "detail": f"双腿基本并拢（双脚间距 {leg_dist:.0f}px）",
            "value": round(leg_dist, 0),
            "standard": f"<{legs_threshold}px",
            "icon": "✓",
        })
    elif leg_dist < warn_threshold:
        issues.append({
            "issue": "全程腿部并拢",
            "phase": "全程",
            "status": "warning",
            "detail": f"双腿略微分开（双脚间距 {leg_dist:.0f}px，建议收紧）",
            "value": round(leg_dist, 0),
            "standard": f"<{legs_threshold}px",
            "icon": "⚠",
        })
    else:
        issues.append({
            "issue": "全程腿部并拢",
            "phase": "全程",
            "status": "bad",
            "detail": f"双腿分开明显（双脚间距 {leg_dist:.0f}px，需要并拢）",
            "value": round(leg_dist, 0),
            "standard": f"<{legs_threshold}px",
            "icon": "✗",
        })
    
    # ========== 5. 全程检测：身体稳定性（视角自适应） ==========
    SWING_THRESHOLDS = {"front": 4, "side": 30, "angle": 22}
    swing_threshold = SWING_THRESHOLDS.get(effective_view, 4)
    swing_good = swing_threshold * 0.6   # "非常稳定"的界线
    swing_warn = swing_threshold * 2.0   # "晃动明显"的界线
    
    body_stable, swing_amplitude = _check_body_swing(stu_frames, effective_view)
    if swing_amplitude < swing_good:
        issues.append({
            "issue": "全程身体稳定性",
            "phase": "全程",
            "status": "good",
            "detail": f"身体非常稳定（晃动 {swing_amplitude:.1f}°，很标准）",
            "value": round(swing_amplitude, 1),
            "standard": f"<{swing_threshold}°",
            "icon": "✓",
        })
    elif swing_amplitude < swing_threshold:
        issues.append({
            "issue": "全程身体稳定性",
            "phase": "全程",
            "status": "good",
            "detail": f"身体基本稳定（晃动 {swing_amplitude:.1f}°）",
            "value": round(swing_amplitude, 1),
            "standard": f"<{swing_threshold}°",
            "icon": "✓",
        })
    elif swing_amplitude < swing_warn:
        issues.append({
            "issue": "全程身体稳定性",
            "phase": "全程",
            "status": "warning",
            "detail": f"身体有些晃动（晃动 {swing_amplitude:.1f}°，注意控制）",
            "value": round(swing_amplitude, 1),
            "standard": f"<{swing_threshold}°",
            "icon": "⚠",
        })
    else:
        issues.append({
            "issue": "全程身体稳定性",
            "phase": "全程",
            "status": "bad",
            "detail": f"身体晃动明显（晃动 {swing_amplitude:.1f}°，需加强核心）",
            "value": round(swing_amplitude, 1),
            "standard": f"<{swing_threshold}°",
            "icon": "✗",
        })
    
    # ========== 6. 全程检测：摆浪蹬腿借力（新增） ==========
    is_kipping, hip_ratio, knee_std = _check_kipping(stu_frames, effective_view)
    if is_kipping:
        # 判定严重程度
        HIP_KIP_THRESHOLDS = {"front": 0.35, "side": 0.80, "angle": 0.35}
        ht = HIP_KIP_THRESHOLDS.get(effective_view, 0.35)
        severity_parts = []
        if hip_ratio > ht * 1.5:
            severity_parts.append(f"身体大幅前后摆荡（摆幅 {hip_ratio:.2f}）")
        elif hip_ratio > ht:
            severity_parts.append(f"身体前后摆动（摆幅 {hip_ratio:.2f}）")
        if knee_std > 15:
            severity_parts.append(f"蹬腿借力明显（膝角变化 {knee_std:.1f}°）")
        elif knee_std > 10:
            severity_parts.append(f"轻微蹬腿借力（膝角变化 {knee_std:.1f}°）")
        
        detail_str = "、".join(severity_parts) if severity_parts else f"存在摆浪蹬腿借力（摆幅 {hip_ratio:.2f}，膝角波动 {knee_std:.1f}°）"
        
        issues.append({
            "issue": "摆浪蹬腿借力",
            "phase": "全程",
            "status": "bad",
            "detail": f"{detail_str}，动作依靠惯性而非肌肉发力，最高仅评65分",
            "value": round(hip_ratio, 3),
            "standard": "无摆浪蹬腿",
            "icon": "✗",
        })
    else:
        issues.append({
            "issue": "摆浪蹬腿借力",
            "phase": "全程",
            "status": "good",
            "detail": f"未检测到明显摆浪蹬腿（摆幅 {hip_ratio:.2f}，膝角波动 {knee_std:.1f}°）",
            "value": round(hip_ratio, 3),
            "standard": "无摆浪蹬腿",
            "icon": "✓",
        })
    
    # 按状态排序：bad > warning > good（问题优先）
    status_order = {"bad": 0, "warning": 1, "good": 2}
    issues.sort(key=lambda x: status_order.get(x["status"], 2))
    
    return issues


def _joint_label_cn(joint: str) -> str:
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
        "chin_over_bar": "下巴过杠",
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


def _check_chin_over_bar(peak_frame: dict[str, Any], view_angle: str = "front") -> tuple[bool, float]:
    """
    检查顶点帧是否下巴过杠（视角自适应）
    
    正面(front)：
      鼻子y < 手腕平均y + 20px → 过杠
      两只手腕都可见，取平均作为杠位
    
    侧面(side) / 斜面(angle)：
      问题：顶点时手臂举到头部附近，手容易遮挡鼻子，导致鼻子检测偏低
      解决：
        1. 面部位置用多点融合（鼻子+可见侧耳朵+可见侧眼睛），取最高点
        2. 杠位用可见侧手腕（置信度更高的那只），而非两只平均
        3. 容差更大（30px），因为侧面角度造成的像素偏差更大
    
    返回：(是否过杠, 差距像素值)
    """
    lms = peak_frame.get("landmarks") or {}
    
    nose = lms.get("nose", {})
    left_wrist = lms.get("left_wrist", {})
    right_wrist = lms.get("right_wrist", {})
    
    nose_y = float(nose.get("y", 9999))
    nose_v = float(nose.get("v", 0))  # 置信度
    lw_y = float(left_wrist.get("y", 0))
    lw_v = float(left_wrist.get("v", 0))
    rw_y = float(right_wrist.get("y", 0))
    rw_v = float(right_wrist.get("v", 0))
    
    if view_angle in ("side", "angle"):
        # ========== 侧面/斜面：多策略融合 ==========
        # 
        # 侧面的核心难题：手腕y坐标不能可靠地代表杠位
        # - 透视角度导致手腕可能出现在脸的上方或下方
        # - 手臂伸直悬挂时 vs 弯曲拉上去时，手腕y位置差异巨大
        # 
        # 策略：
        # A) 面部 vs 手腕 （传统方法，仅部分场景可靠）
        # B) 面部 vs 肩膀 （肩膀在杠附近，更稳定）
        # C) 肘角判定法 （肘角<70°说明人已经拉到最高，几乎一定过杠）
        # 任一策略判定过杠即为过杠
        
        # 1. 面部最高点：综合鼻子、耳朵、眼睛（取最高即y最小的有效点）
        face_candidates = []
        for fkey in ("nose", "left_ear", "right_ear", "left_eye", "right_eye", "mouth_left", "mouth_right"):
            pt = lms.get(fkey, {})
            py = float(pt.get("y", 9999))
            pv = float(pt.get("v", 0))
            if py < 9000 and pv > 0.2:
                face_candidates.append(py)
        
        face_top_y = min(face_candidates) if face_candidates else nose_y
        
        # ---- 策略A：面部 vs 手腕 ----
        # 用y更大的手腕（更低的 = 更接近身体而非头顶的手腕）
        # 避免取到举过头顶的手腕值
        wrist_candidates = []
        if lw_v > 0.3:
            wrist_candidates.append(lw_y)
        if rw_v > 0.3:
            wrist_candidates.append(rw_y)
        
        if wrist_candidates:
            # 取y更大的（位置更低的手腕，更可能是靠近杠的位置）
            bar_y_wrist = max(wrist_candidates)
        else:
            bar_y_wrist = (lw_y + rw_y) / 2
        
        gap_wrist = face_top_y - bar_y_wrist
        over_by_wrist = gap_wrist < 30
        
        # ---- 策略B：面部 vs 肩膀 ----
        # 拉到顶点时，肩膀约在杠的高度附近
        # 如果面部(眼睛/耳朵)高于肩膀，说明下巴大概率过杠
        ls = lms.get("left_shoulder", {})
        rs = lms.get("right_shoulder", {})
        ls_y = float(ls.get("y", 9999))
        rs_y = float(rs.get("y", 9999))
        shoulder_ys = []
        if ls_y < 9000:
            shoulder_ys.append(ls_y)
        if rs_y < 9000:
            shoulder_ys.append(rs_y)
        
        over_by_shoulder = False
        gap_shoulder = 999.0
        if shoulder_ys:
            # 取较高的肩膀（y更小）
            shoulder_top_y = min(shoulder_ys)
            gap_shoulder = face_top_y - shoulder_top_y
            # 如果面部最高点高于肩膀（或接近），说明过杠
            over_by_shoulder = gap_shoulder < 15
        
        # ---- 策略C：肘角判定 ----
        # 如果顶点帧肘角 < 70°，说明手臂弯曲很大，人已经拉到了最高点
        # 标准引体向上顶点肘角通常在 20-50°，必然过杠
        feats = peak_frame.get("features") or {}
        elbow_angles = []
        for ekey in ("visible_elbow_angle", "left_elbow_angle", "right_elbow_angle"):
            ea = feats.get(ekey)
            if ea is not None and not math.isnan(float(ea)) and float(ea) > 5:
                elbow_angles.append(float(ea))
        
        over_by_elbow = False
        min_elbow = 180.0
        if elbow_angles:
            min_elbow = min(elbow_angles)
            over_by_elbow = min_elbow < 70
        
        # ---- 综合判定：任一策略认为过杠即为过杠 ----
        chin_over = over_by_wrist or over_by_shoulder or over_by_elbow
        
        # gap取最有利的值（最小的gap）
        gaps = [gap_wrist]
        if shoulder_ys:
            gaps.append(gap_shoulder)
        if over_by_elbow:
            gaps.append(-10.0)  # 肘角过杠时，给一个"明确过杠"的gap值
        gap = min(gaps)
        
        print(f"[DEBUG] _check_chin_over_bar({view_angle}): "
              f"face_top={face_top_y:.1f}, "
              f"wrist(gap={gap_wrist:.1f},over={over_by_wrist}), "
              f"shoulder(gap={gap_shoulder:.1f},over={over_by_shoulder}), "
              f"elbow(min={min_elbow:.0f}°,over={over_by_elbow}), "
              f"→ final: over={chin_over}, gap={gap:.1f}")
    
    else:
        # ========== 正面：原有逻辑 ==========
        bar_y = (lw_y + rw_y) / 2
        gap = nose_y - bar_y
        chin_over = gap < 20
        
        print(f"[DEBUG] _check_chin_over_bar(front): nose_y={nose_y:.1f}, bar_y={bar_y:.1f}, gap={gap:.1f}, over={chin_over}")
    
    return chin_over, gap


def _check_arm_straight(bottom_frame: dict[str, Any]) -> tuple[bool, float]:
    """
    检查底部帧（下放时）手臂是否完全伸直
    判定标准：肘角接近180度（允许165度以上为伸直）
    返回：(是否伸直, 平均肘角)
    """
    feats = bottom_frame.get("features") or {}
    
    left_elbow = feats.get("left_elbow_angle", 0)
    right_elbow = feats.get("right_elbow_angle", 0)
    
    # 取有效值的平均
    angles = []
    if left_elbow and not math.isnan(left_elbow):
        angles.append(float(left_elbow))
    if right_elbow and not math.isnan(right_elbow):
        angles.append(float(right_elbow))
    
    if not angles:
        return False, 0.0
    
    avg_elbow = sum(angles) / len(angles)
    
    # 肘角>=165度认为手臂伸直
    is_straight = avg_elbow >= 165
    
    print(f"[DEBUG] _check_arm_straight: avg_elbow={avg_elbow:.1f}°, straight={is_straight}")
    return is_straight, avg_elbow


def _check_legs_together(frames: list[dict[str, Any]], view_angle: str = "front") -> tuple[bool, float]:
    """
    检查整个动作过程中腿是否并拢（视角自适应）
    
    不同视角使用不同的检测方法和阈值（基于老师标准视频校准）：
    
    正面(front):
      - 检测方法：左右踝/膝关节 X 轴距离
      - 老师数据：ankle X avg=44.6px, max=52.6px
      - 阈值：65px (max * 1.23)
    
    侧面(side):
      - 检测方法：踝关节欧几里得距离（X轴重叠，需要Y分量）
      - 老师数据：euclidean avg=19.6px, max=37.1px
      - 阈值：45px (max * 1.21)
      - 注意：纯侧面无法完全判断腿是否水平分开，但可以检测前后错开
    
    斜侧面(angle):
      - 检测方法：踝/膝关节 X 轴距离（视角压缩，阈值降低）
      - 老师数据：ankle X avg=27.0px, max=35.7px
      - 阈值：45px (max * 1.26)
    
    返回：(是否并拢, 距离数值)
    """
    # 视角对应的阈值（基于老师标准视频校准）
    THRESHOLDS = {
        "front": 65,   # 老师正面 max 52.6px → 65px
        "side":  45,   # 老师侧面 euclidean max 37.1px → 45px
        "angle": 45,   # 老师斜面 max 35.7px → 45px
    }
    threshold = THRESHOLDS.get(view_angle, 65)
    
    distances = []
    
    if view_angle == "side":
        # 侧面：使用欧几里得距离（包含Y分量）
        for frame in frames:
            lms = frame.get("landmarks") or {}
            la = lms.get("left_ankle", {})
            ra = lms.get("right_ankle", {})
            
            la_x = float(la.get("x", 0))
            la_y = float(la.get("y", 0))
            ra_x = float(ra.get("x", 0))
            ra_y = float(ra.get("y", 0))
            
            if la_x > 0 and ra_x > 0 and la_y > 0 and ra_y > 0:
                dist = math.sqrt((la_x - ra_x)**2 + (la_y - ra_y)**2)
                distances.append(dist)
            
            # 膝关节也做欧几里得
            lk = lms.get("left_knee", {})
            rk = lms.get("right_knee", {})
            lk_x = float(lk.get("x", 0))
            lk_y = float(lk.get("y", 0))
            rk_x = float(rk.get("x", 0))
            rk_y = float(rk.get("y", 0))
            if lk_x > 0 and rk_x > 0 and lk_y > 0 and rk_y > 0:
                dist = math.sqrt((lk_x - rk_x)**2 + (lk_y - rk_y)**2)
                distances.append(dist)
    else:
        # 正面 / 斜侧面：使用 X 轴距离
        for frame in frames:
            lms = frame.get("landmarks") or {}
            left_ankle = lms.get("left_ankle", {})
            right_ankle = lms.get("right_ankle", {})
            left_knee = lms.get("left_knee", {})
            right_knee = lms.get("right_knee", {})
            
            la_x = float(left_ankle.get("x", 0))
            ra_x = float(right_ankle.get("x", 0))
            lk_x = float(left_knee.get("x", 0))
            rk_x = float(right_knee.get("x", 0))
            
            if la_x > 0 and ra_x > 0:
                distances.append(abs(la_x - ra_x))
            if lk_x > 0 and rk_x > 0:
                distances.append(abs(lk_x - rk_x))
    
    if not distances:
        return True, 0.0
    
    avg_dist = sum(distances) / len(distances)
    is_together = avg_dist < threshold
    
    print(f"[DEBUG] _check_legs_together({view_angle}): avg_dist={avg_dist:.1f}px, threshold={threshold}px, together={is_together}")
    return is_together, avg_dist


def _check_body_swing(frames: list[dict[str, Any]], view_angle: str = "front") -> tuple[bool, float]:
    """
    检查整个动作过程中身体是否大幅度晃动（视角自适应）
    
    不同视角使用不同的阈值（基于老师标准视频校准）：
    
    正面(front):
      - 主要看左右晃动 (torso_lean_std + hip_x_range)
      - 老师数据：amplitude=2.13°
      - 阈值：4° (amplitude * 1.9)
    
    侧面(side):
      - 看到了前后摆动（正面看不到的 sagittal 平面运动）
      - 引体向上时身体自然前后摆动，这是正常的
      - 老师数据：amplitude=25.28°
      - 阈值：30° (amplitude * 1.19)
    
    斜侧面(angle):
      - 混合视角，前后摆动部分可见
      - 老师数据：amplitude=17.72°
      - 阈值：22° (amplitude * 1.24)
    
    返回：(是否稳定无大幅晃动, 晃动幅度)
    """
    # 视角对应的阈值（基于老师标准视频校准）
    THRESHOLDS = {
        "front": 4,    # 老师正面 2.13° → 4°
        "side":  30,   # 老师侧面 25.28° → 30°
        "angle": 22,   # 老师斜面 17.72° → 22°
    }
    threshold = THRESHOLDS.get(view_angle, 4)
    
    torso_leans = []
    body_swings = []
    hip_x_positions = []
    
    for frame in frames:
        feats = frame.get("features") or {}
        lms = frame.get("landmarks") or {}
        
        torso_lean = feats.get("torso_lean", 0)
        if torso_lean and not math.isnan(torso_lean):
            torso_leans.append(float(torso_lean))
        
        body_swing = feats.get("body_swing", 0)
        if body_swing and not math.isnan(body_swing):
            body_swings.append(float(body_swing))
        
        left_hip = lms.get("left_hip", {})
        right_hip = lms.get("right_hip", {})
        lh_x = float(left_hip.get("x", 0))
        rh_x = float(right_hip.get("x", 0))
        if lh_x > 0 and rh_x > 0:
            hip_x_positions.append((lh_x + rh_x) / 2)
    
    swing_amplitude = 0.0
    
    # torso_lean（左右倾斜）- 所有视角都使用
    if torso_leans:
        lean_std = float(np.std(torso_leans))
        swing_amplitude = max(swing_amplitude, lean_std)
    
    # body_swing（前后倾斜）- 仅侧面/斜面使用
    # 注意：body_swing 在特征提取时仅对 side/angle 帧计算，
    # 但由于每帧视角可能与评分视角不同，正面评分时不应使用此指标
    if view_angle in ("side", "angle") and body_swings:
        max_swing = max(body_swings)
        swing_amplitude = max(swing_amplitude, max_swing)
    
    # hip_x 水平移动 - 所有视角都使用
    if len(hip_x_positions) > 5:
        hip_range = max(hip_x_positions) - min(hip_x_positions)
        swing_amplitude = max(swing_amplitude, hip_range / 10)
    
    is_stable = swing_amplitude < threshold
    
    print(f"[DEBUG] _check_body_swing({view_angle}): amplitude={swing_amplitude:.1f}°, threshold={threshold}°, stable={is_stable}"
          f" (lean_std={'%.1f' % float(np.std(torso_leans)) if torso_leans else 'N/A'}"
          f", body_swing_max={'%.1f' % max(body_swings) if body_swings else 'N/A'}"
          f", hip_range={'%.1f' % (max(hip_x_positions)-min(hip_x_positions)) if len(hip_x_positions)>5 else 'N/A'}px)")
    return is_stable, swing_amplitude


def _check_kipping(frames: list[dict[str, Any]], view_angle: str = "front") -> tuple[bool, float, float]:
    """
    检测摆浪蹬腿借力（kipping）
    
    摆浪蹬腿 vs 普通不稳定：
    - 普通不稳定：小幅度随机晃动
    - 摆浪蹬腿：大幅度的钟摆式身体摆动 + 腿部蹬伸借力
      即使腿并拢、下巴过杠、手臂伸直，也是借力动作，不标准
    
    检测两个核心指标：
    1. 髋部水平摆动幅度（身体像钟摆一样前后摆）
       - 归一化为 hip_range / torso_height，消除分辨率和视角影响
       - 老师正面标准：hip_range=21.3px, torso~195px → ratio≈0.11
       - 老师侧面标准：hip_range=72.6px, torso~190px → ratio≈0.38
       - 老师斜面标准：hip_range=26.7px, torso~200px → ratio≈0.13
    
    2. 膝角变化幅度（蹬腿：膝盖突然弯曲再伸直来借力）
       - 标准动作：膝盖始终保持直腿，膝角变化很小
       - 蹬腿借力：膝角大幅波动（弯→蹬→直）
    
    返回：(是否摆浪蹬腿, 髋部摆幅ratio, 膝角变化std)
    """
    if len(frames) < 10:
        return False, 0.0, 0.0
    
    # ========== 1. 髋部水平摆动幅度 ==========
    hip_x_positions = []
    torso_heights = []
    
    for frame in frames:
        lms = frame.get("landmarks") or {}
        lh = lms.get("left_hip", {})
        rh = lms.get("right_hip", {})
        ls = lms.get("left_shoulder", {})
        rs = lms.get("right_shoulder", {})
        
        lh_x = float(lh.get("x", 0))
        rh_x = float(rh.get("x", 0))
        ls_y = float(ls.get("y", 0))
        rs_y = float(rs.get("y", 0))
        lh_y = float(lh.get("y", 0))
        rh_y = float(rh.get("y", 0))
        
        if lh_x > 0 and rh_x > 0:
            hip_x_positions.append((lh_x + rh_x) / 2)
        
        if ls_y > 0 and lh_y > 0:
            shoulder_y = (ls_y + rs_y) / 2
            hip_y = (lh_y + rh_y) / 2
            th = abs(hip_y - shoulder_y)
            if th > 20:
                torso_heights.append(th)
    
    hip_swing_ratio = 0.0
    if hip_x_positions and torso_heights:
        hip_range = max(hip_x_positions) - min(hip_x_positions)
        avg_torso = sum(torso_heights) / len(torso_heights)
        if avg_torso > 20:
            hip_swing_ratio = hip_range / avg_torso
    
    # ========== 2. 膝角变化幅度（蹬腿检测） ==========
    knee_angles = []
    for frame in frames:
        feats = frame.get("features") or {}
        lk = feats.get("left_knee_angle")
        rk = feats.get("right_knee_angle")
        
        angles = []
        if lk is not None and not math.isnan(float(lk)):
            angles.append(float(lk))
        if rk is not None and not math.isnan(float(rk)):
            angles.append(float(rk))
        
        if angles:
            knee_angles.append(sum(angles) / len(angles))
    
    knee_angle_std = 0.0
    knee_angle_range = 0.0
    if knee_angles:
        knee_angle_std = float(np.std(knee_angles))
        knee_angle_range = max(knee_angles) - min(knee_angles)
    
    # ========== 判定阈值（基于老师标准视频校准） ==========
    # 
    # 老师标准视频的 hip_swing_ratio:
    #   正面: ~0.11  侧面: ~0.38  斜面: ~0.13
    # 摆浪借力的 hip_swing_ratio 应该远大于老师标准
    #
    # 阈值设计：
    #   hip_swing_ratio > 0.35 (正面/斜面) 或 > 0.80 (侧面) → 大幅摆浪
    #   knee_angle_std > 15° 或 knee_range > 40° → 蹬腿
    #   两项中任一项严重，或两项都偏高 → 判定为摆浪蹬腿
    
    HIP_SWING_THRESHOLDS = {
        "front": 0.35,  # 老师正面 0.11 → 阈值 3x
        "side":  0.80,  # 老师侧面 0.38 → 阈值 2.1x
        "angle": 0.35,  # 老师斜面 0.13 → 阈值 2.7x
    }
    hip_threshold = HIP_SWING_THRESHOLDS.get(view_angle, 0.35)
    
    # 判定逻辑：
    # 条件1: 髋部大幅摆动
    hip_kipping = hip_swing_ratio > hip_threshold
    # 条件2: 蹬腿借力（膝角大幅变化）
    leg_kicking = knee_angle_std > 15 or knee_angle_range > 40
    # 条件3: 两项都偏高（各自未达阈值但合起来严重）
    combined = hip_swing_ratio > hip_threshold * 0.7 and (knee_angle_std > 10 or knee_angle_range > 30)
    
    is_kipping = hip_kipping or leg_kicking or combined
    
    print(f"[DEBUG] _check_kipping({view_angle}): hip_ratio={hip_swing_ratio:.3f}(threshold={hip_threshold}), "
          f"knee_std={knee_angle_std:.1f}°, knee_range={knee_angle_range:.1f}°, "
          f"kipping={is_kipping} (hip={hip_kipping}, kick={leg_kicking}, combined={combined})")
    
    return is_kipping, hip_swing_ratio, knee_angle_std


def _make_tips(*, score_total: int, diff_joint: str, diff_top: list[dict[str, Any]] | None = None, chin_over_bar: bool = True, view_angle: str = "front", has_kipping: bool = False) -> list[str]:
    """
    生成专业个性化教学建议
    基于老师提供的专业评分标准和训练建议
    
    评分等级：
    - 特殊：摆浪蹬腿借力（最高65分，优先于其他等级）
    - 等级一（90%+）：动作标准，手臂伸直+下巴过杠+腿并拢+无晃动+无摆浪
    - 等级二（80-89%）：手臂伸直+下巴过杠+腿未完全并拢+轻微晃动
    - 等级三（70-79%）：手臂伸直+下巴过杠+发力不顺畅/蹬腿借力
    - 等级四（60-69%）：手臂伸直+下巴过杠+身体前后摆动明显
    - 等级五/六（60%以下）：手臂未伸直或下巴未过杠
    """
    tips: list[str] = []
    
    # ========== 特殊等级：摆浪蹬腿借力（最高65分） ==========
    if has_kipping:
        tips.append("【动作评价】检测到明显的摆浪蹬腿借力！身体在引体过程中产生了较大幅度的前后摆动或蹬腿动作，"
                     "通过身体惯性辅助完成上拉，而非完全依靠背部和手臂肌肉主动发力。"
                     "即使手臂能伸直、下巴能过杠、腿也能并拢，但由于动作依靠惯性完成，评分上限为65分。")
        tips.append("【存在问题】1.身体前后大幅摆动（摆浪），利用钟摆效应借力上拉。"
                     "2.蹬腿借力：通过下肢的蹬伸动作产生向上的惯性力。"
                     "3.这种借力模式会降低目标肌群的训练效果，且长期可能增加肩关节和腰椎的受伤风险。")
        tips.append("【原因分析】1.背部及手臂相对力量不足，无法纯靠肌肉力量完成标准动作。"
                     "2.缺乏核心收紧意识，未能在动作全程保持躯干刚性。"
                     "3.动作习惯问题：习惯性地用身体惯性代替肌肉发力，需要重新建立正确的发力模式。"
                     "4.可能存在急于完成次数的心理，忽视了动作质量。")
        tips.append("【训练建议】1.【首要】消除摆浪：在杠上做引体时，请同伴在身后轻按你的下背部/臀部以限制摆动，"
                     "或在脚下夹一个哑铃片/药球，迫使身体保持垂直。"
                     "2.【核心稳定】练习悬垂静止：双手握杠自然悬垂，刻意绷紧腹部、夹紧臀部，保持身体完全静止30秒，"
                     "感受核心收紧的状态，每次训练前做3组。"
                     "3.【力量提升】使用弹力带辅助做严格引体：选择合适的弹力带，确保能在不摆动的情况下完成5-8次标准动作。"
                     "4.【离心控制】做控制性下放训练：跳起或用辅助到最高点，然后用5-8秒的速度匀速缓慢下放，"
                     "全程保持身体垂直，不允许任何前后摆动。"
                     "5.【辅助训练】加强水平划船、哑铃划船等水平面拉的力量，"
                     "这些动作能在较少借力的情况下有效增强背部力量。")
        return tips
    
    # ========== 等级一：90分以上 ==========
    if score_total >= 90:
        tips.append("【动作评价】动作已非常标准！手臂屈伸角度标准，下放时手臂伸直，上拉时下巴过杠，两腿伸直并拢，身体前后晃动角度合理，无明显晃动。")
        tips.append("【可能存在】在高次数训练或力竭时可能出现轻微变形，或存在动作节奏（如离心收缩速度控制）的提升空间。")
        tips.append("【原因分析】核心肌群（腹肌、下背部、臀部）和肩胛稳定性好，背部主导发力的模式正确，动作控制力强。")
        tips.append("【训练建议】1.增加训练强度：尝试负重引体向上（穿负重背心）、单臂辅助引体向上或爆发力引体向上（胸触杠、腾空换手等）。2.优化动作节奏：采用'慢离心'模式，用2秒上拉，在最高点停顿1秒，然后用3-4秒有控制地缓慢下放。3.提升耐力：进行更多组高次数训练，或缩短组间休息时间。")
    
    # ========== 等级二：80-89分 ==========
    elif score_total >= 80:
        tips.append("【动作评价】动作很标准。手臂屈伸角度标准，下放时手臂伸直，上拉时下巴过杠，但两腿未完全伸直并拢，身体前后晃动角度合理，身体轻微晃动。")
        tips.append("【存在问题】核心肌群收紧不足或下肢松散，核心稳定和控制能力弱导致力量传递效率降低。")
        tips.append("【原因分析】1.核心意识薄弱：没有主动绷紧腹部和臀部收腹夹臀的意识，导致下肢成为'悬垂的负担'。2.呼吸配合不佳：上拉时未憋气或呼气以维持腹内压，影响了躯干的刚性。")
        tips.append("【训练建议】1.强化核心稳定性训练：在悬垂状态下练习，如悬垂举腿、悬垂收腹、哥本哈根支撑，感受核心收紧状态下身体的稳定。2.进行'静态锁定'练习：在引体向上的最高点和最低点分别保持静止5-10秒，全程有意识地绷紧全身，尤其是腿和脚并拢。")
    
    # ========== 等级三：70-79分 ==========
    elif score_total >= 70:
        tips.append("【动作评价】动作一般，手臂屈伸角度合理，下放时手臂伸直，上拉时下巴过杠，但发力不太顺畅，存在蹬腿发力现象。")
        tips.append("【存在问题】主动肌群力量不足，为完成动作而通过下肢的蹬伸来借力。")
        tips.append("【原因分析】1.主动肌群力量薄弱。2.动作模式不熟练，不知道如何主动地用背部发起和主导动作。3.核心肌群稳定性及动作节奏差。")
        tips.append("【训练建议】1.强化辅助练习：以弹力带辅助或器械引体向上、器械辅助引体向上为主，选择能标准完成5-8次的辅助力度。2.加强相关肌群：重点练习高位下拉（模仿引体发力）、坐姿划船、面拉（强化肩后束与肩袖肌群稳定）。")
    
    # ========== 等级四：60-69分 ==========
    elif score_total >= 60:
        tips.append("【动作评价】动作不标准，仅做到手臂伸直和下巴过杠，但在动作过程中身体前后晃动角度较大，身体前后摆动明显。")
        tips.append("【存在问题】借力现象严重，动作效率低，容易养成错误的动作习惯，长期可能增加肩关节受伤风险。")
        tips.append("【原因分析】1.相对力量不足，背部力量无法独立完成目标次数，必须靠摆动来借力。2.完全缺乏核心收紧的意识。3.对动作标准的要求不明确。")
        tips.append("【训练建议】1.进行'定格训练'（可辅助）：在动作的顶部、中间、底部各停顿1秒，彻底消除惯性。2.进行离心训练引体控制下放（可辅助）：跳起或用凳子辅助到最高点，然后用尽可能慢的速度（5-8秒）下放到手臂伸直，强迫核心和背部控制身体。")
    
    # ========== 等级五/六：60分以下 ==========
    else:
        if not chin_over_bar:
            tips.append("【动作评价】下放时手臂未完全伸直，上拉时下巴未过杠。动作行程不完整，肌肉无法得到充分的拉伸与收缩，训练效果差。")
        else:
            tips.append("【动作评价】下放时手臂弯曲幅度较大，手臂未完全伸直。动作行程不完整，需要改进。")
        
        tips.append("【存在问题】动作行程不完整，肌肉无法得到充分的拉伸与收缩，训练效果差。")
        tips.append("【原因分析】1.基础力量严重匮乏：背部、手臂、核心的相对力量严重不足，绝对力量不足以移动自身体重。2.关节活动度可能受限：如肩关节灵活性差，影响手臂完全上举。3.对动作的认知有误，不知道什么是完整动作。")
        tips.append("【训练建议】1.垂直悬垂：目标是能轻松悬垂30秒以上，增强握力和肩部适应力。2.离心训练：在辅助下到达最高点，然后以最慢速度下放，即使只能控制2-3秒也是巨大进步。3.强化辅助：使用能提供大量助力的弹力带或器械，确保能完成完整的动作行程（直上直下）。4.灵活性训练：强化肩、胸部位的灵活性。")
    
    return tips


def _get_frame_near_index(data: dict[str, Any], frame_index: int) -> dict[str, Any]:
    frames: list[dict[str, Any]] = list(data.get("frames") or [])
    if not frames:
        raise AnalysisError(POSE_LOW_CONFIDENCE, "姿态关键点检测失败或整体置信度过低，请按拍摄规范重新录制后上传")

    target = int(frame_index)
    best: dict[str, Any] | None = None
    best_dist: int | None = None
    for f in frames:
        fi = int(f.get("frame_index") or -1)
        if fi == target:
            return f
        dist = abs(fi - target)
        if best_dist is None or dist < best_dist:
            best = f
            best_dist = dist

    return best or frames[0]


def _frame_landmarks_from_frame(frame: dict[str, Any]) -> dict[str, tuple[float, float, float]]:
    lms = frame.get("landmarks") or {}
    out: dict[str, tuple[float, float, float]] = {}
    for k, v in lms.items():
        out[k] = (float(v.get("x")), float(v.get("y")), float(v.get("v")))
    return out


def _render_keyframe(
    *,
    cv2: Any,
    video_path: Path,
    frame_index: int,
    landmarks: dict[str, tuple[float, float, float]],
    highlight_joint: str,
    out_path: Path,
    title: str,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise AnalysisError(POSE_LOW_CONFIDENCE, "姿态关键点检测失败或整体置信度过低，请按拍摄规范重新录制后上传")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise AnalysisError(POSE_LOW_CONFIDENCE, "姿态关键点检测失败或整体置信度过低，请按拍摄规范重新录制后上传")

    _draw_skeleton(cv2=cv2, frame=frame, landmarks=landmarks, highlight_joint=highlight_joint)
    _draw_title_text(cv2=cv2, frame=frame, title=title)
    ensure_parent_dir(out_path)
    _write_image_bytes(cv2=cv2, out_path=out_path, frame=frame)


def _draw_title_text(*, cv2: Any, frame: Any, title: str) -> None:
    # OpenCV putText 对中文支持很差；这里用 PIL 叠字，避免“????”。
    try:
        from PIL import ImageFont  # type: ignore

        candidates = [
            os.environ.get("PULLUP_FONT_PATH", "").strip(),
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyh.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
        ]
        font_path = next((p for p in candidates if p and Path(p).exists()), "")
        font = ImageFont.truetype(font_path, 32) if font_path else ImageFont.load_default()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        draw.text((16, 10), title, font=font, fill=(231, 236, 255))
        frame[:, :, :] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception:
        # 兜底：不画标题也不影响结果
        return


def _write_image_bytes(*, cv2: Any, out_path: Path, frame: Any) -> None:
    # OpenCV 在 Windows 上对“绝对路径 + 中文路径”支持不稳定，改为 imencode + Python 写文件，避免 cv2.imwrite 失败。
    ext = (out_path.suffix or ".png").lower()
    ok, buf = cv2.imencode(ext, frame)
    if not ok:
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            raise AnalysisError(INTERNAL_ERROR, "生成关键帧图片失败，请联系管理员")
    out_path.write_bytes(buf.tobytes())


def _draw_skeleton(*, cv2: Any, frame: Any, landmarks: dict[str, tuple[float, float, float]], highlight_joint: str) -> None:
    def p(name: str) -> tuple[int, int]:
        x, y, _ = landmarks[name]
        return int(round(x)), int(round(y))

    def ok(name: str) -> bool:
        if name not in landmarks:
            return False
        x, y, v = landmarks[name]
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        return float(v) >= 0.15

    connections = [
        ("nose", "left_shoulder"),
        ("nose", "right_shoulder"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("left_ankle", "left_heel"),
        ("left_heel", "left_foot_index"),
        ("left_ankle", "left_foot_index"),
        ("right_ankle", "right_heel"),
        ("right_heel", "right_foot_index"),
        ("right_ankle", "right_foot_index"),
    ]

    for a, b in connections:
        if ok(a) and ok(b):
            cv2.line(frame, p(a), p(b), (110, 255, 198), 2)

    for name in landmarks.keys():
        if ok(name):
            cv2.circle(frame, p(name), 3, (110, 255, 198), -1)

    # 高亮差异关节（支持侧面视角的简化名称）
    # 侧面视角返回 "elbow" / "shoulder" 等，需要映射到实际的 left/right 关节
    highlight_targets = []
    if highlight_joint in landmarks:
        highlight_targets = [highlight_joint]
    else:
        # 尝试匹配 left_xxx 和 right_xxx
        for prefix in ["left_", "right_"]:
            full_name = prefix + highlight_joint
            if full_name in landmarks:
                highlight_targets.append(full_name)
        # 处理 torso/torso_forward/body_swing 等躯干指标
        if "torso" in highlight_joint or "swing" in highlight_joint:
            # 高亮肩膀和髋部连线的中点
            for name in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]:
                if name in landmarks:
                    highlight_targets.append(name)
    
    # 绘制高亮圆圈（红色，更大更明显）
    for target in highlight_targets:
        if ok(target):
            cv2.circle(frame, p(target), 16, (0, 0, 255), 4)  # 红色大圆
            cv2.circle(frame, p(target), 8, (255, 106, 122), -1)  # 粉色实心
