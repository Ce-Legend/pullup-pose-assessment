from __future__ import annotations

from dataclasses import dataclass


INVALID_MEDIA = "INVALID_MEDIA"
INVALID_DURATION = "INVALID_DURATION"
POSE_LOW_CONFIDENCE = "POSE_LOW_CONFIDENCE"
VIEW_MISMATCH = "VIEW_MISMATCH"
INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class AnalysisError(Exception):
    error_code: str
    error_message: str

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.error_code}: {self.error_message}"


def user_message_for(code: str) -> str:
    mapping = {
        INVALID_MEDIA: "视频格式不支持或无法解析，请上传 MP4 文件",
        INVALID_DURATION: "视频时长需在 1秒–5分钟内，请检查视频文件",
        POSE_LOW_CONFIDENCE: "姿态关键点检测失败或整体置信度过低，请按拍摄规范重新录制后上传",
        VIEW_MISMATCH: "视角不一致，请按标准正面视角重新上传",
        INTERNAL_ERROR: "系统内部错误，请联系管理员",
    }
    return mapping.get(code, "分析失败，请稍后重试")

