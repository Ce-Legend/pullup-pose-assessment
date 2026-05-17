from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Mp4ParseError(Exception):
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


def get_mp4_duration_ms(path: Path) -> int:
    """
    以最小依赖解析 MP4 的 moov/mvhd，返回时长（毫秒）。
    仅用于“5–20 秒”校验与基础信息，不做完整媒体解析。
    """

    file_size = path.stat().st_size
    if file_size < 16:
        raise Mp4ParseError("文件过小，无法解析")

    with path.open("rb") as f:
        moov_start, moov_size, moov_header = _find_box(f, file_size, b"moov")
        if moov_start is None:
            raise Mp4ParseError("未找到 moov box")
        f.seek(moov_start + moov_header)
        moov_end = moov_start + moov_size

        mvhd_start, mvhd_size, mvhd_header = _find_box(f, moov_end, b"mvhd")
        if mvhd_start is None:
            raise Mp4ParseError("未找到 mvhd box")
        f.seek(mvhd_start + mvhd_header)
        mvhd_payload = f.read(mvhd_size - mvhd_header)
        if len(mvhd_payload) < 20:
            raise Mp4ParseError("mvhd 数据不足")

        version = mvhd_payload[0]
        if version == 0:
            # version(1) + flags(3) + creation(4) + modification(4) + timescale(4) + duration(4)
            if len(mvhd_payload) < 24:
                raise Mp4ParseError("mvhd v0 数据不足")
            timescale = struct.unpack(">I", mvhd_payload[12:16])[0]
            duration = struct.unpack(">I", mvhd_payload[16:20])[0]
        elif version == 1:
            # version(1) + flags(3) + creation(8) + modification(8) + timescale(4) + duration(8)
            if len(mvhd_payload) < 36:
                raise Mp4ParseError("mvhd v1 数据不足")
            timescale = struct.unpack(">I", mvhd_payload[20:24])[0]
            duration = struct.unpack(">Q", mvhd_payload[24:32])[0]
        else:
            raise Mp4ParseError(f"不支持的 mvhd version: {version}")

        if timescale <= 0:
            raise Mp4ParseError("timescale 无效")

        duration_sec = duration / timescale
        duration_ms = int(duration_sec * 1000)
        if duration_ms <= 0:
            raise Mp4ParseError("duration 无效")
        return duration_ms


def _read_u32(f) -> int:
    b = f.read(4)
    if len(b) != 4:
        raise Mp4ParseError("读取 box size 失败")
    return struct.unpack(">I", b)[0]


def _read_type(f) -> bytes:
    t = f.read(4)
    if len(t) != 4:
        raise Mp4ParseError("读取 box type 失败")
    return t


def _find_box(f, end_pos: int, target_type: bytes) -> tuple[int | None, int, int]:
    """
    从当前位置开始，扫描到 end_pos（文件结束或父 box 结束），找到目标 box。
    返回：(box_start, box_size, header_size)。找不到则返回 (None, 0, 0)。
    """

    while f.tell() + 8 <= end_pos:
        box_start = f.tell()
        size = _read_u32(f)
        box_type = _read_type(f)

        header_size = 8
        if size == 1:
            largesize_bytes = f.read(8)
            if len(largesize_bytes) != 8:
                raise Mp4ParseError("读取 largesize 失败")
            size = struct.unpack(">Q", largesize_bytes)[0]
            header_size = 16
        elif size == 0:
            size = end_pos - box_start

        if size < header_size:
            raise Mp4ParseError("box size 无效")

        if box_type == target_type:
            return box_start, int(size), header_size

        # 跳到下一个 box
        f.seek(box_start + int(size))

    return None, 0, 0

