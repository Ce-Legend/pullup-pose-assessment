from pathlib import Path

import pytest

from server.analysis.mp4 import Mp4ParseError, get_mp4_duration_ms


def test_get_mp4_duration_rejects_tiny_file(tmp_path: Path):
    video = tmp_path / "tiny.mp4"
    video.write_bytes(b"mp4")

    with pytest.raises(Mp4ParseError):
        get_mp4_duration_ms(video)
