from pathlib import Path

from server.analysis.pipeline import run_mock
from server.settings import load_settings


def test_load_settings_uses_data_dir_from_environment(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "runtime-data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    settings = load_settings()

    assert settings.data_dir == data_dir.resolve()
    assert settings.db_path == (data_dir / "app.db").resolve()
    assert settings.uploads_dir == (data_dir / "uploads").resolve()
    assert settings.results_dir == (data_dir / "results").resolve()


def test_run_mock_writes_result_and_keyframes(tmp_path: Path):
    artifacts = run_mock(tmp_path)

    assert artifacts.score_total == 86
    assert artifacts.diff_joint == "left_elbow"
    assert artifacts.result_json_path.exists()
    assert artifacts.image_standard_path.exists()
    assert artifacts.image_student_path.exists()
