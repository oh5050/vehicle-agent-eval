"""Tests for run directory and JSONL logging utilities."""

from pathlib import Path

from src.run_log import RunLogger, append_log, create_run_dir, generate_run_id


def test_generate_run_id_format():
    from datetime import datetime, timezone

    run_id = generate_run_id("e1", timestamp=datetime(2026, 7, 9, 7, 43, 0, tzinfo=timezone.utc))
    assert run_id == "e1_20260709_074300"


def test_create_run_dir_and_append_log(tmp_path: Path):
    run_id = "test_run"
    run_dir = create_run_dir(run_id, base_dir=tmp_path)

    assert run_dir.is_dir()
    assert (run_dir / "logs.jsonl").is_file()

    append_log({"event": "test", "latency_ms": 12.3}, run_dir=run_dir)
    lines = (run_dir / "logs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"event":"test"' in lines[0]


def test_run_logger(tmp_path: Path):
    logger = RunLogger("e1_smoke", base_dir=tmp_path)
    logger.log({"step": "inference", "model": "llama3.2"})

    content = logger.logs_path.read_text(encoding="utf-8")
    assert '"step":"inference"' in content
