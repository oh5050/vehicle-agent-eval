"""Run directory creation and JSONL logging for experiment traces."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LOGS_FILENAME = "logs.jsonl"


def generate_run_id(experiment_name: str, *, timestamp: datetime | None = None) -> str:
    """Return a deterministic-ish run id: ``{experiment}_{YYYYMMDD_HHMMSS}``."""
    ts = timestamp or datetime.now(timezone.utc)
    stamp = ts.strftime("%Y%m%d_%H%M%S")
    return f"{experiment_name}_{stamp}"


def create_run_dir(
    run_id: str,
    *,
    base_dir: Path | str = "runs",
    logs_filename: str = DEFAULT_LOGS_FILENAME,
) -> Path:
    """Create ``runs/{run_id}/`` and return its path.

    Creates the directory if it does not exist. Does not overwrite existing runs.
    """
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Touch logs file so downstream code can rely on the path existing.
    (run_dir / logs_filename).touch(exist_ok=True)
    return run_dir


def append_log(
    record: dict[str, Any],
    *,
    run_dir: Path,
    logs_filename: str = DEFAULT_LOGS_FILENAME,
) -> None:
    """Append one JSON-serializable record as a single JSONL line."""
    log_path = run_dir / logs_filename
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


class RunLogger:
    """Append-only JSONL logger bound to a single run directory."""

    def __init__(
        self,
        run_id: str,
        *,
        base_dir: Path | str = "runs",
        logs_filename: str = DEFAULT_LOGS_FILENAME,
    ) -> None:
        self.run_id = run_id
        self.logs_filename = logs_filename
        self.run_dir = create_run_dir(run_id, base_dir=base_dir, logs_filename=logs_filename)

    @property
    def logs_path(self) -> Path:
        """Absolute path to ``runs/{run_id}/logs.jsonl``."""
        return self.run_dir / self.logs_filename

    def log(self, record: dict[str, Any]) -> None:
        """Append a record; callers should include latency and parse metadata."""
        append_log(record, run_dir=self.run_dir, logs_filename=self.logs_filename)
