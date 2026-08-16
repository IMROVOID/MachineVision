"""Lightweight structured JSONL logger for infrastructure operations."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


class InfrastructureLogger:
    def __init__(self, log_file_path: str | Path):
        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        event: str,
        level: str = "INFO",
        run_id: Optional[str] = None,
        chunk_id: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.upper(),
            "event": event,
            "run_id": run_id,
            "chunk_id": chunk_id,
            **kwargs,
        }
        line = json.dumps(entry) + "\n"
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
