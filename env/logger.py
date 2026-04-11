import json
import os
from datetime import datetime, timezone
from typing import Optional


class EpisodeLogger:
    def __init__(self, log_dir: str = "logs", task: str = "unknown", run_id: Optional[str] = None):
        os.makedirs(log_dir, exist_ok=True)
        self.task = task
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._path = os.path.join(log_dir, f"{task}_{self.run_id}.jsonl")
        self._file = open(self._path, "w", encoding="utf-8")
        self._step = 0

    def log_step(self, obs_dict: dict, action: str, reward: float, done: bool, info: dict) -> None:
        self._step += 1
        entry = {
            "run_id": self.run_id,
            "task": self.task,
            "step": self._step,
            "ts": datetime.now(timezone.utc).isoformat(),
            "observation": obs_dict,
            "action": action,
            "reward": reward,
            "done": done,
            "info": info,
        }
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def close(self) -> str:
        self._file.close()
        return self._path

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
