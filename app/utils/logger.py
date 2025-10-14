from __future__ import annotations

import logging
import json
import uuid
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Iterator
from contextlib import contextmanager


def build_logger(name: str = "preprocess", level: int = logging.INFO) -> logging.Logger:
    """
    Configure a standard Python logger for human-readable console output.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setLevel(level)
        # timestamp | LEVEL | message
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


class PrepLogger:
    """
    Lightweight structured event logger for preprocessing pipelines.

    - pl.add("missing_imputation", msg="handled missing data", cols=["age","salary"], strategy="mean")
    - with pl.step("encode_categoricals", cols=["gender","city"]): ...
    - pl.dump_jsonl("logs/run.jsonl")
    """
    def __init__(
        self,
        name: str = "preprocess",
        run_id: Optional[str] = None,
        level: int = logging.INFO,
        json_console: bool = False,  # if True, console logs are JSON too
    ):
        self.logger = build_logger(name, level)
        self.events: List[Dict[str, Any]] = []
        self.run_id = run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.json_console = json_console
        # mark run start
        self.add("_run_start", msg="preprocessing run started")

    def _emit(self, event: Dict[str, Any]) -> None:
        """
        Emit to console in human or JSON form.
        """
        if self.json_console:
            self.logger.info(json.dumps(event, ensure_ascii=False))
        else:
            msg = event.get("msg", "")
            step = event.get("step", "")
            status = event.get("status", "")
            extra = {k: v for k, v in event.items() if k not in {"ts", "run_id", "step", "msg", "status"}}
            if extra:
                self.logger.info(f"[{step}] {status} {msg} | {extra}")
            else:
                self.logger.info(f"[{step}] {status} {msg}")

    def add(self, step: str, msg: str = "", status: str = "ok", **fields: Any) -> None:
        """
        Record a one-off event (e.g., 'missing data handled').
        """
        event = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "run_id": self.run_id,
            "step": step,
            "status": status,
            "msg": msg,      \
            **fields,
        }
        self.events.append(event)
        self._emit(event)

    @contextmanager
    def step(self, step: str, **fields: Any) -> Iterator[None]:
        """
        Context manager to log duration and errors for a block of work.
        Usage:
            with pl.step("impute_numerical", strategy="mean"):
                ... work ...
        """
        start = time.perf_counter()
        self.add(step, msg="start", status="start", **fields)
        try:
            yield
        except Exception as e:
            dur_ms = int((time.perf_counter() - start) * 1000)
            self.add(step, msg=f"error: {type(e).__name__}: {e}", status="error", duration_ms=dur_ms, **fields)
            raise
        else:
            dur_ms = int((time.perf_counter() - start) * 1000)
            self.add(step, msg="done", status="ok", duration_ms=dur_ms, **fields)

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self.events)

    def dump_jsonl(self, path: str) -> None:
        """
        Persist the history as JSON Lines (one event per line).
        """
        with open(path, "a", encoding="utf-8") as f:
            for ev in self.events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def reset(self) -> None:
        self.events.clear()

    def end_run(self, status: str = "success", **fields: Any) -> None:
        self.add("_run_end", msg="preprocessing run finished", status=status, **fields)
