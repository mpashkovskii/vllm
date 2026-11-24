# SPDX-License-Identifier: Apache-2.0
"""iter-16 observability: env-gated per-stage timestamps for the stall
investigation. Inert unless VLLM_ITER16_TELEM points at a directory.
Timing is host-only (time.monotonic_ns) and capture-safe: no device reads,
no synchronization.
"""
import os
import time

ITER16_DIR = os.environ.get("VLLM_ITER16_TELEM")
_fd = None


def enabled() -> bool:
    return bool(ITER16_DIR)


class Sect:
    """iter-21: cheap section timer. Marks record microseconds since the
    previous mark; flush() emits one JSON line. Host-only, capture-safe."""

    __slots__ = ("t0", "_prev", "rec")

    def __init__(self, **kv):
        self.rec = dict(kv)
        self.t0 = self._prev = time.monotonic_ns()

    def m(self, name: str) -> None:
        now = time.monotonic_ns()
        self.rec[name] = (now - self._prev) // 1_000
        self._prev = now

    def flush(self) -> None:
        import json

        self.rec["tot"] = (time.monotonic_ns() - self.t0) // 1_000
        self.rec["t0abs"] = self.t0
        line(json.dumps(self.rec))


def new_sect(**kv) -> "Sect | None":
    if not ITER16_DIR:
        return None
    return Sect(**kv)


def line(rec: str) -> None:
    if not ITER16_DIR:
        return
    global _fd
    if _fd is None:
        _fd = os.open(
            os.path.join(ITER16_DIR, f"proc_{os.getpid()}.jsonl"),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
    try:
        os.write(_fd, rec.encode() + b"\n")
    except OSError:
        pass
