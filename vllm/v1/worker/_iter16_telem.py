# SPDX-License-Identifier: Apache-2.0
"""iter-16 observability: env-gated per-stage timestamps for the stall
investigation. Inert unless VLLM_ITER16_TELEM points at a directory.
Timing is host-only (time.monotonic_ns) and capture-safe: no device reads,
no synchronization.
"""
import os

ITER16_DIR = os.environ.get("VLLM_ITER16_TELEM")
_fd = None


def enabled() -> bool:
    return bool(ITER16_DIR)


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
