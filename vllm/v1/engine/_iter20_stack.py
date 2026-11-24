# SPDX-License-Identifier: Apache-2.0
"""iter-20 observability: in-process main-thread stack sampler + GC timing
for the engine-side fleet-pause root cause. Inert unless VLLM_ITER16_TELEM
points at a directory (same gate as _iter16_telem). Host-only, capture-safe,
no device access.
"""
import gc
import json
import os
import sys
import threading
import time

_DIR = os.environ.get("VLLM_ITER16_TELEM")
_started = False


def _stack_lines(frames):
    out = []
    f = frames
    while f is not None and len(out) < 10:
        co = f.f_code
        fn = co.co_filename.rsplit("/", 3)[-1]
        out.append(f"{fn}:{f.f_lineno}:{co.co_name}")
        f = f.f_back
    return " <| ".join(out)


def start() -> None:
    """Start the engine-side samplers. Called once per process; no-op when
    the env is unset."""
    global _started
    if _started or not _DIR:
        return
    _started = True
    main_ident = threading.main_thread().ident
    fd = os.open(
        os.path.join(_DIR, f"stacks_{os.getpid()}.jsonl"),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )

    def emit(rec: dict) -> None:
        try:
            os.write(fd, json.dumps(rec).encode() + b"\n")
        except OSError:
            pass

    # GC collection timing: names the collector as the op if engine CPU burns
    # coincide with collection windows.
    _gc_t0 = [0]

    def _gc_cb(phase: str, info: dict) -> None:
        if phase == "start":
            _gc_t0[0] = time.monotonic_ns()
        elif phase == "stop" and _gc_t0[0]:
            emit({
                "gc": 1,
                "ts": _gc_t0[0],
                "us": (time.monotonic_ns() - _gc_t0[0]) // 1000,
                "gen": info.get("generation"),
                "collected": info.get("collected"),
            })
            _gc_t0[0] = 0

    gc.callbacks.append(_gc_cb)
    emit({"boot": 1, "ts": time.monotonic_ns(), "pid": os.getpid()})

    def sampler() -> None:
        tf = sys._current_frames
        while True:
            time.sleep(0.005)
            fr = tf().get(main_ident)
            if fr is None:
                continue
            emit({"st": 1, "ts": time.monotonic_ns(), "stack": _stack_lines(fr)})

    threading.Thread(target=sampler, daemon=True).start()
