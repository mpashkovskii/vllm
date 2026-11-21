# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model gate's verdict: a candidate must be faster somewhere, slower
nowhere beyond noise, valid, and no less accurate on GSM8K."""

import json
from pathlib import Path

from tools.model_gate import load_results, main


def write_run(root: Path, side: str, round_: int, concurrency: int, **fields):
    result = {
        "num_prompts": 100,
        "failed": 0,
        "output_throughput": 1000.0,
        "median_ttft_ms": 100.0,
        "median_itl_ms": 10.0,
    } | fields
    path = root / side / f"r{round_}" / f"c{concurrency}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))


def write_gsm8k(root: Path, base: float, candidate: float, n: int = 1319):
    for side, accuracy in (("base", base), ("candidate", candidate)):
        (root / side / "gsm8k.json").write_text(
            json.dumps({"accuracy": accuracy, "num_questions": n})
        )


def write_ab(root: Path, base: list[float], candidate: list[float], **kwargs):
    """Throughputs per round at c8; other metrics equal on both sides."""
    for round_, (a, b) in enumerate(zip(base, candidate)):
        write_run(root, "base", round_, 8, output_throughput=a)
        write_run(root, "candidate", round_, 8, output_throughput=b, **kwargs)


def test_faster_and_as_accurate_passes(tmp_path: Path):
    write_ab(tmp_path, [1000, 1005, 995], [1100, 1105, 1095])
    write_gsm8k(tmp_path, base=0.80, candidate=0.795)
    assert main(["decide", str(tmp_path)]) == 0


def test_regression_beyond_one_percent_fails(tmp_path: Path):
    write_ab(tmp_path, [1000] * 3, [1100] * 3, median_itl_ms=10.2)
    write_gsm8k(tmp_path, 0.8, 0.8)
    verdict = load_results(tmp_path)
    assert [(p.metric, round(p.change, 3)) for p in verdict.regressions] == [
        ("median_itl_ms", -0.02)
    ]
    assert not verdict.passed


def test_noise_widens_the_tolerance(tmp_path: Path):
    # The base varies by 10% between rounds, so a 5% gain is not a gain.
    write_ab(tmp_path, [950, 1050, 1000], [1050] * 3)
    write_gsm8k(tmp_path, 0.8, 0.8)
    verdict = load_results(tmp_path)
    [throughput] = [p for p in verdict.points if p.metric == "output_throughput"]
    assert round(throughput.tolerance, 3) == 0.1
    assert verdict.gains == [] and not verdict.passed


def test_failed_requests_invalidate_the_run(tmp_path: Path):
    write_ab(tmp_path, [1000] * 3, [1100] * 3, failed=2)
    write_gsm8k(tmp_path, 0.8, 0.8)
    verdict = load_results(tmp_path)
    assert len(verdict.invalid) == 3 and not verdict.passed


def test_accuracy_drop_beyond_one_standard_error_fails(tmp_path: Path):
    write_ab(tmp_path, [1000] * 3, [1100] * 3)
    write_gsm8k(tmp_path, base=0.80, candidate=0.77)
    verdict = load_results(tmp_path)
    assert round(verdict.accuracy.allowed_drop, 3) == 0.016
    assert not verdict.accuracy.passed and not verdict.passed
