# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B gate for model optimization PRs: faster, and no less accurate.

`measure` benchmarks a base and a candidate server that serve the same model
with the same command, side by side, in alternating order over several rounds,
then runs GSM8K on both. `decide` reads those results and passes only if:

- every run is valid (at most 1% failed requests),
- no point regresses beyond the tolerance (RFC #59665: 1%, or the measured
  round-to-round spread if larger),
- at least one point improves beyond the tolerance,
- GSM8K drops by at most `max(1 point, one combined standard error)`.

Usage:
    python tools/model_gate.py measure --model M --base http://host:8000 \
        --candidate http://host:8001 --concurrency 1 8 64 --out gate/
    python tools/model_gate.py decide gate/
"""

import argparse
import json
import math
import statistics
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

SIDES = ("base", "candidate")
# Result field of `vllm bench serve` -> whether higher is better.
METRICS = {
    "output_throughput": True,
    "median_ttft_ms": False,
    "median_itl_ms": False,
}
MAX_FAILED_FRACTION = 0.01
MIN_TOLERANCE = 0.01
MIN_ACCURACY_DROP = 0.01
GSM8K_EVAL = Path(__file__).parents[1] / "tests/evals/gsm8k/gsm8k_eval.py"


def bench_command(
    args: argparse.Namespace, url: str, concurrency: int, seed: int, out: Path
) -> list[str]:
    num_prompts = max(100, 10 * concurrency)
    return [
        "vllm", "bench", "serve",
        "--model", args.model,
        "--base-url", url,
        "--dataset-name", "random",
        "--random-input-len", str(args.input_len),
        "--random-output-len", str(args.output_len),
        "--ignore-eos",
        "--seed", str(seed),
        "--max-concurrency", str(concurrency),
        "--num-prompts", str(num_prompts),
        "--num-warmups", str(max(2 * concurrency, 16)),
        "--save-result",
        "--result-dir", str(out.parent),
        "--result-filename", out.name,
    ]  # fmt: skip


def gsm8k_command(args: argparse.Namespace, url: str, out: Path) -> list[str]:
    parsed = urllib.parse.urlsplit(url)
    return [
        sys.executable, str(GSM8K_EVAL),
        "--host", f"{parsed.scheme}://{parsed.hostname}",
        "--port", str(parsed.port),
        "--num-questions", str(args.gsm8k_questions),
        "--save-results", str(out),
    ]  # fmt: skip


def measure(args: argparse.Namespace) -> None:
    urls = {"base": args.base, "candidate": args.candidate}
    for round_ in range(args.rounds):
        # Alternate which side goes first so drift over time hits both.
        order = SIDES if round_ % 2 == 0 else SIDES[::-1]
        for concurrency in args.concurrency:
            for side in order:
                out = args.out / side / f"r{round_}" / f"c{concurrency}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                # The same prompts on both sides, never repeated on one server,
                # so no run is served from the prefix cache of an earlier one.
                seed = args.seed + 1000 * round_ + concurrency
                subprocess.run(
                    bench_command(args, urls[side], concurrency, seed, out),
                    check=True,
                )
    for side in SIDES:
        out = args.out / side / "gsm8k.json"
        subprocess.run(gsm8k_command(args, urls[side], out), check=True)


@dataclass(frozen=True)
class Point:
    concurrency: int
    metric: str
    base: list[float]
    candidate: list[float]

    @property
    def change(self) -> float:
        """Relative change of the candidate's mean, positive is better."""
        base, candidate = statistics.mean(self.base), statistics.mean(self.candidate)
        change = (candidate - base) / base
        return change if METRICS[self.metric] else -change

    @property
    def tolerance(self) -> float:
        spreads = [
            (max(values) - min(values)) / statistics.mean(values)
            for values in (self.base, self.candidate)
        ]
        return max(MIN_TOLERANCE, *spreads)


@dataclass(frozen=True)
class Accuracy:
    base: float
    candidate: float
    num_questions: int

    @property
    def allowed_drop(self) -> float:
        variance = sum(p * (1 - p) for p in (self.base, self.candidate))
        return max(MIN_ACCURACY_DROP, math.sqrt(variance / self.num_questions))

    @property
    def passed(self) -> bool:
        return self.base - self.candidate <= self.allowed_drop


@dataclass
class Verdict:
    points: list[Point]
    accuracy: Accuracy | None
    invalid: list[str]

    @property
    def regressions(self) -> list[Point]:
        return [p for p in self.points if p.change < -p.tolerance]

    @property
    def gains(self) -> list[Point]:
        return [p for p in self.points if p.change > p.tolerance]

    @property
    def passed(self) -> bool:
        return (
            not self.invalid
            and not self.regressions
            and bool(self.gains)
            and self.accuracy is not None
            and self.accuracy.passed
        )


def load_results(root: Path) -> Verdict:
    runs: dict[tuple[str, int], list[dict]] = {}
    invalid = []
    for path in sorted(root.glob("*/r*/c*.json")):
        side = path.parts[-3]
        result = json.loads(path.read_text())
        failed, prompts = result.get("failed", 0), result["num_prompts"]
        if failed > MAX_FAILED_FRACTION * prompts:
            invalid.append(f"{path.relative_to(root)}: {failed}/{prompts} failed")
        runs.setdefault((side, int(path.stem[1:])), []).append(result)

    points = []
    for concurrency in sorted({c for _, c in runs}):
        if not all((side, concurrency) in runs for side in SIDES):
            invalid.append(f"c{concurrency}: missing a side")
            continue
        for metric in METRICS:
            base, candidate = (
                [run[metric] for run in runs[(side, concurrency)]] for side in SIDES
            )
            points.append(Point(concurrency, metric, base, candidate))

    accuracy = None
    gsm8k = [root / side / "gsm8k.json" for side in SIDES]
    if all(path.exists() for path in gsm8k):
        base, candidate = (json.loads(path.read_text()) for path in gsm8k)
        accuracy = Accuracy(
            base["accuracy"], candidate["accuracy"], base["num_questions"]
        )
    return Verdict(points, accuracy, invalid)


def render(verdict: Verdict) -> str:
    lines = [
        "| Concurrency | Metric | Base | Candidate | Change | Tolerance |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for p in verdict.points:
        lines.append(
            f"| {p.concurrency} | {p.metric} | {statistics.mean(p.base):.4g} | "
            f"{statistics.mean(p.candidate):.4g} | {p.change:+.1%} | "
            f"±{p.tolerance:.1%} |"
        )
    lines.append("")
    if verdict.accuracy:
        a = verdict.accuracy
        lines.append(
            f"GSM8K: base {a.base:.3f}, candidate {a.candidate:.3f}, "
            f"allowed drop {a.allowed_drop:.3f}: {'ok' if a.passed else 'FAIL'}"
        )
    else:
        lines.append("GSM8K: missing")
    for problem in verdict.invalid:
        lines.append(f"Invalid: {problem}")
    for p in verdict.regressions:
        lines.append(f"Regression: c{p.concurrency} {p.metric} {p.change:+.1%}")
    if not verdict.gains:
        lines.append("No point improves beyond its tolerance.")
    lines.append(f"Verdict: {'PASS' if verdict.passed else 'FAIL'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("measure", help="benchmark both servers")
    run.add_argument("--model", required=True)
    run.add_argument("--base", required=True, help="base server URL")
    run.add_argument("--candidate", required=True, help="candidate server URL")
    run.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 64])
    run.add_argument("--input-len", type=int, default=1024)
    run.add_argument("--output-len", type=int, default=256)
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--gsm8k-questions", type=int, default=1319)
    run.add_argument("--out", type=Path, required=True)
    decide = commands.add_parser("decide", help="judge the measured results")
    decide.add_argument("out", type=Path)
    args = parser.parse_args(argv)

    if args.command == "measure":
        measure(args)
        return 0
    verdict = load_results(args.out)
    print(render(verdict))
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
