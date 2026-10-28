from __future__ import annotations

import asyncio
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_lifecycle import AgentContext, HookRegistry


async def noop(context):
    return None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


async def benchmark(repetitions: int = 1200) -> dict[str, float | int | bool]:
    baseline, instrumented = HookRegistry(), HookRegistry()
    for index in range(3):
        instrumented.register_pre(noop, priority=index)
    context = AgentContext()
    base_samples, hooked_samples = [], []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        await baseline.dispatch_pre(context)
        base_samples.append((time.perf_counter_ns() - start) / 1_000_000)
        start = time.perf_counter_ns()
        await instrumented.dispatch_pre(context)
        hooked_samples.append((time.perf_counter_ns() - start) / 1_000_000)
    mean_overhead = max(0.0, statistics.mean(hooked_samples) - statistics.mean(base_samples))
    p95_overhead = max(0.0, percentile(hooked_samples, 0.95) - percentile(base_samples, 0.95))
    return {
        "repetitions": repetitions,
        "hooks_added": 3,
        "baseline_mean_ms": round(statistics.mean(base_samples), 6),
        "three_hook_mean_ms": round(statistics.mean(hooked_samples), 6),
        "mean_overhead_ms": round(mean_overhead, 6),
        "p95_overhead_ms": round(p95_overhead, 6),
        "under_5ms_target": p95_overhead < 5.0,
    }


def main() -> int:
    result = asyncio.run(benchmark())
    report_path = ROOT / "reports" / "benchmark.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Benchmark report: reports/benchmark.json")
    return 0 if result["under_5ms_target"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
