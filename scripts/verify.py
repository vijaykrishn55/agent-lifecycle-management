from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "verification.json"


def run(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return completed.returncode, (completed.stdout + "\n" + completed.stderr).strip()


def main() -> int:
    test_code, test_output = run([sys.executable, "-m", "pytest", "-q"])
    benchmark_code, benchmark_output = run([sys.executable, "scripts/benchmark.py"])
    report = {
        "project": "Topic 9 - Agent Lifecycle Management",
        "test_command": "python -m pytest -q",
        "exit_code": test_code if test_code else benchmark_code,
        "passed": test_code == 0 and benchmark_code == 0,
        "output": test_output,
        "benchmark_output": benchmark_output,
        "checks": ["hook order", "sync/async hooks", "multi-turn context", "token tracking",
                    "prompt/session budget guard", "context trace", "JSONL telemetry",
                    "fallback handling", "circuit breaker", "three-hook overhead"],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(test_output)
    print("\nBenchmark:\n" + benchmark_output)
    print("\nVerification:", "PASS" if report["passed"] else "FAIL")
    print("Report: reports/verification.json")
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
