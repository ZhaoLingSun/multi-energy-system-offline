#!/usr/bin/env python3
"""Run full MILP and all thought-question experiments in sequence."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
THOUGHT_DIR = PROJECT_ROOT / "thought_questions"


def _latest_summary(root: Path) -> Optional[Path]:
    summaries = sorted(root.glob("full_milp_*/full_milp_summary.json"), reverse=True)
    return summaries[0] if summaries else None


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full MILP and all thought questions")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--renewable-dir", default=None)
    parser.add_argument("--full-milp-output", default="runs/full_milp")
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=0.0)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--skip-full-milp", action="store_true")
    parser.add_argument("--skip-q1", action="store_true")
    parser.add_argument("--skip-q2", action="store_true")
    parser.add_argument("--skip-q3", action="store_true")
    parser.add_argument("--skip-q4", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    renewable_dir = Path(args.renewable_dir) if args.renewable_dir else data_dir
    full_output = Path(args.full_milp_output)

    if not args.skip_full_milp:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "run_full_milp.py"),
            "--data-dir",
            str(data_dir),
            "--renewable-dir",
            str(renewable_dir),
            "--output-dir",
            str(full_output),
            "--mip-gap",
            str(args.mip_gap),
            "--days",
            str(args.days),
            "--carbon-mode",
            "aligned",
            "--objective",
            "cost",
        ]
        if args.threads > 0:
            cmd += ["--threads", str(args.threads)]
        if args.time_limit > 0:
            cmd += ["--time-limit", str(args.time_limit)]
        _run(cmd)

    summary = _latest_summary(full_output)
    if summary is None or not summary.exists():
        raise FileNotFoundError("无法找到 full_milp_summary.json，请确认主问题已运行")

    if not args.skip_q1:
        cmd = [
            sys.executable,
            str(THOUGHT_DIR / "q1_retrofit" / "run_q1_retrofit.py"),
            "--baseline-summary",
            str(summary),
            "--data-dir",
            str(data_dir),
            "--mip-gap",
            str(max(args.mip_gap, 1e-3)),
        ]
        if args.threads > 0:
            cmd += ["--threads", str(args.threads)]
        if args.time_limit > 0:
            cmd += ["--time-limit", str(args.time_limit)]
        _run(cmd)

    if not args.skip_q2:
        cmd = [
            sys.executable,
            str(THOUGHT_DIR / "q2_seasonal_storage" / "run_q2_seasonal_storage.py"),
            "--baseline-summary",
            str(summary),
            "--data-dir",
            str(data_dir),
        ]
        if args.threads > 0:
            cmd += ["--threads", str(args.threads)]
        _run(cmd)

    if not args.skip_q3:
        cmd = [
            sys.executable,
            str(THOUGHT_DIR / "q3_linepack" / "run_milp_linepack.py"),
            "--plan18-summary",
            str(summary),
            "--data-dir",
            str(data_dir),
            "--days",
            str(args.days),
        ]
        if args.threads > 0:
            cmd += ["--threads", str(args.threads)]
        if args.time_limit > 0:
            cmd += ["--time-limit", str(args.time_limit)]
        _run(cmd)

    if not args.skip_q4:
        cmd = [
            sys.executable,
            str(THOUGHT_DIR / "q4_line_capacity" / "run_milp_line_capacity.py"),
            "--plan18-summary",
            str(summary),
            "--data-dir",
            str(data_dir),
            "--days",
            str(args.days),
        ]
        if args.threads > 0:
            cmd += ["--threads", str(args.threads)]
        if args.time_limit > 0:
            cmd += ["--time-limit", str(args.time_limit)]
        _run(cmd)


if __name__ == "__main__":
    main()
