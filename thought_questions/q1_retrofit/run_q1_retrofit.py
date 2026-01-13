#!/usr/bin/env python3
"""
思考题1：建筑节能改造效益量化（基于主优化基线）。

基线=主优化问题的最优解（baseline summary），本脚本不复算基线。
做法：将所有热负荷按 0.11/0.35 缩放后，重新运行全年 MILP 优化，
并对比两种情形的总成本/碳成本/OJ 评分与排放变化。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


_SCRIPT_DIR = Path(__file__).resolve().parent
_PYTHON_DIR = _SCRIPT_DIR.parents[1]
_PROJECT_ROOT = _PYTHON_DIR.parent


def _latest_full_milp_summary(root: Path) -> Optional[Path]:
    summaries = sorted(root.glob("full_milp_*/full_milp_summary.json"), reverse=True)
    return summaries[0] if summaries else None


def _load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scale_heat_loads(data_dir: Path, output_dir: Path, scale: float) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(data_dir, output_dir)

    heat_files = [
        "负荷曲线_热_学生区.csv",
        "负荷曲线_热_教学办公区.csv",
        "负荷曲线_热_教工区.csv",
        "汇总_负荷曲线_热.csv",
    ]
    for csv_name in heat_files:
        csv_path = output_dir / csv_name
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce") * scale
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return output_dir


def _run_full_milp(
    data_dir: Path,
    output_dir: Path,
    mip_gap: float,
    threads: int,
    time_limit: float,
) -> Path:
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "python" / "scripts" / "run_full_milp.py"),
        "--data-dir",
        str(data_dir),
        "--renewable-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--mip-gap",
        str(mip_gap),
        "--carbon-mode",
        "aligned",
        "--objective",
        "cost",
    ]
    if threads > 0:
        cmd += ["--threads", str(threads)]
    if time_limit > 0:
        cmd += ["--time-limit", str(time_limit)]

    subprocess.run(cmd, check=True, cwd=str(_PROJECT_ROOT))

    summaries = sorted(output_dir.glob("full_milp_*/full_milp_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not summaries:
        raise FileNotFoundError("未找到 full_milp_summary.json")
    return summaries[0]


def _extract_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    carbon = data.get("carbon", {})
    return {
        "C_total": data.get("C_total"),
        "C_OP": data.get("C_OP"),
        "C_Carbon": data.get("C_Carbon"),
        "Score": data.get("Score"),
        "emission_total": carbon.get("emission_total"),
        "emission_excess": carbon.get("emission_excess"),
        "plan18": data.get("plan18"),
        "summary_path": data.get("run_dir"),
    }


def _compute_delta(base: Dict[str, Any], retrofit: Dict[str, Any]) -> Dict[str, Any]:
    delta = {}
    for key in ("C_total", "C_OP", "C_Carbon", "Score", "emission_total", "emission_excess"):
        if base.get(key) is None or retrofit.get(key) is None:
            continue
        delta[key] = base[key] - retrofit[key]
    return delta


def main() -> None:
    parser = argparse.ArgumentParser(description="思考题1：建筑节能改造效益量化（重新优化）")
    parser.add_argument("--baseline-summary", default=None, help="主优化 summary.json（baseline）")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="runs/thought_questions")
    parser.add_argument("--intensity-before", type=float, default=0.35)
    parser.add_argument("--intensity-after", type=float, default=0.11)
    parser.add_argument("--mip-gap", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=0.0)
    args = parser.parse_args()

    baseline_path = Path(args.baseline_summary) if args.baseline_summary else _latest_full_milp_summary(_PROJECT_ROOT / "runs" / "full_milp")
    if not baseline_path or not baseline_path.exists():
        raise FileNotFoundError("未找到 baseline summary，请用 --baseline-summary 指定")

    baseline = _load_summary(baseline_path)
    baseline_metrics = _extract_metrics(baseline)

    scale = args.intensity_after / args.intensity_before
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"q1_retrofit_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    scaled_data_dir = _scale_heat_loads(Path(args.data_dir), run_dir / "data_scaled", scale)

    retrofit_summary_path = _run_full_milp(
        scaled_data_dir,
        run_dir,
        args.mip_gap,
        args.threads,
        args.time_limit,
    )

    retrofit = _load_summary(retrofit_summary_path)
    retrofit_metrics = _extract_metrics(retrofit)

    summary = {
        "baseline_summary": str(baseline_path),
        "retrofit_summary": str(retrofit_summary_path),
        "intensity_before": args.intensity_before,
        "intensity_after": args.intensity_after,
        "scale_heat": scale,
        "baseline": baseline_metrics,
        "retrofit": retrofit_metrics,
        "delta": _compute_delta(baseline_metrics, retrofit_metrics),
        "timestamp": datetime.now().isoformat(),
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Q1 summary saved to: {run_dir / 'summary.json'}")
    print(f"delta C_total={summary['delta'].get('C_total')}, C_Carbon={summary['delta'].get('C_Carbon')}, Score={summary['delta'].get('Score')}")


if __name__ == "__main__":
    main()
