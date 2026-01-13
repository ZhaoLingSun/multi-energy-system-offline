#!/usr/bin/env python3
"""
思考题1（方案A）：建筑节能改造对比实验。

两个场景分别运行完整MILP优化：
- baseline: 当前状态，热负荷缩放=1.0
- retrofit: 改造后，热负荷缩放=0.11/0.35≈0.314

通过命令行调用 run_full_milp.py 并临时修改负荷数据。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


def scale_heat_loads(data_dir: Path, output_dir: Path, scale: float) -> Path:
    """复制数据目录并缩放热负荷。"""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(data_dir, output_dir)
    
    # 缩放热负荷CSV（三个分区 + 汇总）
    heat_files = [
        "负荷曲线_热_学生区.csv",
        "负荷曲线_热_教学办公区.csv",
        "负荷曲线_热_教工区.csv",
        "汇总_负荷曲线_热.csv",
    ]
    for csv_name in heat_files:
        csv_path = output_dir / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # 跳过第一列（时间/日期），缩放数值列
            for col in df.columns[1:]:
                if df[col].dtype in (np.float64, np.int64, float, int):
                    df[col] = df[col] * scale
            df.to_csv(csv_path, index=False)
            print(f"  已缩放 {csv_name}: scale={scale:.4f}")
    
    return output_dir


def run_milp(data_dir: Path, label: str, output_base: Path, extra_args: list) -> Dict[str, Any]:
    """运行 run_full_milp.py 并返回结果摘要。"""
    output_dir = output_base / f"q1_{label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "scripts/run_full_milp.py",
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--mip-gap", "0.01",
    ] + extra_args
    
    print(f"\n{'='*60}")
    print(f"运行场景: {label}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    # 查找最新的 summary.json
    summaries = list(output_dir.glob("full_milp_*/full_milp_summary.json"))
    if summaries:
        latest = max(summaries, key=lambda p: p.stat().st_mtime)
        return json.loads(latest.read_text(encoding="utf-8"))
    return {"error": f"未找到结果文件，返回码={result.returncode}"}


def main():
    parser = argparse.ArgumentParser(description="思考题1：建筑节能改造对比（方案A）")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="runs/thought_questions")
    parser.add_argument("--intensity-before", type=float, default=0.35, help="改造前能耗强度 GJ/m²")
    parser.add_argument("--intensity-after", type=float, default=0.11, help="改造后能耗强度 GJ/m²")
    parser.add_argument("--mip-gap", type=float, default=0.0001)
    parser.add_argument("--time-limit", type=float, default=0)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_base = Path(args.output_dir)
    scale_retrofit = args.intensity_after / args.intensity_before
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / f"q1_retrofit_planA_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    extra_args = ["--mip-gap", str(args.mip_gap)]
    if args.time_limit > 0:
        extra_args += ["--time-limit", str(args.time_limit)]
    if args.threads > 0:
        extra_args += ["--threads", str(args.threads)]

    # 场景1: baseline (原始热负荷)
    print("\n[1/2] 运行 baseline 场景（当前状态，热负荷=100%）...")
    result_baseline = run_milp(data_dir, "baseline", run_dir, extra_args)

    # 场景2: retrofit (缩放热负荷)
    print(f"\n[2/2] 运行 retrofit 场景（改造后，热负荷={scale_retrofit*100:.1f}%）...")
    temp_data_dir = run_dir / "data_retrofit"
    scale_heat_loads(data_dir, temp_data_dir, scale_retrofit)
    result_retrofit = run_milp(temp_data_dir, "retrofit", run_dir, extra_args)

    # 汇总对比
    summary = {
        "intensity_before": args.intensity_before,
        "intensity_after": args.intensity_after,
        "scale_retrofit": scale_retrofit,
        "baseline": {
            "C_total": result_baseline.get("C_total"),
            "C_CAP": result_baseline.get("C_CAP"),
            "C_OP": result_baseline.get("C_OP"),
            "C_Carbon": result_baseline.get("C_Carbon"),
            "Score": result_baseline.get("Score"),
            "plan18": result_baseline.get("plan18"),
        },
        "retrofit": {
            "C_total": result_retrofit.get("C_total"),
            "C_CAP": result_retrofit.get("C_CAP"),
            "C_OP": result_retrofit.get("C_OP"),
            "C_Carbon": result_retrofit.get("C_Carbon"),
            "Score": result_retrofit.get("Score"),
            "plan18": result_retrofit.get("plan18"),
        },
        "timestamp": timestamp,
    }
    
    # 计算差异
    if result_baseline.get("C_total") and result_retrofit.get("C_total"):
        summary["delta"] = {
            "C_total": result_baseline["C_total"] - result_retrofit["C_total"],
            "C_CAP": result_baseline.get("C_CAP", 0) - result_retrofit.get("C_CAP", 0),
            "C_OP": result_baseline.get("C_OP", 0) - result_retrofit.get("C_OP", 0),
        }
    
    (run_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    
    print(f"\n{'='*60}")
    print("对比结果:")
    print(f"  Baseline C_total: {summary['baseline'].get('C_total', 'N/A')}")
    print(f"  Retrofit C_total: {summary['retrofit'].get('C_total', 'N/A')}")
    if "delta" in summary:
        print(f"  差异 (baseline - retrofit): {summary['delta']['C_total']:.2f}")
    print(f"\n结果保存至: {run_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
