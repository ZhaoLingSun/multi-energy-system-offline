#!/usr/bin/env python3
"""
思考题3：分析批量实验结果并生成对比图表

用法:
    python analyze_q3_results.py --results-dir runs/thought_questions/q3_milp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_all_summaries(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all linepack_summary.json files from results directory."""
    summaries = []
    for subdir in results_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith("q3_"):
            summary_file = subdir / "linepack_summary.json"
            if summary_file.exists():
                data = json.loads(summary_file.read_text(encoding="utf-8"))
                data["run_dir"] = str(subdir.name)
                summaries.append(data)
    return summaries


def group_by_amp(summaries: List[Dict]) -> Dict[float, List[Dict]]:
    """Group summaries by daily_price_amp."""
    groups: Dict[float, List[Dict]] = {}
    for s in summaries:
        amp = round(s.get("daily_price_amp", 0.0), 2)
        if amp not in groups:
            groups[amp] = []
        groups[amp].append(s)
    # Sort each group by linepack_cap_mwh
    for amp in groups:
        groups[amp] = sorted(groups[amp], key=lambda x: x.get("linepack_cap_mwh", 0))
    return groups


def compute_savings(summaries: List[Dict]) -> Dict[str, Any]:
    """Compute savings relative to baseline (cap=0)."""
    baseline = None
    for s in summaries:
        if s.get("linepack_cap_mwh", 0) == 0:
            baseline = s
            break
    
    if baseline is None and summaries:
        baseline = min(summaries, key=lambda x: x.get("linepack_cap_mwh", 0))
    
    results = []
    for s in summaries:
        cap_mwh = s.get("linepack_cap_mwh", 0)
        cap_hours = cap_mwh / 50.73 if cap_mwh > 0 else 0  # avg demand ~ 50.73 MWh/h
        obj = s.get("objective", 0)
        c_op_gas = s.get("C_OP_gas", 0)
        
        if baseline:
            base_obj = baseline.get("objective", obj)
            base_gas = baseline.get("C_OP_gas", c_op_gas)
            savings_obj = base_obj - obj
            savings_gas = base_gas - c_op_gas
            savings_ratio = savings_obj / base_obj if base_obj > 0 else 0
        else:
            savings_obj = 0
            savings_gas = 0
            savings_ratio = 0
        
        results.append({
            "cap_hours": cap_hours,
            "cap_mwh": cap_mwh,
            "objective": obj,
            "C_OP_gas": c_op_gas,
            "savings_obj": savings_obj,
            "savings_gas": savings_gas,
            "savings_ratio": savings_ratio,
            "S_lp_mean": s.get("S_lp_mean", 0),
            "S_lp_max": s.get("S_lp_max", 0),
        })
    
    return {
        "baseline_objective": baseline.get("objective", 0) if baseline else 0,
        "baseline_C_OP_gas": baseline.get("C_OP_gas", 0) if baseline else 0,
        "results": results,
    }


def plot_savings_by_amp(grouped: Dict[float, List[Dict]], output_dir: Path):
    """Plot cost savings for different price amplitudes."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {0.0: 'blue', 0.15: 'green', 0.30: 'red'}
    markers = {0.0: 'o', 0.15: 's', 0.30: '^'}
    
    for amp, summaries in sorted(grouped.items()):
        savings_data = compute_savings(summaries)
        results = savings_data["results"]
        
        caps = [r["cap_hours"] for r in results]
        savings_ratio = [r["savings_ratio"] * 100 for r in results]  # percentage
        savings_gas = [r["savings_gas"] / 1e6 for r in results]  # million yuan
        
        label = f"amp={int(amp*100)}%"
        color = colors.get(amp, 'black')
        marker = markers.get(amp, 'o')
        
        axes[0].plot(caps, savings_ratio, f'-{marker}', color=color, label=label, linewidth=2, markersize=8)
        axes[1].plot(caps, savings_gas, f'-{marker}', color=color, label=label, linewidth=2, markersize=8)
    
    axes[0].set_xlabel('Linepack Capacity (hours of avg demand)', fontsize=12)
    axes[0].set_ylabel('Total Cost Savings (%)', fontsize=12)
    axes[0].set_title('Total Cost Reduction vs Linepack Capacity', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-0.5, 25)
    
    axes[1].set_xlabel('Linepack Capacity (hours of avg demand)', fontsize=12)
    axes[1].set_ylabel('Gas Cost Savings (Million Yuan)', fontsize=12)
    axes[1].set_title('Gas Purchase Cost Reduction vs Linepack Capacity', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-0.5, 25)
    
    plt.tight_layout()
    
    output_path = output_dir / "q3_linepack_savings.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()


def plot_linepack_utilization(grouped: Dict[float, List[Dict]], output_dir: Path):
    """Plot linepack SOC statistics."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = {0.0: 'blue', 0.15: 'green', 0.30: 'red'}
    
    for amp, summaries in sorted(grouped.items()):
        results = [s for s in summaries if s.get("linepack_cap_mwh", 0) > 0]
        
        caps = [s.get("linepack_cap_mwh", 0) for s in results]
        utilization = [s.get("S_lp_mean", 0) / s.get("linepack_cap_mwh", 1) * 100 
                       for s in results if s.get("linepack_cap_mwh", 0) > 0]
        
        if caps and utilization:
            label = f"amp={int(amp*100)}%"
            color = colors.get(amp, 'black')
            ax.bar(np.array(range(len(caps))) + amp * 0.25, utilization, 
                   width=0.2, color=color, label=label, alpha=0.7)
    
    ax.set_xlabel('Linepack Capacity Configuration', fontsize=12)
    ax.set_ylabel('Average SOC Utilization (%)', fontsize=12)
    ax.set_title('Linepack Storage Utilization', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = output_dir / "q3_linepack_utilization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()


def generate_summary_table(grouped: Dict[float, List[Dict]], output_dir: Path):
    """Generate markdown summary table."""
    lines = [
        "# 思考题3：管道线包储能实验结果汇总",
        "",
        "## 实验配置",
        "",
        "- 仿真周期: 全年365天 (8760小时)",
        "- MIP Gap容差: 0.01%",
        "- 线包损耗率: 0.05%/h",
        "- 平均气体需求: ~50.73 MWh/h",
        "",
        "## 结果汇总",
        "",
    ]
    
    for amp in sorted(grouped.keys()):
        summaries = grouped[amp]
        savings_data = compute_savings(summaries)
        
        lines.append(f"### 气价波动幅度: {int(amp*100)}%")
        lines.append("")
        lines.append("| 线包容量(h) | 容量(MWh) | 总成本(亿元) | 购气成本(亿元) | 成本节约(%) |")
        lines.append("|------------|----------|-------------|--------------|------------|")
        
        for r in savings_data["results"]:
            cap_h = r["cap_hours"]
            cap_mwh = r["cap_mwh"]
            obj = r["objective"] / 1e8
            gas = r["C_OP_gas"] / 1e8
            ratio = r["savings_ratio"] * 100
            
            lines.append(f"| {cap_h:.0f} | {cap_mwh:.1f} | {obj:.4f} | {gas:.4f} | {ratio:.2f}% |")
        
        lines.append("")
    
    output_path = output_dir / "q3_results_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved summary to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="分析思考题3实验结果")
    parser.add_argument("--results-dir", default="runs/thought_questions/q3_milp",
                        help="结果目录")
    parser.add_argument("--output-dir", default=None, help="输出目录(默认同results-dir)")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    
    print(f"Loading results from: {results_dir}")
    summaries = load_all_summaries(results_dir)
    print(f"Found {len(summaries)} experiment results")
    
    if not summaries:
        print("No results found!")
        return
    
    grouped = group_by_amp(summaries)
    print(f"Grouped by amplitude: {list(grouped.keys())}")
    
    # Generate plots
    plot_savings_by_amp(grouped, output_dir)
    plot_linepack_utilization(grouped, output_dir)
    
    # Generate summary table
    generate_summary_table(grouped, output_dir)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()
