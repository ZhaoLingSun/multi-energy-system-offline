#!/usr/bin/env python3
"""
分析思考题4：传输容量敏感性分析结果
生成对比表格和可视化图表
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_all_results(base_dir: Path) -> List[Dict[str, Any]]:
    """Load all q4 results from multiple runs."""
    all_results = []
    for run_dir in base_dir.glob("q4_milp_*"):
        summary_file = run_dir / "summary.json"
        if summary_file.exists():
            data = json.loads(summary_file.read_text(encoding="utf-8"))
            for r in data.get("results", []):
                if "error" not in r:
                    all_results.append(r)
    return all_results


def analyze_heat_capacity_sensitivity(results: List[Dict[str, Any]], output_dir: Path):
    """Analyze heat capacity sensitivity."""
    # Filter results with line_cap = 50 for heat analysis
    heat_results = [r for r in results if abs(r["line_capacity"] - 50.0) < 1]
    
    if not heat_results:
        print("No results for heat capacity analysis")
        return
    
    # Sort by heat capacity
    heat_results.sort(key=lambda x: x["heat_capacity"])
    
    print("\n=== 热网容量敏感性分析 (电力容量固定50MW) ===")
    print(f"{'热网容量(MW)':<12} {'总成本(亿元)':<14} {'失负荷(MWh)':<14} {'运行成本(亿元)':<14} {'碳成本(亿元)':<14}")
    print("-" * 70)
    
    for r in heat_results:
        heat_cap = r["heat_capacity"]
        objective = r.get("objective", 0) / 1e8
        shed = r.get("shed_total", 0)
        op_cost = r.get("C_OP", 0) / 1e8
        carbon = r.get("C_Carbon", 0) / 1e8
        print(f"{heat_cap:<12.0f} {objective:<14.2f} {shed:<14.1f} {op_cost:<14.2f} {carbon:<14.4f}")
    
    # Plot if matplotlib available
    if HAS_MPL and heat_results:
        heat_caps = [r["heat_capacity"] for r in heat_results]
        objectives = [r.get("objective", 0) / 1e8 for r in heat_results]
        sheds = [r.get("shed_total", 0) for r in heat_results]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Cost vs heat capacity
        ax1 = axes[0]
        ax1.plot(heat_caps, objectives, 'o-', linewidth=2, markersize=8, color='#1f77b4')
        ax1.set_xlabel('Heat Network Capacity (MW)', fontsize=12)
        ax1.set_ylabel('Total Annual Cost (100M RMB)', fontsize=12)
        ax1.set_title('Cost vs Heat Capacity (Line Capacity = 50 MW)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, max(heat_caps) * 1.1])
        
        # Shed vs heat capacity
        ax2 = axes[1]
        ax2.plot(heat_caps, sheds, 's-', linewidth=2, markersize=8, color='#d62728')
        ax2.set_xlabel('Heat Network Capacity (MW)', fontsize=12)
        ax2.set_ylabel('Load Shedding (MWh/year)', fontsize=12)
        ax2.set_title('Load Shedding vs Heat Capacity', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, max(heat_caps) * 1.1])
        
        plt.tight_layout()
        fig_path = output_dir / "q4_heat_capacity_sensitivity.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存: {fig_path}")
        plt.close()


def analyze_line_capacity_sensitivity(results: List[Dict[str, Any]], output_dir: Path):
    """Analyze line capacity sensitivity."""
    # Get unique heat caps
    heat_caps = sorted(set(r["heat_capacity"] for r in results))
    
    print("\n=== 电力线路容量敏感性分析 ===")
    
    for heat_cap in heat_caps[:2]:  # Show first two heat caps
        line_results = [r for r in results if abs(r["heat_capacity"] - heat_cap) < 1]
        if len(line_results) < 2:
            continue
        line_results.sort(key=lambda x: x["line_capacity"])
        
        print(f"\n热网容量 = {heat_cap} MW:")
        print(f"{'电力容量(MW)':<12} {'总成本(亿元)':<14} {'失负荷(MWh)':<14}")
        print("-" * 40)
        
        for r in line_results:
            line_cap = r["line_capacity"]
            objective = r.get("objective", 0) / 1e8
            shed = r.get("shed_total", 0)
            print(f"{line_cap:<12.0f} {objective:<14.2f} {shed:<14.1f}")


def find_optimal_capacity(results: List[Dict[str, Any]], base_line: float, base_heat: float,
                          capex_line: float, capex_heat: float):
    """Find optimal expansion strategy."""
    print("\n=== 扩容经济性分析 ===")
    print(f"基准容量: 电力={base_line} MW, 热网={base_heat} MW")
    print(f"扩容成本: 电力={capex_line/10000:.1f}万元/MW, 热网={capex_heat/10000:.1f}万元/MW")
    
    # Find baseline cost (or minimum cost if baseline not available)
    baseline = None
    for r in results:
        if abs(r["line_capacity"] - base_line) < 1 and abs(r["heat_capacity"] - base_heat) < 1:
            baseline = r
            break
    
    if baseline is None:
        # Use minimum cost result as reference
        baseline = min(results, key=lambda x: x.get("objective", float('inf')))
        print(f"未找到基准场景，使用最低成本场景作为参考")
    
    base_cost = baseline.get("objective", 0)
    print(f"基准总成本: {base_cost/1e8:.2f} 亿元")
    
    print(f"\n{'电力容量':<10} {'热网容量':<10} {'运行成本节约(亿)':<16} {'扩容投资(亿)':<14} {'净收益(亿)':<12}")
    print("-" * 65)
    
    best_net_benefit = -float('inf')
    best_config = None
    
    for r in results:
        line_cap = r["line_capacity"]
        heat_cap = r["heat_capacity"]
        total_cost = r.get("objective", float('inf'))
        
        line_expansion = max(0, line_cap - base_line)
        heat_expansion = max(0, heat_cap - base_heat)
        expansion_cost = line_expansion * capex_line + heat_expansion * capex_heat
        
        savings = base_cost - total_cost
        net_benefit = savings - expansion_cost
        
        print(f"{line_cap:<10.0f} {heat_cap:<10.0f} {savings/1e8:<16.2f} {expansion_cost/1e8:<14.4f} {net_benefit/1e8:<12.2f}")
        
        if net_benefit > best_net_benefit:
            best_net_benefit = net_benefit
            best_config = (line_cap, heat_cap, net_benefit, expansion_cost)
    
    if best_config:
        print(f"\n最优扩容方案: 电力={best_config[0]:.0f}MW, 热网={best_config[1]:.0f}MW")
        print(f"净收益: {best_config[2]/1e8:.2f} 亿元, 扩容投资: {best_config[3]/1e8:.4f} 亿元")


def generate_comprehensive_plot(results: List[Dict[str, Any]], output_dir: Path):
    """Generate comprehensive analysis plot."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plots")
        return
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Heat capacity sensitivity (top left)
    heat_results = sorted([r for r in results if abs(r["line_capacity"] - 50.0) < 1],
                          key=lambda x: x["heat_capacity"])
    if heat_results:
        heat_caps = [r["heat_capacity"] for r in heat_results]
        costs = [r.get("objective", 0) / 1e8 for r in heat_results]
        axes[0, 0].plot(heat_caps, costs, 'o-', linewidth=2.5, markersize=10, color='#2ca02c')
        axes[0, 0].set_xlabel('Heat Network Capacity (MW)', fontsize=11)
        axes[0, 0].set_ylabel('Total Cost (100M RMB)', fontsize=11)
        axes[0, 0].set_title('(a) Total Cost vs Heat Capacity', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=6.88, color='r', linestyle='--', alpha=0.7, label='Baseline (no constraint)')
        axes[0, 0].legend()
    
    # 2. Load shedding vs heat capacity (top right)
    if heat_results:
        sheds = [r.get("shed_total", 0) / 1000 for r in heat_results]  # Convert to GWh
        axes[0, 1].plot(heat_caps, sheds, 's-', linewidth=2.5, markersize=10, color='#d62728')
        axes[0, 1].set_xlabel('Heat Network Capacity (MW)', fontsize=11)
        axes[0, 1].set_ylabel('Load Shedding (GWh/year)', fontsize=11)
        axes[0, 1].set_title('(b) Load Shedding vs Heat Capacity', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(bottom=0)
    
    # 3. Marginal cost analysis (bottom left)
    if len(heat_results) > 1:
        marginal_costs = []
        cap_midpoints = []
        for i in range(1, len(heat_results)):
            delta_cap = heat_results[i]["heat_capacity"] - heat_results[i-1]["heat_capacity"]
            delta_cost = (heat_results[i-1].get("objective", 0) - heat_results[i].get("objective", 0))
            if delta_cap > 0:
                marginal = delta_cost / delta_cap / 1e4  # 万元/MW
                marginal_costs.append(marginal)
                cap_midpoints.append((heat_results[i]["heat_capacity"] + heat_results[i-1]["heat_capacity"]) / 2)
        
        if marginal_costs:
            axes[1, 0].bar(cap_midpoints, marginal_costs, width=5, color='#9467bd', alpha=0.7)
            axes[1, 0].axhline(y=3.0, color='r', linestyle='--', alpha=0.7, label='Heat expansion cost (3万/MW)')
            axes[1, 0].set_xlabel('Heat Capacity (MW)', fontsize=11)
            axes[1, 0].set_ylabel('Marginal Cost Saving (10k RMB/MW)', fontsize=11)
            axes[1, 0].set_title('(c) Marginal Cost Saving vs Heat Capacity', fontsize=12)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
    
    # 4. Heat transfer utilization (bottom right)
    if heat_results:
        h_max = [r.get("H_transfer_max", 0) for r in heat_results]
        h_mean = [r.get("H_transfer_mean", 0) for r in heat_results]
        
        x = np.arange(len(heat_caps))
        width = 0.35
        
        bars1 = axes[1, 1].bar(x - width/2, h_max, width, label='Max Transfer', color='#1f77b4')
        bars2 = axes[1, 1].bar(x + width/2, h_mean, width, label='Mean Transfer', color='#ff7f0e')
        axes[1, 1].plot(x, heat_caps, 'k--', linewidth=2, label='Capacity Limit')
        
        axes[1, 1].set_xlabel('Heat Capacity Setting (MW)', fontsize=11)
        axes[1, 1].set_ylabel('Heat Transfer Power (MW)', fontsize=11)
        axes[1, 1].set_title('(d) Heat Transfer Utilization', fontsize=12)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f'{c:.0f}' for c in heat_caps], rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = output_dir / "q4_comprehensive_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n综合分析图已保存: {fig_path}")
    plt.close()


def main():
    base_dir = Path("runs/thought_questions")
    output_dir = Path("docs/reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("加载实验结果...")
    results = load_all_results(base_dir)
    print(f"共加载 {len(results)} 个有效结果")
    
    if not results:
        print("未找到有效结果")
        return
    
    # Remove duplicates by (line_cap, heat_cap)
    seen = set()
    unique_results = []
    for r in results:
        key = (r["line_capacity"], r["heat_capacity"])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    print(f"去重后 {len(unique_results)} 个唯一配置")
    
    analyze_heat_capacity_sensitivity(unique_results, output_dir)
    analyze_line_capacity_sensitivity(unique_results, output_dir)
    find_optimal_capacity(unique_results, base_line=100.0, base_heat=50.0,
                          capex_line=50000.0, capex_heat=30000.0)
    generate_comprehensive_plot(unique_results, output_dir)
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()
