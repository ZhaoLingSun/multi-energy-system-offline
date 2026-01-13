#!/usr/bin/env python3
"""
思考题2扩展：跨季节冷热联储多场景敏感性分析。

探索不同损耗率、效率、价格波动下的经济性边界。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple
import itertools

import numpy as np
from scipy.optimize import linprog

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parents[2]

import sys
sys.path.insert(0, str(_project_root))

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from meos.ga.codec import PLAN18_BOUNDS, PLAN18_BASE_CAPACITY
from meos.export.platform_full_exporter import _parse_meos_inputs
from meos.simulate.annual_summary import calculate_annuity_factor


def _month_index_series(n_hours: int, year: int = 2020) -> np.ndarray:
    start = datetime(year, 1, 1)
    months = np.zeros(n_hours, dtype=int)
    for i in range(n_hours):
        months[i] = (start + timedelta(hours=i)).month
    return months


def _monthly_aggregate(values: np.ndarray, months: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    totals = np.zeros(12, dtype=float)
    counts = np.zeros(12, dtype=int)
    for val, m in zip(values, months):
        idx = int(m) - 1
        totals[idx] += float(val)
        counts[idx] += 1
    return totals, counts


def _load_score_spec(path: Path) -> Dict[str, Any]:
    if not HAS_YAML:
        return {
            "carbon_price": 600.0,
            "gas_to_MWh": 0.01,
            "discount_rate": 0.04,
        }
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    score = data.get("score_spec", {})
    return {
        "carbon_price": score.get("carbon", {}).get("price", 600.0),
        "gas_to_MWh": score.get("units", {}).get("gas_m3_to_MWh", 0.01),
        "discount_rate": score.get("capex", {}).get("discount_rate", 0.04),
    }


def _load_device_catalog(path: Path) -> List[Dict[str, Any]]:
    if not HAS_YAML:
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    devices = data.get("device_catalog", {}).get("devices", [])
    return sorted(devices, key=lambda d: d.get("idx", 0))


def _storage_defaults(score_spec: Dict[str, Any]) -> Tuple[float, float]:
    device_catalog = _load_device_catalog(_project_root / "spec" / "device_catalog.yaml")
    capex_heat = 90000.0
    capex_cool = 90000.0
    for device in device_catalog:
        device_id = str(device.get("device_id", ""))
        unit_cost = float(device.get("unit_cost", 0.0))
        lifespan = int(device.get("lifespan", 20))
        ann = calculate_annuity_factor(float(score_spec["discount_rate"]), lifespan)
        if device_id == "ThermalStorage":
            capex_heat = unit_cost * ann
        elif device_id == "ColdStorage":
            capex_cool = unit_cost * ann
    return capex_heat, capex_cool


def _storage_cap_max() -> float:
    max_heat = PLAN18_BOUNDS[11][1] * PLAN18_BASE_CAPACITY[11]
    max_cool = PLAN18_BOUNDS[12][1] * PLAN18_BASE_CAPACITY[12]
    return max(max_heat, max_cool)

def _build_lp(
    heat_demand: np.ndarray,
    cool_demand: np.ndarray,
    heat_cost: np.ndarray,
    cool_cost: np.ndarray,
    capex_heat: float,
    capex_cool: float,
    cap_max_heat: float,
    cap_max_cool: float,
    eta_ch: float,
    eta_dis: float,
    loss_rate: float,
    monthly_rate: float,
) -> Dict[str, Any]:
    n = 12
    offset = 0
    idx = {}
    for name in ("H_buy", "H_ch", "H_dis", "S_h", "C_buy", "C_ch", "C_dis", "S_c"):
        idx[name] = np.arange(offset, offset + n)
        offset += n
    idx["E_cap_h"] = np.array([offset])
    offset += 1
    idx["E_cap_c"] = np.array([offset])
    offset += 1

    n_vars = offset
    c = np.zeros(n_vars, dtype=float)
    c[idx["H_buy"]] = heat_cost
    c[idx["C_buy"]] = cool_cost
    c[idx["E_cap_h"]] = capex_heat
    c[idx["E_cap_c"]] = capex_cool

    A_eq = []
    b_eq = []

    for m in range(n):
        row = np.zeros(n_vars, dtype=float)
        row[idx["H_buy"][m]] = 1.0
        row[idx["H_dis"][m]] = 1.0
        row[idx["H_ch"][m]] = -1.0
        A_eq.append(row)
        b_eq.append(float(heat_demand[m]))

        row = np.zeros(n_vars, dtype=float)
        row[idx["C_buy"][m]] = 1.0
        row[idx["C_dis"][m]] = 1.0
        row[idx["C_ch"][m]] = -1.0
        A_eq.append(row)
        b_eq.append(float(cool_demand[m]))

    for m in range(n):
        m_next = (m + 1) % n
        row = np.zeros(n_vars, dtype=float)
        row[idx["S_h"][m_next]] = 1.0
        row[idx["S_h"][m]] = -(1.0 - loss_rate)
        row[idx["H_ch"][m]] = -eta_ch
        row[idx["H_dis"][m]] = 1.0 / eta_dis
        A_eq.append(row)
        b_eq.append(0.0)

        row = np.zeros(n_vars, dtype=float)
        row[idx["S_c"][m_next]] = 1.0
        row[idx["S_c"][m]] = -(1.0 - loss_rate)
        row[idx["C_ch"][m]] = -eta_ch
        row[idx["C_dis"][m]] = 1.0 / eta_dis
        A_eq.append(row)
        b_eq.append(0.0)

    A_ub = []
    b_ub = []

    for m in range(n):
        row = np.zeros(n_vars, dtype=float)
        row[idx["S_h"][m]] = 1.0
        row[idx["E_cap_h"][0]] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

        row = np.zeros(n_vars, dtype=float)
        row[idx["S_c"][m]] = 1.0
        row[idx["E_cap_c"][0]] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

        if monthly_rate > 0:
            for var_name in ["H_ch", "H_dis"]:
                row = np.zeros(n_vars, dtype=float)
                row[idx[var_name][m]] = 1.0
                row[idx["E_cap_h"][0]] = -monthly_rate
                A_ub.append(row)
                b_ub.append(0.0)

            for var_name in ["C_ch", "C_dis"]:
                row = np.zeros(n_vars, dtype=float)
                row[idx[var_name][m]] = 1.0
                row[idx["E_cap_c"][0]] = -monthly_rate
                A_ub.append(row)
                b_ub.append(0.0)

    row = np.zeros(n_vars, dtype=float)
    row[idx["E_cap_h"][0]] = 1.0
    A_ub.append(row)
    b_ub.append(cap_max_heat)

    row = np.zeros(n_vars, dtype=float)
    row[idx["E_cap_c"][0]] = 1.0
    A_ub.append(row)
    b_ub.append(cap_max_cool)

    bounds = [(0.0, None)] * n_vars
    bounds[idx["E_cap_h"][0]] = (0.0, cap_max_heat)
    bounds[idx["E_cap_c"][0]] = (0.0, cap_max_cool)

    res = linprog(
        c,
        A_ub=np.asarray(A_ub, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        A_eq=np.asarray(A_eq, dtype=float),
        b_eq=np.asarray(b_eq, dtype=float),
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return {"status": "failed", "message": res.message}

    sol = res.x
    return {
        "objective": float(res.fun),
        "status": "optimal",
        "E_cap_h": float(sol[idx["E_cap_h"][0]]),
        "E_cap_c": float(sol[idx["E_cap_c"][0]]),
        "S_h_max": float(np.max(sol[idx["S_h"]])),
        "S_c_max": float(np.max(sol[idx["S_c"]])),
    }


def run_scenario(
    heat_month: np.ndarray,
    cool_month: np.ndarray,
    elec_month_avg: np.ndarray,
    gas_month_avg: np.ndarray,
    eta_ch: float,
    eta_dis: float,
    loss_rate: float,
    capex_heat: float,
    capex_cool: float,
    summer_elec_mult: float,
    winter_gas_mult: float,
    cop_heat: float = 5.0,
    cop_cool: float = 5.0,
    cap_max: float = 500000.0,
) -> Dict[str, Any]:
    """运行单个场景并返回结果。"""
    elec_adj = elec_month_avg.copy()
    gas_adj = gas_month_avg.copy()
    
    summer_months = [5, 6, 7, 8]  # 5-8月为夏季
    winter_months = [11, 12, 1, 2]  # 11-2月为冬季
    
    for m in summer_months:
        elec_adj[m - 1] *= summer_elec_mult
    for m in winter_months:
        gas_adj[m - 1] *= winter_gas_mult

    gas_heat_cost = gas_adj / 0.95
    elec_heat_cost = elec_adj / cop_heat
    heat_cost = np.minimum(gas_heat_cost, elec_heat_cost)
    cool_cost = elec_adj / cop_cool

    baseline_cost = float(np.sum(heat_month * heat_cost + cool_month * cool_cost))

    result = _build_lp(
        heat_month, cool_month,
        heat_cost, cool_cost,
        capex_heat, capex_cool,
        cap_max, cap_max,
        eta_ch, eta_dis, loss_rate, 1.0,
    )

    if result.get("status") != "optimal":
        return {"status": "failed"}

    optimized_cost = result["objective"]
    savings = baseline_cost - optimized_cost
    capex = capex_heat * result["E_cap_h"] + capex_cool * result["E_cap_c"]
    
    # 计算6个月后的综合效率
    efficiency_6m = eta_ch * eta_dis * ((1 - loss_rate) ** 6)
    
    return {
        "status": "optimal",
        "baseline_cost": baseline_cost,
        "optimized_cost": optimized_cost,
        "savings": savings,
        "savings_ratio": savings / baseline_cost if baseline_cost > 0 else 0,
        "capex": capex,
        "payback_years": capex / savings if savings > 1e3 else float("inf"),
        "E_cap_h_MWh": result["E_cap_h"],
        "E_cap_c_MWh": result["E_cap_c"],
        "efficiency_6m": efficiency_6m,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="思考题2扩展：跨季节储能敏感性分析")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--renewable-dir", default=None)
    parser.add_argument("--output-dir", default="runs/thought_questions")
    args = parser.parse_args()

    project_root = _project_root
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "matlab" / "data" / "raw"
    renewable_dir = Path(args.renewable_dir) if args.renewable_dir else data_dir

    parsed = _parse_meos_inputs(data_dir, renewable_dir)
    score_spec = _load_score_spec(project_root / "matlab" / "configs" / "oj_score.yaml")
    loads_heat = np.asarray(parsed.loads_heat, dtype=float)
    loads_cool = np.asarray(parsed.loads_cool, dtype=float)
    prices_e = np.asarray(parsed.prices_electricity, dtype=float)
    prices_g = np.asarray(parsed.prices_gas, dtype=float)
    carbon_e = np.asarray(parsed.carbon_electricity, dtype=float)
    carbon_g = np.asarray(parsed.carbon_gas, dtype=float)

    aligned_e = prices_e + float(score_spec["carbon_price"]) * carbon_e
    aligned_g = prices_g + float(score_spec["carbon_price"]) * (carbon_g / float(score_spec["gas_to_MWh"]))

    heat_total = np.sum(loads_heat, axis=1)
    cool_total = np.sum(loads_cool, axis=1)
    months = _month_index_series(heat_total.size)

    heat_month, heat_counts = _monthly_aggregate(heat_total, months)
    cool_month, cool_counts = _monthly_aggregate(cool_total, months)
    elec_month_sum, _ = _monthly_aggregate(aligned_e, months)
    gas_month_sum, _ = _monthly_aggregate(aligned_g, months)

    elec_month_avg = elec_month_sum / np.maximum(1, heat_counts)
    gas_month_avg = gas_month_sum / np.maximum(1, heat_counts)

    cap_max_default = _storage_cap_max()

    # ==================== 敏感性分析场景 ====================
    
    # 场景1：损耗率敏感性（CAPEX=0，价格波动适中）
    print("=" * 60)
    print("场景1：损耗率敏感性分析（CAPEX=0，夏电+30%，冬气+30%）")
    print("=" * 60)
    
    loss_rates = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]  # 0.5%~5%/月
    loss_rate_results = []
    
    for lr in loss_rates:
        res = run_scenario(
            heat_month, cool_month, elec_month_avg, gas_month_avg,
            eta_ch=0.95, eta_dis=0.95, loss_rate=lr,
            capex_heat=0, capex_cool=0,
            summer_elec_mult=1.3, winter_gas_mult=1.3,
            cap_max=cap_max_default,
        )
        loss_rate_results.append({
            "loss_rate_monthly": lr,
            "loss_rate_annual": 1 - (1 - lr) ** 12,
            **res,
        })
        if res["status"] == "optimal":
            print(f"  损耗 {lr*100:.1f}%/月（年{(1-(1-lr)**12)*100:.1f}%）: "
                  f"节约 {res['savings']/1e6:.2f} 百万元 ({res['savings_ratio']*100:.2f}%), "
                  f"容量 H={res['E_cap_h_MWh']:.0f} MWh, 6月效率={res['efficiency_6m']*100:.1f}%")

    # 场景2：价格波动敏感性（低损耗，CAPEX=0）
    print("\n" + "=" * 60)
    print("场景2：价格波动敏感性分析（损耗0.5%/月，CAPEX=0）")
    print("=" * 60)
    
    price_mults = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    price_results = []
    
    for pm in price_mults:
        res = run_scenario(
            heat_month, cool_month, elec_month_avg, gas_month_avg,
            eta_ch=0.95, eta_dis=0.95, loss_rate=0.005,
            capex_heat=0, capex_cool=0,
            summer_elec_mult=pm, winter_gas_mult=pm,
            cap_max=cap_max_default,
        )
        price_results.append({
            "price_mult": pm,
            **res,
        })
        if res["status"] == "optimal":
            print(f"  价格波动 ×{pm:.1f}: "
                  f"节约 {res['savings']/1e6:.2f} 百万元 ({res['savings_ratio']*100:.2f}%), "
                  f"容量 H={res['E_cap_h_MWh']:.0f} MWh")

    # 场景3：投资成本敏感性（低损耗，高价格波动）
    print("\n" + "=" * 60)
    print("场景3：投资成本敏感性分析（损耗0.5%/月，价格×1.5）")
    print("=" * 60)
    
    capex_values = [0, 10000, 20000, 50000, 100000, 150000]  # 元/MWh
    capex_results = []
    
    for cx in capex_values:
        res = run_scenario(
            heat_month, cool_month, elec_month_avg, gas_month_avg,
            eta_ch=0.95, eta_dis=0.95, loss_rate=0.005,
            capex_heat=cx, capex_cool=cx,
            summer_elec_mult=1.5, winter_gas_mult=1.5,
            cap_max=cap_max_default,
        )
        capex_results.append({
            "capex_per_MWh": cx,
            **res,
        })
        if res["status"] == "optimal":
            payback_str = f"{res['payback_years']:.1f}年" if res['payback_years'] < 100 else "不可行"
            print(f"  CAPEX {cx/10000:.1f}万元/MWh: "
                  f"节约 {res['savings']/1e6:.2f} 百万元, "
                  f"容量 {res['E_cap_h_MWh']:.0f} MWh, "
                  f"回收期 {payback_str}")

    # 场景4：技术路线对比（不同储能技术参数）
    print("\n" + "=" * 60)
    print("场景4：储能技术路线对比（价格×1.5，CAPEX=5万元/MWh）")
    print("=" * 60)
    
    tech_scenarios = [
        {"name": "地下含水层储热(ATES)", "eta_ch": 0.90, "eta_dis": 0.90, "loss_rate": 0.005},
        {"name": "钻孔储热(BTES)", "eta_ch": 0.85, "eta_dis": 0.85, "loss_rate": 0.008},
        {"name": "水箱储热(保温)", "eta_ch": 0.95, "eta_dis": 0.95, "loss_rate": 0.02},
        {"name": "相变材料储热(PCM)", "eta_ch": 0.92, "eta_dis": 0.92, "loss_rate": 0.01},
        {"name": "理想储热(理论极限)", "eta_ch": 1.0, "eta_dis": 1.0, "loss_rate": 0.0},
    ]
    tech_results = []
    
    for tech in tech_scenarios:
        res = run_scenario(
            heat_month, cool_month, elec_month_avg, gas_month_avg,
            eta_ch=tech["eta_ch"], eta_dis=tech["eta_dis"], loss_rate=tech["loss_rate"],
            capex_heat=50000, capex_cool=50000,
            summer_elec_mult=1.5, winter_gas_mult=1.5,
            cap_max=cap_max_default,
        )
        tech_results.append({
            "technology": tech["name"],
            "eta_ch": tech["eta_ch"],
            "eta_dis": tech["eta_dis"],
            "loss_rate": tech["loss_rate"],
            **res,
        })
        if res["status"] == "optimal":
            payback_str = f"{res['payback_years']:.1f}年" if res['payback_years'] < 100 else "不可行"
            print(f"  {tech['name']}: "
                  f"6月效率={res['efficiency_6m']*100:.1f}%, "
                  f"节约 {res['savings']/1e6:.2f} 百万元, "
                  f"回收期 {payback_str}")

    # 场景5：临界可行条件分析
    print("\n" + "=" * 60)
    print("场景5：经济可行临界条件（回收期<15年）")
    print("=" * 60)
    
    # 扫描找到使回收期=15年的边界条件
    critical_results = []
    for pm in [1.3, 1.5, 2.0]:
        for lr in [0.005, 0.01, 0.02]:
            for cx in [10000, 30000, 50000]:
                res = run_scenario(
                    heat_month, cool_month, elec_month_avg, gas_month_avg,
                    eta_ch=0.90, eta_dis=0.90, loss_rate=lr,
                    capex_heat=cx, capex_cool=cx,
                    summer_elec_mult=pm, winter_gas_mult=pm,
                    cap_max=cap_max_default,
                )
                if res["status"] == "optimal" and res["payback_years"] < 15:
                    critical_results.append({
                        "price_mult": pm,
                        "loss_rate": lr,
                        "capex": cx,
                        **res,
                    })
    
    if critical_results:
        print(f"  找到 {len(critical_results)} 个可行组合（回收期<15年）：")
        for cr in sorted(critical_results, key=lambda x: x["payback_years"])[:5]:
            print(f"    价格×{cr['price_mult']:.1f}, 损耗{cr['loss_rate']*100:.1f}%/月, "
                  f"CAPEX={cr['capex']/10000:.0f}万元/MWh → 回收期 {cr['payback_years']:.1f}年")
    else:
        print("  未找到可行组合")

    # 保存结果
    summary = {
        "heat_month_MWh": heat_month.tolist(),
        "cool_month_MWh": cool_month.tolist(),
        "elec_month_avg": elec_month_avg.tolist(),
        "gas_month_avg": gas_month_avg.tolist(),
        "cap_max_default_MWh": cap_max_default,
        "analysis": {
            "loss_rate_sensitivity": loss_rate_results,
            "price_sensitivity": price_results,
            "capex_sensitivity": capex_results,
            "technology_comparison": tech_results,
            "critical_conditions": critical_results,
        },
        "objective_mode": "aligned",
        "timestamp": datetime.now().isoformat(),
    }

    run_dir = Path(args.output_dir) / f"q2_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果保存至: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
