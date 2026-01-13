#!/usr/bin/env python3
"""
思考题2：跨季节冷热联储的建模与经济性评估（基于主优化基线）。

基线=主优化问题的最优解（baseline summary），本脚本不复算基线。
做法：在月尺度上引入全区共通的跨季节冷热联储（无投资成本），
保持主问题一致的目标口径（aligned 价格），并对不同容量进行敏感性分析。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parents[2]

import sys
sys.path.insert(0, str(_project_root))

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("gurobipy 未安装，思考题2需要使用 Gurobi 求解器") from exc

from meos.export.platform_full_exporter import _parse_meos_inputs
from meos.simulate.annual_summary import calculate_score


def _latest_full_milp_summary(root: Path) -> Path | None:
    summaries = sorted(root.glob("full_milp_*/full_milp_summary.json"), reverse=True)
    return summaries[0] if summaries else None


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


def _parse_float_list(text: str) -> List[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    values = []
    for p in parts:
        try:
            values.append(float(p))
        except ValueError:
            continue
    return values


def _load_score_spec(path: Path) -> Dict[str, Any]:
    if not HAS_YAML:
        return {
            "carbon_price": 600.0,
            "gas_to_MWh": 0.01,
        }
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    score = data.get("score_spec", {})
    return {
        "carbon_price": score.get("carbon", {}).get("price", 600.0),
        "gas_to_MWh": score.get("units", {}).get("gas_m3_to_MWh", 0.01),
    }


def _load_baseline(path: Path) -> Dict[str, Any]:
    baseline = json.loads(path.read_text(encoding="utf-8"))
    carbon = baseline.get("carbon", {})
    return {
        "path": str(path),
        "C_CAP": baseline.get("C_CAP"),
        "C_OP": baseline.get("C_OP"),
        "C_Carbon": baseline.get("C_Carbon"),
        "C_total": baseline.get("C_total"),
        "Score": baseline.get("Score"),
        "emission_total": carbon.get("emission_total"),
        "emission_excess": carbon.get("emission_excess"),
        "carbon_threshold": baseline.get("carbon_threshold", carbon.get("emission_threshold")),
        "carbon_price": baseline.get("carbon_price", carbon.get("carbon_price")),
    }


def _solve_monthly_storage(
    heat_demand: np.ndarray,
    cool_demand: np.ndarray,
    heat_cost: np.ndarray,
    cool_cost: np.ndarray,
    cap_heat: float,
    cap_cool: float,
    eta_ch: float,
    eta_dis: float,
    loss_rate: float,
    monthly_rate: float,
    threads: int,
    verbose: bool,
) -> Dict[str, Any]:
    n = 12
    model = gp.Model("seasonal_storage")
    model.Params.OutputFlag = 1 if verbose else 0
    if threads > 0:
        model.Params.Threads = threads

    H_buy = model.addMVar(n, lb=0.0, name="H_buy")
    H_ch = model.addMVar(n, lb=0.0, name="H_ch")
    H_dis = model.addMVar(n, lb=0.0, name="H_dis")
    S_h = model.addMVar(n, lb=0.0, ub=cap_heat, name="S_h")

    C_buy = model.addMVar(n, lb=0.0, name="C_buy")
    C_ch = model.addMVar(n, lb=0.0, name="C_ch")
    C_dis = model.addMVar(n, lb=0.0, name="C_dis")
    S_c = model.addMVar(n, lb=0.0, ub=cap_cool, name="S_c")

    model.addConstr(H_buy + H_dis - H_ch == heat_demand, name="heat_balance")
    model.addConstr(C_buy + C_dis - C_ch == cool_demand, name="cool_balance")

    for m in range(n):
        m_next = (m + 1) % n
        model.addConstr(
            S_h[m_next] == (1.0 - loss_rate) * S_h[m] + eta_ch * H_ch[m] - H_dis[m] / eta_dis,
            name=f"soc_heat_{m}",
        )
        model.addConstr(
            S_c[m_next] == (1.0 - loss_rate) * S_c[m] + eta_ch * C_ch[m] - C_dis[m] / eta_dis,
            name=f"soc_cool_{m}",
        )

    if monthly_rate > 0:
        model.addConstr(H_ch <= cap_heat * monthly_rate, name="heat_ch_rate")
        model.addConstr(H_dis <= cap_heat * monthly_rate, name="heat_dis_rate")
        model.addConstr(C_ch <= cap_cool * monthly_rate, name="cool_ch_rate")
        model.addConstr(C_dis <= cap_cool * monthly_rate, name="cool_dis_rate")

    objective = gp.quicksum(heat_cost[m] * H_buy[m] + cool_cost[m] * C_buy[m] for m in range(n))
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi solve failed: status={model.Status}")

    return {
        "objective": float(model.ObjVal),
        "H_buy": H_buy.X.tolist(),
        "H_ch": H_ch.X.tolist(),
        "H_dis": H_dis.X.tolist(),
        "S_h": S_h.X.tolist(),
        "C_buy": C_buy.X.tolist(),
        "C_ch": C_ch.X.tolist(),
        "C_dis": C_dis.X.tolist(),
        "S_c": S_c.X.tolist(),
    }


def _heat_supply_choice(
    heat_supply: str,
    gas_cost: np.ndarray,
    elec_cost: np.ndarray,
) -> List[str]:
    if heat_supply == "gas":
        return ["gas"] * len(gas_cost)
    if heat_supply == "electric":
        return ["electric"] * len(gas_cost)
    return ["gas" if gas_cost[i] <= elec_cost[i] else "electric" for i in range(len(gas_cost))]


def _calc_energy_cost_and_emission(
    solution: Dict[str, Any],
    heat_choice: List[str],
    prices_e: np.ndarray,
    prices_g: np.ndarray,
    emission_e: np.ndarray,
    emission_g: np.ndarray,
    gas_eff: float,
    cop_heat: float,
    cop_cool: float,
) -> Tuple[float, float]:
    H_buy = np.asarray(solution["H_buy"], dtype=float)
    C_buy = np.asarray(solution["C_buy"], dtype=float)

    econ_cost = 0.0
    emission = 0.0
    for i in range(12):
        if heat_choice[i] == "gas":
            gas_input = H_buy[i] / gas_eff
            econ_cost += prices_g[i] * gas_input
            emission += emission_g[i] * gas_input
        else:
            elec_input = H_buy[i] / cop_heat
            econ_cost += prices_e[i] * elec_input
            emission += emission_e[i] * elec_input

        cool_input = C_buy[i] / cop_cool
        econ_cost += prices_e[i] * cool_input
        emission += emission_e[i] * cool_input

    return float(econ_cost), float(emission)


def _calc_aligned_cost(
    solution: Dict[str, Any],
    heat_choice: List[str],
    aligned_e: np.ndarray,
    aligned_g: np.ndarray,
    gas_eff: float,
    cop_heat: float,
    cop_cool: float,
) -> float:
    H_buy = np.asarray(solution["H_buy"], dtype=float)
    C_buy = np.asarray(solution["C_buy"], dtype=float)

    cost = 0.0
    for i in range(12):
        if heat_choice[i] == "gas":
            gas_input = H_buy[i] / gas_eff
            cost += aligned_g[i] * gas_input
        else:
            elec_input = H_buy[i] / cop_heat
            cost += aligned_e[i] * elec_input

        cool_input = C_buy[i] / cop_cool
        cost += aligned_e[i] * cool_input

    return float(cost)


def main() -> None:
    parser = argparse.ArgumentParser(description="思考题2：跨季节冷热联储评估（基于主优化基线）")
    parser.add_argument("--baseline-summary", default=None, help="主优化 summary.json（baseline）")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--renewable-dir", default=None)
    parser.add_argument("--heat-supply", default="min", choices=("gas", "electric", "min"))
    parser.add_argument("--gas-eff", type=float, default=0.95)
    parser.add_argument("--cop-heat", type=float, default=5.0)
    parser.add_argument("--cop-cool", type=float, default=5.0)
    parser.add_argument("--eta-ch", type=float, default=0.95)
    parser.add_argument("--eta-dis", type=float, default=0.95)
    parser.add_argument("--loss-rate", type=float, default=0.005, help="monthly loss rate")
    parser.add_argument("--monthly-rate", type=float, default=1.0, help="max charge/discharge ratio per month")
    parser.add_argument("--cap-list", default="0,20000,50000,100000", help="heat/cool capacity list (MWh)")
    parser.add_argument("--cap-heat-list", default=None, help="heat capacity list (MWh)")
    parser.add_argument("--cap-cool-list", default=None, help="cool capacity list (MWh)")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output-dir", default="runs/thought_questions")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    baseline_path = Path(args.baseline_summary) if args.baseline_summary else _latest_full_milp_summary(_project_root / "runs" / "full_milp")
    if not baseline_path or not baseline_path.exists():
        raise FileNotFoundError("未找到 baseline summary，请用 --baseline-summary 指定")

    baseline = _load_baseline(baseline_path)
    score_spec = _load_score_spec(_project_root / "matlab" / "configs" / "oj_score.yaml")

    data_dir = Path(args.data_dir)
    renewable_dir = Path(args.renewable_dir) if args.renewable_dir else data_dir
    parsed = _parse_meos_inputs(data_dir, renewable_dir)

    loads_heat = np.asarray(parsed.loads_heat, dtype=float)
    loads_cool = np.asarray(parsed.loads_cool, dtype=float)
    prices_e = np.asarray(parsed.prices_electricity, dtype=float)
    prices_g = np.asarray(parsed.prices_gas, dtype=float)
    carbon_e = np.asarray(parsed.carbon_electricity, dtype=float)
    carbon_g = np.asarray(parsed.carbon_gas, dtype=float)

    carbon_price = float(baseline["carbon_price"] or score_spec["carbon_price"])
    gas_to_MWh = float(score_spec["gas_to_MWh"])

    aligned_e = prices_e + carbon_price * carbon_e
    aligned_g = prices_g + carbon_price * (carbon_g / gas_to_MWh)

    months = _month_index_series(prices_e.size)
    heat_total = np.sum(loads_heat, axis=1)
    cool_total = np.sum(loads_cool, axis=1)

    heat_month, heat_counts = _monthly_aggregate(heat_total, months)
    cool_month, _ = _monthly_aggregate(cool_total, months)

    price_e_month_sum, _ = _monthly_aggregate(prices_e, months)
    price_g_month_sum, _ = _monthly_aggregate(prices_g, months)
    aligned_e_month_sum, _ = _monthly_aggregate(aligned_e, months)
    aligned_g_month_sum, _ = _monthly_aggregate(aligned_g, months)
    emission_e_month_sum, _ = _monthly_aggregate(carbon_e, months)
    emission_g_month_sum, _ = _monthly_aggregate(carbon_g / gas_to_MWh, months)

    counts = np.maximum(1, heat_counts)
    price_e_month = price_e_month_sum / counts
    price_g_month = price_g_month_sum / counts
    aligned_e_month = aligned_e_month_sum / counts
    aligned_g_month = aligned_g_month_sum / counts
    emission_e_month = emission_e_month_sum / counts
    emission_g_month = emission_g_month_sum / counts

    gas_heat_cost_aligned = aligned_g_month / args.gas_eff
    elec_heat_cost_aligned = aligned_e_month / max(1e-6, args.cop_heat)

    heat_choice = _heat_supply_choice(args.heat_supply, gas_heat_cost_aligned, elec_heat_cost_aligned)

    if args.heat_supply == "gas":
        heat_cost_aligned = gas_heat_cost_aligned
    elif args.heat_supply == "electric":
        heat_cost_aligned = elec_heat_cost_aligned
    else:
        heat_cost_aligned = np.minimum(gas_heat_cost_aligned, elec_heat_cost_aligned)

    cool_cost_aligned = aligned_e_month / max(1e-6, args.cop_cool)

    if args.cap_heat_list:
        caps_heat = _parse_float_list(args.cap_heat_list)
    else:
        caps_heat = _parse_float_list(args.cap_list)
    if args.cap_cool_list:
        caps_cool = _parse_float_list(args.cap_cool_list)
    else:
        caps_cool = _parse_float_list(args.cap_list)

    if len(caps_heat) != len(caps_cool):
        raise ValueError("cap-heat-list 与 cap-cool-list 长度不一致")

    cap_pairs = list(zip(caps_heat, caps_cool))
    if not cap_pairs:
        raise ValueError("未提供可用的容量列表")

    baseline_solution = _solve_monthly_storage(
        heat_month,
        cool_month,
        heat_cost_aligned,
        cool_cost_aligned,
        0.0,
        0.0,
        args.eta_ch,
        args.eta_dis,
        args.loss_rate,
        args.monthly_rate,
        args.threads,
        args.verbose,
    )

    baseline_aligned_cost = _calc_aligned_cost(
        baseline_solution,
        heat_choice,
        aligned_e_month,
        aligned_g_month,
        args.gas_eff,
        args.cop_heat,
        args.cop_cool,
    )

    baseline_econ_cost, baseline_emission = _calc_energy_cost_and_emission(
        baseline_solution,
        heat_choice,
        price_e_month,
        price_g_month,
        emission_e_month,
        emission_g_month,
        args.gas_eff,
        args.cop_heat,
        args.cop_cool,
    )

    results = []
    for cap_heat, cap_cool in cap_pairs:
        solution = _solve_monthly_storage(
            heat_month,
            cool_month,
            heat_cost_aligned,
            cool_cost_aligned,
            cap_heat,
            cap_cool,
            args.eta_ch,
            args.eta_dis,
            args.loss_rate,
            args.monthly_rate,
            args.threads,
            args.verbose,
        )

        aligned_cost = _calc_aligned_cost(
            solution,
            heat_choice,
            aligned_e_month,
            aligned_g_month,
            args.gas_eff,
            args.cop_heat,
            args.cop_cool,
        )

        econ_cost, emission = _calc_energy_cost_and_emission(
            solution,
            heat_choice,
            price_e_month,
            price_g_month,
            emission_e_month,
            emission_g_month,
            args.gas_eff,
            args.cop_heat,
            args.cop_cool,
        )

        delta_econ = econ_cost - baseline_econ_cost
        delta_emission = emission - baseline_emission

        new_C_OP = (baseline["C_OP"] or 0.0) + delta_econ
        new_emission_total = (baseline["emission_total"] or 0.0) + delta_emission
        carbon_threshold = baseline["carbon_threshold"] or 100000.0
        C_Carbon = carbon_price * max(0.0, new_emission_total - carbon_threshold)
        new_C_total = (baseline["C_CAP"] or 0.0) + new_C_OP + C_Carbon
        new_score = calculate_score(new_C_total)

        results.append(
            {
                "cap_heat": cap_heat,
                "cap_cool": cap_cool,
                "aligned_cost": aligned_cost,
                "economic_cost": econ_cost,
                "emission_total": emission,
                "delta_vs_no_storage": {
                    "aligned_cost": aligned_cost - baseline_aligned_cost,
                    "economic_cost": delta_econ,
                    "emission_total": delta_emission,
                },
                "C_OP": new_C_OP,
                "C_Carbon": C_Carbon,
                "C_total": new_C_total,
                "Score": new_score,
                "solution": solution,
            }
        )

    summary = {
        "baseline": baseline,
        "baseline_no_storage": {
            "aligned_cost": baseline_aligned_cost,
            "economic_cost": baseline_econ_cost,
            "emission_total": baseline_emission,
            "solution": baseline_solution,
        },
        "params": {
            "heat_supply": args.heat_supply,
            "gas_eff": args.gas_eff,
            "cop_heat": args.cop_heat,
            "cop_cool": args.cop_cool,
            "eta_ch": args.eta_ch,
            "eta_dis": args.eta_dis,
            "loss_rate": args.loss_rate,
            "monthly_rate": args.monthly_rate,
            "cap_pairs": cap_pairs,
            "objective_mode": "aligned",
        },
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    run_dir = Path(args.output_dir) / f"q2_seasonal_storage_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Q2 summary saved to: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
