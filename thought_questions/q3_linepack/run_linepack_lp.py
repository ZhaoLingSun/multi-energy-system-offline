#!/usr/bin/env python3
"""
思考题3：热/气管道等效储能（线包）建模与运行影响评估。

以 OJ 输出的气源消耗序列作为需求，构建线包储能 LP，
比较不同线包容量下的运行成本。
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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

from meos.export.platform_full_exporter import _parse_meos_inputs


def _load_oj_gas(oj_csv: Path) -> np.ndarray:
    with oj_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        gas = []
        for row in reader:
            gas.append(float(row.get("ans_gas", 0.0)))
    return np.asarray(gas, dtype=float)


def _parse_cap_hours(text: str) -> List[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [float(p) for p in parts]


def _apply_price_modulation(
    price: np.ndarray,
    daily_amp: float,
    seasonal_amp: float,
) -> np.ndarray:
    if daily_amp == 0.0 and seasonal_amp == 0.0:
        return price
    n = price.size
    t = np.arange(n, dtype=float)
    mod = np.ones(n, dtype=float)
    if daily_amp != 0.0:
        mod += daily_amp * np.sin(2 * np.pi * t / 24.0)
    if seasonal_amp != 0.0:
        mod += seasonal_amp * np.sin(2 * np.pi * t / 8760.0)
    mod = np.maximum(mod, 0.0)
    return price * mod


def _load_score_spec(path: Path) -> Dict[str, float]:
    if not HAS_YAML:
        return {"carbon_price": 600.0, "gas_to_MWh": 0.01}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    score = data.get("score_spec", {})
    return {
        "carbon_price": float(score.get("carbon", {}).get("price", 600.0)),
        "gas_to_MWh": float(score.get("units", {}).get("gas_m3_to_MWh", 0.01)),
    }


def _solve_linepack(
    demand: np.ndarray,
    price: np.ndarray,
    cap_mwh: float,
    loss_rate: float,
) -> Tuple[float, np.ndarray]:
    n = demand.size
    n_vars = 2 * n  # [p_t, s_t]
    idx_p = np.arange(0, n)
    idx_s = np.arange(n, 2 * n)

    c = np.zeros(n_vars, dtype=float)
    c[idx_p] = price

    A_eq = []
    b_eq = []
    for t in range(n):
        row = np.zeros(n_vars, dtype=float)
        t_next = (t + 1) % n
        row[idx_s[t_next]] = 1.0
        row[idx_s[t]] = -(1.0 - loss_rate)
        row[idx_p[t]] = -1.0
        A_eq.append(row)
        b_eq.append(-float(demand[t]))

    A_ub = []
    b_ub = []
    for t in range(n):
        row = np.zeros(n_vars, dtype=float)
        row[idx_s[t]] = 1.0
        A_ub.append(row)
        b_ub.append(cap_mwh)

    bounds = [(0.0, None)] * n_vars

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
        raise RuntimeError(f"LP solve failed: {res.message}")
    return float(res.fun), res.x


def main() -> None:
    parser = argparse.ArgumentParser(description="思考题3：管道线包等效储能评估")
    parser.add_argument("--oj-csv", default=None, help="OJ 输出 csv（含 ans_gas）")
    parser.add_argument("--data-dir", default=None, help="input data dir")
    parser.add_argument("--renewable-dir", default=None, help="renewable data dir")
    parser.add_argument("--loss-rate", type=float, default=0.0005, help="hourly loss rate")
    parser.add_argument("--daily-price-amp", type=float, default=0.0, help="daily price modulation amplitude")
    parser.add_argument("--seasonal-price-amp", type=float, default=0.0, help="seasonal price modulation amplitude")
    parser.add_argument("--cap-hours", default="0,6,12,24", help="线包容量等于平均需求的小时数")
    parser.add_argument("--output-dir", default="runs/thought_questions", help="output base dir")
    args = parser.parse_args()

    project_root = _project_root
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "matlab" / "data" / "raw"
    renewable_dir = Path(args.renewable_dir) if args.renewable_dir else data_dir
    score_spec = _load_score_spec(project_root / "matlab" / "configs" / "oj_score.yaml")

    if args.oj_csv:
        oj_csv = Path(args.oj_csv)
    else:
        # 自动查找最新的 full_milp 结果
        full_milp_dir = project_root / "runs" / "full_milp"
        run_dirs = sorted([d for d in full_milp_dir.iterdir() if d.is_dir() and d.name.startswith("full_milp_")], reverse=True)
        oj_csv = None
        for run_dir in run_dirs:
            for sub in run_dir.iterdir():
                if sub.is_dir():
                    for f in sub.iterdir():
                        if f.name.endswith("_oj.csv"):
                            oj_csv = f
                            break
                    if oj_csv:
                        break
            if oj_csv:
                break
        if oj_csv is None:
            raise FileNotFoundError("未找到 OJ CSV 文件，请使用 --oj-csv 指定")
    demand = _load_oj_gas(oj_csv)

    parsed = _parse_meos_inputs(data_dir, renewable_dir)
    prices_g = np.asarray(parsed.prices_gas, dtype=float)
    carbon_g = np.asarray(parsed.carbon_gas, dtype=float)
    prices_g = prices_g + score_spec["carbon_price"] * (carbon_g / score_spec["gas_to_MWh"])
    prices_g = _apply_price_modulation(prices_g, args.daily_price_amp, args.seasonal_price_amp)
    if prices_g.size != demand.size:
        raise ValueError("gas price length mismatch with demand")

    baseline_cost = float(np.sum(demand * prices_g))
    avg_demand = float(np.mean(demand))

    cap_hours = _parse_cap_hours(args.cap_hours)
    results = []
    for h in cap_hours:
        cap_mwh = avg_demand * h
        if cap_mwh <= 0:
            cost = baseline_cost
            savings = 0.0
        else:
            cost, _ = _solve_linepack(demand, prices_g, cap_mwh, args.loss_rate)
            savings = baseline_cost - cost
        results.append({
            "cap_hours": h,
            "cap_mwh": cap_mwh,
            "cost": cost,
            "savings": savings,
            "savings_ratio": savings / baseline_cost if baseline_cost > 0 else 0.0,
        })

    summary = {
        "oj_csv": str(oj_csv),
        "baseline_cost": baseline_cost,
        "avg_demand_mwh": avg_demand,
        "loss_rate_hourly": args.loss_rate,
        "daily_price_amp": args.daily_price_amp,
        "seasonal_price_amp": args.seasonal_price_amp,
        "results": results,
        "objective_mode": "aligned",
        "timestamp": datetime.now().isoformat(),
    }

    run_dir = Path(args.output_dir) / f"q3_linepack_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Q3 summary saved to: {run_dir / 'summary.json'}")
    for item in results:
        print(f"cap={item['cap_hours']}h savings={item['savings']:.3f} ratio={item['savings_ratio']:.6f}")


if __name__ == "__main__":
    main()
