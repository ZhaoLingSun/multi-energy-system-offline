#!/usr/bin/env python3
"""
思考题4：传输容量受限下的分区规划与扩容评估（电力线容量扫描）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parents[2]
sys.path.insert(0, str(_project_root))

from meos.dispatch.zonal_dispatcher import DispatchOptions, ZonalDailySolver
from meos.export.platform_full_exporter import _parse_meos_inputs
from meos.ga.evaluator import CandidateEvaluator, EvaluatorConfig


_GLOBAL_SOLVER: Optional[ZonalDailySolver] = None
_GLOBAL_PLAN18: Optional[np.ndarray] = None


def _load_plan18(plan18_path: Optional[str], summary_path: Optional[str]) -> np.ndarray:
    if summary_path:
        data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        plan = np.asarray(data.get("plan18", []), dtype=float)
    elif plan18_path:
        path = Path(plan18_path)
        if path.suffix.lower() in (".yaml", ".yml") and HAS_YAML:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            plan = np.asarray(data.get("plan18", []), dtype=float)
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            plan = np.asarray(data.get("plan18", []), dtype=float)
    else:
        raise ValueError("plan18 is required (--plan18 or --plan18-summary)")

    if plan.size != 18:
        raise ValueError(f"plan18 length mismatch: {plan.size}")
    return plan


def _load_score_config(score_spec_path: Optional[str]) -> Dict[str, Any]:
    if score_spec_path:
        path = Path(score_spec_path)
    else:
        path = _project_root / "configs" / "oj_score.yaml"
    if path.exists() and HAS_YAML:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        score = raw.get("score_spec", {})
        return {
            "discount_rate": score.get("capex", {}).get("discount_rate", 0.04),
            "carbon_threshold": score.get("carbon", {}).get("threshold", 100000.0),
            "carbon_price": score.get("carbon", {}).get("price", 600.0),
            "gas_emission_factor": score.get("carbon", {}).get("gas_emission_factor", 0.002),
            "gas_to_MWh": score.get("units", {}).get("gas_m3_to_MWh", 0.01),
        }
    return {
        "discount_rate": 0.04,
        "carbon_threshold": 100000.0,
        "carbon_price": 600.0,
        "gas_emission_factor": 0.002,
        "gas_to_MWh": 0.01,
    }


def _compute_aligned_prices(
    prices_e: np.ndarray,
    prices_g: np.ndarray,
    carbon_e: np.ndarray,
    carbon_g: np.ndarray,
    carbon_price: float,
    gas_to_MWh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    aligned_e = prices_e + carbon_price * carbon_e
    aligned_g = prices_g + carbon_price * (carbon_g / gas_to_MWh)
    return aligned_e, aligned_g


def _init_worker(
    options_dict: Dict[str, Any],
    plan18: np.ndarray,
    guidance_e: np.ndarray,
    guidance_g: np.ndarray,
) -> None:
    global _GLOBAL_SOLVER, _GLOBAL_PLAN18
    options = DispatchOptions(**options_dict)
    options.guidance_price_matrix = guidance_e
    options.guidance_gas_matrix = guidance_g
    _GLOBAL_SOLVER = ZonalDailySolver(options)
    _GLOBAL_PLAN18 = plan18


def _solve_day(day_index: int) -> Tuple[int, Dict[str, Any]]:
    if _GLOBAL_SOLVER is None or _GLOBAL_PLAN18 is None:
        raise RuntimeError("worker not initialized")
    result = _GLOBAL_SOLVER.solve_day(day_index, _GLOBAL_PLAN18)
    if isinstance(result, dict):
        result = dict(result)
        result.setdefault("day_index", day_index)
    return day_index, result


def _run_days(
    n_days: int,
    workers: int,
    options_dict: Dict[str, Any],
    plan18: np.ndarray,
    guidance_e: np.ndarray,
    guidance_g: np.ndarray,
) -> List[Dict[str, Any]]:
    results: List[Optional[Dict[str, Any]]] = [None] * n_days
    if workers <= 1:
        _init_worker(options_dict, plan18, guidance_e, guidance_g)
        for day in range(1, n_days + 1):
            idx, res = _solve_day(day)
            results[idx - 1] = res
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(options_dict, plan18, guidance_e, guidance_g),
        ) as executor:
            futures = {executor.submit(_solve_day, day): day for day in range(1, n_days + 1)}
            for fut in as_completed(futures):
                idx, res = fut.result()
                results[idx - 1] = res
    missing = [i + 1 for i, r in enumerate(results) if r is None]
    if missing:
        raise RuntimeError(f"missing results for days: {missing}")
    return results  # type: ignore


def _total_shed(daily_results: List[Dict[str, Any]]) -> float:
    total = 0.0
    for dr in daily_results:
        shed = np.asarray(dr.get("L_shed", 0.0), dtype=float)
        total += float(np.sum(shed))
    return total


def _parse_cap_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_cap_list_optional(text: Optional[str]) -> List[Optional[float]]:
    if not text:
        return [None]
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="思考题4：线路容量受限规划评估")
    parser.add_argument("--plan18", default=None, help="plan18 yaml/json file")
    parser.add_argument("--plan18-summary", default=None, help="full_milp_summary.json path")
    parser.add_argument("--data-dir", default=None, help="input data dir")
    parser.add_argument("--renewable-dir", default=None, help="renewable data dir")
    parser.add_argument("--score-spec-path", default=None, help="score spec yaml path")
    parser.add_argument("--workers", type=int, default=0, help="parallel workers (0=auto)")
    parser.add_argument("--days-per-worker", type=int, default=2, help="days per worker for auto")
    parser.add_argument("--line-cap-list", default="200,400,600,800,1200", help="line capacity list (MW)")
    parser.add_argument("--heat-cap-list", default=None, help="heat transfer capacity list (MW), optional")
    parser.add_argument("--base-cap", type=float, default=400.0, help="base capacity (MW)")
    parser.add_argument("--capex-per-mw", type=float, default=50000.0, help="expansion cost per MW")
    parser.add_argument("--output-dir", default="runs/thought_questions", help="output base directory")
    args = parser.parse_args()

    plan18 = _load_plan18(args.plan18, args.plan18_summary)
    score_config = _load_score_config(args.score_spec_path)

    project_root = _project_root
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "matlab" / "data" / "raw"
    renewable_dir = Path(args.renewable_dir) if args.renewable_dir else data_dir

    parsed = _parse_meos_inputs(data_dir, renewable_dir)
    prices_e = np.asarray(parsed.prices_electricity, dtype=float)
    prices_g = np.asarray(parsed.prices_gas, dtype=float)
    carbon_e = np.asarray(parsed.carbon_electricity, dtype=float)
    carbon_g = np.asarray(parsed.carbon_gas, dtype=float)

    guidance_e, guidance_g = _compute_aligned_prices(
        prices_e,
        prices_g,
        carbon_e,
        carbon_g,
        float(score_config.get("carbon_price", 600.0)),
        float(score_config.get("gas_to_MWh", 0.01)),
    )
    guidance_e = guidance_e.reshape(365, 24)
    guidance_g = guidance_g.reshape(365, 24)

    if args.workers and args.workers > 0:
        workers = args.workers
    else:
        workers = int(math.ceil(365 / max(1, int(args.days_per_worker))))

    results = []
    line_caps = _parse_cap_list(args.line_cap_list)
    heat_caps = _parse_cap_list_optional(args.heat_cap_list)
    for cap in line_caps:
        for heat_cap in heat_caps:
            options = {
                "solver": "gurobi",
                "gurobi_threads": 0,
                "storage_mutual_exclusive": True,
                "data_dir": str(data_dir),
                "renewable_dir": str(renewable_dir),
                "line_capacity": cap,
            }
            if heat_cap is not None:
                options["heat_transfer_cap"] = heat_cap
            t0 = time.time()
            daily_results = _run_days(365, workers, options, plan18, guidance_e, guidance_g)
            evaluator = CandidateEvaluator(EvaluatorConfig(data_dir=str(data_dir), renewable_dir=str(renewable_dir)))
            daily_results = [evaluator._normalize_day_result(r) for r in daily_results]
            costs = evaluator._calculate_costs(daily_results, plan18)
            shed = _total_shed(daily_results)
            elapsed = time.time() - t0

            expansion = max(0.0, cap - args.base_cap)
            capex = expansion * args.capex_per_mw
            total_with_capex = costs["C_total"] + capex

            results.append({
                "line_capacity": cap,
                "heat_capacity": heat_cap,
                "C_total": costs["C_total"],
                "C_OP": costs["C_OP"],
                "C_Carbon": costs["C_Carbon"],
                "C_CAP": costs["C_CAP"],
                "Score": costs["Score"],
                "shed_total": shed,
                "elapsed_sec": elapsed,
                "expansion": expansion,
                "expansion_cost": capex,
                "total_with_capex": total_with_capex,
            })

    summary = {
        "base_cap": args.base_cap,
        "capex_per_mw": args.capex_per_mw,
        "line_caps": line_caps,
        "heat_caps": heat_caps,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    run_dir = Path(args.output_dir) / f"q4_line_capacity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Q4 summary saved to: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
