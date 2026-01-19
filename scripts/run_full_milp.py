#!/usr/bin/env python3
"""
Build and solve a full-year MILP with plan18 + dispatch in a single model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

import sys

_script_dir = Path(__file__).parent
_python_dir = _script_dir.parent
sys.path.insert(0, str(_python_dir))

from meos.dispatch.zonal_dispatcher import (
    DispatchOptions,
    load_simulation_data,
    load_topology,
    load_source_constraints,
)
from meos.export.oj_exporter import DispatchResult8760, export_oj_csv
from meos.export.platform_full_exporter import _parse_meos_inputs
from meos.ga.codec import PLAN18_BOUNDS, PLAN18_BASE_CAPACITY
from meos.ga.evaluator import OutputGenerator
from meos.simulate.annual_summary import calculate_annuity_factor, summarize_annual


def _load_device_catalog(path: Path) -> List[Dict[str, Any]]:
    if not HAS_YAML:
        raise ImportError("需要 PyYAML 读取设备目录")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    devices = data.get("device_catalog", {}).get("devices", [])
    return sorted(devices, key=lambda d: d.get("idx", 0))


def _load_device_max_units(path: Path) -> List[int]:
    devices = _load_device_catalog(path)
    max_units: List[int] = [0] * 18
    for idx, device in enumerate(devices):
        max_val = device.get("max_units")
        if max_val is None:
            max_val = device.get("max")
        if max_val is None:
            max_val = device.get("max_unit")
        if max_val is None:
            continue
        try:
            max_units[idx] = int(max_val)
        except (TypeError, ValueError):
            continue
    return max_units


def _load_score_spec(path: Path, data_dir: Optional[Path] = None) -> Dict[str, Any]:
    if not HAS_YAML:
        raise ImportError("需要 PyYAML 读取评分配置")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    score = data.get("score_spec", {})
    gas_factor = score.get("carbon", {}).get("gas_emission_factor", 0.002)
    try:
        base_dir = data_dir or Path("data/raw")
        parsed = _parse_meos_inputs(base_dir, base_dir)
        gas_vals = np.asarray(parsed.carbon_gas, dtype=float).reshape(-1)
        if gas_vals.size:
            mean_factor = float(np.mean(gas_vals))
            if mean_factor > 0:
                gas_factor = mean_factor
    except Exception:
        pass
    return {
        "carbon_threshold": score.get("carbon", {}).get("threshold", 100000.0),
        "carbon_price": score.get("carbon", {}).get("price", 600.0),
        "gas_emission_factor": gas_factor,
        "gas_to_MWh": score.get("units", {}).get("gas_m3_to_MWh", 0.01),
        "discount_rate": score.get("capex", {}).get("discount_rate", 0.04),
        "shed_penalty": score.get("load_shedding_penalty", 500000.0),
    }


def _scale_renewable_to_mw(values: np.ndarray, cap_mw: float, base_cap_mw: float = 500.0) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    max_val = float(np.nanmax(values))
    if max_val > 2.0:
        scale = cap_mw / base_cap_mw if base_cap_mw > 0 else 0.0
    else:
        scale = cap_mw
    return values * scale


def _load_curve_dates(data_dir: Path) -> Optional[List[str]]:
    for name in ("出力曲线_光伏.csv", "出力曲线_风电.csv"):
        path = data_dir / name
        if path.exists():
            df = pd.read_csv(path)
            if "日期" in df.columns and df["日期"].shape[0] >= 365:
                return df["日期"].astype(str).tolist()[:365]
    return None


def _load_plan18(plan18_path: Optional[str], best_individual_path: Optional[str]) -> Optional[np.ndarray]:
    if best_individual_path:
        data = json.loads(Path(best_individual_path).read_text(encoding="utf-8"))
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
        return None

    if plan.size != 18:
        raise ValueError(f"plan18 length mismatch: {plan.size}")
    return plan


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


def _compute_cache_key(
    plan18: np.ndarray,
    guide_e: np.ndarray,
    guide_g: np.ndarray,
    storage_units: Optional[np.ndarray] = None,
    precision: int = 6,
) -> str:
    data = np.round(plan18, precision).tobytes()
    data += np.round(guide_e, precision).tobytes()
    data += np.round(guide_g, precision).tobytes()
    if storage_units is not None:
        data += np.round(storage_units, precision).tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _load_plan18_from_summary(summary_path: Optional[str]) -> Optional[np.ndarray]:
    if not summary_path:
        return None
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"summary not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    plan = np.asarray(data.get("plan18", []), dtype=float)
    if plan.size != 18:
        raise ValueError(f"summary plan18 length mismatch: {plan.size}")
    return plan


def _build_plan18_bounds(
    base_bounds: List[Tuple[int, int]],
    relax_zero_bounds: bool,
    zero_bound_max: int,
    upper_scale: float,
    upper_add: int,
    relax_from_plan18: Optional[np.ndarray],
    device_max_units: Optional[List[int]],
) -> List[Tuple[int, int]]:
    bounds = [(int(lb), int(ub)) for lb, ub in base_bounds]
    if relax_zero_bounds:
        for i, (lb, ub) in enumerate(bounds):
            if ub == 0:
                max_from_catalog = 0
                if device_max_units and i < len(device_max_units):
                    max_from_catalog = device_max_units[i]
                new_ub = max(zero_bound_max, max_from_catalog)
                if new_ub > 0:
                    bounds[i] = (lb, new_ub)
    if relax_from_plan18 is not None and (upper_scale > 1.0 or upper_add > 0):
        tol = 1e-6
        for i, (lb, ub) in enumerate(bounds):
            if ub <= 0:
                continue
            if relax_from_plan18[i] >= ub - tol:
                scaled = int(math.ceil(ub * upper_scale))
                added = ub + int(upper_add)
                bounds[i] = (lb, max(ub, scaled, added))
    return bounds


def _build_daily_results(
    P_thermal: np.ndarray,
    G_buy: np.ndarray,
    L_shed: np.ndarray,
    prices_e: np.ndarray,
    prices_g: np.ndarray,
    shed_penalty: float,
    n_days: int,
) -> List[Dict[str, Any]]:
    daily_results: List[Dict[str, Any]] = []
    for day in range(n_days):
        h_start = day * 24
        h_end = (day + 1) * 24
        p_thermal = P_thermal[h_start:h_end, :]
        g_buy = G_buy[h_start:h_end]
        l_shed = L_shed[h_start:h_end, :]
        p_total = np.sum(p_thermal, axis=1)
        l_shed_total = np.sum(l_shed, axis=1)
        cost_e = float(np.sum(p_total * prices_e[h_start:h_end]))
        cost_g = float(np.sum(g_buy * prices_g[h_start:h_end]))
        shed_total = float(np.sum(l_shed_total))
        cost_pen = float(shed_penalty * shed_total)
        daily_results.append(
            {
                "day_index": day + 1,
                "cost": {
                    "electricity": cost_e,
                    "gas": cost_g,
                    "penalty": cost_pen,
                    "total": cost_e + cost_g + cost_pen,
                },
                "P_thermal": p_thermal,
                "P_grid": p_total.tolist(),
                "G_buy": g_buy.tolist(),
                "L_shed": l_shed_total.tolist(),
                "shed_load": l_shed,
            }
        )
    return daily_results


def _stack_storage_series(storage: np.ndarray) -> np.ndarray:
    """Flatten storage series to match exporter index_map ordering."""
    arr = np.asarray(storage, dtype=float)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError(f"storage series must be (n_hours, 3, 3), got {arr.shape}")
    return np.column_stack(
        [
            arr[:, 0, 0],  # elec_s
            arr[:, 2, 0],  # elec_t
            arr[:, 1, 0],  # elec_f
            arr[:, 0, 1],  # heat_s
            arr[:, 2, 1],  # heat_t
            arr[:, 1, 1],  # heat_f
            arr[:, 0, 2],  # cool_s
            arr[:, 2, 2],  # cool_t
        ]
    )


def _build_dispatch_data(
    P_thermal: np.ndarray,
    G_buy: np.ndarray,
    L_shed: np.ndarray,
    P_pv: Optional[np.ndarray] = None,
    P_wind: Optional[np.ndarray] = None,
    P_p2g: Optional[np.ndarray] = None,
    P_e_boiler: Optional[np.ndarray] = None,
    P_hp_b: Optional[np.ndarray] = None,
    P_chiller_a: Optional[np.ndarray] = None,
    P_hp_a: Optional[np.ndarray] = None,
    P_gt: Optional[np.ndarray] = None,
    P_chiller_b: Optional[np.ndarray] = None,
    P_gas_boiler: Optional[np.ndarray] = None,
    P_cchp: Optional[np.ndarray] = None,
    P_absorption: Optional[np.ndarray] = None,
    H_transfer_s: Optional[np.ndarray] = None,
    H_transfer_f: Optional[np.ndarray] = None,
    line_flow: Optional[np.ndarray] = None,
    storage_ch: Optional[np.ndarray] = None,
    storage_dis: Optional[np.ndarray] = None,
    soc: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    dispatch: Dict[str, np.ndarray] = {
        "shed_load": np.asarray(L_shed, dtype=float),
        "thermal_power": np.asarray(P_thermal, dtype=float),
        "gas_source": np.asarray(G_buy, dtype=float).reshape(-1, 1),
    }

    def _set(key: str, val: Optional[np.ndarray]) -> None:
        if val is None:
            return
        arr = np.asarray(val, dtype=float)
        dispatch[key] = arr

    _set("pv_power", P_pv)
    _set("wind_power", P_wind)
    _set("p2g", P_p2g)
    _set("e_boiler", P_e_boiler)
    _set("hp_b", P_hp_b)
    _set("chiller_a", P_chiller_a)
    _set("hp_a", P_hp_a)
    _set("gt", P_gt)
    _set("chiller_b", P_chiller_b)
    _set("gas_boiler", P_gas_boiler)
    _set("cchp", P_cchp)
    _set("absorption", P_absorption)
    _set("heat_transfer_s", H_transfer_s)
    _set("heat_transfer_f", H_transfer_f)
    _set("line_flow", line_flow)
    _set("storage_ch", storage_ch)
    _set("storage_dis", storage_dis)
    _set("SOC", soc)

    for key, arr in list(dispatch.items()):
        arr = np.asarray(arr, dtype=float)
        arr[np.abs(arr) < 1e-6] = 0.0
        arr[arr < 0] = 0.0
        dispatch[key] = arr
    return dispatch


def build_and_solve(
    n_days: int,
    time_limit: float,
    mip_gap: float,
    threads: int,
    storage_mutex: bool,
    storage_daily_balance: bool,
    data_dir: Path,
    renewable_dir: Path,
    score_spec_path: Optional[Path],
    fixed_plan18: Optional[np.ndarray],
    carbon_mode: str,
    carbon_regime: str,
    objective_mode: str,
    carbon_threshold_override: Optional[float],
    carbon_price_override: Optional[float],
    relax_zero_bounds: bool,
    zero_bound_max: int,
    upper_bound_scale: float,
    upper_bound_add: int,
    relax_from_summary: Optional[str],
    platform_template: Optional[str],
    branch_reference: Optional[str],
    export_outputs: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    if not HAS_GUROBI:
        raise RuntimeError("缺少 gurobipy")

    run_dir = output_dir / f"full_milp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    options = DispatchOptions()
    options.data_dir = str(data_dir)
    options.renewable_dir = str(renewable_dir)
    data = load_simulation_data(options)
    grid = load_source_constraints(load_topology(options), options)
    parsed = _parse_meos_inputs(data_dir, renewable_dir)

    loads = data.loads.total
    prices_e = data.prices.electricity
    prices_g = data.prices.gas
    pv_avail = data.renewable.pv_avail
    wind_avail = data.renewable.wind_avail
    carbon_e = parsed.carbon_electricity
    carbon_g = parsed.carbon_gas

    n_hours = min(n_days * 24, loads.shape[0])
    loads = loads[:n_hours]
    prices_e = prices_e[:n_hours]
    prices_g = prices_g[:n_hours]
    pv_avail = pv_avail[:n_hours]
    wind_avail = wind_avail[:n_hours]
    carbon_e = carbon_e[:n_hours]
    carbon_g = carbon_g[:n_hours]

    score_path = score_spec_path if score_spec_path else Path("configs/oj_score.yaml")
    score_spec = _load_score_spec(score_path, data_dir=data_dir)
    if carbon_threshold_override is not None:
        score_spec["carbon_threshold"] = float(carbon_threshold_override)
    if carbon_price_override is not None:
        score_spec["carbon_price"] = float(carbon_price_override)
    prices_e_guided, prices_g_guided = _compute_aligned_prices(
        prices_e,
        prices_g,
        carbon_e,
        carbon_g,
        float(score_spec["carbon_price"]),
        float(score_spec["gas_to_MWh"]),
    )
    if carbon_mode not in ("aligned", "threshold"):
        raise ValueError(f"unsupported carbon_mode: {carbon_mode}")
    if carbon_regime not in ("auto", "relax", "penalty"):
        raise ValueError(f"unsupported carbon_regime: {carbon_regime}")
    if objective_mode not in ("cost", "emission"):
        raise ValueError(f"unsupported objective_mode: {objective_mode}")
    relax_plan18 = _load_plan18_from_summary(relax_from_summary)
    device_max_units = _load_device_max_units(Path("spec/device_catalog.yaml"))

    output_dir.mkdir(parents=True, exist_ok=True)
    model = gp.Model("full_year_milp")
    model.Params.OutputFlag = 1
    if time_limit > 0:
        model.Params.TimeLimit = time_limit
    if mip_gap > 0:
        model.Params.MIPGap = mip_gap
    if threads > 0:
        model.Params.Threads = threads

    plan_bounds = _build_plan18_bounds(
        PLAN18_BOUNDS,
        relax_zero_bounds=relax_zero_bounds,
        zero_bound_max=zero_bound_max,
        upper_scale=upper_bound_scale,
        upper_add=upper_bound_add,
        relax_from_plan18=relax_plan18,
        device_max_units=device_max_units,
    )
    plan_lo = [b[0] for b in plan_bounds]
    plan_hi = [b[1] for b in plan_bounds]
    plan18 = model.addMVar(shape=18, vtype=GRB.INTEGER, lb=plan_lo, ub=plan_hi, name="plan18")
    if fixed_plan18 is not None:
        fixed_vals = np.round(fixed_plan18).astype(int)
        for i, val in enumerate(fixed_vals):
            model.addConstr(plan18[i] == float(val))
    base_caps = np.array(PLAN18_BASE_CAPACITY, dtype=float)

    n_thermal = 3
    n_storage_types = 3
    n_zones = 3
    n_nodes = grid.n_nodes
    n_lines = grid.n_lines

    P_thermal = model.addMVar((n_hours, n_thermal), lb=0.0, name="P_thermal")
    G_buy = model.addMVar(n_hours, lb=0.0, name="G_buy")
    P_pv = model.addMVar(n_hours, lb=0.0, name="P_pv")
    P_wind = model.addMVar(n_hours, lb=0.0, name="P_wind")
    P_e_boiler = model.addMVar(n_hours, lb=0.0, name="P_e_boiler")
    P_hp_b = model.addMVar(n_hours, lb=0.0, name="P_hp_b")
    P_chiller_a = model.addMVar(n_hours, lb=0.0, name="P_chiller_a")
    P_p2g = model.addMVar(n_hours, lb=0.0, name="P_p2g")
    P_hp_a = model.addMVar(n_hours, lb=0.0, name="P_hp_a")
    P_gt = model.addMVar(n_hours, lb=0.0, name="P_gt")
    P_chiller_b = model.addMVar(n_hours, lb=0.0, name="P_chiller_b")
    P_gas_boiler = model.addMVar(n_hours, lb=0.0, name="P_gas_boiler")
    P_cchp = model.addMVar(n_hours, lb=0.0, name="P_cchp")
    P_absorption = model.addMVar(n_hours, lb=0.0, name="P_absorption")
    P_grid_f = model.addMVar(n_hours, lb=-GRB.INFINITY, name="P_grid_f")

    P_storage_ch = model.addMVar((n_hours, n_zones, n_storage_types), lb=0.0, name="P_storage_ch")
    P_storage_dis = model.addMVar((n_hours, n_zones, n_storage_types), lb=0.0, name="P_storage_dis")
    if storage_daily_balance:
        SOC = model.addMVar((n_days, 25, n_zones, n_storage_types), lb=0.0, name="SOC")
    else:
        SOC = model.addMVar((n_hours + 1, n_zones, n_storage_types), lb=0.0, name="SOC")

    L_shed = model.addMVar((n_hours, 9), lb=0.0, name="L_shed")
    H_transfer_s = model.addMVar(n_hours, lb=0.0, name="H_transfer_s")
    H_transfer_f = model.addMVar(n_hours, lb=0.0, name="H_transfer_f")

    F_ij = model.addMVar((n_hours, n_lines), lb=-GRB.INFINITY, name="F_ij")
    theta_i = model.addMVar((n_hours, n_nodes), lb=-GRB.INFINITY, name="theta_i")

    if storage_mutex:
        y_ch = model.addMVar((n_hours, n_zones, n_storage_types), vtype=GRB.BINARY, name="y_ch")
    else:
        y_ch = None

    # Device capacities (linear in plan18)
    cap_e_boiler = plan18[3] * base_caps[3]
    cap_chiller_a = plan18[4] * base_caps[4]
    cap_chiller_b = plan18[5] * base_caps[5]
    cap_absorption = plan18[6] * base_caps[6]
    cap_gas_boiler = plan18[7] * base_caps[7]
    cap_hp_a = plan18[8] * base_caps[8]
    cap_hp_b = plan18[9] * base_caps[9]
    storage_units = model.addMVar((n_zones, n_storage_types), vtype=GRB.INTEGER, lb=0.0, name="storage_units")
    for s in range(n_storage_types):
        model.addConstr(gp.quicksum(storage_units[z, s] for z in range(n_zones)) == plan18[10 + s])
    storage_base_caps = base_caps[10:13]
    # 风电/光伏在 OJ 中按 MW 直接计入
    cap_wind = plan18[13]
    cap_pv = plan18[14]
    cap_p2g = plan18[15] * base_caps[15]
    cap_gt = plan18[16] * base_caps[16]
    cap_cchp = plan18[17] * base_caps[17]

    storage_power_ratio = np.array(options.storage_power_ratio, dtype=float)
    storage_eta_ch = np.array(options.storage_eta_ch, dtype=float)
    storage_eta_dis = np.array(options.storage_eta_dis, dtype=float)

    # Bounds and capacity constraints
    for t in range(n_hours):
        for k in range(n_thermal):
            model.addConstr(P_thermal[t, k] >= grid.thermal_min[k])
            model.addConstr(P_thermal[t, k] <= grid.thermal_max[k])
        model.addConstr(G_buy[t] >= grid.gas_min)
        model.addConstr(G_buy[t] <= grid.gas_max)

        model.addConstr(P_pv[t] <= cap_pv * pv_avail[t])
        model.addConstr(P_wind[t] <= cap_wind * wind_avail[t])
        model.addConstr(P_e_boiler[t] <= cap_e_boiler)
        model.addConstr(P_hp_b[t] <= cap_hp_b)
        model.addConstr(P_chiller_a[t] <= cap_chiller_a)
        model.addConstr(P_p2g[t] <= cap_p2g)
        model.addConstr(P_hp_a[t] <= cap_hp_a)
        model.addConstr(P_gt[t] <= cap_gt)
        model.addConstr(P_chiller_b[t] <= cap_chiller_b)
        model.addConstr(P_gas_boiler[t] <= cap_gas_boiler)
        model.addConstr(P_cchp[t] <= cap_cchp)
        model.addConstr(P_absorption[t] <= cap_absorption)

        day = t // 24
        hour = t % 24
        for z in range(n_zones):
            for s in range(n_storage_types):
                cap_E = storage_units[z, s] * storage_base_caps[s]
                cap_P = cap_E * storage_power_ratio[s]
                model.addConstr(P_storage_ch[t, z, s] <= cap_P)
                model.addConstr(P_storage_dis[t, z, s] <= cap_P)
                if storage_daily_balance:
                    model.addConstr(SOC[day, hour, z, s] >= cap_E * options.storage_soc_min)
                    model.addConstr(SOC[day, hour, z, s] <= cap_E * options.storage_soc_max)
                else:
                    model.addConstr(SOC[t, z, s] >= cap_E * options.storage_soc_min)
                    model.addConstr(SOC[t, z, s] <= cap_E * options.storage_soc_max)
                if y_ch is not None:
                    model.addConstr(P_storage_ch[t, z, s] <= cap_P * y_ch[t, z, s])
                    model.addConstr(P_storage_dis[t, z, s] <= cap_P * (1 - y_ch[t, z, s]))

        for l in range(n_lines):
            model.addConstr(F_ij[t, l] <= grid.F_max[l])
            model.addConstr(F_ij[t, l] >= -grid.F_max[l])

        model.addConstr(theta_i[t, :] >= grid.theta_min)
        model.addConstr(theta_i[t, :] <= grid.theta_max)

        if not grid.zone_elec_export[1]:
            model.addConstr(P_grid_f[t] >= 0)

    if storage_daily_balance:
        for day in range(n_days):
            for z in range(n_zones):
                for s in range(n_storage_types):
                    cap_E = storage_units[z, s] * storage_base_caps[s]
                    model.addConstr(SOC[day, 24, z, s] >= cap_E * options.storage_soc_min)
                    model.addConstr(SOC[day, 24, z, s] <= cap_E * options.storage_soc_max)
    else:
        for z in range(n_zones):
            for s in range(n_storage_types):
                cap_E = storage_units[z, s] * storage_base_caps[s]
                model.addConstr(SOC[n_hours, z, s] >= cap_E * options.storage_soc_min)
                model.addConstr(SOC[n_hours, z, s] <= cap_E * options.storage_soc_max)

    # Energy balance and dynamics
    for t in range(n_hours):
        L_elec_s, L_cool_s, L_heat_s, L_elec_f, L_cool_f, L_heat_f, L_elec_t, L_heat_t, L_cool_t = loads[t]

        storage_e = 0
        storage_h = 1
        storage_c = 2

        for n in range(n_nodes):
            flow_sum = gp.quicksum(-grid.incidence[n, l] * F_ij[t, l] for l in range(n_lines))
            thermal_units = [i for i, bus in enumerate(grid.thermal_to_bus) if bus == n + 1]
            thermal_sum = gp.quicksum(P_thermal[t, k] for k in thermal_units)
            if n + 1 == grid.zone_to_bus[0]:
                model.addConstr(
                    flow_sum + thermal_sum + P_pv[t]
                    - P_e_boiler[t] - P_hp_b[t] - P_chiller_a[t] - P_p2g[t]
                    - P_storage_ch[t, 0, storage_e] + P_storage_dis[t, 0, storage_e] + L_shed[t, 0]
                    == L_elec_s
                )
            elif n + 1 == grid.zone_to_bus[1]:
                model.addConstr(flow_sum + thermal_sum - P_grid_f[t] == 0)
            else:
                model.addConstr(
                    flow_sum + thermal_sum + P_wind[t] + P_cchp[t] * 0.4
                    - P_storage_ch[t, 2, storage_e] + P_storage_dis[t, 2, storage_e]
                    + L_shed[t, 6] == L_elec_t
                )

        model.addConstr(
            P_grid_f[t] + P_gt[t] * 0.7 - P_hp_a[t] - P_chiller_b[t]
            - P_storage_ch[t, 1, storage_e] + P_storage_dis[t, 1, storage_e]
            + L_shed[t, 3] == L_elec_f
        )

        model.addConstr(
            P_e_boiler[t] * 0.9 + P_hp_b[t] * 6 + H_transfer_s[t]
            + P_storage_dis[t, 0, storage_h] - P_storage_ch[t, 0, storage_h] + L_shed[t, 2] == L_heat_s
        )
        model.addConstr(
            P_hp_a[t] * 5 + H_transfer_f[t]
            + P_storage_dis[t, 1, storage_h] - P_storage_ch[t, 1, storage_h]
            + L_shed[t, 5] == L_heat_f
        )
        model.addConstr(
            P_gas_boiler[t] * 0.95 + P_cchp[t] * 0.3 - P_absorption[t]
            - H_transfer_s[t] - H_transfer_f[t]
            + P_storage_dis[t, 2, storage_h] - P_storage_ch[t, 2, storage_h]
            + L_shed[t, 7] == L_heat_t
        )

        model.addConstr(
            P_chiller_a[t] * 4
            + P_storage_dis[t, 0, storage_c] - P_storage_ch[t, 0, storage_c]
            + L_shed[t, 1] == L_cool_s
        )
        model.addConstr(
            P_chiller_b[t] * 5
            + P_storage_dis[t, 1, storage_c] - P_storage_ch[t, 1, storage_c]
            + L_shed[t, 4] == L_cool_f
        )
        model.addConstr(
            P_absorption[t] * 0.8 + P_cchp[t] * 0.3
            + P_storage_dis[t, 2, storage_c] - P_storage_ch[t, 2, storage_c]
            + L_shed[t, 8] == L_cool_t
        )

        model.addConstr(
            G_buy[t] + P_p2g[t] * 0.4 - P_gas_boiler[t] - P_cchp[t] - P_gt[t] == 0
        )

        if grid.heat_capacity is not None and np.isfinite(grid.heat_capacity):
            model.addConstr(H_transfer_s[t] + H_transfer_f[t] <= grid.heat_capacity)
        if not grid.zone_heat_export[2]:
            model.addConstr(H_transfer_s[t] == 0)
            model.addConstr(H_transfer_f[t] == 0)
        if not grid.zone_heat_import[0]:
            model.addConstr(H_transfer_s[t] == 0)
        if not grid.zone_heat_import[1]:
            model.addConstr(H_transfer_f[t] == 0)

        day = t // 24
        hour = t % 24
        for z in range(n_zones):
            for s in range(n_storage_types):
                if storage_daily_balance:
                    model.addConstr(
                        SOC[day, hour + 1, z, s] - SOC[day, hour, z, s]
                        - storage_eta_ch[s] * P_storage_ch[t, z, s]
                        + (1 / storage_eta_dis[s]) * P_storage_dis[t, z, s] == 0
                    )
                else:
                    model.addConstr(
                        SOC[t + 1, z, s] - SOC[t, z, s]
                        - storage_eta_ch[s] * P_storage_ch[t, z, s]
                        + (1 / storage_eta_dis[s]) * P_storage_dis[t, z, s] == 0
                    )

        for l in range(n_lines):
            i = grid.line_from[l] - 1
            j = grid.line_to[l] - 1
            model.addConstr(F_ij[t, l] - grid.B_ij[l] * (theta_i[t, i] - theta_i[t, j]) == 0)

        ref_bus = 2 if n_nodes >= 3 else 0
        model.addConstr(theta_i[t, ref_bus] == 0)

    if storage_daily_balance:
        for day in range(n_days):
            for z in range(n_zones):
                for s in range(n_storage_types):
                    model.addConstr(SOC[day, 0, z, s] == SOC[day, 24, z, s])

    # Objective
    device_catalog = _load_device_catalog(Path("spec/device_catalog.yaml"))
    discount_rate = score_spec["discount_rate"]
    annuity = []
    for device in device_catalog:
        years = int(device.get("lifespan", 20))
        annuity.append(calculate_annuity_factor(discount_rate, years))
    annuity = np.array(annuity, dtype=float)
    unit_cost = np.array([d.get("unit_cost", 0.0) for d in device_catalog], dtype=float)
    capex_base = base_caps.copy()
    for i, device in enumerate(device_catalog):
        device_id = str(device.get("device_id", ""))
        if device_id in ("WindTurbine", "PV"):
            # OJ scoring treats wind/pv plan18 as MW directly.
            capex_base[i] = 1.0
    capex_coeff = annuity * unit_cost * capex_base
    C_CAP = gp.quicksum(capex_coeff[i] * plan18[i] for i in range(18))

    shed_penalty = score_spec["shed_penalty"]
    if carbon_mode == "threshold":
        C_OP_ele = gp.quicksum(
            prices_e[t] * gp.quicksum(P_thermal[t, k] for k in range(n_thermal))
            for t in range(n_hours)
        )
        C_OP_gas = gp.quicksum(prices_g[t] * G_buy[t] for t in range(n_hours))
    else:
        C_OP_ele = gp.quicksum(
            prices_e_guided[t] * gp.quicksum(P_thermal[t, k] for k in range(n_thermal))
            for t in range(n_hours)
        )
        C_OP_gas = gp.quicksum(prices_g_guided[t] * G_buy[t] for t in range(n_hours))
    C_OP_pen = shed_penalty * gp.quicksum(L_shed[t, i] for t in range(n_hours) for i in range(9))
    C_OP = C_OP_ele + C_OP_gas + C_OP_pen

    gas_to_MWh = score_spec["gas_to_MWh"]
    gas_factor = score_spec["gas_emission_factor"]
    carbon_threshold = score_spec["carbon_threshold"]
    carbon_price = score_spec["carbon_price"]

    E_elec = gp.quicksum(
        carbon_e[t] * gp.quicksum(P_thermal[t, k] for k in range(n_thermal))
        for t in range(n_hours)
    )
    E_gas = gp.quicksum(G_buy[t] / gas_to_MWh * gas_factor for t in range(n_hours))
    E_total = E_elec + E_gas

    if objective_mode == "emission":
        model.setObjective(E_total, GRB.MINIMIZE)
    else:
        if carbon_mode == "threshold":
            if carbon_regime == "auto":
                E_excess = model.addVar(lb=0.0, name="E_excess")
                model.addConstr(E_excess >= E_total - carbon_threshold)
                C_Carbon = carbon_price * E_excess
                model.setObjective(C_CAP + C_OP + C_Carbon, GRB.MINIMIZE)
            elif carbon_regime == "relax":
                model.addConstr(E_total <= carbon_threshold)
                model.setObjective(C_CAP + C_OP, GRB.MINIMIZE)
            else:
                model.addConstr(E_total >= carbon_threshold)
                C_Carbon = carbon_price * (E_total - carbon_threshold)
                model.setObjective(C_CAP + C_OP + C_Carbon, GRB.MINIMIZE)
        else:
            model.setObjective(C_CAP + C_OP, GRB.MINIMIZE)

    t0 = time.time()
    model.optimize()
    elapsed = time.time() - t0

    status = model.Status
    status_name = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INTERRUPTED: "INTERRUPTED",
    }.get(status, f"STATUS_{status}")

    summary: Dict[str, Any] = {
        "status": status,
        "status_name": status_name,
        "elapsed_sec": elapsed,
        "n_days": n_days,
        "n_hours": n_hours,
        "carbon_mode": carbon_mode,
        "carbon_regime": carbon_regime,
        "objective_mode": objective_mode,
        "carbon_threshold": carbon_threshold,
        "carbon_price": carbon_price,
        "plan18_bounds": plan_bounds,
        "run_dir": str(run_dir),
    }
    if hasattr(model, "ObjVal") and status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        objective = float(model.ObjVal)
        if objective_mode == "cost" and carbon_mode == "aligned":
            objective -= carbon_price * carbon_threshold
        summary["objective"] = objective
    if hasattr(model, "ObjBound") and status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        objective_bound = float(model.ObjBound)
        if objective_mode == "cost" and carbon_mode == "aligned":
            objective_bound -= carbon_price * carbon_threshold
        summary["objective_bound"] = objective_bound
    if hasattr(model, "MIPGap") and status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        summary["mip_gap"] = float(model.MIPGap)

    if status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        plan18_val = fixed_plan18 if fixed_plan18 is not None else plan18.X
        plan18_val = np.asarray(plan18_val, dtype=float).reshape(18)
        summary["plan18"] = plan18_val.tolist()
        storage_units_val = np.asarray(storage_units.X, dtype=float)
        storage_caps_val = storage_units_val * storage_base_caps
        storage_power_ratio = np.array(options.storage_power_ratio, dtype=float)
        storage_power_val = storage_caps_val * storage_power_ratio.reshape(1, -1)
        summary["storage_zone_names"] = ["学生区", "教工区", "教学办公区"]
        summary["storage_type_names"] = ["电储能", "热储能", "冷储能"]
        summary["storage_units_by_zone"] = storage_units_val.tolist()
        summary["storage_capacity_by_zone"] = storage_caps_val.tolist()
        summary["storage_power_by_zone"] = storage_power_val.tolist()

        P_thermal_val = np.asarray(P_thermal.X, dtype=float)
        G_buy_val = np.asarray(G_buy.X, dtype=float).reshape(-1)
        L_shed_val = np.asarray(L_shed.X, dtype=float)

        daily_results = _build_daily_results(
            P_thermal_val,
            G_buy_val,
            L_shed_val,
            prices_e,
            prices_g,
            float(shed_penalty),
            n_days,
        )
        plan_capacity = plan18_val * base_caps
        if plan_capacity.size >= 15:
            plan_capacity[13] = plan18_val[13]
            plan_capacity[14] = plan18_val[14]

        annual = summarize_annual(
            daily_results=daily_results,
            plan18=plan_capacity.tolist(),
            device_catalog=device_catalog,
            carbon_factors=carbon_e.tolist(),
            config=score_spec,
            day_weights=None,
        )
        summary.update(
            {
                "C_CAP": annual.C_CAP,
                "C_OP": annual.C_OP_total,
                "C_Carbon": annual.C_Carbon,
                "C_total": annual.C_total,
                "Score": annual.Score,
                "carbon": asdict(annual.carbon),
            }
        )

        if export_outputs and n_hours == 8760:
            output_gen = OutputGenerator(str(run_dir))
            cache_key = _compute_cache_key(plan18_val, prices_e_guided, prices_g_guided, storage_units_val)
            summary["cache_key"] = cache_key

            summary["capacity_yaml"] = output_gen.export_capacity_yaml(plan18_val, cache_key)
            summary["guide_price_csv"] = output_gen.export_guide_price_csv(
                {"electricity": prices_e_guided.tolist(), "gas": prices_g_guided.tolist()},
                cache_key,
            )
            summary["guide_gas_csv"] = output_gen.export_guide_gas_csv(
                prices_g_guided.tolist(),
                cache_key,
            )

            curve_dates = _load_curve_dates(data_dir)
            pv_mw = _scale_renewable_to_mw(pv_avail, float(plan18_val[14]))
            wind_mw = _scale_renewable_to_mw(wind_avail, float(plan18_val[13]))
            pv_paths = output_gen.export_renewable_curve_csv(pv_mw, cache_key, "pv_curve", dates=curve_dates)
            wind_paths = output_gen.export_renewable_curve_csv(wind_mw, cache_key, "wind_curve", dates=curve_dates)
            summary["pv_curve_csv"] = pv_paths["csv"]
            summary["pv_curve_xlsx"] = pv_paths["xlsx"]
            summary["wind_curve_csv"] = wind_paths["csv"]
            summary["wind_curve_xlsx"] = wind_paths["xlsx"]

            from meos.export.platform_full_exporter import PlatformExportOptions
            platform_options = PlatformExportOptions()
            if platform_template:
                platform_options.template_path = platform_template
            if branch_reference:
                platform_options.branch_reference_path = branch_reference

            P_pv_val = np.asarray(P_pv.X, dtype=float).reshape(-1)
            P_wind_val = np.asarray(P_wind.X, dtype=float).reshape(-1)
            P_p2g_val = np.asarray(P_p2g.X, dtype=float).reshape(-1)
            P_e_boiler_val = np.asarray(P_e_boiler.X, dtype=float).reshape(-1)
            P_hp_b_val = np.asarray(P_hp_b.X, dtype=float).reshape(-1)
            P_chiller_a_val = np.asarray(P_chiller_a.X, dtype=float).reshape(-1)
            P_hp_a_val = np.asarray(P_hp_a.X, dtype=float).reshape(-1)
            P_gt_val = np.asarray(P_gt.X, dtype=float).reshape(-1)
            P_chiller_b_val = np.asarray(P_chiller_b.X, dtype=float).reshape(-1)
            P_gas_boiler_val = np.asarray(P_gas_boiler.X, dtype=float).reshape(-1)
            P_cchp_val = np.asarray(P_cchp.X, dtype=float).reshape(-1)
            P_absorption_val = np.asarray(P_absorption.X, dtype=float).reshape(-1)
            H_transfer_s_val = np.asarray(H_transfer_s.X, dtype=float).reshape(-1)
            H_transfer_f_val = np.asarray(H_transfer_f.X, dtype=float).reshape(-1)
            F_ij_val = np.asarray(F_ij.X, dtype=float)
            storage_ch_val = np.asarray(P_storage_ch.X, dtype=float)
            storage_dis_val = np.asarray(P_storage_dis.X, dtype=float)

            if storage_daily_balance:
                soc_raw = np.asarray(SOC.X, dtype=float)
                soc_hours = soc_raw[:, :24, :, :].reshape(n_hours, n_zones, n_storage_types)
            else:
                soc_hours = np.asarray(SOC.X, dtype=float)[:n_hours, :, :]

            storage_ch_flat = _stack_storage_series(storage_ch_val)
            storage_dis_flat = _stack_storage_series(storage_dis_val)
            soc_flat = _stack_storage_series(soc_hours)

            dispatch_data = _build_dispatch_data(
                P_thermal_val,
                G_buy_val,
                L_shed_val,
                P_pv=P_pv_val,
                P_wind=P_wind_val,
                P_p2g=P_p2g_val,
                P_e_boiler=P_e_boiler_val,
                P_hp_b=P_hp_b_val,
                P_chiller_a=P_chiller_a_val,
                P_hp_a=P_hp_a_val,
                P_gt=P_gt_val,
                P_chiller_b=P_chiller_b_val,
                P_gas_boiler=P_gas_boiler_val,
                P_cchp=P_cchp_val,
                P_absorption=P_absorption_val,
                H_transfer_s=H_transfer_s_val,
                H_transfer_f=H_transfer_f_val,
                line_flow=F_ij_val,
                storage_ch=storage_ch_flat,
                storage_dis=storage_dis_flat,
                soc=soc_flat,
            )
            summary["platform_csv"] = output_gen.export_platform_csv(
                dispatch_data,
                plan18_val,
                cache_key,
                platform_options,
            )

            oj_path = output_gen._get_output_path(cache_key, "oj.csv")
            dispatch_result = DispatchResult8760(
                shed_load=dispatch_data["shed_load"],
                thermal_power=dispatch_data["thermal_power"],
                gas_source=dispatch_data["gas_source"],
            )
            export_oj_csv(dispatch_result, plan18_val, oj_path, validate=True)
            summary["oj_csv"] = str(oj_path)
        elif export_outputs and n_hours != 8760:
            summary["export_skipped"] = "n_hours != 8760"

    output_path = run_dir / "full_milp_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-year MILP with plan18 + dispatch")
    parser.add_argument("--days", type=int, default=365, help="number of days")
    parser.add_argument("--time-limit", type=float, default=0.0, help="time limit (s)")
    parser.add_argument("--mip-gap", type=float, default=0.01, help="MIP gap")
    parser.add_argument("--threads", type=int, default=0, help="Gurobi threads")
    parser.add_argument("--storage-mutex", dest="storage_mutex", action="store_true", help="enable storage mutex")
    parser.add_argument("--no-storage-mutex", dest="storage_mutex", action="store_false", help="disable storage mutex")
    parser.add_argument("--storage-daily-balance", dest="storage_daily_balance", action="store_true", help="daily SOC balance (platform mode)")
    parser.add_argument("--storage-cross-day", dest="storage_daily_balance", action="store_false", help="cross-day SOC continuity")
    parser.add_argument("--data-dir", default="data/raw", help="input data dir")
    parser.add_argument("--renewable-dir", default=None, help="renewable data dir")
    parser.add_argument("--score-spec-path", default=None, help="score spec yaml path")
    parser.add_argument("--plan18", default=None, help="fixed plan18 yaml/json path")
    parser.add_argument("--best-individual", default=None, help="best_individual.json path")
    parser.add_argument("--carbon-mode", default="aligned", choices=["aligned", "threshold"], help="aligned or threshold")
    parser.add_argument("--carbon-regime", default="auto", choices=["auto", "relax", "penalty"], help="carbon regime for threshold mode")
    parser.add_argument("--objective", default="cost", choices=["cost", "emission"], help="objective: cost or emission")
    parser.add_argument("--carbon-threshold", type=float, default=None, help="override carbon threshold")
    parser.add_argument("--carbon-price", type=float, default=None, help="override carbon price")
    parser.add_argument("--relax-zero-bounds", action="store_true", help="relax zero upper bounds using catalog")
    parser.add_argument("--zero-bound-max", type=int, default=0, help="fallback max for zero-bound devices")
    parser.add_argument("--upper-bound-scale", type=float, default=1.0, help="scale upper bounds for capped devices")
    parser.add_argument("--upper-bound-add", type=int, default=0, help="additive upper bound slack")
    parser.add_argument("--relax-from-summary", default=None, help="summary json to detect capped devices")
    parser.add_argument("--platform-template", default=None, help="platform export template (xls/xlsx/csv)")
    parser.add_argument("--branch-reference", default=None, help="branch reference file for export")
    parser.add_argument("--export", dest="export_outputs", action="store_true", help="export platform/OJ outputs")
    parser.add_argument("--no-export", dest="export_outputs", action="store_false", help="skip exports")
    parser.add_argument("--output-dir", default="runs/full_milp", help="output directory")
    parser.set_defaults(storage_mutex=True, storage_daily_balance=True, export_outputs=True)
    args = parser.parse_args()

    fixed_plan18 = _load_plan18(args.plan18, args.best_individual)
    data_dir = Path(args.data_dir)
    renewable_dir = Path(args.renewable_dir) if args.renewable_dir else data_dir
    score_path = Path(args.score_spec_path) if args.score_spec_path else None

    build_and_solve(
        n_days=args.days,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        storage_mutex=args.storage_mutex,
        storage_daily_balance=args.storage_daily_balance,
        data_dir=data_dir,
        renewable_dir=renewable_dir,
        score_spec_path=score_path,
        fixed_plan18=fixed_plan18,
        carbon_mode=args.carbon_mode,
        carbon_regime=args.carbon_regime,
        objective_mode=args.objective,
        carbon_threshold_override=args.carbon_threshold,
        carbon_price_override=args.carbon_price,
        relax_zero_bounds=args.relax_zero_bounds,
        zero_bound_max=args.zero_bound_max,
        upper_bound_scale=args.upper_bound_scale,
        upper_bound_add=args.upper_bound_add,
        relax_from_summary=args.relax_from_summary,
        platform_template=args.platform_template,
        branch_reference=args.branch_reference,
        export_outputs=args.export_outputs,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
