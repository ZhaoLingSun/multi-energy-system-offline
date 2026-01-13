#!/usr/bin/env python3
"""
思考题3：管道线包等效储能 - 完整MILP模型分析

在完整综合能源系统MILP模型中引入气网线包储能，
比较不同线包容量对系统总成本与运行策略的影响。

用法:
    python run_milp_linepack.py --linepack-cap-hours 0 --days 365 --mip-gap 1e-4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
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
_project_root = _script_dir.parents[2]
sys.path.insert(0, str(_project_root))

from meos.dispatch.zonal_dispatcher import (
    DispatchOptions,
    load_simulation_data,
    load_topology,
    load_source_constraints,
)
from meos.export.platform_full_exporter import _parse_meos_inputs
from meos.ga.codec import PLAN18_BOUNDS, PLAN18_BASE_CAPACITY
from meos.simulate.annual_summary import calculate_annuity_factor


def _load_device_catalog(path: Path) -> List[Dict[str, Any]]:
    if not HAS_YAML:
        raise ImportError("需要 PyYAML 读取设备目录")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    devices = data.get("device_catalog", {}).get("devices", [])
    return sorted(devices, key=lambda d: d.get("idx", 0))


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


def _load_plan18_from_summary(summary_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load plan18 and storage_units_by_zone from summary."""
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"summary not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    plan = np.asarray(data.get("plan18", []), dtype=float)
    if plan.size != 18:
        raise ValueError(f"plan18 length mismatch: {plan.size}")
    
    storage_units = None
    if "storage_units_by_zone" in data:
        storage_units = np.asarray(data["storage_units_by_zone"], dtype=float)
    
    return plan, storage_units


def _apply_price_modulation(
    price: np.ndarray,
    daily_amp: float,
) -> np.ndarray:
    """Apply daily sinusoidal price modulation."""
    if daily_amp == 0.0:
        return price
    n = price.size
    t = np.arange(n, dtype=float)
    mod = 1.0 + daily_amp * np.sin(2 * np.pi * t / 24.0)
    mod = np.maximum(mod, 0.0)
    return price * mod


def build_and_solve_with_linepack(
    n_days: int,
    time_limit: float,
    mip_gap: float,
    threads: int,
    storage_mutex: bool,
    storage_daily_balance: bool,
    data_dir: Path,
    renewable_dir: Path,
    fixed_plan18: np.ndarray,
    fixed_storage_units: Optional[np.ndarray],
    linepack_cap_mwh: float,
    linepack_loss_rate: float,
    daily_price_amp: float,
    objective_mode: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Build and solve MILP with linepack storage."""
    if not HAS_GUROBI:
        raise RuntimeError("缺少 gurobipy")

    run_dir = output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    options = DispatchOptions()
    options.data_dir = str(data_dir)
    options.renewable_dir = str(renewable_dir)
    data = load_simulation_data(options)
    grid = load_source_constraints(load_topology(options), options)
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

    # Apply price modulation for sensitivity analysis
    prices_g = _apply_price_modulation(prices_g, daily_price_amp)

    score_path = _project_root / "matlab" / "configs" / "oj_score.yaml"
    score_spec = _load_score_spec(score_path, data_dir=data_dir)

    gas_to_MWh = score_spec["gas_to_MWh"]
    carbon_price = score_spec["carbon_price"]
    aligned_e = prices_e + carbon_price * carbon_e
    aligned_g = prices_g + carbon_price * (carbon_g / gas_to_MWh)

    model = gp.Model("milp_with_linepack")
    model.Params.OutputFlag = 1
    if time_limit > 0:
        model.Params.TimeLimit = time_limit
    if mip_gap > 0:
        model.Params.MIPGap = mip_gap
    if threads > 0:
        model.Params.Threads = threads

    base_caps = np.array(PLAN18_BASE_CAPACITY, dtype=float)
    plan18 = np.round(fixed_plan18).astype(float)

    n_thermal = 3
    n_storage_types = 3
    n_zones = 3
    n_nodes = grid.n_nodes
    n_lines = grid.n_lines

    # Decision variables
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

    # ==== LINEPACK STORAGE VARIABLES ====
    # S_lp[t]: linepack SOC at beginning of hour t (MWh)
    S_lp = model.addMVar(n_hours + 1, lb=0.0, ub=linepack_cap_mwh, name="S_lp")
    # G_demand[t]: gas consumed by devices (燃气锅炉 + CCHP + GT - P2G产气)
    G_demand = model.addMVar(n_hours, lb=0.0, name="G_demand")

    # Device capacities (fixed from plan18)
    cap_e_boiler = plan18[3] * base_caps[3]
    cap_chiller_a = plan18[4] * base_caps[4]
    cap_chiller_b = plan18[5] * base_caps[5]
    cap_absorption = plan18[6] * base_caps[6]
    cap_gas_boiler = plan18[7] * base_caps[7]
    cap_hp_a = plan18[8] * base_caps[8]
    cap_hp_b = plan18[9] * base_caps[9]
    
    # Storage allocation from summary or default
    if fixed_storage_units is not None:
        storage_units = np.maximum(fixed_storage_units, 0.0)  # Ensure non-negative
    else:
        storage_units = np.array([
            [0.0, 10.0, 7.0],
            [0.0, 9.0, 0.0],
            [0.0, 3.0, 15.0],
        ], dtype=float)
    print(f"Storage units by zone:\n{storage_units}")
    
    storage_base_caps = base_caps[10:13]
    cap_wind = plan18[13]
    cap_pv = plan18[14]
    cap_p2g = plan18[15] * base_caps[15]
    cap_gt = plan18[16] * base_caps[16]
    cap_cchp = plan18[17] * base_caps[17]

    storage_power_ratio = np.array(options.storage_power_ratio, dtype=float)
    storage_eta_ch = np.array(options.storage_eta_ch, dtype=float)
    storage_eta_dis = np.array(options.storage_eta_dis, dtype=float)

    # Build constraints for each hour
    for t in range(n_hours):
        L_ele_s, L_ele_f, L_ele_t = loads[t, 0], loads[t, 1], loads[t, 2]
        L_heat_s, L_heat_f, L_heat_t = loads[t, 3], loads[t, 4], loads[t, 5]
        L_cool_s, L_cool_f, L_cool_t = loads[t, 6], loads[t, 7], loads[t, 8]

        storage_e, storage_h, storage_c = 0, 1, 2

        # Thermal plant bounds
        for k in range(n_thermal):
            model.addConstr(P_thermal[t, k] >= grid.thermal_min[k])
            model.addConstr(P_thermal[t, k] <= grid.thermal_max[k])

        # Device capacity bounds
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
        model.addConstr(G_buy[t] >= grid.gas_min)
        model.addConstr(G_buy[t] <= grid.gas_max)

        # Storage bounds
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

        # Line flow bounds
        for l in range(n_lines):
            model.addConstr(F_ij[t, l] <= grid.F_max[l])
            model.addConstr(F_ij[t, l] >= -grid.F_max[l])
        model.addConstr(theta_i[t, :] >= grid.theta_min)
        model.addConstr(theta_i[t, :] <= grid.theta_max)

        if not grid.zone_elec_export[1]:
            model.addConstr(P_grid_f[t] >= 0)

        # Energy balance constraints
        # Zone S (学生区)
        model.addConstr(
            P_thermal[t, 0] + P_pv[t] + P_wind[t]
            - P_e_boiler[t] - P_hp_a[t] - P_hp_b[t] - P_chiller_a[t] - P_p2g[t]
            + P_storage_dis[t, 0, storage_e] - P_storage_ch[t, 0, storage_e]
            + L_shed[t, 0] == L_ele_s
        )
        # Zone F (教工区)
        model.addConstr(
            P_thermal[t, 1] + P_gt[t] * 0.35 + P_cchp[t] * 0.3
            - P_chiller_b[t]
            + P_storage_dis[t, 1, storage_e] - P_storage_ch[t, 1, storage_e]
            - P_grid_f[t]
            + L_shed[t, 3] == L_ele_f
        )
        # Zone T (教学办公区)
        model.addConstr(
            P_thermal[t, 2] + P_grid_f[t]
            + P_storage_dis[t, 2, storage_e] - P_storage_ch[t, 2, storage_e]
            + L_shed[t, 6] == L_ele_t
        )

        # Heat balance
        model.addConstr(
            P_e_boiler[t] * 0.98 + P_hp_a[t] * 3 + P_hp_b[t] * 3
            + P_storage_dis[t, 0, storage_h] - P_storage_ch[t, 0, storage_h]
            + H_transfer_s[t]
            + L_shed[t, 2] == L_heat_s
        )
        model.addConstr(
            P_storage_dis[t, 1, storage_h] - P_storage_ch[t, 1, storage_h]
            + H_transfer_f[t]
            + L_shed[t, 5] == L_heat_f
        )
        model.addConstr(
            P_gas_boiler[t] * 0.95 + P_cchp[t] * 0.3
            - P_absorption[t]
            - H_transfer_s[t] - H_transfer_f[t]
            + P_storage_dis[t, 2, storage_h] - P_storage_ch[t, 2, storage_h]
            + L_shed[t, 7] == L_heat_t
        )

        # Cooling balance
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

        # ==== GAS DEMAND DEFINITION ====
        # G_demand = 燃气锅炉 + CCHP + GT 的气体消耗
        model.addConstr(
            G_demand[t] == P_gas_boiler[t] + P_cchp[t] + P_gt[t] - P_p2g[t] * 0.4
        )

        # ==== LINEPACK STATE EQUATION ====
        # S_lp[t+1] = (1 - λ) * S_lp[t] + G_buy[t] - G_demand[t]
        model.addConstr(
            S_lp[t + 1] == (1 - linepack_loss_rate) * S_lp[t] + G_buy[t] - G_demand[t]
        )

        # Heat capacity constraint
        if grid.heat_capacity is not None and np.isfinite(grid.heat_capacity):
            model.addConstr(H_transfer_s[t] + H_transfer_f[t] <= grid.heat_capacity)
        if not grid.zone_heat_export[2]:
            model.addConstr(H_transfer_s[t] == 0)
            model.addConstr(H_transfer_f[t] == 0)

        # DC power flow
        for l in range(n_lines):
            i = grid.line_from[l] - 1
            j = grid.line_to[l] - 1
            model.addConstr(F_ij[t, l] - grid.B_ij[l] * (theta_i[t, i] - theta_i[t, j]) == 0)

        ref_bus = 2 if n_nodes >= 3 else 0
        model.addConstr(theta_i[t, ref_bus] == 0)

        # Storage SOC dynamics
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

    # ==== LINEPACK CYCLIC BOUNDARY ====
    model.addConstr(S_lp[n_hours] == S_lp[0])

    # Storage daily balance
    if storage_daily_balance:
        for day in range(n_days):
            for z in range(n_zones):
                for s in range(n_storage_types):
                    cap_E = storage_units[z, s] * storage_base_caps[s]
                    model.addConstr(SOC[day, 24, z, s] >= cap_E * options.storage_soc_min)
                    model.addConstr(SOC[day, 24, z, s] <= cap_E * options.storage_soc_max)
                    model.addConstr(SOC[day, 0, z, s] == SOC[day, 24, z, s])

    # ==== OBJECTIVE FUNCTION ====
    device_catalog = _load_device_catalog(_project_root / "spec" / "device_catalog.yaml")
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
            capex_base[i] = 1.0
    capex_coeff = annuity * unit_cost * capex_base
    C_CAP = sum(capex_coeff[i] * plan18[i] for i in range(18))

    shed_penalty = score_spec["shed_penalty"]
    C_OP_ele = gp.quicksum(
        prices_e[t] * gp.quicksum(P_thermal[t, k] for k in range(n_thermal))
        for t in range(n_hours)
    )
    C_OP_gas = gp.quicksum(prices_g[t] * G_buy[t] for t in range(n_hours))
    C_OP_pen = shed_penalty * gp.quicksum(L_shed[t, i] for t in range(n_hours) for i in range(9))
    C_OP = C_OP_ele + C_OP_gas + C_OP_pen

    C_OP_ele_aligned = gp.quicksum(
        aligned_e[t] * gp.quicksum(P_thermal[t, k] for k in range(n_thermal))
        for t in range(n_hours)
    )
    C_OP_gas_aligned = gp.quicksum(aligned_g[t] * G_buy[t] for t in range(n_hours))
    C_OP_aligned = C_OP_ele_aligned + C_OP_gas_aligned + C_OP_pen

    gas_factor = score_spec["gas_emission_factor"]
    carbon_threshold = score_spec["carbon_threshold"]

    E_elec = gp.quicksum(
        carbon_e[t] * gp.quicksum(P_thermal[t, k] for k in range(n_thermal))
        for t in range(n_hours)
    )
    E_gas = gp.quicksum(G_buy[t] / gas_to_MWh * gas_factor for t in range(n_hours))
    E_total = E_elec + E_gas

    E_excess = model.addVar(lb=0.0, name="E_excess")
    model.addConstr(E_excess >= E_total - carbon_threshold)
    C_Carbon = carbon_price * E_excess

    if objective_mode == "aligned":
        model.setObjective(C_CAP + C_OP_aligned, GRB.MINIMIZE)
    else:
        model.setObjective(C_CAP + C_OP + C_Carbon, GRB.MINIMIZE)

    # Solve
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
        "linepack_cap_mwh": linepack_cap_mwh,
        "linepack_loss_rate": linepack_loss_rate,
        "daily_price_amp": daily_price_amp,
        "objective_mode": objective_mode,
    }

    if hasattr(model, "ObjVal") and status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        summary["objective"] = float(model.ObjVal)
        summary["C_CAP"] = float(C_CAP)
        summary["C_OP_ele"] = float(C_OP_ele.getValue())
        summary["C_OP_gas"] = float(C_OP_gas.getValue())
        summary["C_OP_pen"] = float(C_OP_pen.getValue())
        summary["C_OP"] = float(C_OP.getValue())
        summary["C_OP_aligned"] = float(C_OP_aligned.getValue())
        summary["C_Carbon"] = float(C_Carbon.getValue())
        summary["E_total"] = float(E_total.getValue())
        summary["G_buy_total"] = float(sum(G_buy[t].X for t in range(n_hours)))

        # Linepack statistics
        S_lp_vals = np.array([S_lp[t].X for t in range(n_hours + 1)])
        summary["S_lp_mean"] = float(np.mean(S_lp_vals))
        summary["S_lp_max"] = float(np.max(S_lp_vals))
        summary["S_lp_min"] = float(np.min(S_lp_vals))

    if hasattr(model, "ObjBound") and status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        summary["objective_bound"] = float(model.ObjBound)
    if hasattr(model, "MIPGap") and status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        summary["mip_gap"] = float(model.MIPGap)

    summary["plan18"] = plan18.tolist()
    summary["timestamp"] = datetime.now().isoformat()

    # Save summary
    output_path = run_dir / "linepack_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Summary saved to: {output_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="思考题3：管道线包完整MILP分析")
    parser.add_argument("--days", type=int, default=365, help="仿真天数")
    parser.add_argument("--time-limit", type=float, default=0.0, help="求解时限(秒)")
    parser.add_argument("--mip-gap", type=float, default=1e-4, help="MIP Gap容差")
    parser.add_argument("--threads", type=int, default=0, help="Gurobi线程数")
    parser.add_argument("--storage-mutex", dest="storage_mutex", action="store_true")
    parser.add_argument("--no-storage-mutex", dest="storage_mutex", action="store_false")
    parser.add_argument("--storage-daily-balance", dest="storage_daily_balance", action="store_true")
    parser.add_argument("--storage-cross-day", dest="storage_daily_balance", action="store_false")
    parser.add_argument("--data-dir", default="data/raw", help="数据目录")
    parser.add_argument("--renewable-dir", default=None, help="可再生能源数据目录")
    parser.add_argument("--plan18-summary", required=True, help="Plan18 summary.json路径")
    parser.add_argument("--linepack-cap-hours", type=float, default=0.0,
                        help="线包容量(相对于平均气体需求的小时数)")
    parser.add_argument("--linepack-cap-mwh", type=float, default=None,
                        help="线包容量(MWh直接指定)")
    parser.add_argument("--linepack-loss-rate", type=float, default=0.0005,
                        help="线包小时损耗率")
    parser.add_argument("--daily-price-amp", type=float, default=0.0,
                        help="气价日内波动振幅(0.3=30%)")
    parser.add_argument("--objective-mode", default="aligned", choices=("aligned", "score"),
                        help="优化目标：aligned(与主问题一致) 或 score(含碳阈值惩罚)")
    parser.add_argument("--output-dir", default="runs/thought_questions/q3_milp",
                        help="输出目录")
    parser.set_defaults(storage_mutex=True, storage_daily_balance=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    renewable_dir = Path(args.renewable_dir) if args.renewable_dir else data_dir

    # Load plan18 and storage units
    plan18, storage_units = _load_plan18_from_summary(args.plan18_summary)

    # Compute linepack capacity
    if args.linepack_cap_mwh is not None:
        linepack_cap_mwh = args.linepack_cap_mwh
    else:
        # Estimate average demand from previous run
        avg_demand = 50.73  # MWh/h from baseline MILP
        linepack_cap_mwh = avg_demand * args.linepack_cap_hours

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cap_str = f"cap{args.linepack_cap_hours:.0f}h" if args.linepack_cap_mwh is None else f"cap{args.linepack_cap_mwh:.0f}mwh"
    amp_str = f"amp{int(args.daily_price_amp*100)}"
    run_dir = Path(args.output_dir) / f"q3_{cap_str}_{amp_str}_{timestamp}"

    print(f"Linepack capacity: {linepack_cap_mwh:.2f} MWh")
    print(f"Daily price amplitude: {args.daily_price_amp*100:.1f}%")
    print(f"Output directory: {run_dir}")

    build_and_solve_with_linepack(
        n_days=args.days,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        storage_mutex=args.storage_mutex,
        storage_daily_balance=args.storage_daily_balance,
        data_dir=data_dir,
        renewable_dir=renewable_dir,
        fixed_plan18=plan18,
        fixed_storage_units=storage_units,
        linepack_cap_mwh=linepack_cap_mwh,
        linepack_loss_rate=args.linepack_loss_rate,
        daily_price_amp=args.daily_price_amp,
        objective_mode=args.objective_mode,
        output_dir=run_dir,
    )


if __name__ == "__main__":
    main()
