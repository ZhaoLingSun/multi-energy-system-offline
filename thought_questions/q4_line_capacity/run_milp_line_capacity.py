#!/usr/bin/env python3
"""
思考题4：传输容量约束敏感性分析 - 完整MILP模型

在固定plan18规划方案下，扫描不同电力线路容量与热网传输容量，
分析传输约束对系统运行成本的影响。

用法:
    python run_milp_line_capacity.py --plan18-summary PATH --line-caps "10,20,50,100,200" --heat-caps "10,20,50,100" --mip-gap 1e-4
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
from meos.ga.codec import PLAN18_BASE_CAPACITY
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


def _parse_cap_list(text: str) -> List[float]:
    """Parse comma-separated capacity list."""
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def build_and_solve_with_capacity(
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
    line_capacity: float,
    heat_capacity: float,
    objective_mode: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Build and solve MILP with specified line/heat capacity."""
    if not HAS_GUROBI:
        raise RuntimeError("缺少 gurobipy")

    run_dir = output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    options = DispatchOptions()
    options.data_dir = str(data_dir)
    options.renewable_dir = str(renewable_dir)
    options.line_capacity = line_capacity
    options.heat_transfer_cap = heat_capacity
    
    data = load_simulation_data(options)
    grid = load_source_constraints(load_topology(options), options)
    parsed = _parse_meos_inputs(data_dir, renewable_dir)

    # Override grid capacities
    grid.F_max = np.full(grid.n_lines, line_capacity, dtype=float)
    grid.heat_capacity = heat_capacity

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

    score_path = _project_root / "matlab" / "configs" / "oj_score.yaml"
    score_spec = _load_score_spec(score_path, data_dir=data_dir)

    gas_to_MWh = score_spec["gas_to_MWh"]
    carbon_price = score_spec["carbon_price"]
    aligned_e = prices_e + carbon_price * carbon_e
    aligned_g = prices_g + carbon_price * (carbon_g / gas_to_MWh)

    model = gp.Model("milp_line_capacity")
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
        storage_units = np.maximum(fixed_storage_units, 0.0)
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

    print(f"Building model with {n_hours} hours, line_cap={line_capacity}, heat_cap={heat_capacity}")

    # Build constraints for each hour
    for t in range(n_hours):
        L_ele_s, L_ele_f, L_ele_t = loads[t, 0], loads[t, 3], loads[t, 6]
        L_heat_s, L_heat_f, L_heat_t = loads[t, 2], loads[t, 5], loads[t, 7]
        L_cool_s, L_cool_f, L_cool_t = loads[t, 1], loads[t, 4], loads[t, 8]

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

        # Line flow bounds - KEY CONSTRAINT FOR THIS STUDY
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

        # Gas balance
        model.addConstr(
            G_buy[t] + P_p2g[t] * 0.4 - P_gas_boiler[t] - P_cchp[t] - P_gt[t] == 0
        )

        # Heat capacity constraint - KEY CONSTRAINT FOR THIS STUDY
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

    # Storage daily balance
    if storage_daily_balance:
        for day in range(n_days):
            for z in range(n_zones):
                for s in range(n_storage_types):
                    cap_E = storage_units[z, s] * storage_base_caps[s]
                    model.addConstr(SOC[day, 24, z, s] >= cap_E * options.storage_soc_min)
                    model.addConstr(SOC[day, 24, z, s] <= cap_E * options.storage_soc_max)
                    model.addConstr(SOC[day, 0, z, s] == SOC[day, 24, z, s])

    # Continue to objective function...

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
        "line_capacity": line_capacity,
        "heat_capacity": heat_capacity,
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
        
        # Shed statistics
        shed_vals = np.array([[L_shed[t, i].X for i in range(9)] for t in range(n_hours)])
        summary["shed_total"] = float(np.sum(shed_vals))
        summary["shed_by_type"] = [float(np.sum(shed_vals[:, i])) for i in range(9)]
        
        # Line flow statistics
        F_vals = np.array([[F_ij[t, l].X for l in range(n_lines)] for t in range(n_hours)])
        summary["F_max_actual"] = float(np.max(np.abs(F_vals)))
        summary["F_mean_actual"] = float(np.mean(np.abs(F_vals)))
        
        # Congestion statistics (拥塞分析)
        # 达到容量上限95%以上的小时数
        congestion_threshold = 0.95 * line_capacity
        congested_hours = np.sum(np.abs(F_vals) >= congestion_threshold, axis=0)
        summary["congested_hours_by_line"] = congested_hours.tolist()
        summary["congested_hours_total"] = int(np.sum(congested_hours))
        summary["congestion_rate"] = float(np.sum(congested_hours) / (n_hours * n_lines))
        
        # 线路利用率
        utilization = np.abs(F_vals) / line_capacity if line_capacity > 0 else np.zeros_like(F_vals)
        summary["line_utilization_max"] = float(np.max(utilization))
        summary["line_utilization_mean"] = float(np.mean(utilization))
        
        # Heat transfer statistics
        H_s_vals = np.array([H_transfer_s[t].X for t in range(n_hours)])
        H_f_vals = np.array([H_transfer_f[t].X for t in range(n_hours)])
        summary["H_transfer_max"] = float(np.max(H_s_vals + H_f_vals))
        summary["H_transfer_mean"] = float(np.mean(H_s_vals + H_f_vals))

    if hasattr(model, "ObjBound") and status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        summary["objective_bound"] = float(model.ObjBound)
    if hasattr(model, "MIPGap") and status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        summary["mip_gap"] = float(model.MIPGap)

    summary["plan18"] = plan18.tolist()
    summary["timestamp"] = datetime.now().isoformat()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="思考题4：传输容量约束完整MILP分析")
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
    parser.add_argument("--line-caps", default="5,10,20,50,100,200,400",
                        help="电力线路容量列表(MW)")
    parser.add_argument("--heat-caps", default="5,10,20,50,100,200",
                        help="热网传输容量列表(MW)")
    parser.add_argument("--base-line-cap", type=float, default=100.0, help="基准电力容量(MW)")
    parser.add_argument("--base-heat-cap", type=float, default=50.0, help="基准热网容量(MW)")
    parser.add_argument("--capex-line", type=float, default=50000.0, help="电力扩容成本(元/MW)")
    parser.add_argument("--capex-heat", type=float, default=30000.0, help="热网扩容成本(元/MW)")
    parser.add_argument("--objective-mode", default="aligned", choices=("aligned", "score"),
                        help="优化目标：aligned(与主问题一致) 或 score(含碳阈值惩罚)")
    parser.add_argument("--output-dir", default="runs/thought_questions",
                        help="输出目录")
    parser.set_defaults(storage_mutex=True, storage_daily_balance=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    renewable_dir = Path(args.renewable_dir) if args.renewable_dir else data_dir

    # Load plan18 and storage units
    plan18, storage_units = _load_plan18_from_summary(args.plan18_summary)

    # Parse capacity lists
    line_caps = _parse_cap_list(args.line_caps)
    heat_caps = _parse_cap_list(args.heat_caps)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"q4_milp_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_runs = len(line_caps) * len(heat_caps)
    run_idx = 0

    for line_cap in line_caps:
        for heat_cap in heat_caps:
            run_idx += 1
            print(f"\n{'='*60}")
            print(f"Run {run_idx}/{total_runs}: line_cap={line_cap} MW, heat_cap={heat_cap} MW")
            print(f"{'='*60}")

            case_dir = run_dir / f"line{int(line_cap)}_heat{int(heat_cap)}"
            
            try:
                result = build_and_solve_with_capacity(
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
                    line_capacity=line_cap,
                    heat_capacity=heat_cap,
                    objective_mode=args.objective_mode,
                    output_dir=case_dir,
                )
                
                # Add expansion cost analysis
                line_expansion = max(0.0, line_cap - args.base_line_cap)
                heat_expansion = max(0.0, heat_cap - args.base_heat_cap)
                expansion_cost = line_expansion * args.capex_line + heat_expansion * args.capex_heat
                
                result["line_expansion"] = line_expansion
                result["heat_expansion"] = heat_expansion
                result["expansion_cost"] = expansion_cost
                if "objective" in result:
                    result["total_with_expansion"] = result["objective"] + expansion_cost
                
                results.append(result)
                
                # Save individual result
                (case_dir / "result.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
                
            except Exception as e:
                print(f"Error in run: {e}")
                results.append({
                    "line_capacity": line_cap,
                    "heat_capacity": heat_cap,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })

    # Save summary
    summary = {
        "line_caps": line_caps,
        "heat_caps": heat_caps,
        "objective_mode": args.objective_mode,
        "base_line_cap": args.base_line_cap,
        "base_heat_cap": args.base_heat_cap,
        "capex_line": args.capex_line,
        "capex_heat": args.capex_heat,
        "n_days": args.days,
        "mip_gap": args.mip_gap,
        "plan18_summary": args.plan18_summary,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"Summary saved to: {summary_path}")
    print(f"Total runs: {len(results)}")


if __name__ == "__main__":
    main()
