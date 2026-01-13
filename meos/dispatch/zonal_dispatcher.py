"""
分区域日调度优化。

目标：复刻平台分区约束/潮流约束的日内 LP 求解流程，
用于生成与平台一致的 dispatch_data（购电/购气/切负荷等）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import re

import numpy as np
from scipy.optimize import linprog

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

from meos.export.platform_full_exporter import _parse_meos_inputs


_WARNED: set[str] = set()


@dataclass
class DispatchOptions:
    guidance_price_factor: float = 1.0
    guidance_gas_factor: float = 1.0
    guidance_price_matrix: Optional[np.ndarray] = None
    guidance_gas_matrix: Optional[np.ndarray] = None
    day_indices: Optional[Sequence[int]] = None
    verbose: bool = False
    shed_penalty: float = 500000.0
    gas_to_MWh: float = 0.01
    heat_transfer_cap: float = float("inf")
    renewable_scale: float = 1.0
    renewable_scale_pv: Optional[float] = None
    renewable_scale_wind: Optional[float] = None
    topology_path: Optional[str] = None
    line_capacity: Any = 400.0
    line_reactance: float = 0.000281
    theta_min: float = -3.14
    theta_max: float = 3.15
    gas_pressure_min: float = 5.0
    gas_pressure_max: float = 100.0
    gas_pressure_base: float = 10.0
    attributes_path: Optional[str] = None
    storage_soc_min: float = 0.0
    storage_soc_max: float = 1.0
    storage_daily_balance: bool = True
    storage_eta_ch: Sequence[float] = (0.92, 0.95, 0.95)
    storage_eta_dis: Sequence[float] = (0.92, 0.95, 0.95)
    storage_power_ratio: Sequence[float] = (0.5, 0.2, 0.2)
    solver: str = "linprog"
    thermal_order: Sequence[int] = (2, 1, 3)
    thermal_bias: Sequence[float] = (0.0, 0.0, 0.0)
    gurobi_threads: int = 0
    gurobi_mip_gap: float = 0.0
    storage_mutual_exclusive: bool = False
    optimize_price: bool = False
    price_multipliers: Optional[Sequence[float]] = None
    price_penalty: float = 0.0
    load_scale_electric: float = 1.0
    load_scale_heat: float = 1.0
    load_scale_cool: float = 1.0
    data_dir: Optional[str] = None
    renewable_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.renewable_scale_pv is None:
            self.renewable_scale_pv = self.renewable_scale
        if self.renewable_scale_wind is None:
            self.renewable_scale_wind = self.renewable_scale
        if self.price_multipliers is None:
            self.price_multipliers = (0.8, 1.0, 1.2)


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_topology_path(options: DispatchOptions) -> Path:
    if options.topology_path:
        return Path(options.topology_path)
    return _resolve_project_root() / "example" / "topology.json"


def _resolve_attributes_path(options: DispatchOptions) -> Path:
    if options.attributes_path:
        return Path(options.attributes_path)
    return _resolve_project_root() / "example" / "attributes.json"


def _resolve_data_dir(options: DispatchOptions) -> Path:
    if options.data_dir:
        return Path(options.data_dir)
    return _resolve_project_root() / "data" / "raw"


def _resolve_renewable_dir(options: DispatchOptions, data_dir: Path) -> Path:
    if options.renewable_dir:
        return Path(options.renewable_dir)
    return data_dir


_ELEC_STORAGE_SHARE = np.array([1.0, 1.0, 1.0], dtype=float)
_HEAT_STORAGE_SHARE = np.array([10.0, 4.0, 10.0], dtype=float)
_COLD_STORAGE_SHARE = np.array([7.0, 24.0], dtype=float)


def _split_capacity(total: float, shares: np.ndarray) -> np.ndarray:
    shares = np.asarray(shares, dtype=float)
    if total <= 0 or shares.size == 0:
        return np.zeros_like(shares)
    denom = float(np.sum(shares))
    if denom <= 0:
        return np.zeros_like(shares)
    return total * shares / denom


def _pad_params(values: Sequence[float], size: int, default: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.full(size, default, dtype=float)
    if arr.size == 1:
        return np.full(size, float(arr[0]), dtype=float)
    if arr.size < size:
        return np.pad(arr, (0, size - arr.size), mode="edge")
    return arr[:size]


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(message)


def _create_gurobi_env(options: DispatchOptions) -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.setParam("OutputFlag", 0)
    env.start()
    return env


def _solve_lp_gurobi(
    f: np.ndarray,
    A_ub: Optional[np.ndarray],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[np.ndarray],
    b_eq: Optional[np.ndarray],
    lb: np.ndarray,
    ub: np.ndarray,
    options: DispatchOptions,
) -> SimpleNamespace:
    n_vars = f.shape[0]
    env = _create_gurobi_env(options)
    model = None
    try:
        model = gp.Model(env=env)
        model.Params.OutputFlag = 0
        if options.gurobi_threads and options.gurobi_threads > 0:
            model.Params.Threads = int(options.gurobi_threads)
        if options.gurobi_mip_gap and options.gurobi_mip_gap > 0:
            model.Params.MIPGap = float(options.gurobi_mip_gap)
        if options.gurobi_mip_gap and options.gurobi_mip_gap > 0:
            model.Params.MIPGap = float(options.gurobi_mip_gap)

        lb_g = np.where(np.isneginf(lb), -GRB.INFINITY, lb)
        ub_g = np.where(np.isposinf(ub), GRB.INFINITY, ub)
        x = model.addMVar(shape=n_vars, lb=lb_g, ub=ub_g, name="x")

        if A_eq is not None and b_eq is not None:
            model.addMConstr(A_eq, x, "=", b_eq)
        if A_ub is not None and b_ub is not None:
            model.addMConstr(A_ub, x, "<", b_ub)

        model.setObjective(f @ x, GRB.MINIMIZE)
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            return SimpleNamespace(success=False, message=f"status={model.Status}")
        return SimpleNamespace(success=True, x=x.X, message="optimal")
    finally:
        if model is not None:
            model.dispose()
        env.dispose()


def _solve_milp_gurobi(
    f: np.ndarray,
    A_ub: Optional[np.ndarray],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[np.ndarray],
    b_eq: Optional[np.ndarray],
    lb: np.ndarray,
    ub: np.ndarray,
    options: DispatchOptions,
    storage_ch_idx: np.ndarray,
    storage_dis_idx: np.ndarray,
    storage_cap: np.ndarray,
) -> SimpleNamespace:
    n_vars = f.shape[0]
    env = _create_gurobi_env(options)
    model = None
    try:
        model = gp.Model(env=env)
        model.Params.OutputFlag = 0
        if options.gurobi_threads and options.gurobi_threads > 0:
            model.Params.Threads = int(options.gurobi_threads)
        if options.gurobi_mip_gap and options.gurobi_mip_gap > 0:
            model.Params.MIPGap = float(options.gurobi_mip_gap)

        lb_g = np.where(np.isneginf(lb), -GRB.INFINITY, lb)
        ub_g = np.where(np.isposinf(ub), GRB.INFINITY, ub)
        x = model.addMVar(shape=n_vars, lb=lb_g, ub=ub_g, name="x")

        if A_eq is not None and b_eq is not None:
            model.addMConstr(A_eq, x, "=", b_eq)
        if A_ub is not None and b_ub is not None:
            model.addMConstr(A_ub, x, "<", b_ub)

        n_hours, n_storage = storage_ch_idx.shape
        y_ch = model.addMVar(shape=(n_hours, n_storage), vtype=GRB.BINARY, name="y_ch")
        for t in range(n_hours):
            for s in range(n_storage):
                cap = float(storage_cap[s])
                if cap <= 0:
                    continue
                model.addConstr(x[storage_ch_idx[t, s]] <= cap * y_ch[t, s])
                model.addConstr(x[storage_dis_idx[t, s]] <= cap * (1 - y_ch[t, s]))

        model.setObjective(f @ x, GRB.MINIMIZE)
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            return SimpleNamespace(success=False, message=f"status={model.Status}")
        return SimpleNamespace(success=True, x=x.X, message="optimal")
    finally:
        if model is not None:
            model.dispose()
        env.dispose()


def _solve_milp_gurobi_with_price(
    f: np.ndarray,
    A_ub: Optional[np.ndarray],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[np.ndarray],
    b_eq: Optional[np.ndarray],
    lb: np.ndarray,
    ub: np.ndarray,
    options: DispatchOptions,
    storage_ch_idx: np.ndarray,
    storage_dis_idx: np.ndarray,
    storage_cap: np.ndarray,
    thermal_idx: np.ndarray,
    price_levels: np.ndarray,
    price_penalty_levels: Optional[np.ndarray],
    thermal_max_sum: float,
) -> SimpleNamespace:
    n_vars = f.shape[0]
    env = _create_gurobi_env(options)
    model = None
    try:
        model = gp.Model(env=env)
        model.Params.OutputFlag = 0
        if options.gurobi_threads and options.gurobi_threads > 0:
            model.Params.Threads = int(options.gurobi_threads)

        lb_g = np.where(np.isneginf(lb), -GRB.INFINITY, lb)
        ub_g = np.where(np.isposinf(ub), GRB.INFINITY, ub)
        x = model.addMVar(shape=n_vars, lb=lb_g, ub=ub_g, name="x")

        if A_eq is not None and b_eq is not None:
            model.addMConstr(A_eq, x, "=", b_eq)
        if A_ub is not None and b_ub is not None:
            model.addMConstr(A_ub, x, "<", b_ub)

        if options.storage_mutual_exclusive:
            n_hours, n_storage = storage_ch_idx.shape
            y_ch = model.addMVar(shape=(n_hours, n_storage), vtype=GRB.BINARY, name="y_ch")
            for t in range(n_hours):
                for s in range(n_storage):
                    cap = float(storage_cap[s])
                    if cap <= 0:
                        continue
                    model.addConstr(x[storage_ch_idx[t, s]] <= cap * y_ch[t, s])
                    model.addConstr(x[storage_dis_idx[t, s]] <= cap * (1 - y_ch[t, s]))

        n_hours, n_levels = price_levels.shape
        p_level = model.addMVar(shape=(n_hours, n_levels), lb=0.0, name="p_level")
        y_price = model.addMVar(shape=(n_hours, n_levels), vtype=GRB.BINARY, name="y_price")

        for t in range(n_hours):
            total_thermal = gp.quicksum(x[int(thermal_idx[t, k])] for k in range(thermal_idx.shape[1]))
            model.addConstr(gp.quicksum(p_level[t, k] for k in range(n_levels)) == total_thermal)
            model.addConstr(gp.quicksum(y_price[t, k] for k in range(n_levels)) == 1)
            for k in range(n_levels):
                model.addConstr(p_level[t, k] <= thermal_max_sum * y_price[t, k])

        obj = f @ x
        obj += gp.quicksum(
            float(price_levels[t, k]) * p_level[t, k]
            for t in range(n_hours)
            for k in range(n_levels)
        )
        if price_penalty_levels is not None:
            obj += gp.quicksum(
                float(price_penalty_levels[t, k]) * y_price[t, k]
                for t in range(n_hours)
                for k in range(n_levels)
            )

        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            return SimpleNamespace(success=False, message=f"status={model.Status}")

        return SimpleNamespace(
            success=True,
            x=x.X,
            message="optimal",
            extra={
                "y_price": y_price.X,
                "p_level": p_level.X,
                "price_levels": price_levels,
            },
        )
    finally:
        if model is not None:
            model.dispose()
        env.dispose()


def _solve_lp(
    f: np.ndarray,
    A_ub: Optional[np.ndarray],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[np.ndarray],
    b_eq: Optional[np.ndarray],
    lb: np.ndarray,
    ub: np.ndarray,
    options: DispatchOptions,
) -> SimpleNamespace:
    solver = str(options.solver or "linprog").lower()
    if solver == "gurobi":
        if HAS_GUROBI:
            try:
                result = _solve_lp_gurobi(f, A_ub, b_ub, A_eq, b_eq, lb, ub, options)
                if result.success:
                    return result
                _warn_once("gurobi_failed", f"Gurobi 求解失败，回退到 linprog: {result.message}")
            except Exception as exc:
                _warn_once("gurobi_failed", f"Gurobi 求解异常，回退到 linprog: {exc}")
        else:
            _warn_once("gurobi_unavailable", "Gurobi 不可用，回退到 linprog")

    result = linprog(
        f,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=list(zip(lb, ub)),
        method="highs",
    )
    return SimpleNamespace(success=result.success, x=result.x, message=result.message)


def _sanitize_json_text(raw: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", raw)


def _map_bus_id(node_id: object) -> Tuple[Optional[int], bool]:
    if not isinstance(node_id, (str, bytes)):
        return None, False
    text = node_id.decode() if isinstance(node_id, bytes) else node_id
    match = re.search(r"elec_bus_(\d+)", text)
    if not match:
        return None, False
    return int(match.group(1)), True


def _map_thermal_unit(node_id: object) -> Tuple[Optional[int], bool]:
    if not isinstance(node_id, (str, bytes)):
        return None, False
    text = node_id.decode() if isinstance(node_id, bytes) else node_id
    match = re.search(r"thermal_gen_(\d+)", text)
    if not match:
        return None, False
    return int(match.group(1)), True


def _select_topology_object(topo_data: Any) -> Optional[dict]:
    if topo_data is None:
        return None
    if isinstance(topo_data, dict):
        objs = [topo_data]
    elif isinstance(topo_data, list):
        objs = topo_data
    else:
        objs = [topo_data]
    for candidate in objs:
        links = candidate.get("links")
        if links is None:
            continue
        if isinstance(links, dict):
            link_list = list(links.values())
        elif isinstance(links, list):
            link_list = links
        else:
            link_list = [links]
        for link in link_list:
            if link.get("category") == "Transmission_Line":
                return candidate
    return None


def _extract_lines(system_obj: dict) -> Tuple[List[int], List[int]]:
    line_from: List[int] = []
    line_to: List[int] = []
    links = system_obj.get("links", [])
    if isinstance(links, dict):
        link_list = list(links.values())
    elif isinstance(links, list):
        link_list = links
    else:
        link_list = [links]
    for link in link_list:
        if link.get("category") != "Transmission_Line":
            continue
        if link.get("medium") and link.get("medium") != "Electricity":
            continue
        from_idx, ok1 = _map_bus_id(link.get("source"))
        to_idx, ok2 = _map_bus_id(link.get("target"))
        if ok1 and ok2:
            line_from.append(from_idx)
            line_to.append(to_idx)
    if line_from:
        pairs = []
        for f, t in zip(line_from, line_to):
            if f > t:
                f, t = t, f
            pairs.append((f, t))
        pairs.sort(key=lambda x: (x[0], x[1]))
        line_from = [p[0] for p in pairs]
        line_to = [p[1] for p in pairs]
    return line_from, line_to


def _extract_thermal_mapping(system_obj: dict) -> List[Optional[int]]:
    links = system_obj.get("links", [])
    if isinstance(links, dict):
        link_list = list(links.values())
    elif isinstance(links, list):
        link_list = links
    else:
        link_list = [links]
    temp_map: List[Optional[int]] = [None, None, None]
    for link in link_list:
        unit_idx, ok_unit = _map_thermal_unit(link.get("source"))
        bus_idx, ok_bus = _map_bus_id(link.get("target"))
        if ok_unit and ok_bus and 1 <= unit_idx <= 3:
            temp_map[unit_idx - 1] = bus_idx
    if any(v is not None for v in temp_map):
        return temp_map
    return []


def _extract_zone_ports(system_obj: dict, zone_ids: Sequence[str]) -> Tuple[List[Optional[int]], SimpleNamespace]:
    zone_to_bus: List[Optional[int]] = [None] * len(zone_ids)
    ports = SimpleNamespace(
        elec_import=[False] * len(zone_ids),
        elec_export=[False] * len(zone_ids),
        heat_import=[False] * len(zone_ids),
        heat_export=[False] * len(zone_ids),
        gas_import=[False] * len(zone_ids),
        gas_export=[False] * len(zone_ids),
    )
    nodes = system_obj.get("nodes", [])
    if isinstance(nodes, dict):
        node_list = list(nodes.values())
    elif isinstance(nodes, list):
        node_list = nodes
    else:
        node_list = [nodes]

    def map_zone_index(node_id: object, node_name: object) -> Optional[int]:
        if isinstance(node_id, (str, bytes)):
            node_text = node_id.decode() if isinstance(node_id, bytes) else node_id
            for idx, zone_id in enumerate(zone_ids, start=1):
                if node_text == zone_id:
                    return idx
            if "student" in node_text:
                return 1
            if "faculty" in node_text:
                return 2
            if "teaching" in node_text:
                return 3
        if isinstance(node_name, (str, bytes)):
            name_text = node_name.decode() if isinstance(node_name, bytes) else node_name
            if "学生" in name_text:
                return 1
            if "教工" in name_text:
                return 2
            if "教学" in name_text:
                return 3
        return None

    for node in node_list:
        zone_idx = map_zone_index(node.get("id"), node.get("name"))
        if zone_idx is None:
            continue
        node_ports = node.get("ports", [])
        if isinstance(node_ports, dict):
            port_list = list(node_ports.values())
        elif isinstance(node_ports, list):
            port_list = node_ports
        else:
            port_list = [node_ports]
        for port in port_list:
            medium = str(port.get("medium", "")).lower()
            ptype = str(port.get("type", "")).lower()
            is_input = ptype == "input"
            is_output = ptype == "output"
            idx = zone_idx - 1
            if medium == "electricity":
                if is_input:
                    ports.elec_import[idx] = True
                if is_output:
                    ports.elec_export[idx] = True
                bus_idx, ok_bus = _map_bus_id(port.get("node_ref"))
                if ok_bus:
                    zone_to_bus[idx] = bus_idx
            elif medium == "heat":
                if is_input:
                    ports.heat_import[idx] = True
                if is_output:
                    ports.heat_export[idx] = True
            elif medium == "gas":
                if is_input:
                    ports.gas_import[idx] = True
                if is_output:
                    ports.gas_export[idx] = True
    if all(v is None for v in zone_to_bus):
        zone_to_bus = []
    return zone_to_bus, ports


def _extract_network_params(system_obj: dict) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    network_params = system_obj.get("network_params", {})
    electric = network_params.get("electric", {})
    if "line_reactance" in electric:
        params["line_reactance"] = electric["line_reactance"]
    if "line_capacity" in electric:
        params["line_capacity"] = electric["line_capacity"]
    if "theta_min" in electric:
        params["theta_min"] = electric["theta_min"]
    if "theta_max" in electric:
        params["theta_max"] = electric["theta_max"]
    gas = network_params.get("gas", {})
    if "pressure_min_MPa" in gas:
        params["gas_pressure_min"] = gas["pressure_min_MPa"]
    if "pressure_max_MPa" in gas:
        params["gas_pressure_max"] = gas["pressure_max_MPa"]
    if "pressure_base_MPa" in gas:
        params["gas_pressure_base"] = gas["pressure_base_MPa"]
    heat = network_params.get("heat", {})
    if "capacity_MW" in heat:
        params["heat_capacity"] = heat["capacity_MW"]
    return params


def _build_incidence(n_nodes: int, line_from: Sequence[int], line_to: Sequence[int]) -> np.ndarray:
    n_lines = len(line_from)
    incidence = np.zeros((n_nodes, n_lines))
    for l in range(n_lines):
        incidence[line_from[l] - 1, l] = 1
        incidence[line_to[l] - 1, l] = -1
    return incidence


def _normalize_line_capacity(line_capacity: Any, n_lines: int) -> np.ndarray:
    if line_capacity is None or (isinstance(line_capacity, float) and np.isnan(line_capacity)):
        return np.full(n_lines, 100.0)
    if np.isscalar(line_capacity):
        return np.full(n_lines, float(line_capacity))
    values = np.asarray(line_capacity, dtype=float).reshape(-1)
    if values.size == n_lines:
        return values
    return np.full(n_lines, 100.0)


def load_topology(options: DispatchOptions) -> SimpleNamespace:
    grid = SimpleNamespace()
    grid.n_nodes = 3
    grid.zone_names = ["学生区", "教工区", "教学办公区"]
    grid.zone_ids = ["student_zone", "faculty_zone", "teaching_zone"]
    grid.zone_to_bus = [1, 2, 3]
    grid.zone_elec_import = [True, True, True]
    grid.zone_elec_export = [True, True, True]
    grid.zone_heat_import = [True, True, True]
    grid.zone_heat_export = [True, True, True]
    grid.zone_gas_import = [True, True, True]
    grid.zone_gas_export = [True, True, True]
    grid.thermal_unit_names = ["火电机组1", "火电机组2", "火电机组3"]
    grid.thermal_to_bus = [1, 3, 2]
    grid.line_from = [1, 2, 3]
    grid.line_to = [2, 3, 1]
    grid.topology_source = "fallback"

    line_reactance = options.line_reactance
    line_capacity = options.line_capacity
    theta_min = options.theta_min
    theta_max = options.theta_max
    gas_pressure_min = options.gas_pressure_min
    gas_pressure_max = options.gas_pressure_max
    gas_pressure_base = options.gas_pressure_base
    heat_capacity = None

    topo_path = _resolve_topology_path(options)
    if topo_path.exists():
        try:
            raw = topo_path.read_text(encoding="utf-8")
            topo_data = json.loads(_sanitize_json_text(raw))
            system_obj = _select_topology_object(topo_data)
            if system_obj:
                params = _extract_network_params(system_obj)
                if "line_reactance" in params:
                    line_reactance = params["line_reactance"]
                if "line_capacity" in params:
                    line_capacity = params["line_capacity"]
                if "theta_min" in params:
                    theta_min = params["theta_min"]
                if "theta_max" in params:
                    theta_max = params["theta_max"]
                if "gas_pressure_min" in params:
                    gas_pressure_min = params["gas_pressure_min"]
                if "gas_pressure_max" in params:
                    gas_pressure_max = params["gas_pressure_max"]
                if "gas_pressure_base" in params:
                    gas_pressure_base = params["gas_pressure_base"]
                if "heat_capacity" in params:
                    heat_capacity = params["heat_capacity"]

                line_from, line_to = _extract_lines(system_obj)
                if line_from:
                    grid.line_from = line_from
                    grid.line_to = line_to

                thermal_to_bus = _extract_thermal_mapping(system_obj)
                if thermal_to_bus:
                    for idx, value in enumerate(thermal_to_bus):
                        if value is None:
                            thermal_to_bus[idx] = grid.thermal_to_bus[idx]
                    grid.thermal_to_bus = thermal_to_bus

                zone_to_bus, zone_ports = _extract_zone_ports(system_obj, grid.zone_ids)
                if zone_to_bus:
                    for idx, value in enumerate(zone_to_bus):
                        if value is None:
                            zone_to_bus[idx] = grid.zone_to_bus[idx]
                    grid.zone_to_bus = zone_to_bus
                if zone_ports:
                    grid.zone_elec_import = zone_ports.elec_import
                    grid.zone_elec_export = zone_ports.elec_export
                    grid.zone_heat_import = zone_ports.heat_import
                    grid.zone_heat_export = zone_ports.heat_export
                    grid.zone_gas_import = zone_ports.gas_import
                    grid.zone_gas_export = zone_ports.gas_export
                grid.topology_source = str(topo_path)
        except Exception as exc:
            _warn_once("topology_parse_failed", f"拓扑解析失败，使用默认拓扑: {exc}")
    else:
        _warn_once("topology_missing", f"拓扑文件不存在，使用默认拓扑: {topo_path}")

    grid.n_lines = len(grid.line_from)
    grid.incidence = _build_incidence(grid.n_nodes, grid.line_from, grid.line_to)
    grid.line_reactance = line_reactance
    grid.B_ij = np.ones(grid.n_lines) / line_reactance
    grid.F_max = _normalize_line_capacity(line_capacity, grid.n_lines)
    grid.theta_min = theta_min
    grid.theta_max = theta_max
    grid.gas_pressure_min = gas_pressure_min
    grid.gas_pressure_max = gas_pressure_max
    grid.gas_pressure_base = gas_pressure_base
    grid.heat_capacity = heat_capacity
    return grid


def load_source_constraints(grid: SimpleNamespace, options: DispatchOptions) -> SimpleNamespace:
    attr_path = _resolve_attributes_path(options)
    grid.thermal_max = np.full(3, np.inf, dtype=float)
    grid.thermal_min = np.zeros(3, dtype=float)
    grid.gas_max = np.inf
    grid.gas_min = 0.0
    grid.gas_base = 0.0

    if not attr_path.exists():
        _warn_once("attributes_missing", f"attributes.json 不存在，使用默认上限: {attr_path}")
        return grid

    try:
        raw = attr_path.read_text(encoding="utf-8")
        data = json.loads(_sanitize_json_text(raw))
    except Exception as exc:
        _warn_once("attributes_parse_failed", f"attributes.json 解析失败，使用默认上限: {exc}")
        return grid

    sources = data.get("top_level", {}).get("sources", [])
    if isinstance(sources, dict):
        source_list = list(sources.values())
    elif isinstance(sources, list):
        source_list = sources
    else:
        source_list = [sources]

    for src in source_list:
        src_id = src.get("id")
        params = src.get("parameters", {})
        if not src_id or not isinstance(params, dict):
            continue
        if str(src_id).startswith("thermal_gen_"):
            match = re.search(r"thermal_gen_(\d+)", str(src_id))
            if not match:
                continue
            idx = int(match.group(1))
            if 1 <= idx <= 3:
                if "max_output_MW" in params:
                    grid.thermal_max[idx - 1] = params["max_output_MW"]
                if "min_output_MW" in params:
                    grid.thermal_min[idx - 1] = params["min_output_MW"]
                if "max_output" in params:
                    grid.thermal_max[idx - 1] = params["max_output"]
                if "min_output" in params:
                    grid.thermal_min[idx - 1] = params["min_output"]
        elif str(src_id) == "gas_source_1":
            if "max_output_MW" in params:
                grid.gas_max = params["max_output_MW"]
            if "min_output_MW" in params:
                grid.gas_min = params["min_output_MW"]
            if "base_injection_MW" in params:
                grid.gas_base = params["base_injection_MW"]
            if "max_output" in params:
                grid.gas_max = params["max_output"]
            if "min_output" in params:
                grid.gas_min = params["min_output"]
            if "base_injection" in params:
                grid.gas_base = params["base_injection"]
    return grid


def load_device_parameters(plan18: np.ndarray, options: DispatchOptions) -> SimpleNamespace:
    plan = np.asarray(plan18, dtype=float).reshape(-1)
    if plan.size != 18:
        raise ValueError(f"plan18 length must be 18, got {plan.size}")

    base_caps = np.array([
        2, 8, 10, 2, 0.5, 12, 12, 2, 2, 10, 2, 40, 40, 0.482, 0.356, 0.5, 5, 3
    ], dtype=float)
    real_caps = plan * base_caps
    if real_caps.size >= 15:
        # 风电/光伏在 OJ 中按 MW 直接计入，不做基准容量倍数折算。
        real_caps[13] = plan[13]
        real_caps[14] = plan[14]

    devices = SimpleNamespace()
    devices.e_boiler = SimpleNamespace(cap=real_caps[3], eta=0.9)
    devices.chiller_a = SimpleNamespace(cap=real_caps[4], COP=4)
    devices.hp_b = SimpleNamespace(cap=real_caps[9], COP=6)
    devices.pv = SimpleNamespace(cap=real_caps[14])
    devices.p2g = SimpleNamespace(cap=real_caps[15], eta=0.4)
    devices.hp_a = SimpleNamespace(cap=real_caps[8], COP=5)
    devices.gas_turbine = SimpleNamespace(cap=real_caps[16], eta=0.7)
    devices.chiller_b = SimpleNamespace(cap=real_caps[5], COP=5)
    devices.gas_boiler = SimpleNamespace(cap=real_caps[7], eta=0.95)
    devices.cchp = SimpleNamespace(cap=real_caps[17], eta_e=0.4, eta_h=0.3, eta_c=0.3)
    devices.absorption = SimpleNamespace(cap=real_caps[6], COP=0.8)
    devices.wind = SimpleNamespace(cap=real_caps[13])

    cap_e_total = float(real_caps[10])
    cap_h_total = float(real_caps[11])
    cap_c_total = float(real_caps[12])
    cap_e = _split_capacity(cap_e_total, _ELEC_STORAGE_SHARE)
    cap_h = _split_capacity(cap_h_total, _HEAT_STORAGE_SHARE)
    cap_c = _split_capacity(cap_c_total, _COLD_STORAGE_SHARE)

    power_ratio = _pad_params(options.storage_power_ratio, 3, 0.2)
    ratio_e, ratio_h, ratio_c = power_ratio[:3]
    cap_E = np.concatenate([cap_e, cap_h, cap_c])
    cap_P = np.concatenate([cap_e * ratio_e, cap_h * ratio_h, cap_c * ratio_c])

    eta_ch_base = _pad_params(options.storage_eta_ch, 3, 0.95)
    eta_dis_base = _pad_params(options.storage_eta_dis, 3, 0.95)
    eta_ch = np.array([
        eta_ch_base[0], eta_ch_base[0], eta_ch_base[0],
        eta_ch_base[1], eta_ch_base[1], eta_ch_base[1],
        eta_ch_base[2], eta_ch_base[2],
    ])
    eta_dis = np.array([
        eta_dis_base[0], eta_dis_base[0], eta_dis_base[0],
        eta_dis_base[1], eta_dis_base[1], eta_dis_base[1],
        eta_dis_base[2], eta_dis_base[2],
    ])

    devices.storage = SimpleNamespace(
        cap_E=np.asarray(cap_E, dtype=float),
        cap_P=np.asarray(cap_P, dtype=float),
        eta_ch=np.asarray(eta_ch, dtype=float),
        eta_dis=np.asarray(eta_dis, dtype=float),
        soc_min=float(options.storage_soc_min),
        soc_max=float(options.storage_soc_max),
        index_map={
            "elec_s": 0,
            "elec_t": 1,
            "elec_f": 2,
            "heat_s": 3,
            "heat_t": 4,
            "heat_f": 5,
            "cool_s": 6,
            "cool_t": 7,
        },
    )
    devices.plan18 = plan
    devices.real_caps = real_caps
    return devices


def _normalize_renewable_series(series: np.ndarray, cap: float) -> np.ndarray:
    series = np.asarray(series, dtype=float).reshape(-1)
    if series.size == 0:
        return series
    if np.nanmax(series) > 2 and cap > 0:
        return series / cap
    return series


def load_simulation_data(options: DispatchOptions) -> SimpleNamespace:
    data_dir = _resolve_data_dir(options)
    renewable_dir = _resolve_renewable_dir(options, data_dir)
    parsed = _parse_meos_inputs(data_dir, renewable_dir)
    loads_total = np.asarray(parsed.loads_total, dtype=float).copy()
    if loads_total.ndim == 2 and loads_total.shape[1] >= 9:
        if options.load_scale_electric != 1.0:
            loads_total[:, [0, 3, 6]] *= float(options.load_scale_electric)
        if options.load_scale_cool != 1.0:
            loads_total[:, [1, 4, 8]] *= float(options.load_scale_cool)
        if options.load_scale_heat != 1.0:
            loads_total[:, [2, 5, 7]] *= float(options.load_scale_heat)
    return SimpleNamespace(
        loads=SimpleNamespace(total=loads_total),
        prices=SimpleNamespace(
            electricity=parsed.prices_electricity,
            gas=parsed.prices_gas,
        ),
        renewable=SimpleNamespace(
            pv_avail=parsed.renewable_pv_avail,
            wind_avail=parsed.renewable_wind_avail,
        ),
    )


def build_day_data(
    data: SimpleNamespace,
    day_index: int,
    options: DispatchOptions,
    devices: SimpleNamespace,
) -> Dict[str, Any]:
    hour_start = (day_index - 1) * 24
    hour_end = day_index * 24
    loads = data.loads.total[hour_start:hour_end, :]

    if options.guidance_price_matrix is not None:
        price_matrix = np.asarray(options.guidance_price_matrix, dtype=float)
        prices_e = price_matrix[day_index - 1, :].reshape(-1)
    else:
        prices_e = data.prices.electricity[hour_start:hour_end] * options.guidance_price_factor
    if options.guidance_gas_matrix is not None:
        gas_matrix = np.asarray(options.guidance_gas_matrix, dtype=float)
        prices_g = gas_matrix[day_index - 1, :].reshape(-1)
    else:
        prices_g = data.prices.gas[hour_start:hour_end] * options.guidance_gas_factor

    prices_orig = {
        "electricity": data.prices.electricity[hour_start:hour_end],
        "gas": data.prices.gas[hour_start:hour_end],
    }

    pv_avail = data.renewable.pv_avail[hour_start:hour_end] * options.renewable_scale_pv
    wind_avail = data.renewable.wind_avail[hour_start:hour_end] * options.renewable_scale_wind
    pv_avail = _normalize_renewable_series(pv_avail, devices.pv.cap)
    wind_avail = _normalize_renewable_series(wind_avail, devices.wind.cap)

    return {
        "loads": loads,
        "prices": {"electricity": prices_e, "gas": prices_g},
        "prices_original": prices_orig,
        "renewable": {"pv_avail": pv_avail, "wind_avail": wind_avail},
    }


def _build_variable_index(
    n_hours: int,
    n_nodes: int,
    n_lines: int,
    n_thermal: int,
    thermal_order: Sequence[int],
    n_storage: int = 3,
) -> Tuple[SimpleNamespace, int]:
    idx = SimpleNamespace()
    base = 0

    order = [int(x) for x in thermal_order] if thermal_order else list(range(1, n_thermal + 1))
    order = [o for o in order if 1 <= o <= n_thermal]
    if len(set(order)) != n_thermal:
        order = list(range(1, n_thermal + 1))
    order = [o - 1 for o in order]

    idx.P_thermal = np.zeros((n_hours, n_thermal), dtype=int)
    for t in range(n_hours):
        for k in order:
            idx.P_thermal[t, k] = base
            base += 1

    def _vec(name: str, size: int) -> np.ndarray:
        nonlocal base
        arr = np.arange(base, base + size, dtype=int)
        base += size
        setattr(idx, name, arr)
        return arr

    _vec("G_buy", n_hours)
    _vec("P_pv", n_hours)
    _vec("P_e_boiler", n_hours)
    _vec("P_hp_b", n_hours)
    _vec("P_chiller_a", n_hours)
    _vec("P_p2g", n_hours)
    _vec("P_hp_a", n_hours)
    _vec("P_gt", n_hours)
    _vec("P_chiller_b", n_hours)
    _vec("P_wind", n_hours)
    _vec("P_gas_boiler", n_hours)
    _vec("P_cchp", n_hours)
    _vec("P_absorption", n_hours)
    _vec("P_grid_f", n_hours)

    idx.P_storage_ch = np.arange(base, base + n_hours * n_storage, dtype=int).reshape(n_hours, n_storage)
    base += n_hours * n_storage
    idx.P_storage_dis = np.arange(base, base + n_hours * n_storage, dtype=int).reshape(n_hours, n_storage)
    base += n_hours * n_storage
    idx.SOC = np.arange(base, base + (n_hours + 1) * n_storage, dtype=int).reshape(n_hours + 1, n_storage)
    base += (n_hours + 1) * n_storage

    _vec("L_shed_e_s", n_hours)
    _vec("L_shed_c_s", n_hours)
    _vec("L_shed_h_s", n_hours)
    _vec("L_shed_e_f", n_hours)
    _vec("L_shed_c_f", n_hours)
    _vec("L_shed_h_f", n_hours)
    _vec("L_shed_e_t", n_hours)
    _vec("L_shed_h_t", n_hours)
    _vec("L_shed_c_t", n_hours)
    _vec("H_transfer_s", n_hours)
    _vec("H_transfer_f", n_hours)

    idx.F_ij = np.arange(base, base + n_hours * n_lines, dtype=int).reshape(n_hours, n_lines)
    base += n_hours * n_lines
    idx.theta_i = np.arange(base, base + n_hours * n_nodes, dtype=int).reshape(n_hours, n_nodes)
    base += n_hours * n_nodes

    return idx, base


def optimize_daily_zonal(
    day_data: Dict[str, Any],
    devices: SimpleNamespace,
    options: DispatchOptions,
    grid: SimpleNamespace,
) -> Dict[str, Any]:
    n_hours = 24
    loads = np.asarray(day_data["loads"], dtype=float)

    L_elec_s = loads[:, 0]
    L_cool_s = loads[:, 1]
    L_heat_s = loads[:, 2]
    L_elec_f = loads[:, 3]
    L_cool_f = loads[:, 4]
    L_heat_f = loads[:, 5]
    L_elec_t = loads[:, 6]
    L_heat_t = loads[:, 7]
    L_cool_t = loads[:, 8]

    P_elec = np.asarray(day_data["prices"]["electricity"], dtype=float)
    P_gas = np.asarray(day_data["prices"]["gas"], dtype=float)
    P_elec_orig = np.asarray(day_data["prices_original"]["electricity"], dtype=float)
    P_gas_orig = np.asarray(day_data["prices_original"]["gas"], dtype=float)
    pv_avail = np.asarray(day_data["renewable"]["pv_avail"], dtype=float)
    wind_avail = np.asarray(day_data["renewable"]["wind_avail"], dtype=float)

    storage = devices.storage
    storage_index = storage.index_map
    shed_penalty = options.shed_penalty

    n_nodes = grid.n_nodes
    n_lines = grid.n_lines
    n_thermal = len(grid.thermal_to_bus)

    n_storage = int(storage.cap_E.size)
    idx, n_vars = _build_variable_index(
        n_hours, n_nodes, n_lines, n_thermal, options.thermal_order, n_storage
    )

    f = np.zeros(n_vars)
    thermal_bias = list(options.thermal_bias) if options.thermal_bias else []
    if len(thermal_bias) < n_thermal:
        thermal_bias = thermal_bias + [0.0] * (n_thermal - len(thermal_bias))
    thermal_bias = thermal_bias[:n_thermal]

    for t in range(n_hours):
        for k in range(n_thermal):
            if options.optimize_price:
                f[idx.P_thermal[t, k]] = thermal_bias[k]
            else:
                f[idx.P_thermal[t, k]] = P_elec[t] + thermal_bias[k]
        f[idx.G_buy[t]] = P_gas[t]

    shed_vars = np.concatenate([
        idx.L_shed_e_s, idx.L_shed_c_s, idx.L_shed_h_s,
        idx.L_shed_e_f, idx.L_shed_c_f, idx.L_shed_h_f,
        idx.L_shed_e_t, idx.L_shed_h_t, idx.L_shed_c_t,
    ])
    f[shed_vars] = shed_penalty

    n_eq_per_hour = n_nodes + 3 + 3 + 1 + n_lines + 1 + 1 + n_storage
    n_eq = n_hours * n_eq_per_hour + n_storage
    Aeq = np.zeros((n_eq, n_vars))
    beq = np.zeros(n_eq)
    eq_idx = 0

    incidence = grid.incidence
    B_ij = grid.B_ij
    F_max = grid.F_max

    for t in range(n_hours):
        for n in range(n_nodes):
            for l in range(n_lines):
                Aeq[eq_idx, idx.F_ij[t, l]] = -incidence[n, l]

            thermal_units = [i for i, bus in enumerate(grid.thermal_to_bus) if bus == n + 1]
            for k in thermal_units:
                Aeq[eq_idx, idx.P_thermal[t, k]] = 1

            if n + 1 == grid.zone_to_bus[0]:
                Aeq[eq_idx, idx.P_pv[t]] = 1
                Aeq[eq_idx, idx.P_e_boiler[t]] = -1
                Aeq[eq_idx, idx.P_hp_b[t]] = -1
                Aeq[eq_idx, idx.P_chiller_a[t]] = -1
                Aeq[eq_idx, idx.P_p2g[t]] = -1
                Aeq[eq_idx, idx.P_storage_ch[t, storage_index["elec_s"]]] = -1
                Aeq[eq_idx, idx.P_storage_dis[t, storage_index["elec_s"]]] = 1
                Aeq[eq_idx, idx.L_shed_e_s[t]] = 1
                beq[eq_idx] = L_elec_s[t]
            elif n + 1 == grid.zone_to_bus[1]:
                Aeq[eq_idx, idx.P_grid_f[t]] = -1
                beq[eq_idx] = 0
            else:
                Aeq[eq_idx, idx.P_wind[t]] = 1
                Aeq[eq_idx, idx.P_cchp[t]] = devices.cchp.eta_e
                Aeq[eq_idx, idx.P_storage_dis[t, storage_index["elec_t"]]] = 1
                Aeq[eq_idx, idx.P_storage_ch[t, storage_index["elec_t"]]] = -1
                Aeq[eq_idx, idx.L_shed_e_t[t]] = 1
                beq[eq_idx] = L_elec_t[t]
            eq_idx += 1

        Aeq[eq_idx, idx.P_grid_f[t]] = 1
        Aeq[eq_idx, idx.P_gt[t]] = devices.gas_turbine.eta
        Aeq[eq_idx, idx.P_hp_a[t]] = -1
        Aeq[eq_idx, idx.P_chiller_b[t]] = -1
        Aeq[eq_idx, idx.P_storage_dis[t, storage_index["elec_f"]]] = 1
        Aeq[eq_idx, idx.P_storage_ch[t, storage_index["elec_f"]]] = -1
        Aeq[eq_idx, idx.L_shed_e_f[t]] = 1
        beq[eq_idx] = L_elec_f[t]
        eq_idx += 1

        Aeq[eq_idx, idx.P_e_boiler[t]] = devices.e_boiler.eta
        Aeq[eq_idx, idx.P_hp_b[t]] = devices.hp_b.COP
        Aeq[eq_idx, idx.H_transfer_s[t]] = 1
        Aeq[eq_idx, idx.P_storage_dis[t, storage_index["heat_s"]]] = 1
        Aeq[eq_idx, idx.P_storage_ch[t, storage_index["heat_s"]]] = -1
        Aeq[eq_idx, idx.L_shed_h_s[t]] = 1
        beq[eq_idx] = L_heat_s[t]
        eq_idx += 1

        Aeq[eq_idx, idx.P_hp_a[t]] = devices.hp_a.COP
        Aeq[eq_idx, idx.H_transfer_f[t]] = 1
        Aeq[eq_idx, idx.P_storage_dis[t, storage_index["heat_f"]]] = 1
        Aeq[eq_idx, idx.P_storage_ch[t, storage_index["heat_f"]]] = -1
        Aeq[eq_idx, idx.L_shed_h_f[t]] = 1
        beq[eq_idx] = L_heat_f[t]
        eq_idx += 1

        Aeq[eq_idx, idx.P_gas_boiler[t]] = devices.gas_boiler.eta
        Aeq[eq_idx, idx.P_cchp[t]] = devices.cchp.eta_h
        Aeq[eq_idx, idx.P_absorption[t]] = -1
        Aeq[eq_idx, idx.H_transfer_s[t]] = -1
        Aeq[eq_idx, idx.H_transfer_f[t]] = -1
        Aeq[eq_idx, idx.P_storage_dis[t, storage_index["heat_t"]]] = 1
        Aeq[eq_idx, idx.P_storage_ch[t, storage_index["heat_t"]]] = -1
        Aeq[eq_idx, idx.L_shed_h_t[t]] = 1
        beq[eq_idx] = L_heat_t[t]
        eq_idx += 1

        Aeq[eq_idx, idx.P_chiller_a[t]] = devices.chiller_a.COP
        Aeq[eq_idx, idx.P_storage_dis[t, storage_index["cool_s"]]] = 1
        Aeq[eq_idx, idx.P_storage_ch[t, storage_index["cool_s"]]] = -1
        Aeq[eq_idx, idx.L_shed_c_s[t]] = 1
        beq[eq_idx] = L_cool_s[t]
        eq_idx += 1

        Aeq[eq_idx, idx.P_chiller_b[t]] = devices.chiller_b.COP
        Aeq[eq_idx, idx.L_shed_c_f[t]] = 1
        beq[eq_idx] = L_cool_f[t]
        eq_idx += 1

        Aeq[eq_idx, idx.P_absorption[t]] = devices.absorption.COP
        Aeq[eq_idx, idx.P_cchp[t]] = devices.cchp.eta_c
        Aeq[eq_idx, idx.P_storage_dis[t, storage_index["cool_t"]]] = 1
        Aeq[eq_idx, idx.P_storage_ch[t, storage_index["cool_t"]]] = -1
        Aeq[eq_idx, idx.L_shed_c_t[t]] = 1
        beq[eq_idx] = L_cool_t[t]
        eq_idx += 1

        Aeq[eq_idx, idx.G_buy[t]] = 1
        Aeq[eq_idx, idx.P_p2g[t]] = devices.p2g.eta
        Aeq[eq_idx, idx.P_gas_boiler[t]] = -1
        Aeq[eq_idx, idx.P_cchp[t]] = -1
        Aeq[eq_idx, idx.P_gt[t]] = -1
        beq[eq_idx] = 0
        eq_idx += 1

        for s in range(n_storage):
            Aeq[eq_idx, idx.SOC[t + 1, s]] = 1
            Aeq[eq_idx, idx.SOC[t, s]] = -1
            Aeq[eq_idx, idx.P_storage_ch[t, s]] = -storage.eta_ch[s]
            Aeq[eq_idx, idx.P_storage_dis[t, s]] = 1 / storage.eta_dis[s]
            beq[eq_idx] = 0
            eq_idx += 1

        for l in range(n_lines):
            i = grid.line_from[l] - 1
            j = grid.line_to[l] - 1
            Aeq[eq_idx, idx.F_ij[t, l]] = 1
            Aeq[eq_idx, idx.theta_i[t, i]] = -B_ij[l]
            Aeq[eq_idx, idx.theta_i[t, j]] = B_ij[l]
            beq[eq_idx] = 0
            eq_idx += 1

        ref_bus = 2 if n_nodes >= 3 else 0
        Aeq[eq_idx, idx.theta_i[t, ref_bus]] = 1
        beq[eq_idx] = 0
        eq_idx += 1

    if options.storage_daily_balance:
        for s in range(n_storage):
            Aeq[eq_idx, idx.SOC[0, s]] = 1
            Aeq[eq_idx, idx.SOC[n_hours, s]] = -1
            beq[eq_idx] = 0
            eq_idx += 1

    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)

    heat_cap = options.heat_transfer_cap
    if grid.heat_capacity is not None:
        heat_cap = grid.heat_capacity
    if heat_cap is None or (isinstance(heat_cap, float) and np.isnan(heat_cap)):
        heat_cap = float("inf")
    lb[idx.H_transfer_s] = 0
    lb[idx.H_transfer_f] = 0
    ub[idx.H_transfer_s] = heat_cap
    ub[idx.H_transfer_f] = heat_cap
    if hasattr(grid, "zone_heat_export") and not grid.zone_heat_export[2]:
        ub[idx.H_transfer_s] = 0
        ub[idx.H_transfer_f] = 0
    if hasattr(grid, "zone_heat_import"):
        if not grid.zone_heat_import[0]:
            ub[idx.H_transfer_s] = 0
        if not grid.zone_heat_import[1]:
            ub[idx.H_transfer_f] = 0

    for t in range(n_hours):
        for l in range(n_lines):
            lb[idx.F_ij[t, l]] = -F_max[l]
            ub[idx.F_ij[t, l]] = F_max[l]

    lb[idx.theta_i] = grid.theta_min
    ub[idx.theta_i] = grid.theta_max

    if hasattr(grid, "zone_elec_export") and not grid.zone_elec_export[1]:
        lb[idx.P_grid_f] = 0
    else:
        lb[idx.P_grid_f] = -np.inf

    for s in range(n_storage):
        ub[idx.P_storage_ch[:, s]] = storage.cap_P[s]
        ub[idx.P_storage_dis[:, s]] = storage.cap_P[s]
        lb[idx.SOC[:, s]] = storage.cap_E[s] * storage.soc_min
        ub[idx.SOC[:, s]] = storage.cap_E[s] * storage.soc_max

    for t in range(n_hours):
        ub[idx.P_pv[t]] = pv_avail[t] * devices.pv.cap
        ub[idx.P_wind[t]] = wind_avail[t] * devices.wind.cap
        ub[idx.P_e_boiler[t]] = devices.e_boiler.cap
        ub[idx.P_hp_b[t]] = devices.hp_b.cap
        ub[idx.P_chiller_a[t]] = devices.chiller_a.cap
        ub[idx.P_p2g[t]] = devices.p2g.cap
        ub[idx.P_hp_a[t]] = devices.hp_a.cap
        ub[idx.P_gt[t]] = devices.gas_turbine.cap
        ub[idx.P_chiller_b[t]] = devices.chiller_b.cap
        ub[idx.P_gas_boiler[t]] = devices.gas_boiler.cap
        ub[idx.P_cchp[t]] = devices.cchp.cap
        ub[idx.P_absorption[t]] = devices.absorption.cap
        ub[idx.G_buy[t]] = grid.gas_max
        lb[idx.G_buy[t]] = grid.gas_min
        for k in range(n_thermal):
            ub[idx.P_thermal[t, k]] = grid.thermal_max[k]
            lb[idx.P_thermal[t, k]] = grid.thermal_min[k]
        ub[idx.L_shed_e_s[t]] = L_elec_s[t]
        ub[idx.L_shed_c_s[t]] = L_cool_s[t]
        ub[idx.L_shed_h_s[t]] = L_heat_s[t]
        ub[idx.L_shed_e_f[t]] = L_elec_f[t]
        ub[idx.L_shed_c_f[t]] = L_cool_f[t]
        ub[idx.L_shed_h_f[t]] = L_heat_f[t]
        ub[idx.L_shed_e_t[t]] = L_elec_t[t]
        ub[idx.L_shed_h_t[t]] = L_heat_t[t]
        ub[idx.L_shed_c_t[t]] = L_cool_t[t]

    A_ub = None
    b_ub = None
    if np.isfinite(heat_cap):
        A_ub = np.zeros((n_hours, n_vars))
        b_ub = np.full(n_hours, heat_cap)
        for t in range(n_hours):
            A_ub[t, idx.H_transfer_s[t]] = 1
            A_ub[t, idx.H_transfer_f[t]] = 1

    if options.optimize_price:
        if not HAS_GUROBI or str(options.solver or "").lower() != "gurobi":
            raise RuntimeError("optimize_price 需要使用 Gurobi 求解")
        multipliers = np.asarray(options.price_multipliers, dtype=float).reshape(-1)
        price_levels = P_elec.reshape(-1, 1) * multipliers.reshape(1, -1)
        penalty_levels = None
        if options.price_penalty and options.price_penalty > 0:
            penalty_levels = (
                np.abs(multipliers - 1.0).reshape(1, -1) * P_elec.reshape(-1, 1) * options.price_penalty
            )
        thermal_max_sum = float(np.sum(grid.thermal_max)) if hasattr(grid, "thermal_max") else 1.0
        thermal_max_sum = max(1.0, thermal_max_sum)
        result = _solve_milp_gurobi_with_price(
            f,
            A_ub,
            b_ub,
            Aeq,
            beq,
            lb,
            ub,
            options,
            idx.P_storage_ch,
            idx.P_storage_dis,
            storage.cap_P,
            idx.P_thermal,
            price_levels,
            penalty_levels,
            thermal_max_sum,
        )
    elif options.storage_mutual_exclusive:
        if not HAS_GUROBI or str(options.solver or "").lower() != "gurobi":
            raise RuntimeError("storage_mutual_exclusive 需要使用 Gurobi 求解")
        result = _solve_milp_gurobi(
            f,
            A_ub,
            b_ub,
            Aeq,
            beq,
            lb,
            ub,
            options,
            idx.P_storage_ch,
            idx.P_storage_dis,
            storage.cap_P,
        )
    else:
        result = _solve_lp(f, A_ub, b_ub, Aeq, beq, lb, ub, options)
    if not result.success:
        raise RuntimeError(f"优化失败: {result.message}")

    x = result.x

    result_dict: Dict[str, Any] = {}
    result_dict["P_thermal"] = x[idx.P_thermal]
    result_dict["G_buy"] = x[idx.G_buy]
    result_dict["P_e_boiler"] = x[idx.P_e_boiler]
    result_dict["P_hp_a"] = x[idx.P_hp_a]
    result_dict["P_hp_b"] = x[idx.P_hp_b]
    result_dict["P_gas_boiler"] = x[idx.P_gas_boiler]
    result_dict["P_cchp"] = x[idx.P_cchp]
    result_dict["P_gt"] = x[idx.P_gt]
    result_dict["P_pv"] = x[idx.P_pv]
    result_dict["P_wind"] = x[idx.P_wind]
    result_dict["P_chiller_a"] = x[idx.P_chiller_a]
    result_dict["P_chiller_b"] = x[idx.P_chiller_b]
    result_dict["P_absorption"] = x[idx.P_absorption]
    result_dict["P_p2g"] = x[idx.P_p2g]
    result_dict["P_grid_f"] = x[idx.P_grid_f]
    result_dict["P_storage_ch"] = x[idx.P_storage_ch]
    result_dict["P_storage_dis"] = x[idx.P_storage_dis]
    result_dict["SOC"] = x[idx.SOC]
    result_dict["H_transfer_s"] = x[idx.H_transfer_s]
    result_dict["H_transfer_f"] = x[idx.H_transfer_f]
    result_dict["F_ij"] = x[idx.F_ij]
    result_dict["theta_i"] = x[idx.theta_i]
    result_dict["L_shed"] = np.column_stack([
        x[idx.L_shed_e_s], x[idx.L_shed_c_s], x[idx.L_shed_h_s],
        x[idx.L_shed_e_f], x[idx.L_shed_c_f], x[idx.L_shed_h_f],
        x[idx.L_shed_e_t], x[idx.L_shed_h_t], x[idx.L_shed_c_t],
    ])

    price_selected = None
    if options.optimize_price and getattr(result, "extra", None):
        y_price = np.asarray(result.extra.get("y_price"), dtype=float)
        price_levels = np.asarray(result.extra.get("price_levels"), dtype=float)
        if y_price.ndim == 2 and price_levels.shape == y_price.shape:
            idx_level = np.argmax(y_price, axis=1)
            price_selected = price_levels[np.arange(n_hours), idx_level]
            result_dict["price_electricity"] = price_selected

    result_dict["energy"] = {
        "elec_buy": float(np.sum(result_dict["P_thermal"])),
        "gas_buy": float(np.sum(result_dict["G_buy"])),
        "shed": float(np.sum(result_dict["L_shed"])),
    }

    P_thermal_total = np.sum(result_dict["P_thermal"], axis=1)
    if price_selected is not None:
        result_dict["cost"] = {
            "electricity": float(np.sum(P_thermal_total * price_selected)),
            "gas": float(np.sum(result_dict["G_buy"] * P_gas)),
            "penalty": float(result_dict["energy"]["shed"] * shed_penalty),
        }
        result_dict["cost"]["total"] = (
            result_dict["cost"]["electricity"]
            + result_dict["cost"]["gas"]
            + result_dict["cost"]["penalty"]
        )
    result_dict["cost_original"] = {
        "electricity": float(np.sum(P_thermal_total * P_elec_orig)),
        "gas": float(np.sum(result_dict["G_buy"] * P_gas_orig)),
        "penalty": float(result_dict["energy"]["shed"] * shed_penalty),
    }
    result_dict["cost_original"]["total"] = (
        result_dict["cost_original"]["electricity"]
        + result_dict["cost_original"]["gas"]
        + result_dict["cost_original"]["penalty"]
    )
    return result_dict


def build_dispatch_8760(daily_results: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    dispatch = {
        "shed_load": np.zeros((8760, 9)),
        "thermal_power": np.zeros((8760, 3)),
        "gas_source": np.zeros(8760),
        "pv_power": np.zeros(8760),
        "wind_power": np.zeros(8760),
        "p2g": np.zeros(8760),
        "e_boiler": np.zeros(8760),
        "hp_b": np.zeros(8760),
        "chiller_a": np.zeros(8760),
        "hp_a": np.zeros(8760),
        "gt": np.zeros(8760),
        "chiller_b": np.zeros(8760),
        "gas_boiler": np.zeros(8760),
        "cchp": np.zeros(8760),
        "absorption": np.zeros(8760),
        "heat_transfer_s": np.zeros(8760),
        "heat_transfer_f": np.zeros(8760),
        "line_flow": np.array([]),
        "theta": np.array([]),
        "storage_ch": np.array([]),
        "storage_dis": np.array([]),
        "SOC": np.array([]),
    }

    for day_idx, dr in enumerate(daily_results, start=1):
        if not isinstance(dr, dict):
            continue
        day_index = int(dr.get("day_index", day_idx))
        hour_start = (day_index - 1) * 24
        hour_end = day_index * 24
        if hour_end > 8760:
            break
        if dr.get("status") not in (None, "optimal"):
            continue
        dispatch["shed_load"][hour_start:hour_end, :] = dr.get("L_shed", 0)
        dispatch["thermal_power"][hour_start:hour_end, :] = dr.get("P_thermal", 0)
        dispatch["gas_source"][hour_start:hour_end] = dr.get("G_buy", 0)

        if "P_pv" in dr:
            dispatch["pv_power"][hour_start:hour_end] = dr["P_pv"]
        if "P_wind" in dr:
            dispatch["wind_power"][hour_start:hour_end] = dr["P_wind"]
        if "P_p2g" in dr:
            dispatch["p2g"][hour_start:hour_end] = dr["P_p2g"]
        if "P_e_boiler" in dr:
            dispatch["e_boiler"][hour_start:hour_end] = dr["P_e_boiler"]
        if "P_hp_b" in dr:
            dispatch["hp_b"][hour_start:hour_end] = dr["P_hp_b"]
        if "P_chiller_a" in dr:
            dispatch["chiller_a"][hour_start:hour_end] = dr["P_chiller_a"]
        if "P_hp_a" in dr:
            dispatch["hp_a"][hour_start:hour_end] = dr["P_hp_a"]
        if "P_gt" in dr:
            dispatch["gt"][hour_start:hour_end] = dr["P_gt"]
        if "P_chiller_b" in dr:
            dispatch["chiller_b"][hour_start:hour_end] = dr["P_chiller_b"]
        if "P_gas_boiler" in dr:
            dispatch["gas_boiler"][hour_start:hour_end] = dr["P_gas_boiler"]
        if "P_cchp" in dr:
            dispatch["cchp"][hour_start:hour_end] = dr["P_cchp"]
        if "P_absorption" in dr:
            dispatch["absorption"][hour_start:hour_end] = dr["P_absorption"]
        if "H_transfer_s" in dr:
            dispatch["heat_transfer_s"][hour_start:hour_end] = dr["H_transfer_s"]
        if "H_transfer_f" in dr:
            dispatch["heat_transfer_f"][hour_start:hour_end] = dr["H_transfer_f"]

        if "P_storage_ch" in dr:
            storage_ch = np.asarray(dr["P_storage_ch"], dtype=float)
            if dispatch["storage_ch"].size == 0:
                dispatch["storage_ch"] = np.zeros((8760, storage_ch.shape[1]))
            dispatch["storage_ch"][hour_start:hour_end, :] = storage_ch
        if "P_storage_dis" in dr:
            storage_dis = np.asarray(dr["P_storage_dis"], dtype=float)
            if dispatch["storage_dis"].size == 0:
                dispatch["storage_dis"] = np.zeros((8760, storage_dis.shape[1]))
            dispatch["storage_dis"][hour_start:hour_end, :] = storage_dis
        if "SOC" in dr:
            soc = np.asarray(dr["SOC"], dtype=float)
            if soc.ndim == 2 and soc.shape[0] >= 24:
                if dispatch["SOC"].size == 0:
                    dispatch["SOC"] = np.zeros((8760, soc.shape[1]))
                dispatch["SOC"][hour_start:hour_end, :] = soc[:24, :]

        if "F_ij" in dr:
            flows = np.asarray(dr["F_ij"], dtype=float)
            if dispatch["line_flow"].size == 0:
                dispatch["line_flow"] = np.zeros((8760, flows.shape[1]))
            dispatch["line_flow"][hour_start:hour_end, :] = flows
        if "theta_i" in dr:
            theta = np.asarray(dr["theta_i"], dtype=float)
            if dispatch["theta"].size == 0:
                dispatch["theta"] = np.zeros((8760, theta.shape[1]))
            dispatch["theta"][hour_start:hour_end, :] = theta

    return dispatch


class ZonalDailySolver:
    def __init__(self, options: Optional[DispatchOptions] = None):
        self.options = options or DispatchOptions()
        self.data = load_simulation_data(self.options)
        self.grid = load_source_constraints(load_topology(self.options), self.options)

    def solve_day(self, day_index: int, plan18: np.ndarray) -> Dict[str, Any]:
        devices = load_device_parameters(plan18, self.options)
        day_data = build_day_data(self.data, day_index, self.options, devices)
        result = optimize_daily_zonal(day_data, devices, self.options, self.grid)
        result["day_index"] = day_index
        result["status"] = "optimal"
        return result


def run_annual_simulation(
    plan18: np.ndarray,
    options: Optional[DispatchOptions] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, np.ndarray]]:
    options = options or DispatchOptions()
    solver = ZonalDailySolver(options)

    day_list = list(range(1, 366))
    if options.day_indices:
        day_list = [d for d in options.day_indices if 1 <= d <= 365]
        if not day_list:
            day_list = list(range(1, 366))

    daily_results: List[Dict[str, Any]] = []
    annual_summary = {
        "total_elec_cost": 0.0,
        "total_gas_cost": 0.0,
        "total_shed_penalty": 0.0,
        "total_elec_buy": 0.0,
        "total_gas_buy": 0.0,
        "total_shed": 0.0,
        "failed_days": 0,
        "day_indices": day_list,
    }

    for day in day_list:
        try:
            result = solver.solve_day(day, plan18)
            daily_results.append(result)
            annual_summary["total_elec_cost"] += result["cost_original"]["electricity"]
            annual_summary["total_gas_cost"] += result["cost_original"]["gas"]
            annual_summary["total_shed_penalty"] += result["cost_original"]["penalty"]
            annual_summary["total_elec_buy"] += result["energy"]["elec_buy"]
            annual_summary["total_gas_buy"] += result["energy"]["gas_buy"]
            annual_summary["total_shed"] += result["energy"]["shed"]
        except Exception as exc:
            daily_results.append({"day_index": day, "status": "failed", "error": str(exc)})
            annual_summary["failed_days"] += 1

    dispatch = build_dispatch_8760(daily_results)
    return daily_results, annual_summary, dispatch
