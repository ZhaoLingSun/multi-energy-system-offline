"""
MEOS 电网约束模块
实现电网节点平衡与线路约束（DC 潮流模型）

对应 MATLAB daily_dispatcher.m 中的：
- 变量索引: var_idx.F_ij, var_idx.theta_i, var_idx.P_grid, var_idx.P_thermal
- DC 潮流约束: A1.5 节
- 节点功率平衡: A1.6 节
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ElectricIndex:
    """
    电网变量索引结构

    对应 MATLAB var_idx 中的电网相关部分
    """
    n_hours: int = 24           # 时间步数
    n_nodes: int = 3            # 电节点数（母线数）
    n_lines: int = 3            # 线路数
    n_thermal: int = 3          # 火电机组数

    # 变量起始索引（在完整变量向量中的位置）
    P_grid_start: int = 0       # 购电功率 P_grid(t,n)
    P_thermal_start: int = 0    # 火电出力 P_thermal(t,k)
    F_line_start: int = 0       # 线路功率 F_ij(t,l)
    theta_start: int = 0        # 节点相角 theta(t,n)

    def var_count(self) -> int:
        """电网变量总数"""
        return (self.n_hours * self.n_nodes +      # P_grid
                self.n_hours * self.n_thermal +    # P_thermal
                self.n_hours * self.n_lines +      # F_line
                self.n_hours * self.n_nodes)       # theta


@dataclass
class ElectricVariables:
    """
    电网决策变量

    维度说明（对应 MATLAB daily_dispatcher.m）：
    - P_grid[t, n]: 节点 n 在时刻 t 的购电功率 (MW), shape=(n_hours, n_nodes)
    - P_thermal[t, k]: 火电机组 k 在时刻 t 的出力 (MW), shape=(n_hours, n_thermal)
    - F_line[t, l]: 线路 l 在时刻 t 的功率 (MW), shape=(n_hours, n_lines)
    - theta[t, n]: 节点 n 在时刻 t 的相角 (rad), shape=(n_hours, n_nodes)
    """
    P_grid: np.ndarray = None      # 购电功率
    P_thermal: np.ndarray = None   # 火电出力
    F_line: np.ndarray = None      # 线路功率
    theta: np.ndarray = None       # 节点相角


@dataclass
class ElectricNetworkData:
    """
    电网拓扑与参数数据

    对应 MATLAB topology.json 中的 network_params.electric
    """
    # 线路拓扑 (from_node, to_node)
    line_from: List[int] = field(default_factory=lambda: [0, 1, 0])
    line_to: List[int] = field(default_factory=lambda: [1, 2, 2])

    # 线路参数
    line_reactance: float = 0.000281   # 线路电抗 (pu)
    line_capacity: float = 400.0       # 线路容量 (MW)

    # 相角限制
    theta_min: float = -3.14           # 最小相角 (rad)
    theta_max: float = 3.14            # 最大相角 (rad)

    # 火电容量 (MW)
    thermal_caps: List[float] = field(default_factory=lambda: [1000, 1000, 1000])


def _build_incidence_matrix(n_nodes: int, line_from: List[int],
                            line_to: List[int]) -> np.ndarray:
    """
    构建节点-线路关联矩阵

    对应 MATLAB daily_dispatcher.m 第 568-574 行
    """
    n_lines = len(line_from)
    incidence = np.zeros((n_nodes, n_lines))
    for l in range(n_lines):
        incidence[line_from[l], l] = 1
        incidence[line_to[l], l] = -1
    return incidence


def build_electric_constraints(
    idx: ElectricIndex,
    net_data: ElectricNetworkData,
    elec_load: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    构建电网节点平衡与线路约束

    参数:
        idx: 电网变量索引
        net_data: 电网拓扑与参数
        elec_load: 电负荷数据 shape=(n_hours, n_nodes), 可选

    返回:
        Dict 包含:
        - Aeq, beq: 等式约束 (节点平衡 + DC潮流)
        - Aub, bub: 不等式约束 (线容量 + 火电容量)
        - lb, ub: 变量边界
        - var_names: 变量名称列表
    """
    n_hours = idx.n_hours
    n_nodes = idx.n_nodes
    n_lines = idx.n_lines
    n_thermal = idx.n_thermal

    # 变量总数
    n_vars = idx.var_count()

    # 线路导纳
    B_ij = 1.0 / net_data.line_reactance

    # 关联矩阵
    incidence = _build_incidence_matrix(
        n_nodes, net_data.line_from, net_data.line_to
    )

    # =========================================================
    # 等式约束: 节点平衡 + DC潮流
    # =========================================================
    # 约束数: n_hours * n_nodes (节点平衡)
    #       + n_hours * n_lines (DC潮流)
    #       + n_hours (参考节点)
    n_eq = n_hours * (n_nodes + n_lines + 1)
    Aeq = np.zeros((n_eq, n_vars))
    beq = np.zeros(n_eq)

    eq_idx = 0

    # ---------------------------------------------------------
    # 1. 节点功率平衡
    # Σ F_ij(流入) - Σ F_ij(流出) + P_grid + P_thermal = Load
    # ---------------------------------------------------------
    for t in range(n_hours):
        for n in range(n_nodes):
            # 线路潮流贡献
            for l in range(n_lines):
                col = idx.F_line_start + t * n_lines + l
                Aeq[eq_idx, col] = incidence[n, l]

            # 购电 P_grid
            col = idx.P_grid_start + t * n_nodes + n
            Aeq[eq_idx, col] = -1

            # 火电 P_thermal (机组与节点一一对应)
            if n < n_thermal:
                col = idx.P_thermal_start + t * n_thermal + n
                Aeq[eq_idx, col] = -1

            # 右端项: 负荷
            if elec_load is not None:
                beq[eq_idx] = -elec_load[t, n]

            eq_idx += 1

    # ---------------------------------------------------------
    # 2. DC 潮流约束
    # F_ij = B_ij * (theta_i - theta_j)
    # ---------------------------------------------------------
    for t in range(n_hours):
        for l in range(n_lines):
            i = net_data.line_from[l]
            j = net_data.line_to[l]

            # F_ij 系数 = 1
            col = idx.F_line_start + t * n_lines + l
            Aeq[eq_idx, col] = 1

            # theta_i 系数 = -B_ij
            col = idx.theta_start + t * n_nodes + i
            Aeq[eq_idx, col] = -B_ij

            # theta_j 系数 = +B_ij
            col = idx.theta_start + t * n_nodes + j
            Aeq[eq_idx, col] = B_ij

            eq_idx += 1

    # ---------------------------------------------------------
    # 3. 参考节点约束
    # theta_0(t) = 0
    # ---------------------------------------------------------
    for t in range(n_hours):
        col = idx.theta_start + t * n_nodes + 0
        Aeq[eq_idx, col] = 1
        beq[eq_idx] = 0
        eq_idx += 1

    # =========================================================
    # 不等式约束: 线容量 + 火电容量
    # =========================================================
    # 约束数: n_hours * n_lines * 2 (线容量双向)
    #       + n_hours * n_thermal (火电容量)
    n_ub = n_hours * (n_lines * 2 + n_thermal)
    Aub = np.zeros((n_ub, n_vars))
    bub = np.zeros(n_ub)

    ub_idx = 0

    # ---------------------------------------------------------
    # 4. 线容量约束
    # |F_ij| <= F_max => F_ij <= F_max 且 -F_ij <= F_max
    # ---------------------------------------------------------
    F_max = net_data.line_capacity
    for t in range(n_hours):
        for l in range(n_lines):
            col = idx.F_line_start + t * n_lines + l
            # F_ij <= F_max
            Aub[ub_idx, col] = 1
            bub[ub_idx] = F_max
            ub_idx += 1
            # -F_ij <= F_max
            Aub[ub_idx, col] = -1
            bub[ub_idx] = F_max
            ub_idx += 1

    # ---------------------------------------------------------
    # 5. 火电容量约束 (对应 MATLAB 第 965-971 行)
    # P_thermal(t,k) <= Cap_thermal(k)
    # ---------------------------------------------------------
    for t in range(n_hours):
        for k in range(n_thermal):
            col = idx.P_thermal_start + t * n_thermal + k
            Aub[ub_idx, col] = 1
            bub[ub_idx] = net_data.thermal_caps[k]
            ub_idx += 1

    # =========================================================
    # 变量边界
    # =========================================================
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)

    # 相角边界 (对应 MATLAB 第 1060-1065 行)
    for t in range(n_hours):
        for n in range(n_nodes):
            col = idx.theta_start + t * n_nodes + n
            lb[col] = net_data.theta_min
            ub[col] = net_data.theta_max

    # 线路功率边界 (双向)
    for t in range(n_hours):
        for l in range(n_lines):
            col = idx.F_line_start + t * n_lines + l
            lb[col] = -net_data.line_capacity
            ub[col] = net_data.line_capacity

    # =========================================================
    # 变量名称列表
    # =========================================================
    var_names = []
    for t in range(n_hours):
        for n in range(n_nodes):
            var_names.append(f"P_grid[{t},{n}]")
    for t in range(n_hours):
        for k in range(n_thermal):
            var_names.append(f"P_thermal[{t},{k}]")
    for t in range(n_hours):
        for l in range(n_lines):
            var_names.append(f"F_line[{t},{l}]")
    for t in range(n_hours):
        for n in range(n_nodes):
            var_names.append(f"theta[{t},{n}]")

    return {
        "Aeq": Aeq,
        "beq": beq,
        "Aub": Aub,
        "bub": bub,
        "lb": lb,
        "ub": ub,
        "var_names": var_names,
        "n_vars": n_vars,
        "incidence": incidence,
    }


def create_electric_index(n_hours: int = 24, start_offset: int = 0) -> ElectricIndex:
    """
    创建电网变量索引

    参数:
        n_hours: 时间步数
        start_offset: 在完整变量向量中的起始偏移

    返回:
        ElectricIndex 实例
    """
    idx = ElectricIndex(n_hours=n_hours)
    n_nodes = idx.n_nodes
    n_lines = idx.n_lines
    n_thermal = idx.n_thermal

    current = start_offset
    idx.P_grid_start = current
    current += n_hours * n_nodes

    idx.P_thermal_start = current
    current += n_hours * n_thermal

    idx.F_line_start = current
    current += n_hours * n_lines

    idx.theta_start = current

    return idx


def create_network_data_from_topology(
    topology: Dict[str, Any],
    attributes: Optional[Dict[str, Any]] = None,
) -> ElectricNetworkData:
    """
    从拓扑数据创建电网参数

    参数:
        topology: 规范化后的拓扑数据
        attributes: 属性数据（可选）

    返回:
        ElectricNetworkData 实例
    """
    net_data = ElectricNetworkData()

    # 从 network_params 提取电网参数
    if "network_params" in topology:
        elec_params = topology["network_params"].get("electric", {})
        if elec_params:
            net_data.line_reactance = elec_params.get(
                "line_reactance", net_data.line_reactance
            )
            net_data.line_capacity = elec_params.get(
                "line_capacity", net_data.line_capacity
            )
            net_data.theta_min = elec_params.get("theta_min", net_data.theta_min)
            net_data.theta_max = elec_params.get("theta_max", net_data.theta_max)

    # 从 attributes 提取火电容量
    if attributes and "sources" in attributes:
        thermal_caps = []
        for src in attributes["sources"]:
            if src.get("type") == "Thermal_Generator":
                params = src.get("parameters", {})
                cap = params.get("max_output_MW", 1000)
                thermal_caps.append(cap)
        if thermal_caps:
            net_data.thermal_caps = thermal_caps

    return net_data
