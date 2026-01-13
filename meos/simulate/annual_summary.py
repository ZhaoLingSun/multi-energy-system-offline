"""
MEOS 年度汇总与成本核算模块

对应 MATLAB cost_accounting.m 的统计口径：
- C_CAP: 年化投资成本
- C_OP: 运行成本 (购电 + 购气 + 失负荷惩罚)
- C_Carbon: 碳成本 (超额碳排放 × 碳价)
- Score: 最终分数 (Logistic 函数)
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import math
import numpy as np


# ============ 默认参数 (对应 MATLAB oj_score.yaml) ============

DEFAULT_DISCOUNT_RATE = 0.04          # 贴现率 4%
DEFAULT_CARBON_THRESHOLD = 100000.0   # 碳阈值 tCO2
DEFAULT_CARBON_PRICE = 600.0          # 碳价 元/tCO2
DEFAULT_GAS_EMISSION_FACTOR = 0.002   # 购气碳因子 tCO2/m³
DEFAULT_GAS_TO_MWH = 0.01             # m³ → MWh
DEFAULT_COST_UNIT = 10000             # 万元
DEFAULT_LOGISTIC_X0 = 100000          # Logistic x0
DEFAULT_LOGISTIC_K = 15000            # Logistic k
DEFAULT_SHED_PENALTY = 500000         # 切负荷惩罚 元/MWh


# ============ 数据类定义 ============

@dataclass
class DailyCost:
    """日成本结构"""
    day: int
    electricity: float = 0.0    # 购电成本 (元)
    gas: float = 0.0            # 购气成本 (元)
    penalty: float = 0.0        # 失负荷惩罚 (元)
    total: float = 0.0          # 日总成本 (元)

    # 能量统计
    P_grid_total: float = 0.0   # 日购电量 (MWh)
    G_buy_total: float = 0.0    # 日购气量 (m³)
    L_shed_total: float = 0.0   # 日切负荷量 (MWh)


@dataclass
class CarbonBreakdown:
    """碳排放分解"""
    emission_electricity: float = 0.0  # 购电碳排放 (tCO2)
    emission_gas: float = 0.0          # 购气碳排放 (tCO2)
    emission_total: float = 0.0        # 总碳排放 (tCO2)
    emission_threshold: float = DEFAULT_CARBON_THRESHOLD
    emission_excess: float = 0.0       # 超额碳排放 (tCO2)
    carbon_price: float = DEFAULT_CARBON_PRICE


@dataclass
class AnnualSummary:
    """年度汇总结构"""
    # 成本分项
    C_CAP: float = 0.0          # 年化投资成本 (元)
    C_OP_ele: float = 0.0       # 购电成本 (元)
    C_OP_gas: float = 0.0       # 购气成本 (元)
    C_OP_penalty: float = 0.0   # 失负荷惩罚 (元)
    C_OP_total: float = 0.0     # 运行成本合计 (元)
    C_Carbon: float = 0.0       # 碳成本 (元)
    C_total: float = 0.0        # 总成本 (元)
    Score: float = 0.0          # 最终分数

    # 能量统计
    E_grid_total: float = 0.0   # 年购电量 (MWh)
    G_buy_total: float = 0.0    # 年购气量 (m³)
    L_shed_total: float = 0.0   # 年切负荷量 (MWh)

    # 碳排放
    carbon: CarbonBreakdown = field(default_factory=CarbonBreakdown)

    # 日成本列表
    daily_costs: List[DailyCost] = field(default_factory=list)

    # 元数据
    n_days: int = 0
    timestamp: str = ""


# ============ 辅助函数 ============

def calculate_annuity_factor(rate: float, years: int) -> float:
    """
    计算年化系数

    公式: i(1+i)^n / ((1+i)^n - 1)
    对应 MATLAB cost_accounting.m 第 179 行
    """
    if rate <= 0 or years <= 0:
        return 0.0
    factor = (1 + rate) ** years
    return rate * factor / (factor - 1)


def calculate_capex(
    plan18: List[float],
    device_catalog: Optional[List[Dict[str, Any]]] = None,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
) -> float:
    """
    计算年化投资成本 C_CAP

    对应 MATLAB cost_accounting.m 第 149-188 行
    """
    if device_catalog is None:
        return 0.0

    C_CAP = 0.0
    for k, cap in enumerate(plan18):
        if cap <= 0 or k >= len(device_catalog):
            continue
        device = device_catalog[k]
        unit_cost = device.get("unit_cost", 0)
        lifespan = device.get("lifespan_years")
        if lifespan is None:
            lifespan = device.get("lifespan", 20)
        if unit_cost > 0 and lifespan > 0:
            annuity = calculate_annuity_factor(discount_rate, lifespan)
            C_CAP += cap * unit_cost * annuity
    return C_CAP


def calculate_carbon_cost(
    E_elec: float,
    E_gas: float,
    threshold: float = DEFAULT_CARBON_THRESHOLD,
    price: float = DEFAULT_CARBON_PRICE,
) -> tuple:
    """
    计算碳成本

    对应 MATLAB cost_accounting.m 第 190-265 行
    """
    E_total = E_elec + E_gas
    E_excess = max(0, E_total - threshold)
    C_Carbon = E_excess * price

    carbon = CarbonBreakdown(
        emission_electricity=E_elec,
        emission_gas=E_gas,
        emission_total=E_total,
        emission_threshold=threshold,
        emission_excess=E_excess,
        carbon_price=price,
    )
    return C_Carbon, carbon


def calculate_score(
    C_total: float,
    cost_unit: float = DEFAULT_COST_UNIT,
    x0: float = DEFAULT_LOGISTIC_X0,
    k: float = DEFAULT_LOGISTIC_K,
) -> float:
    """
    计算最终分数 (Logistic 函数)

    公式: Score = 100 / (1 + exp((x - x0) / k))
    对应 MATLAB cost_accounting.m 第 109-114 行
    """
    x = C_total / cost_unit
    z = (x - x0) / k
    if z > 700:
        return 0.0
    if z < -700:
        return 100.0
    return 100.0 / (1.0 + math.exp(z))


# ============ 主函数 ============

def summarize_annual(
    daily_results: List[Dict[str, Any]],
    plan18: Optional[List[float]] = None,
    device_catalog: Optional[List[Dict[str, Any]]] = None,
    carbon_factors: Optional[List[float]] = None,
    config: Optional[Dict[str, Any]] = None,
    day_weights: Optional[List[float]] = None,
) -> AnnualSummary:
    """
    年度汇总与成本核算

    对应 MATLAB cost_accounting.m

    参数:
        daily_results: 日调度结果列表，每项包含:
            - cost: {electricity, gas, penalty, total}
            - P_grid: 购电功率 (24,) 或 (24, n_zones)
            - G_buy: 购气量 (24,)
            - L_shed: 切负荷量 (24,) 或 (24, n_types)
        plan18: 18维规划向量 (可选)
        device_catalog: 设备目录 (可选)
        carbon_factors: 24*365 碳因子 (可选)
        config: 配置参数 (可选)

    返回:
        AnnualSummary 实例
    """
    config = config or {}

    # 提取配置参数
    discount_rate = config.get("discount_rate", DEFAULT_DISCOUNT_RATE)
    carbon_threshold = config.get("carbon_threshold", DEFAULT_CARBON_THRESHOLD)
    carbon_price = config.get("carbon_price", DEFAULT_CARBON_PRICE)
    gas_emission_factor = config.get("gas_emission_factor", DEFAULT_GAS_EMISSION_FACTOR)
    gas_to_MWh = config.get("gas_to_MWh", DEFAULT_GAS_TO_MWH)

    # 初始化汇总
    C_elec_total = 0.0
    C_gas_total = 0.0
    C_penalty_total = 0.0
    E_grid_total = 0.0
    G_buy_total = 0.0
    L_shed_total = 0.0
    E_elec_carbon = 0.0
    E_gas_carbon = 0.0
    daily_costs: List[DailyCost] = []

    # =========================================================
    # 1. 遍历日结果汇总 (对应 MATLAB 第 58-78 行)
    # =========================================================
    if day_weights is None or len(day_weights) != len(daily_results):
        weights = [1.0] * len(daily_results)
    else:
        weights = [float(w) for w in day_weights]

    for d, day_result in enumerate(daily_results):
        weight = weights[d]
        day_idx = day_result.get("day_index", d + 1)
        try:
            day_idx = int(day_idx)
        except (TypeError, ValueError):
            day_idx = d + 1
        if day_idx < 1:
            day_idx = d + 1
        cost = day_result.get("cost", {})
        c_elec = cost.get("electricity", 0.0)
        c_gas = cost.get("gas", 0.0)
        c_penalty = cost.get("penalty", 0.0)
        c_total = cost.get("total", c_elec + c_gas + c_penalty)

        C_elec_total += c_elec * weight
        C_gas_total += c_gas * weight
        C_penalty_total += c_penalty * weight

        # 能量统计
        P_grid = day_result.get("P_grid", [])
        G_buy = day_result.get("G_buy", [])
        L_shed = day_result.get("L_shed", [])

        p_grid_arr = np.asarray(P_grid, dtype=float).reshape(-1) if P_grid is not None else np.array([])
        g_buy_arr = np.asarray(G_buy, dtype=float).reshape(-1) if G_buy is not None else np.array([])
        l_shed_arr = np.asarray(L_shed, dtype=float).reshape(-1) if L_shed is not None else np.array([])

        p_grid_day = float(np.sum(p_grid_arr)) if p_grid_arr.size else 0.0
        g_buy_day = float(np.sum(g_buy_arr)) if g_buy_arr.size else 0.0
        l_shed_day = float(np.sum(l_shed_arr)) if l_shed_arr.size else 0.0

        E_grid_total += p_grid_day * weight
        G_buy_total += g_buy_day * weight
        L_shed_total += l_shed_day * weight

        # 碳排放计算
        if carbon_factors and len(carbon_factors) >= day_idx * 24:
            day_start = (day_idx - 1) * 24
            day_factors = carbon_factors[day_start:day_start + 24]
            if P_grid and len(P_grid) == 24:
                E_elec_carbon += weight * sum(p * f for p, f in zip(P_grid, day_factors))

        # 购气碳排放
        if G_buy:
            g_buy_m3 = g_buy_day / gas_to_MWh
            E_gas_carbon += weight * g_buy_m3 * gas_emission_factor

        # 记录日成本
        daily_costs.append(DailyCost(
            day=day_idx,
            electricity=c_elec,
            gas=c_gas,
            penalty=c_penalty,
            total=c_total,
            P_grid_total=p_grid_day,
            G_buy_total=g_buy_day,
            L_shed_total=l_shed_day,
        ))

    # =========================================================
    # 2. 计算年化投资成本 C_CAP
    # =========================================================
    C_CAP = 0.0
    if plan18 and device_catalog:
        C_CAP = calculate_capex(plan18, device_catalog, discount_rate)

    # =========================================================
    # 3. 运行成本合计
    # =========================================================
    C_OP_total = C_elec_total + C_gas_total + C_penalty_total

    # =========================================================
    # 4. 碳成本
    # =========================================================
    C_Carbon, carbon = calculate_carbon_cost(
        E_elec_carbon, E_gas_carbon, carbon_threshold, carbon_price
    )

    # =========================================================
    # 5. 总成本与分数
    # =========================================================
    C_total = C_CAP + C_OP_total + C_Carbon
    Score = calculate_score(C_total)

    # =========================================================
    # 6. 构建返回结果
    # =========================================================
    return AnnualSummary(
        C_CAP=C_CAP,
        C_OP_ele=C_elec_total,
        C_OP_gas=C_gas_total,
        C_OP_penalty=C_penalty_total,
        C_OP_total=C_OP_total,
        C_Carbon=C_Carbon,
        C_total=C_total,
        Score=Score,
        E_grid_total=E_grid_total,
        G_buy_total=G_buy_total,
        L_shed_total=L_shed_total,
        carbon=carbon,
        daily_costs=daily_costs,
        n_days=len(daily_results),
        timestamp=datetime.now().isoformat(),
    )


# ============ 导出函数 ============

def export_summary_json(
    summary: AnnualSummary,
    output_path: str,
    top_n_days: int = 10,
) -> None:
    """
    导出汇总结果为 JSON 文件

    参数:
        summary: AnnualSummary 实例
        output_path: 输出路径
        top_n_days: 前 N 日摘要数量
    """
    # 构建输出结构
    output = {
        "summary": {
            "C_CAP": summary.C_CAP,
            "C_OP_ele": summary.C_OP_ele,
            "C_OP_gas": summary.C_OP_gas,
            "C_OP_penalty": summary.C_OP_penalty,
            "C_OP_total": summary.C_OP_total,
            "C_Carbon": summary.C_Carbon,
            "C_total": summary.C_total,
            "Score": summary.Score,
        },
        "energy": {
            "E_grid_total_MWh": summary.E_grid_total,
            "G_buy_total_m3": summary.G_buy_total,
            "L_shed_total_MWh": summary.L_shed_total,
        },
        "carbon": asdict(summary.carbon),
        "metadata": {
            "n_days": summary.n_days,
            "timestamp": summary.timestamp,
        },
        "top_days": [
            asdict(dc) for dc in summary.daily_costs[:top_n_days]
        ],
    }

    # 写入文件
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
