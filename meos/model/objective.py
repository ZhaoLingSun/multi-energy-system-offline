"""
MEOS Model - 目标函数模块
Phase 2: 日内优化目标函数（购电/购气/失负荷惩罚）

注意：碳排放不进入日内目标函数，仅作为后处理统计。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class CostComponents:
    """成本分项"""
    electricity_purchase: float = 0.0  # 购电成本 (元)
    gas_purchase: float = 0.0          # 购气成本 (元)
    load_shedding: float = 0.0         # 失负荷惩罚 (元)

    @property
    def total(self) -> float:
        """总成本"""
        return self.electricity_purchase + self.gas_purchase + self.load_shedding

    def to_dict(self) -> Dict[str, float]:
        return {
            "electricity_purchase": self.electricity_purchase,
            "gas_purchase": self.gas_purchase,
            "load_shedding": self.load_shedding,
            "total": self.total,
        }


@dataclass
class HourlyCost:
    """单时段成本"""
    hour: int
    electricity_purchase: float = 0.0
    gas_purchase: float = 0.0
    load_shedding: float = 0.0

    @property
    def total(self) -> float:
        return self.electricity_purchase + self.gas_purchase + self.load_shedding


@dataclass
class ObjectiveConfig:
    """目标函数配置"""
    load_shedding_penalty: float = 500000.0  # 失负荷惩罚系数 (元/MWh)
    gas_to_MWh: float = 0.01                 # 天然气转换系数


class ObjectiveFunction:
    """日内优化目标函数"""

    def __init__(self, config: Optional[ObjectiveConfig] = None):
        self.config = config or ObjectiveConfig()

    def compute_hourly(
        self,
        hour: int,
        grid_purchase_MW: float,
        gas_purchase_MW: float,
        load_shed_MW: float,
        elec_price: float,
        gas_price: float,
    ) -> HourlyCost:
        """计算单时段成本"""
        return HourlyCost(
            hour=hour,
            electricity_purchase=grid_purchase_MW * elec_price,
            gas_purchase=gas_purchase_MW * gas_price,
            load_shedding=load_shed_MW * self.config.load_shedding_penalty,
        )

    def compute_daily(
        self,
        hourly_costs: List[HourlyCost],
    ) -> CostComponents:
        """汇总日成本"""
        result = CostComponents()
        for hc in hourly_costs:
            result.electricity_purchase += hc.electricity_purchase
            result.gas_purchase += hc.gas_purchase
            result.load_shedding += hc.load_shedding
        return result
