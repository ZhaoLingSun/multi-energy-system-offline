"""
设备与储能约束模块。

定义转换设备（P2G、锅炉、热泵、CCHP等）与储能SOC约束。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DeviceType(Enum):
    """设备类型枚举"""
    # 转换设备
    P2G = "P2G"                          # 电制气
    ELECTRIC_BOILER = "ElectricBoiler"   # 电锅炉
    GAS_BOILER = "GasBoiler"             # 燃气锅炉
    HEAT_PUMP = "HeatPump"               # 热泵
    GAS_TURBINE = "GasTurbine"           # 燃气轮机
    CHP = "CHP"                          # 热电联产
    CCHP = "CCHP"                        # 冷热电联供
    CHILLER = "Chiller"                  # 压缩式制冷机
    ABSORPTION_CHILLER = "AbsorptionChiller"  # 吸收式制冷机
    ICE = "ICE"                          # 内燃机
    # 储能设备
    BATTERY = "BatteryStorage"           # 电储能
    THERMAL_STORAGE = "ThermalStorage"   # 热储能
    COLD_STORAGE = "ColdStorage"         # 冷储能
    # 可再生能源
    PV = "PV"                            # 光伏
    WIND = "WindTurbine"                 # 风电


class Medium(Enum):
    """介质类型"""
    ELECTRICITY = "Electricity"
    GAS = "Gas"
    HEAT = "Heat"
    COOLING = "Cooling"


@dataclass
class Variable:
    """优化变量定义"""
    name: str
    device_id: str
    medium: Medium
    direction: str  # "input" or "output"
    time_indexed: bool = True
    lower_bound: float = 0.0
    upper_bound: Optional[float] = None


@dataclass
class Constraint:
    """约束定义"""
    name: str
    expression: str  # 约束表达式描述
    constraint_type: str  # "eq" (等式) or "ineq" (不等式)


@dataclass
class DeviceParams:
    """设备参数"""
    device_id: str
    device_type: DeviceType
    capacity: float  # MW or MWh
    efficiency: Dict[str, float] = field(default_factory=dict)


class DeviceConstraintBase(ABC):
    """设备约束基类"""

    def __init__(self, params: DeviceParams):
        self.params = params
        self._variables: List[Variable] = []
        self._constraints: List[Constraint] = []

    @property
    def device_id(self) -> str:
        return self.params.device_id

    @property
    def capacity(self) -> float:
        return self.params.capacity

    @abstractmethod
    def define_variables(self) -> List[Variable]:
        """定义设备变量"""
        pass

    @abstractmethod
    def define_constraints(self) -> List[Constraint]:
        """定义设备约束"""
        pass

    def get_variables(self) -> List[Variable]:
        if not self._variables:
            self._variables = self.define_variables()
        return self._variables

    def get_constraints(self) -> List[Constraint]:
        if not self._constraints:
            self._constraints = self.define_constraints()
        return self._constraints


# ============================================================
# 转换设备约束
# ============================================================

class P2GConstraint(DeviceConstraintBase):
    """
    电制气 (Power-to-Gas) 约束。

    输入: 电力 P_e_in
    输出: 天然气 P_g_out
    效率: η_p2g (典型值 0.40)

    约束:
    - P_g_out = η_p2g * P_e_in
    - 0 ≤ P_e_in ≤ capacity
    """

    def define_variables(self) -> List[Variable]:
        return [
            Variable(
                name=f"{self.device_id}_P_e_in",
                device_id=self.device_id,
                medium=Medium.ELECTRICITY,
                direction="input",
                upper_bound=self.capacity,
            ),
            Variable(
                name=f"{self.device_id}_P_g_out",
                device_id=self.device_id,
                medium=Medium.GAS,
                direction="output",
            ),
        ]

    def define_constraints(self) -> List[Constraint]:
        eta = self.params.efficiency.get("conversion", 0.40)
        return [
            Constraint(
                name=f"{self.device_id}_efficiency",
                expression=f"P_g_out[t] == {eta} * P_e_in[t]",
                constraint_type="eq",
            ),
        ]


class ElectricBoilerConstraint(DeviceConstraintBase):
    """
    电锅炉约束。

    输入: 电力 P_e_in
    输出: 热量 P_h_out
    效率: η_eb (典型值 0.90)

    约束:
    - P_h_out = η_eb * P_e_in
    - 0 ≤ P_e_in ≤ capacity
    """

    def define_variables(self) -> List[Variable]:
        return [
            Variable(
                name=f"{self.device_id}_P_e_in",
                device_id=self.device_id,
                medium=Medium.ELECTRICITY,
                direction="input",
                upper_bound=self.capacity,
            ),
            Variable(
                name=f"{self.device_id}_P_h_out",
                device_id=self.device_id,
                medium=Medium.HEAT,
                direction="output",
            ),
        ]

    def define_constraints(self) -> List[Constraint]:
        eta = self.params.efficiency.get("thermal", 0.90)
        return [
            Constraint(
                name=f"{self.device_id}_efficiency",
                expression=f"P_h_out[t] == {eta} * P_e_in[t]",
                constraint_type="eq",
            ),
        ]


class GasBoilerConstraint(DeviceConstraintBase):
    """
    燃气锅炉约束。

    输入: 天然气 P_g_in
    输出: 热量 P_h_out
    效率: η_gb (典型值 0.95)
    """

    def define_variables(self) -> List[Variable]:
        return [
            Variable(
                name=f"{self.device_id}_P_g_in",
                device_id=self.device_id,
                medium=Medium.GAS,
                direction="input",
                upper_bound=self.capacity,
            ),
            Variable(
                name=f"{self.device_id}_P_h_out",
                device_id=self.device_id,
                medium=Medium.HEAT,
                direction="output",
            ),
        ]

    def define_constraints(self) -> List[Constraint]:
        eta = self.params.efficiency.get("thermal", 0.95)
        return [
            Constraint(
                name=f"{self.device_id}_efficiency",
                expression=f"P_h_out[t] == {eta} * P_g_in[t]",
                constraint_type="eq",
            ),
        ]


class HeatPumpConstraint(DeviceConstraintBase):
    """
    热泵约束。

    输入: 电力 P_e_in
    输出: 热量 P_h_out
    COP: 制热系数 (典型值 5.0-6.0)

    约束:
    - P_h_out = COP * P_e_in
    - 0 ≤ P_e_in ≤ capacity
    """

    def define_variables(self) -> List[Variable]:
        return [
            Variable(
                name=f"{self.device_id}_P_e_in",
                device_id=self.device_id,
                medium=Medium.ELECTRICITY,
                direction="input",
                upper_bound=self.capacity,
            ),
            Variable(
                name=f"{self.device_id}_P_h_out",
                device_id=self.device_id,
                medium=Medium.HEAT,
                direction="output",
            ),
        ]

    def define_constraints(self) -> List[Constraint]:
        cop = self.params.efficiency.get("COP", 5.0)
        return [
            Constraint(
                name=f"{self.device_id}_cop",
                expression=f"P_h_out[t] == {cop} * P_e_in[t]",
                constraint_type="eq",
            ),
        ]


class ChillerConstraint(DeviceConstraintBase):
    """
    压缩式制冷机约束。

    输入: 电力 P_e_in
    输出: 冷量 P_c_out
    COP: 制冷系数 (典型值 4.0-5.0)
    """

    def define_variables(self) -> List[Variable]:
        return [
            Variable(
                name=f"{self.device_id}_P_e_in",
                device_id=self.device_id,
                medium=Medium.ELECTRICITY,
                direction="input",
                upper_bound=self.capacity,
            ),
            Variable(
                name=f"{self.device_id}_P_c_out",
                device_id=self.device_id,
                medium=Medium.COOLING,
                direction="output",
            ),
        ]

    def define_constraints(self) -> List[Constraint]:
        cop = self.params.efficiency.get("COP", 4.0)
        return [
            Constraint(
                name=f"{self.device_id}_cop",
                expression=f"P_c_out[t] == {cop} * P_e_in[t]",
                constraint_type="eq",
            ),
        ]


class AbsorptionChillerConstraint(DeviceConstraintBase):
    """
    吸收式制冷机约束。

    输入: 热量 P_h_in
    输出: 冷量 P_c_out
    COP: 制冷系数 (典型值 0.8)
    """

    def define_variables(self) -> List[Variable]:
        return [
            Variable(
                name=f"{self.device_id}_P_h_in",
                device_id=self.device_id,
                medium=Medium.HEAT,
                direction="input",
                upper_bound=self.capacity,
            ),
            Variable(
                name=f"{self.device_id}_P_c_out",
                device_id=self.device_id,
                medium=Medium.COOLING,
                direction="output",
            ),
        ]

    def define_constraints(self) -> List[Constraint]:
        cop = self.params.efficiency.get("COP", 0.8)
        return [
            Constraint(
                name=f"{self.device_id}_cop",
                expression=f"P_c_out[t] == {cop} * P_h_in[t]",
                constraint_type="eq",
            ),
        ]


class GasTurbineConstraint(DeviceConstraintBase):
    """
    燃气轮机约束。

    输入: 天然气 P_g_in
    输出: 电力 P_e_out
    效率: η_gt (典型值 0.70)
    """

    def define_variables(self) -> List[Variable]:
        return [
            Variable(
                name=f"{self.device_id}_P_g_in",
                device_id=self.device_id,
                medium=Medium.GAS,
                direction="input",
                upper_bound=self.capacity,
            ),
            Variable(
                name=f"{self.device_id}_P_e_out",
                device_id=self.device_id,
                medium=Medium.ELECTRICITY,
                direction="output",
            ),
        ]

    def define_constraints(self) -> List[Constraint]:
        eta = self.params.efficiency.get("electric", 0.70)
        return [
            Constraint(
                name=f"{self.device_id}_efficiency",
                expression=f"P_e_out[t] == {eta} * P_g_in[t]",
                constraint_type="eq",
            ),
        ]


class CHPConstraint(DeviceConstraintBase):
    """
    热电联产 (CHP) 约束。

    输入: 天然气 P_g_in
    输出: 电力 P_e_out, 热量 P_h_out
    效率: η_e (发电), η_h (供热)
    """

    def define_variables(self) -> List[Variable]:
        return [
            Variable(
                name=f"{self.device_id}_P_g_in",
                device_id=self.device_id,
                medium=Medium.GAS,
                direction="input",
                upper_bound=self.capacity,
            ),
            Variable(
                name=f"{self.device_id}_P_e_out",
                device_id=self.device_id,
                medium=Medium.ELECTRICITY,
                direction="output",
            ),
            Variable(
                name=f"{self.device_id}_P_h_out",
                device_id=self.device_id,
                medium=Medium.HEAT,
                direction="output",
            ),
        ]

    def define_constraints(self) -> List[Constraint]:
        eta_e = self.params.efficiency.get("electric", 0.25)
        eta_h = self.params.efficiency.get("thermal", 0.55)
        return [
            Constraint(
                name=f"{self.device_id}_elec_eff",
                expression=f"P_e_out[t] == {eta_e} * P_g_in[t]",
                constraint_type="eq",
            ),
            Constraint(
                name=f"{self.device_id}_heat_eff",
                expression=f"P_h_out[t] == {eta_h} * P_g_in[t]",
                constraint_type="eq",
            ),
        ]


class CCHPConstraint(DeviceConstraintBase):
    """
    冷热电联供 (CCHP) 约束。

    输入: 天然气 P_g_in
    输出: 电力 P_e_out, 热量 P_h_out, 冷量 P_c_out
    效率: η_e (发电), η_h (供热), η_c (制冷)
    """

    def define_variables(self) -> List[Variable]:
        return [
            Variable(
                name=f"{self.device_id}_P_g_in",
                device_id=self.device_id,
                medium=Medium.GAS,
                direction="input",
                upper_bound=self.capacity,
            ),
            Variable(
                name=f"{self.device_id}_P_e_out",
                device_id=self.device_id,
                medium=Medium.ELECTRICITY,
                direction="output",
            ),
            Variable(
                name=f"{self.device_id}_P_h_out",
                device_id=self.device_id,
                medium=Medium.HEAT,
                direction="output",
            ),
            Variable(
                name=f"{self.device_id}_P_c_out",
                device_id=self.device_id,
                medium=Medium.COOLING,
                direction="output",
            ),
        ]

    def define_constraints(self) -> List[Constraint]:
        eta_e = self.params.efficiency.get("electric", 0.40)
        eta_h = self.params.efficiency.get("thermal", 0.30)
        eta_c = self.params.efficiency.get("cooling", 0.30)
        return [
            Constraint(
                name=f"{self.device_id}_elec_eff",
                expression=f"P_e_out[t] == {eta_e} * P_g_in[t]",
                constraint_type="eq",
            ),
            Constraint(
                name=f"{self.device_id}_heat_eff",
                expression=f"P_h_out[t] == {eta_h} * P_g_in[t]",
                constraint_type="eq",
            ),
            Constraint(
                name=f"{self.device_id}_cool_eff",
                expression=f"P_c_out[t] == {eta_c} * P_g_in[t]",
                constraint_type="eq",
            ),
        ]


# ============================================================
# 储能设备约束
# ============================================================

@dataclass
class StorageParams(DeviceParams):
    """储能设备参数"""
    energy_capacity: float = 0.0  # MWh
    power_capacity: float = 0.0   # MW (充放能功率)
    soc_min: float = 0.0          # 最小 SOC
    soc_max: float = 1.0          # 最大 SOC
    charge_eff: float = 0.95      # 充能效率
    discharge_eff: float = 0.95   # 放能效率


class StorageConstraintBase(DeviceConstraintBase):
    """
    储能设备约束基类。

    变量:
    - P_ch[t]: 充能功率
    - P_dis[t]: 放能功率
    - SOC[t]: 能量状态

    约束:
    - SOC 动态: SOC[t+1] = SOC[t] + η_ch*P_ch[t]*Δt - P_dis[t]/η_dis*Δt
    - SOC 范围: SOC_min ≤ SOC[t] ≤ SOC_max
    - 功率限制: 0 ≤ P_ch[t] ≤ P_max, 0 ≤ P_dis[t] ≤ P_max
    - 日内首尾相等: SOC[0] = SOC[T]
    """

    def __init__(self, params: StorageParams, num_periods: int = 24):
        super().__init__(params)
        self.storage_params = params
        self.num_periods = num_periods

    @property
    def energy_capacity(self) -> float:
        return self.storage_params.energy_capacity

    @property
    def power_capacity(self) -> float:
        return self.storage_params.power_capacity

    @property
    @abstractmethod
    def medium(self) -> Medium:
        """储能介质类型"""
        pass

    def define_variables(self) -> List[Variable]:
        return [
            Variable(
                name=f"{self.device_id}_P_ch",
                device_id=self.device_id,
                medium=self.medium,
                direction="input",
                upper_bound=self.power_capacity,
            ),
            Variable(
                name=f"{self.device_id}_P_dis",
                device_id=self.device_id,
                medium=self.medium,
                direction="output",
                upper_bound=self.power_capacity,
            ),
            Variable(
                name=f"{self.device_id}_SOC",
                device_id=self.device_id,
                medium=self.medium,
                direction="state",
                lower_bound=self.storage_params.soc_min * self.energy_capacity,
                upper_bound=self.storage_params.soc_max * self.energy_capacity,
            ),
        ]

    def define_constraints(self) -> List[Constraint]:
        eta_ch = self.storage_params.charge_eff
        eta_dis = self.storage_params.discharge_eff
        soc_min = self.storage_params.soc_min * self.energy_capacity
        soc_max = self.storage_params.soc_max * self.energy_capacity

        constraints = [
            # SOC 动态约束
            Constraint(
                name=f"{self.device_id}_soc_dynamics",
                expression=f"SOC[t+1] == SOC[t] + {eta_ch}*P_ch[t] - P_dis[t]/{eta_dis}",
                constraint_type="eq",
            ),
            # SOC 下限
            Constraint(
                name=f"{self.device_id}_soc_min",
                expression=f"SOC[t] >= {soc_min}",
                constraint_type="ineq",
            ),
            # SOC 上限
            Constraint(
                name=f"{self.device_id}_soc_max",
                expression=f"SOC[t] <= {soc_max}",
                constraint_type="ineq",
            ),
            # 日内首尾相等
            Constraint(
                name=f"{self.device_id}_soc_cyclic",
                expression="SOC[0] == SOC[T]",
                constraint_type="eq",
            ),
        ]
        return constraints


class BatteryStorageConstraint(StorageConstraintBase):
    """电储能约束"""

    @property
    def medium(self) -> Medium:
        return Medium.ELECTRICITY


class ThermalStorageConstraint(StorageConstraintBase):
    """热储能约束"""

    @property
    def medium(self) -> Medium:
        return Medium.HEAT


class ColdStorageConstraint(StorageConstraintBase):
    """冷储能约束"""

    @property
    def medium(self) -> Medium:
        return Medium.COOLING


# ============================================================
# 设备约束工厂
# ============================================================

DEVICE_CONSTRAINT_MAP = {
    DeviceType.P2G: P2GConstraint,
    DeviceType.ELECTRIC_BOILER: ElectricBoilerConstraint,
    DeviceType.GAS_BOILER: GasBoilerConstraint,
    DeviceType.HEAT_PUMP: HeatPumpConstraint,
    DeviceType.CHILLER: ChillerConstraint,
    DeviceType.ABSORPTION_CHILLER: AbsorptionChillerConstraint,
    DeviceType.GAS_TURBINE: GasTurbineConstraint,
    DeviceType.CHP: CHPConstraint,
    DeviceType.CCHP: CCHPConstraint,
    DeviceType.BATTERY: BatteryStorageConstraint,
    DeviceType.THERMAL_STORAGE: ThermalStorageConstraint,
    DeviceType.COLD_STORAGE: ColdStorageConstraint,
}


def create_device_constraint(
    params: DeviceParams | StorageParams,
) -> DeviceConstraintBase:
    """根据设备类型创建约束实例"""
    constraint_cls = DEVICE_CONSTRAINT_MAP.get(params.device_type)
    if constraint_cls is None:
        raise ValueError(f"未知设备类型: {params.device_type}")
    return constraint_cls(params)
