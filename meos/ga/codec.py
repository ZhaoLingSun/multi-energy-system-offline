"""
GA 编码/解码模块

实现规划向量（18维）+ 引导电价参数（192维）+ 引导气价参数（192维）的编码与解码。
- Plan18: 离散整数编码（设备容量倍数，风/光为 MW 直接值）
- Price192: 连续实数编码（季节×工作日/周末×24小时倍率）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


# ============================================================================
# 默认边界配置 - 对齐 spec/search_space.yaml
# ============================================================================

# Plan18 设备边界 [min_multiple, max_multiple]
# 说明：除风电/光伏外，其余设备不设有效上限（使用较大的整数上界）。
PLAN18_UNBOUNDED_MAX = 4096
PLAN18_BOUNDS = [
    (0, 0),                     # 1: 热电联产A（禁用）
    (0, 0),                     # 2: 热电联产B（禁用）
    (0, 0),                     # 3: 内燃机（禁用）
    (0, 0),                     # 4: 电锅炉（禁用）
    (0, PLAN18_UNBOUNDED_MAX),  # 5: 压缩式制冷机A
    (0, PLAN18_UNBOUNDED_MAX),  # 6: 压缩式制冷机B
    (0, PLAN18_UNBOUNDED_MAX),  # 7: 吸收式制冷机组
    (0, PLAN18_UNBOUNDED_MAX),  # 8: 燃气锅炉
    (0, PLAN18_UNBOUNDED_MAX),  # 9: 地源热泵A
    (0, PLAN18_UNBOUNDED_MAX),  # 10: 地源热泵B
    (0, 0),                     # 11: 电储能（禁用）
    (0, PLAN18_UNBOUNDED_MAX),  # 12: 热储能
    (0, PLAN18_UNBOUNDED_MAX),  # 13: 冷储能
    (0, 500),                   # 14: 风电（上限 500MW）
    (0, 500),                   # 15: 光伏（上限 500MW）
    (0, PLAN18_UNBOUNDED_MAX),  # 16: 电制气
    (0, PLAN18_UNBOUNDED_MAX),  # 17: 燃气轮机
    (0, PLAN18_UNBOUNDED_MAX),  # 18: 冷热电联供
]

# Price192 边界 [min, max]
PRICE12_BOUNDS = [(0.6, 1.5)] * 192
# Gas192 边界 [min, max]
GAS12_BOUNDS = [(0.6, 1.5)] * 192

# Plan18 基准容量（风/光按 MW 直接计）
PLAN18_BASE_CAPACITY = [
    2.0,    # 热电联产A
    8.0,    # 热电联产B
    10.0,   # 内燃机
    2.0,    # 电锅炉
    0.5,    # 压缩式制冷机A
    12.0,   # 压缩式制冷机B
    12.0,   # 吸收式制冷机组
    2.0,    # 燃气锅炉
    2.0,    # 地源热泵A
    10.0,   # 地源热泵B
    2.0,    # 电储能
    40.0,   # 热储能
    40.0,   # 冷储能
    0.482,  # 风电
    0.356,  # 光伏
    0.5,    # 电制气
    5.0,    # 燃气轮机
    3.0,    # 冷热电联供
]


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class CodecConfig:
    """编解码配置"""
    n_plan18: int = 18
    n_price12: int = 192
    n_gas12: int = 192
    plan18_bounds: List[Tuple[int, int]] = field(default_factory=lambda: PLAN18_BOUNDS.copy())
    price12_bounds: List[Tuple[float, float]] = field(default_factory=lambda: PRICE12_BOUNDS.copy())
    gas12_bounds: List[Tuple[float, float]] = field(default_factory=lambda: GAS12_BOUNDS.copy())
    base_capacity: List[float] = field(default_factory=lambda: PLAN18_BASE_CAPACITY.copy())

    @property
    def total_dim(self) -> int:
        return self.n_plan18 + self.n_price12 + self.n_gas12


@dataclass
class Individual:
    """个体表示"""
    plan18: np.ndarray      # (18,) 整数倍数
    price12: np.ndarray     # (12|192,) 连续倍率（电价）
    gas12: np.ndarray       # (12|192,) 连续倍率（气价）
    fitness: float = float('inf')

    @property
    def chromosome(self) -> np.ndarray:
        """合并为染色体"""
        return np.concatenate([self.plan18.astype(float), self.price12, self.gas12])

    def copy(self) -> 'Individual':
        return Individual(
            plan18=self.plan18.copy(),
            price12=self.price12.copy(),
            gas12=self.gas12.copy(),
            fitness=self.fitness
        )


# ============================================================================
# 编解码器
# ============================================================================

class Codec:
    """编解码器"""

    def __init__(self, config: CodecConfig = None):
        self.config = config or CodecConfig()

    def clip_plan18(self, plan18: np.ndarray) -> np.ndarray:
        """裁剪 Plan18 到边界"""
        clipped = plan18.copy()
        for i, (lo, hi) in enumerate(self.config.plan18_bounds):
            clipped[i] = np.clip(int(round(clipped[i])), lo, hi)
        return clipped.astype(int)

    def clip_price12(self, price12: np.ndarray) -> np.ndarray:
        """裁剪 Price12 到边界"""
        clipped = price12.copy()
        for i, (lo, hi) in enumerate(self.config.price12_bounds):
            clipped[i] = np.clip(clipped[i], lo, hi)
        return clipped

    def clip_gas12(self, gas12: np.ndarray) -> np.ndarray:
        """裁剪 Gas12 到边界"""
        clipped = gas12.copy()
        for i, (lo, hi) in enumerate(self.config.gas12_bounds):
            clipped[i] = np.clip(clipped[i], lo, hi)
        return clipped

    def clip_individual(self, ind: Individual) -> Individual:
        """裁剪个体到边界"""
        return Individual(
            plan18=self.clip_plan18(ind.plan18),
            price12=self.clip_price12(ind.price12),
            gas12=self.clip_gas12(ind.gas12),
            fitness=ind.fitness
        )

    def random_plan18(self, rng: np.random.Generator = None) -> np.ndarray:
        """随机生成 Plan18"""
        rng = rng or np.random.default_rng()
        plan18 = np.zeros(self.config.n_plan18, dtype=int)
        for i, (lo, hi) in enumerate(self.config.plan18_bounds):
            plan18[i] = rng.integers(lo, hi + 1)
        return plan18

    def random_price12(self, rng: np.random.Generator = None) -> np.ndarray:
        """随机生成 Price12"""
        rng = rng or np.random.default_rng()
        price12 = np.zeros(self.config.n_price12)
        for i, (lo, hi) in enumerate(self.config.price12_bounds):
            price12[i] = rng.uniform(lo, hi)
        return price12

    def random_gas12(self, rng: np.random.Generator = None) -> np.ndarray:
        """随机生成 Gas12"""
        rng = rng or np.random.default_rng()
        gas12 = np.zeros(self.config.n_gas12)
        for i, (lo, hi) in enumerate(self.config.gas12_bounds):
            gas12[i] = rng.uniform(lo, hi)
        return gas12

    def random_individual(self, rng: np.random.Generator = None) -> Individual:
        """随机生成个体"""
        rng = rng or np.random.default_rng()
        return Individual(
            plan18=self.random_plan18(rng),
            price12=self.random_price12(rng),
            gas12=self.random_gas12(rng),
        )

    def encode(self, plan18: np.ndarray, price12: np.ndarray, gas12: np.ndarray) -> np.ndarray:
        """编码为染色体"""
        return np.concatenate([plan18.astype(float), price12, gas12])

    def decode(self, chromosome: np.ndarray) -> Individual:
        """解码染色体为个体"""
        plan18 = chromosome[:self.config.n_plan18].astype(int)
        price_end = self.config.n_plan18 + self.config.n_price12
        price12 = chromosome[self.config.n_plan18:price_end]
        gas12 = chromosome[price_end:price_end + self.config.n_gas12]
        return self.clip_individual(Individual(plan18=plan18, price12=price12, gas12=gas12))

    def to_capacity(self, plan18: np.ndarray) -> np.ndarray:
        """倍数转实际容量 (MW/MWh)"""
        base = np.array(self.config.base_capacity)
        cap = plan18 * base
        if cap.size >= 15:
            cap[13] = plan18[13]
            cap[14] = plan18[14]
        return cap

    def from_capacity(self, capacity: np.ndarray) -> np.ndarray:
        """实际容量转倍数"""
        base = np.array(self.config.base_capacity)
        plan18 = np.round(capacity / base).astype(int)
        if plan18.size >= 15:
            plan18[13] = int(round(capacity[13]))
            plan18[14] = int(round(capacity[14]))
        return plan18
