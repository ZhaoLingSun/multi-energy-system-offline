"""
GA 核心模块

实现遗传算法的核心操作：
- 种群初始化
- 选择（锦标赛选择）
- 交叉（混合交叉）
- 变异（高斯变异 + 整数变异）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import numpy as np

from .codec import Codec, CodecConfig, Individual


# ============================================================================
# GA 配置
# ============================================================================

@dataclass
class GAConfig:
    """GA 配置参数"""
    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 3
    elitism_count: int = 2
    seed: Optional[int] = 42

    # 变异参数
    plan18_mutation_sigma: float = 5.0
    price12_mutation_sigma: float = 0.1
    gas12_mutation_sigma: float = 0.1


# ============================================================================
# 种群类
# ============================================================================

@dataclass
class Population:
    """种群"""
    individuals: List[Individual] = field(default_factory=list)
    generation: int = 0

    @property
    def size(self) -> int:
        return len(self.individuals)

    @property
    def best(self) -> Individual:
        return min(self.individuals, key=lambda x: x.fitness)

    @property
    def worst(self) -> Individual:
        return max(self.individuals, key=lambda x: x.fitness)

    def stats(self) -> dict:
        """统计信息"""
        fits = [ind.fitness for ind in self.individuals]
        return {
            "gen": self.generation,
            "best": min(fits),
            "worst": max(fits),
            "mean": np.mean(fits),
            "std": np.std(fits),
        }


# ============================================================================
# GA 核心类
# ============================================================================

class GeneticAlgorithm:
    """遗传算法核心"""

    def __init__(
        self,
        config: GAConfig = None,
        codec_config: CodecConfig = None,
    ):
        self.config = config or GAConfig()
        self.codec = Codec(codec_config)
        self.rng = np.random.default_rng(self.config.seed)

    def init_population(self) -> Population:
        """初始化种群"""
        individuals = [
            self.codec.random_individual(self.rng)
            for _ in range(self.config.population_size)
        ]
        return Population(individuals=individuals, generation=0)

    def tournament_select(self, pop: Population) -> Individual:
        """锦标赛选择"""
        candidates = self.rng.choice(
            pop.individuals,
            size=self.config.tournament_size,
            replace=False
        )
        return min(candidates, key=lambda x: x.fitness).copy()

    def crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """混合交叉（SBX for continuous, uniform for discrete）"""
        if self.rng.random() > self.config.crossover_rate:
            return p1.copy(), p2.copy()

        # Plan18: 均匀交叉
        c1_plan = p1.plan18.copy()
        c2_plan = p2.plan18.copy()
        n_plan = self.codec.config.n_plan18
        mask = self.rng.random(n_plan) < 0.5
        c1_plan[mask], c2_plan[mask] = c2_plan[mask], c1_plan[mask]

        # Price: 算术交叉
        n_price = self.codec.config.n_price12
        alpha = self.rng.random(n_price)
        c1_price = alpha * p1.price12 + (1 - alpha) * p2.price12
        c2_price = (1 - alpha) * p1.price12 + alpha * p2.price12

        # Gas: 算术交叉
        n_gas = self.codec.config.n_gas12
        beta = self.rng.random(n_gas)
        c1_gas = beta * p1.gas12 + (1 - beta) * p2.gas12
        c2_gas = (1 - beta) * p1.gas12 + beta * p2.gas12

        return (
            Individual(plan18=c1_plan, price12=c1_price, gas12=c1_gas),
            Individual(plan18=c2_plan, price12=c2_price, gas12=c2_gas)
        )

    def mutate(self, ind: Individual) -> Individual:
        """变异操作"""
        plan18 = ind.plan18.copy()
        price12 = ind.price12.copy()
        gas12 = ind.gas12.copy()

        # Plan18: 整数高斯变异
        for i in range(self.codec.config.n_plan18):
            if self.rng.random() < self.config.mutation_rate:
                delta = int(round(self.rng.normal(0, self.config.plan18_mutation_sigma)))
                plan18[i] += delta

        # Price: 高斯变异
        for i in range(self.codec.config.n_price12):
            if self.rng.random() < self.config.mutation_rate:
                price12[i] += self.rng.normal(0, self.config.price12_mutation_sigma)

        # Gas: 高斯变异
        for i in range(self.codec.config.n_gas12):
            if self.rng.random() < self.config.mutation_rate:
                gas12[i] += self.rng.normal(0, self.config.gas12_mutation_sigma)

        return self.codec.clip_individual(Individual(plan18=plan18, price12=price12, gas12=gas12))

    def evolve(self, pop: Population) -> Population:
        """进化一代"""
        # 精英保留
        sorted_inds = sorted(pop.individuals, key=lambda x: x.fitness)
        new_inds = [ind.copy() for ind in sorted_inds[:self.config.elitism_count]]

        # 生成子代
        while len(new_inds) < self.config.population_size:
            p1 = self.tournament_select(pop)
            p2 = self.tournament_select(pop)
            c1, c2 = self.crossover(p1, p2)
            c1 = self.mutate(c1)
            c2 = self.mutate(c2)
            new_inds.extend([c1, c2])

        return Population(
            individuals=new_inds[:self.config.population_size],
            generation=pop.generation + 1
        )

    def run(
        self,
        fitness_func: Callable[[Individual], float],
        callback: Callable[[Population], None] = None,
    ) -> Population:
        """运行 GA"""
        pop = self.init_population()

        # 评估初始种群
        for ind in pop.individuals:
            ind.fitness = fitness_func(ind)

        if callback:
            callback(pop)

        # 进化循环
        for _ in range(self.config.max_generations):
            pop = self.evolve(pop)
            for ind in pop.individuals:
                if ind.fitness == float('inf'):
                    ind.fitness = fitness_func(ind)
            if callback:
                callback(pop)

        return pop
