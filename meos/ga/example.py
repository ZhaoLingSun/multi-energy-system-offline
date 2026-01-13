"""
GA 最小可运行示例

演示 10 个体 × 3 代的 GA 运行（不调用仿真评估）
"""

import numpy as np
from meos.ga import GeneticAlgorithm, GAConfig, Individual


def mock_fitness(ind: Individual) -> float:
    """Mock 适应度函数（不调用仿真）"""
    # 简单测试：最小化 plan18 总和 + price12 方差
    plan_sum = np.sum(ind.plan18)
    price_var = np.var(ind.price12)
    return plan_sum + price_var * 1000


def print_stats(pop):
    """打印统计信息"""
    stats = pop.stats()
    print(f"Gen {stats['gen']:2d}: best={stats['best']:.2f}, "
          f"mean={stats['mean']:.2f}, std={stats['std']:.2f}")


def main():
    """最小可运行示例：10个体×3代"""
    config = GAConfig(
        population_size=10,
        max_generations=3,
        crossover_rate=0.8,
        mutation_rate=0.1,
        tournament_size=3,
        elitism_count=2,
        seed=42,
    )

    ga = GeneticAlgorithm(config=config)
    print("GA 最小示例: 10个体 × 3代")
    print("-" * 40)

    final_pop = ga.run(mock_fitness, callback=print_stats)

    print("-" * 40)
    best = final_pop.best
    print(f"最优个体:")
    print(f"  Plan18: {best.plan18[:6]}... (前6维)")
    print(f"  Price12: {best.price12[:4]}... (前4维)")
    print(f"  Fitness: {best.fitness:.2f}")


if __name__ == "__main__":
    main()
