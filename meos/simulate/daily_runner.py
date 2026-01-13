#!/usr/bin/env python3
"""
日内仿真并行运行器。

功能：
- 可控并行（joblib/multiprocessing）
- 确定性输出（固定随机种子）
- 运行日志（每日用时、失败日、重试次数）

用法:
    python -m meos.simulate.daily_runner --days 365 --parallel 4
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# 尝试导入并行库
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import multiprocessing as mp
    HAS_MP = True
except ImportError:
    HAS_MP = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================================
# 配置与枚举
# ============================================================================

class ParallelBackend(Enum):
    """并行后端类型"""
    SEQUENTIAL = "sequential"  # 顺序执行
    JOBLIB = "joblib"          # joblib 并行
    MULTIPROCESSING = "multiprocessing"  # multiprocessing 并行


@dataclass
class RunnerConfig:
    """运行器配置"""
    # 并行配置
    parallel_backend: ParallelBackend = ParallelBackend.SEQUENTIAL
    n_jobs: int = 1  # 并行进程数，-1 表示使用所有 CPU

    # 可复现性配置
    random_seed: Optional[int] = 42

    # 重试配置
    max_retries: int = 3
    retry_delay_sec: float = 1.0

    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # 输出配置
    output_dir: str = "output"
    save_intermediate: bool = True


@dataclass
class DayRunLog:
    """单日运行日志"""
    day_index: int
    status: str  # "success", "failed", "skipped"
    elapsed_sec: float = 0.0
    retry_count: int = 0
    error_msg: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day_index": self.day_index,
            "status": self.status,
            "elapsed_sec": self.elapsed_sec,
            "retry_count": self.retry_count,
            "error_msg": self.error_msg,
            "timestamp": self.timestamp,
        }


@dataclass
class RunSummary:
    """运行摘要"""
    total_days: int = 0
    success_count: int = 0
    failed_count: int = 0
    total_elapsed_sec: float = 0.0
    failed_days: List[int] = field(default_factory=list)
    day_logs: List[DayRunLog] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_days": self.total_days,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "total_elapsed_sec": self.total_elapsed_sec,
            "avg_day_sec": self.total_elapsed_sec / max(1, self.total_days),
            "failed_days": self.failed_days,
            "day_logs": [log.to_dict() for log in self.day_logs],
        }


# ============================================================================
# 随机种子管理
# ============================================================================

def set_global_seed(seed: Optional[int]) -> None:
    """设置全局随机种子以保证可复现性"""
    if seed is None:
        return
    random.seed(seed)
    if HAS_NUMPY:
        np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_day_seed(base_seed: Optional[int], day_index: int) -> Optional[int]:
    """为每天生成确定性种子"""
    if base_seed is None:
        return None
    return base_seed + day_index


# ============================================================================
# 日志设置
# ============================================================================

def setup_logger(config: RunnerConfig) -> logging.Logger:
    """配置日志器"""
    logger = logging.getLogger("daily_runner")
    logger.setLevel(getattr(logging, config.log_level.upper()))

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件处理器
    if config.log_file:
        fh = logging.FileHandler(config.log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ============================================================================
# 单日执行函数
# ============================================================================

def run_single_day(
    day_index: int,
    solve_func: Callable[[int], Any],
    config: RunnerConfig,
) -> Tuple[Any, DayRunLog]:
    """
    执行单日仿真（带重试）。

    Args:
        day_index: 日索引
        solve_func: 求解函数，接受 day_index 返回结果
        config: 运行配置

    Returns:
        (result, log): 结果与日志
    """
    # 设置该日的随机种子
    day_seed = get_day_seed(config.random_seed, day_index)
    set_global_seed(day_seed)

    start_time = time.time()
    retry_count = 0
    last_error = None

    for attempt in range(config.max_retries + 1):
        try:
            result = solve_func(day_index)
            elapsed = time.time() - start_time
            log = DayRunLog(
                day_index=day_index,
                status="success",
                elapsed_sec=elapsed,
                retry_count=retry_count,
            )
            return result, log
        except Exception as e:
            last_error = str(e)
            retry_count += 1
            if attempt < config.max_retries:
                time.sleep(config.retry_delay_sec)

    # 所有重试失败
    elapsed = time.time() - start_time
    log = DayRunLog(
        day_index=day_index,
        status="failed",
        elapsed_sec=elapsed,
        retry_count=retry_count,
        error_msg=last_error,
    )
    return None, log


# ============================================================================
# 日内仿真运行器
# ============================================================================

class DailyRunner:
    """
    日内仿真并行运行器。

    支持：
    - 顺序执行 / joblib 并行 / multiprocessing 并行
    - 固定随机种子保证可复现
    - 自动重试与日志记录
    """

    def __init__(self, config: Optional[RunnerConfig] = None):
        self.config = config or RunnerConfig()
        self.logger = setup_logger(self.config)
        self._validate_backend()

    def _validate_backend(self) -> None:
        """验证并行后端可用性"""
        backend = self.config.parallel_backend
        if backend == ParallelBackend.JOBLIB and not HAS_JOBLIB:
            self.logger.warning("joblib 不可用，回退到顺序执行")
            self.config.parallel_backend = ParallelBackend.SEQUENTIAL
        elif backend == ParallelBackend.MULTIPROCESSING and not HAS_MP:
            self.logger.warning("multiprocessing 不可用，回退到顺序执行")
            self.config.parallel_backend = ParallelBackend.SEQUENTIAL

    def run(
        self,
        day_indices: List[int],
        solve_func: Callable[[int], Any],
    ) -> RunSummary:
        """运行多日仿真"""
        set_global_seed(self.config.random_seed)
        total_start = time.time()

        self.logger.info(
            f"开始运行 {len(day_indices)} 天仿真, "
            f"后端: {self.config.parallel_backend.value}, "
            f"进程数: {self.config.n_jobs}"
        )

        backend = self.config.parallel_backend
        if backend == ParallelBackend.SEQUENTIAL:
            results = self._run_sequential(day_indices, solve_func)
        elif backend == ParallelBackend.JOBLIB:
            results = self._run_joblib(day_indices, solve_func)
        else:
            results = self._run_multiprocessing(day_indices, solve_func)

        summary = self._build_summary(results, time.time() - total_start)
        self._log_summary(summary)
        self._save_summary(summary)
        return summary

    def _run_sequential(
        self, day_indices: List[int], solve_func: Callable[[int], Any]
    ) -> List[Tuple[Any, DayRunLog]]:
        """顺序执行"""
        results = []
        for i, day_idx in enumerate(day_indices):
            self.logger.debug(f"处理第 {day_idx} 天 ({i+1}/{len(day_indices)})")
            result, log = run_single_day(day_idx, solve_func, self.config)
            results.append((result, log))
        return results

    def _run_joblib(
        self, day_indices: List[int], solve_func: Callable[[int], Any]
    ) -> List[Tuple[Any, DayRunLog]]:
        """joblib 并行执行"""
        results = Parallel(n_jobs=self.config.n_jobs)(
            delayed(run_single_day)(day_idx, solve_func, self.config)
            for day_idx in day_indices
        )
        return results

    def _run_multiprocessing(
        self, day_indices: List[int], solve_func: Callable[[int], Any]
    ) -> List[Tuple[Any, DayRunLog]]:
        """multiprocessing 并行执行"""
        n_jobs = self.config.n_jobs
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        with mp.Pool(n_jobs) as pool:
            args = [(day_idx, solve_func, self.config) for day_idx in day_indices]
            results = pool.starmap(run_single_day, args)
        return results

    def _build_summary(
        self, results: List[Tuple[Any, DayRunLog]], total_elapsed: float
    ) -> RunSummary:
        """构建运行摘要"""
        logs = [r[1] for r in results]
        failed_days = [log.day_index for log in logs if log.status == "failed"]
        return RunSummary(
            total_days=len(results),
            success_count=sum(1 for log in logs if log.status == "success"),
            failed_count=len(failed_days),
            total_elapsed_sec=total_elapsed,
            failed_days=failed_days,
            day_logs=logs,
        )

    def _log_summary(self, summary: RunSummary) -> None:
        """记录运行摘要"""
        self.logger.info(f"运行完成: {summary.success_count}/{summary.total_days} 成功")
        self.logger.info(f"总耗时: {summary.total_elapsed_sec:.2f}s")
        if summary.failed_days:
            self.logger.warning(f"失败日: {summary.failed_days}")

    def _save_summary(self, summary: RunSummary) -> None:
        """保存运行摘要"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_file = output_dir / "run_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)


# ============================================================================
# 仿真配置类
# ============================================================================

@dataclass
class SimConfig:
    """年度仿真配置"""
    # 时间范围
    year: int = 2024
    start_day: int = 1          # 起始日（1-365）
    end_day: int = 365          # 结束日（1-365）
    hours_per_day: int = 24

    # 数据路径
    topology_path: str = ""
    attributes_path: str = ""
    timeseries_dir: str = ""

    # 缓存配置
    cache_dir: str = "meos/simulate/cache"
    cache_format: str = "json"  # "json" or "parquet"

    # 运行配置
    runner_config: RunnerConfig = field(default_factory=RunnerConfig)
    verbose: bool = True
    save_hourly: bool = True

    def validate(self) -> None:
        """校验配置"""
        if not 1 <= self.start_day <= 365:
            raise ValueError(f"start_day must be 1-365, got {self.start_day}")
        if not 1 <= self.end_day <= 365:
            raise ValueError(f"end_day must be 1-365, got {self.end_day}")
        if self.start_day > self.end_day:
            raise ValueError("start_day must <= end_day")

    @property
    def n_days(self) -> int:
        """仿真天数"""
        return self.end_day - self.start_day + 1

    @property
    def day_range(self) -> range:
        """日期范围迭代器"""
        return range(self.start_day, self.end_day + 1)


# ============================================================================
# 日内结果类
# ============================================================================

@dataclass
class HourlyResult:
    """单时段结果"""
    hour: int
    P_grid: Dict[str, float] = field(default_factory=dict)
    P_thermal: Dict[str, float] = field(default_factory=dict)
    device_outputs: Dict[str, float] = field(default_factory=dict)
    load_shed: Dict[str, float] = field(default_factory=dict)
    cost: float = 0.0


@dataclass
class DailyResult:
    """单日仿真结果"""
    day: int
    year: int
    status: str = "success"
    hourly: List[HourlyResult] = field(default_factory=list)
    total_cost: float = 0.0
    total_grid_purchase: float = 0.0
    total_gas_purchase: float = 0.0
    total_load_shed: float = 0.0
    solve_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典"""
        return {
            "day": self.day,
            "year": self.year,
            "status": self.status,
            "total_cost": self.total_cost,
            "total_grid_purchase": self.total_grid_purchase,
            "total_gas_purchase": self.total_gas_purchase,
            "total_load_shed": self.total_load_shed,
            "solve_time_sec": self.solve_time_sec,
            "hourly": [asdict(h) for h in self.hourly] if self.hourly else [],
        }


# ============================================================================
# 缓存管理
# ============================================================================

class CacheManager:
    """日内结果缓存管理"""

    def __init__(self, cache_dir: str, cache_format: str = "json"):
        self.cache_dir = Path(cache_dir)
        self.cache_format = cache_format
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """确保缓存目录存在"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, year: int, day: int) -> Path:
        """获取缓存文件路径"""
        ext = "json" if self.cache_format == "json" else "parquet"
        return self.cache_dir / f"day_{year}_{day:03d}.{ext}"

    def save_daily(self, result: DailyResult) -> None:
        """保存单日结果"""
        path = self.get_cache_path(result.year, result.day)
        data = result.to_dict()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_daily(self, year: int, day: int) -> Optional[DailyResult]:
        """加载单日结果"""
        path = self.get_cache_path(year, day)
        if not path.exists():
            return None
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return DailyResult(
            day=data["day"],
            year=data["year"],
            status=data["status"],
            total_cost=data["total_cost"],
            total_grid_purchase=data["total_grid_purchase"],
            total_gas_purchase=data["total_gas_purchase"],
            total_load_shed=data["total_load_shed"],
            solve_time_sec=data["solve_time_sec"],
        )

    def exists(self, year: int, day: int) -> bool:
        """检查缓存是否存在"""
        return self.get_cache_path(year, day).exists()


# ============================================================================
# 年度仿真主入口
# ============================================================================

@dataclass
class YearResult:
    """年度仿真结果汇总"""
    year: int
    start_day: int
    end_day: int
    total_cost: float = 0.0
    daily_results: List[DailyResult] = field(default_factory=list)
    run_summary: Optional[RunSummary] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "start_day": self.start_day,
            "end_day": self.end_day,
            "total_cost": self.total_cost,
            "n_days": len(self.daily_results),
        }


def run_year(
    sim_config: SimConfig,
    solve_day_func: Optional[Callable[[int, int], DailyResult]] = None,
) -> YearResult:
    """
    年度仿真主入口。

    Args:
        sim_config: 仿真配置
        solve_day_func: 单日求解函数，签名 (year, day) -> DailyResult
                       若为 None，使用默认 mock 函数

    Returns:
        YearResult: 年度仿真结果
    """
    sim_config.validate()

    logger = logging.getLogger("daily_runner")
    if sim_config.verbose:
        logger.setLevel(logging.INFO)

    cache = CacheManager(sim_config.cache_dir, sim_config.cache_format)
    runner = DailyRunner(sim_config.runner_config)

    # 默认 mock 求解函数
    if solve_day_func is None:
        solve_day_func = _mock_solve_day

    year_result = YearResult(
        year=sim_config.year,
        start_day=sim_config.start_day,
        end_day=sim_config.end_day,
    )

    summary = RunSummary(total_days=sim_config.n_days)
    start_time = time.time()

    if sim_config.verbose:
        logger.info(f"开始仿真: {sim_config.year} 年")
        logger.info(f"日期范围: {sim_config.start_day}-{sim_config.end_day}")

    for day in sim_config.day_range:
        # 调用单日求解
        def _solve(d: int) -> DailyResult:
            return solve_day_func(sim_config.year, d)

        result, log = run_single_day(day, _solve, sim_config.runner_config)
        summary.day_logs.append(log)

        if log.status == "success" and result:
            cache.save_daily(result)
            year_result.daily_results.append(result)
            year_result.total_cost += result.total_cost
            summary.success_count += 1
        else:
            summary.failed_count += 1
            summary.failed_days.append(day)

        if sim_config.verbose and day % 30 == 0:
            logger.info(f"  已完成 {day}/{sim_config.end_day} 天")

    summary.total_elapsed_sec = time.time() - start_time
    year_result.run_summary = summary

    if sim_config.verbose:
        logger.info(f"仿真完成: 成功 {summary.success_count} 天")
        logger.info(f"总耗时: {summary.total_elapsed_sec:.1f} 秒")

    return year_result


def _mock_solve_day(year: int, day: int) -> DailyResult:
    """Mock 单日求解函数（用于测试）"""
    import random
    result = DailyResult(day=day, year=year)
    result.total_cost = random.uniform(10000, 50000)
    result.total_grid_purchase = random.uniform(100, 500)
    result.total_gas_purchase = random.uniform(50, 200)
    result.solve_time_sec = random.uniform(0.1, 0.5)
    return result
