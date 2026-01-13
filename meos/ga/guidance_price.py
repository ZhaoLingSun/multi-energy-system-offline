"""
引导电价生成器

将 12 维引导参数生成 365×24 价格矩阵。
对齐 MATLAB generate_guided_price.m 与 seasonal_tou_pricing.m 逻辑。

12 维参数结构:
- multipliers[0:4]: 峰时段倍率 (春/夏/秋/冬)
- multipliers[4:8]: 平时段倍率 (春/夏/秋/冬)
- multipliers[8:12]: 谷时段倍率 (春/夏/秋/冬)

季节定义:
- 春: 3-5月 (season_idx=0)
- 夏: 6-8月 (season_idx=1)
- 秋: 9-11月 (season_idx=2)
- 冬: 12,1,2月 (season_idx=3)

时段分类:
- peak(1): 价格最高的8小时
- flat(2): 中间8小时
- valley(3): 价格最低的8小时
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# 常量定义
# ============================================================================

# 时间维度
HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365
HOURS_PER_YEAR = HOURS_PER_DAY * DAYS_PER_YEAR

# 季节定义
N_SEASONS = 4
SEASON_NAMES = ['spring', 'summer', 'autumn', 'winter']
SEASON_MONTHS = {
    0: [3, 4, 5],      # 春
    1: [6, 7, 8],      # 夏
    2: [9, 10, 11],    # 秋
    3: [12, 1, 2],     # 冬
}

# 时段定义
SEGMENT_PEAK = 1
SEGMENT_FLAT = 2
SEGMENT_VALLEY = 3
SEGMENT_NAMES = {1: 'peak', 2: 'flat', 3: 'valley'}

# 每个时段的小时数
HOURS_PER_SEGMENT = 8

# 12维参数索引
N_MULTIPLIERS = 12
IDX_PEAK = slice(0, 4)      # 峰时段倍率
IDX_FLAT = slice(4, 8)      # 平时段倍率
IDX_VALLEY = slice(8, 12)   # 谷时段倍率


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class SegmentMap:
    """时段分类映射 (4×24)"""
    data: np.ndarray  # (4, 24), 值为 1/2/3
    season_hours: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)

    def get_segment(self, season_idx: int, hour: int) -> int:
        """获取指定季节和小时的时段类型"""
        return int(self.data[season_idx, hour])

    def validate(self) -> List[str]:
        """校验时段映射"""
        issues = []
        if self.data.shape != (N_SEASONS, HOURS_PER_DAY):
            issues.append(f"形状应为 (4, 24)，实际为 {self.data.shape}")
        for s in range(N_SEASONS):
            counts = {1: 0, 2: 0, 3: 0}
            for h in range(HOURS_PER_DAY):
                seg = self.data[s, h]
                if seg in counts:
                    counts[seg] += 1
            for seg, count in counts.items():
                if count != HOURS_PER_SEGMENT:
                    issues.append(f"季节{s} {SEGMENT_NAMES[seg]}应有8小时，实际{count}")
        return issues


@dataclass
class GuidanceMultipliers:
    """12维引导倍率参数"""
    values: np.ndarray  # (12,)

    @property
    def peak(self) -> np.ndarray:
        """峰时段倍率 (4季节)"""
        return self.values[IDX_PEAK]

    @property
    def flat(self) -> np.ndarray:
        """平时段倍率 (4季节)"""
        return self.values[IDX_FLAT]

    @property
    def valley(self) -> np.ndarray:
        """谷时段倍率 (4季节)"""
        return self.values[IDX_VALLEY]

    def get_multiplier(self, season_idx: int, segment: int) -> float:
        """获取指定季节和时段的倍率"""
        if segment == SEGMENT_PEAK:
            return float(self.peak[season_idx])
        elif segment == SEGMENT_FLAT:
            return float(self.flat[season_idx])
        elif segment == SEGMENT_VALLEY:
            return float(self.valley[season_idx])
        return 1.0

    def validate(self) -> List[str]:
        """校验倍率参数"""
        issues = []
        if self.values.shape != (N_MULTIPLIERS,):
            issues.append(f"形状应为 (12,)，实际为 {self.values.shape}")
        if np.any(self.values <= 0):
            issues.append("倍率必须为正数")
        if np.any(np.isnan(self.values)) or np.any(np.isinf(self.values)):
            issues.append("倍率包含 NaN 或 Inf")
        return issues

    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            'peak': self.peak.tolist(),
            'flat': self.flat.tolist(),
            'valley': self.valley.tolist(),
        }


@dataclass
class GuidancePriceResult:
    """引导电价生成结果"""
    p_guided: np.ndarray       # (365, 24) 引导电价矩阵
    p_base: np.ndarray         # (365, 24) 原始电价矩阵
    multipliers: GuidanceMultipliers
    segment_map: SegmentMap
    dates: np.ndarray          # (365,) 日期数组

    def to_csv(self, path: Union[str, Path]) -> None:
        """导出为 CSV"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_PANDAS:
            columns = [f'Hour{h+1}' for h in range(HOURS_PER_DAY)]
            df = pd.DataFrame(self.p_guided, columns=columns)
            df.insert(0, 'Date', self.dates)
            df.to_csv(path, index=False)
        else:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header = ['Date'] + [f'Hour{h+1}' for h in range(HOURS_PER_DAY)]
                writer.writerow(header)
                for i, row in enumerate(self.p_guided):
                    writer.writerow([self.dates[i]] + row.tolist())

    def summary_stats(self) -> Dict[str, Any]:
        """生成摘要统计"""
        return {
            'shape': self.p_guided.shape,
            'p_guided': {
                'mean': float(np.mean(self.p_guided)),
                'std': float(np.std(self.p_guided)),
                'min': float(np.min(self.p_guided)),
                'max': float(np.max(self.p_guided)),
            },
            'p_base': {
                'mean': float(np.mean(self.p_base)),
                'std': float(np.std(self.p_base)),
                'min': float(np.min(self.p_base)),
                'max': float(np.max(self.p_base)),
            },
            'ratio': {
                'mean': float(np.mean(self.p_guided / (self.p_base + 1e-10))),
                'min': float(np.min(self.p_guided / (self.p_base + 1e-10))),
                'max': float(np.max(self.p_guided / (self.p_base + 1e-10))),
            },
            'multipliers': self.multipliers.to_dict(),
        }


# ============================================================================
# 季节索引函数
# ============================================================================

def get_season_index(dates: np.ndarray) -> np.ndarray:
    """
    获取日期对应的季节索引。

    对齐 MATLAB get_season_index 函数。

    Args:
        dates: (N,) 日期数组，YYYYMMDD 格式整数

    Returns:
        season_idx: (N,) 季节索引，0=春, 1=夏, 2=秋, 3=冬
    """
    # 提取月份
    months = (dates % 10000) // 100

    season_idx = np.zeros(len(dates), dtype=int)
    season_idx[(months >= 3) & (months <= 5)] = 0   # 春
    season_idx[(months >= 6) & (months <= 8)] = 1   # 夏
    season_idx[(months >= 9) & (months <= 11)] = 2  # 秋
    season_idx[(months == 12) | (months <= 2)] = 3  # 冬

    return season_idx


def get_month_from_day(day_of_year: int, year: int = 2024) -> int:
    """根据年内天数获取月份"""
    from datetime import date, timedelta
    d = date(year, 1, 1) + timedelta(days=day_of_year - 1)
    return d.month


# ============================================================================
# 时段分类生成
# ============================================================================

def generate_segment_map(
    p_base: np.ndarray,
    dates: np.ndarray,
) -> SegmentMap:
    """
    生成季节性峰/平/谷时段划分。

    对齐 MATLAB seasonal_tou_pricing.m 逻辑。

    Args:
        p_base: (365, 24) 原始电价矩阵
        dates: (365,) 日期数组，YYYYMMDD 格式

    Returns:
        SegmentMap: 4×24 时段分类映射
    """
    months = (dates % 10000) // 100
    segment_data = np.zeros((N_SEASONS, HOURS_PER_DAY), dtype=int)
    season_hours = {}

    for s in range(N_SEASONS):
        # 获取该季节的日期索引
        season_mask = np.isin(months, SEASON_MONTHS[s])
        season_prices = p_base[season_mask, :]

        # 计算24小时均价
        mean_price = np.mean(season_prices, axis=0)

        # 排序获取峰/平/谷小时
        sorted_idx = np.argsort(mean_price)[::-1]  # 降序

        peak_hours = sorted(sorted_idx[:8].tolist())
        flat_hours = sorted(sorted_idx[8:16].tolist())
        valley_hours = sorted(sorted_idx[16:24].tolist())

        # 填充 segment_data
        segment_data[s, peak_hours] = SEGMENT_PEAK
        segment_data[s, flat_hours] = SEGMENT_FLAT
        segment_data[s, valley_hours] = SEGMENT_VALLEY

        # 存储到字典
        season_hours[SEASON_NAMES[s]] = {
            'peak': peak_hours,
            'flat': flat_hours,
            'valley': valley_hours,
            'mean_price': mean_price.tolist(),
        }

    return SegmentMap(data=segment_data, season_hours=season_hours)


# ============================================================================
# 核心引导电价生成
# ============================================================================

def generate_guided_price(
    p_base: np.ndarray,
    multipliers: np.ndarray,
    segment_map: SegmentMap,
    dates: np.ndarray,
) -> np.ndarray:
    """
    生成引导电价矩阵。

    对齐 MATLAB generate_guided_price.m 逻辑:
    p_guided(d,h) = p_base(d,h) * m_segment(season(d), h)

    Args:
        p_base: (365, 24) 原始电价矩阵
        multipliers: (12,) 倍率向量
        segment_map: 4×24 时段分类映射
        dates: (365,) 日期数组

    Returns:
        p_guided: (365, 24) 引导电价矩阵
    """
    mults = GuidanceMultipliers(values=np.asarray(multipliers))
    season_idx = get_season_index(dates)

    n_days, n_hours = p_base.shape
    p_guided = np.zeros((n_days, n_hours))

    for d in range(n_days):
        s = season_idx[d]
        for h in range(n_hours):
            seg = segment_map.get_segment(s, h)
            m = mults.get_multiplier(s, seg)
            p_guided[d, h] = p_base[d, h] * m

    return p_guided


def _get_day_type(dates: np.ndarray) -> np.ndarray:
    """0=工作日, 1=周末"""
    from datetime import date
    day_types = np.zeros(len(dates), dtype=int)
    for i, d in enumerate(dates):
        y = int(d // 10000)
        m = int((d % 10000) // 100)
        dd = int(d % 100)
        weekday = date(y, m, dd).weekday()
        day_types[i] = 1 if weekday >= 5 else 0
    return day_types


def generate_guided_price_192(
    p_base: np.ndarray,
    multipliers: np.ndarray,
    dates: np.ndarray,
) -> np.ndarray:
    """
    生成 192 维引导电价矩阵。

    维度定义: 4季节 × 2日类型(工作日/周末) × 24小时
    """
    multipliers = np.asarray(multipliers, dtype=float).reshape(-1)
    if multipliers.size != 192:
        raise ValueError(f"multipliers 长度应为 192，实际为 {multipliers.size}")

    mults = multipliers.reshape(4, 2, 24)
    season_idx = get_season_index(dates)
    day_types = _get_day_type(dates)

    n_days, n_hours = p_base.shape
    p_guided = np.zeros((n_days, n_hours))
    for d in range(n_days):
        s = season_idx[d]
        t = day_types[d]
        p_guided[d, :] = p_base[d, :] * mults[s, t, :]

    return p_guided


# ============================================================================
# 主入口函数
# ============================================================================

def create_guidance_price(
    multipliers: np.ndarray,
    p_base: Optional[np.ndarray] = None,
    dates: Optional[np.ndarray] = None,
    price_csv_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    year: int = 2024,
) -> GuidancePriceResult:
    """
    创建引导电价（主入口函数）。

    可独立调用，不依赖 GA 流程。

    Args:
        multipliers: (12,) 倍率向量
        p_base: (365, 24) 原始电价矩阵（可选）
        dates: (365,) 日期数组（可选）
        price_csv_path: 电价 CSV 路径（可选）
        output_path: 输出 CSV 路径（可选）
        year: 年份（默认 2024）

    Returns:
        GuidancePriceResult: 引导电价结果
    """
    multipliers = np.asarray(multipliers).flatten()
    if len(multipliers) != N_MULTIPLIERS:
        raise ValueError(f"multipliers 长度应为 12，实际为 {len(multipliers)}")

    # 加载或生成基础电价
    if p_base is None:
        if price_csv_path and Path(price_csv_path).exists():
            p_base, dates = _load_price_csv(price_csv_path)
        else:
            p_base, dates = _generate_mock_price(year)
    elif dates is None:
        dates = _generate_dates(year)

    # 生成时段映射
    segment_map = generate_segment_map(p_base, dates)

    # 生成引导电价
    p_guided = generate_guided_price(p_base, multipliers, segment_map, dates)

    # 构建结果
    result = GuidancePriceResult(
        p_guided=p_guided,
        p_base=p_base,
        multipliers=GuidanceMultipliers(values=multipliers),
        segment_map=segment_map,
        dates=dates,
    )

    # 导出 CSV
    if output_path:
        result.to_csv(output_path)

    return result


# ============================================================================
# 辅助函数
# ============================================================================

def _generate_dates(year: int = 2024) -> np.ndarray:
    """生成年度日期数组 (YYYYMMDD 格式)"""
    from datetime import date, timedelta
    start = date(year, 1, 1)
    dates = []
    for i in range(DAYS_PER_YEAR):
        d = start + timedelta(days=i)
        dates.append(d.year * 10000 + d.month * 100 + d.day)
    return np.array(dates, dtype=int)


def _generate_mock_price(year: int = 2024) -> Tuple[np.ndarray, np.ndarray]:
    """生成模拟电价数据"""
    dates = _generate_dates(year)
    np.random.seed(42)

    # 基础电价 + 季节波动 + 日内波动
    p_base = np.zeros((DAYS_PER_YEAR, HOURS_PER_DAY))
    season_idx = get_season_index(dates)

    # 季节基准价格
    season_base = {0: 0.5, 1: 0.7, 2: 0.5, 3: 0.6}  # 春夏秋冬

    for d in range(DAYS_PER_YEAR):
        s = season_idx[d]
        base = season_base[s]
        for h in range(HOURS_PER_DAY):
            # 日内波动：早晚高峰
            hour_factor = 1.0
            if 7 <= h <= 11 or 17 <= h <= 21:
                hour_factor = 1.3  # 高峰
            elif 0 <= h <= 6 or 22 <= h <= 23:
                hour_factor = 0.7  # 低谷
            p_base[d, h] = base * hour_factor * (1 + 0.1 * np.random.randn())

    return np.maximum(p_base, 0.1), dates


def _load_price_csv(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """从 CSV 加载电价数据"""
    if HAS_PANDAS:
        df = pd.read_csv(path)
        dates = df.iloc[:, 0].values.astype(int)
        p_base = df.iloc[:, 1:25].values.astype(float)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            rows = list(reader)
        dates = np.array([int(r[0]) for r in rows])
        p_base = np.array([[float(x) for x in r[1:25]] for r in rows])
    return p_base, dates


# ============================================================================
# 可视化函数
# ============================================================================

def plot_guidance_price(
    result: GuidancePriceResult,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Any]:
    """
    可视化引导电价。

    Args:
        result: 引导电价结果
        output_path: 图片保存路径
        show: 是否显示图片

    Returns:
        fig: matplotlib figure 对象
    """
    if not HAS_MATPLOTLIB:
        print("警告: matplotlib 不可用，跳过可视化")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 日均价对比
    ax1 = axes[0, 0]
    daily_base = np.mean(result.p_base, axis=1)
    daily_guided = np.mean(result.p_guided, axis=1)
    ax1.plot(daily_base, label='Base', alpha=0.7)
    ax1.plot(daily_guided, label='Guided', alpha=0.7)
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Price')
    ax1.set_title('Daily Average Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 典型日 24 小时曲线
    ax2 = axes[0, 1]
    for s, name in enumerate(SEASON_NAMES):
        season_mask = get_season_index(result.dates) == s
        idx = np.where(season_mask)[0][0]
        ax2.plot(result.p_guided[idx, :], label=name)
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Guided Price')
    ax2.set_title('Typical Day Price by Season')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 倍率热力图
    ax3 = axes[1, 0]
    mult_matrix = np.array([
        result.multipliers.peak,
        result.multipliers.flat,
        result.multipliers.valley,
    ])
    im = ax3.imshow(mult_matrix, aspect='auto', cmap='RdYlGn_r')
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(SEASON_NAMES)
    ax3.set_yticks(range(3))
    ax3.set_yticklabels(['Peak', 'Flat', 'Valley'])
    ax3.set_title('Multipliers by Season/Segment')
    plt.colorbar(im, ax=ax3)

    # 4. 时段分布
    ax4 = axes[1, 1]
    seg_map = result.segment_map.data
    im2 = ax4.imshow(seg_map, aspect='auto', cmap='coolwarm')
    ax4.set_xlabel('Hour')
    ax4.set_ylabel('Season')
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(SEASON_NAMES)
    ax4.set_title('Segment Map (1=Peak, 2=Flat, 3=Valley)')
    plt.colorbar(im2, ax=ax4)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)

    if show:
        plt.show()

    return fig


# ============================================================================
# 摘要打印
# ============================================================================

def print_summary(result: GuidancePriceResult) -> None:
    """打印引导电价摘要"""
    stats = result.summary_stats()

    print("=" * 60)
    print("引导电价生成摘要")
    print("=" * 60)
    print(f"矩阵形状: {stats['shape']}")
    print()
    print("原始电价 (p_base):")
    print(f"  均值: {stats['p_base']['mean']:.4f}")
    print(f"  范围: [{stats['p_base']['min']:.4f}, {stats['p_base']['max']:.4f}]")
    print()
    print("引导电价 (p_guided):")
    print(f"  均值: {stats['p_guided']['mean']:.4f}")
    print(f"  范围: [{stats['p_guided']['min']:.4f}, {stats['p_guided']['max']:.4f}]")
    print()
    print("倍率比值:")
    print(f"  均值: {stats['ratio']['mean']:.4f}")
    print(f"  范围: [{stats['ratio']['min']:.4f}, {stats['ratio']['max']:.4f}]")
    print()
    print("12维倍率参数:")
    m = stats['multipliers']
    print(f"  Peak:   {m['peak']}")
    print(f"  Flat:   {m['flat']}")
    print(f"  Valley: {m['valley']}")
    print("=" * 60)


# ============================================================================
# 默认倍率
# ============================================================================

def get_default_multipliers() -> np.ndarray:
    """获取默认倍率参数"""
    return np.array([
        # Peak (春/夏/秋/冬)
        1.2, 1.3, 1.2, 1.25,
        # Flat (春/夏/秋/冬)
        1.0, 1.0, 1.0, 1.0,
        # Valley (春/夏/秋/冬)
        0.8, 0.7, 0.8, 0.75,
    ])
