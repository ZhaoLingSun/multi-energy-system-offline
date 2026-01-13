"""
OJ CSV 导出器

将平台导出数据或调度结果转换为 OJ 提交格式。

OJ CSV 格式契约 (HWdata_for_OJ.csv):
- 形状: 8760 × 6
- 列顺序: ans_load1, ans_load2, ans_load3, ans_ele, ans_gas, ans_planning
- ans_load1: 电负荷切除量汇总 (3区合计)
- ans_load2: 热负荷切除量汇总 (3区合计)
- ans_load3: 冷负荷切除量汇总 (3区合计)
- ans_ele: 火电出力汇总 (3机组合计)
- ans_gas: 气源出力 (1气源)
- ans_planning: 规划向量 (1:18 为 plan18, 19:8760 为 0)
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


# ============================================================================
# 常量定义
# ============================================================================

# OJ CSV 列名
OJ_COLUMNS = [
    "ans_load1",     # 电负荷切除量汇总
    "ans_load2",     # 热负荷切除量汇总
    "ans_load3",     # 冷负荷切除量汇总
    "ans_ele",       # 火电出力汇总
    "ans_gas",       # 气源出力
    "ans_planning",  # 规划向量
]

# 时间维度
HOURS_PER_YEAR = 8760
HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365

# 对象数量
N_ZONES = 3           # 3个区域
N_THERMAL_UNITS = 3   # 3个火电机组
N_GAS_SOURCES = 1     # 1个气源
N_PLAN_DEVICES = 18   # 18维规划向量


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class OJData:
    """OJ 提交数据结构"""
    ans_load1: np.ndarray      # (8760,) 电负荷切除
    ans_load2: np.ndarray      # (8760,) 热负荷切除
    ans_load3: np.ndarray      # (8760,) 冷负荷切除
    ans_ele: np.ndarray        # (8760,) 火电出力
    ans_gas: np.ndarray        # (8760,) 气源出力
    ans_planning: np.ndarray   # (8760,) 规划向量

    def validate(self) -> List[str]:
        """校验数据格式，返回问题列表"""
        issues = []

        # 检查形状
        for name in OJ_COLUMNS:
            arr = getattr(self, name)
            if arr.shape != (HOURS_PER_YEAR,):
                issues.append(f"{name} 形状应为 (8760,)，实际为 {arr.shape}")

        # 检查非负
        for name in ["ans_load1", "ans_load2", "ans_load3", "ans_ele", "ans_gas"]:
            arr = getattr(self, name)
            if np.any(arr < 0):
                issues.append(f"{name} 存在负值")

        # 检查 NaN/Inf
        for name in OJ_COLUMNS:
            arr = getattr(self, name)
            if np.any(np.isnan(arr)):
                issues.append(f"{name} 包含 NaN")
            if np.any(np.isinf(arr)):
                issues.append(f"{name} 包含 Inf")

        # 检查 ans_planning 尾部为零
        if np.any(self.ans_planning[N_PLAN_DEVICES:] != 0):
            issues.append("ans_planning(19:8760) 应全为 0")

        return issues

    def to_matrix(self) -> np.ndarray:
        """转换为 8760×6 矩阵"""
        return np.column_stack([
            self.ans_load1,
            self.ans_load2,
            self.ans_load3,
            self.ans_ele,
            self.ans_gas,
            self.ans_planning,
        ])


@dataclass
class DispatchResult8760:
    """全年调度结果"""
    shed_load: np.ndarray       # (8760, 9) 切负荷: 3区×3类
    thermal_power: np.ndarray   # (8760, 3) 火电出力: 3机组
    gas_source: np.ndarray      # (8760, 1) 气源出力: 1气源

    def validate(self) -> List[str]:
        """校验数据维度"""
        issues = []
        if self.shed_load.shape != (HOURS_PER_YEAR, 9):
            issues.append(f"shed_load 形状应为 (8760, 9)，实际为 {self.shed_load.shape}")
        if self.thermal_power.shape != (HOURS_PER_YEAR, N_THERMAL_UNITS):
            issues.append(f"thermal_power 形状应为 (8760, 3)，实际为 {self.thermal_power.shape}")
        if self.gas_source.shape[0] != HOURS_PER_YEAR:
            issues.append(f"gas_source 行数应为 8760，实际为 {self.gas_source.shape[0]}")
        return issues


# ============================================================================
# 核心 reshape 函数
# ============================================================================

def reshape_shed_load(shed_load_9col: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 9 列切负荷数据 reshape 为 3 个汇总列。

    - 9 列对应 3 区 × 3 类负荷
    - 列顺序: 学生区(电/冷/热), 教工区(电/冷/热), 教学办公区(电/热/冷)
    - 按负荷类型汇总: 电(col 0,3,6), 热(col 2,5,7), 冷(col 1,4,8)

    Args:
        shed_load_9col: (8760, 9) 切负荷数据

    Returns:
        (ans_load1, ans_load2, ans_load3): 电/热/冷负荷切除汇总
    """
    # 电负荷列索引: 学生区-电(0), 教工区-电(3), 教学办公区-电(6)
    ele_cols = [0, 3, 6]
    # 热负荷列索引: 学生区-热(2), 教工区-热(5), 教学办公区-热(7)
    heat_cols = [2, 5, 7]
    # 冷负荷列索引: 学生区-冷(1), 教工区-冷(4), 教学办公区-冷(8)
    cold_cols = [1, 4, 8]

    ans_load1 = np.sum(shed_load_9col[:, ele_cols], axis=1)   # 电
    ans_load2 = np.sum(shed_load_9col[:, heat_cols], axis=1)  # 热
    ans_load3 = np.sum(shed_load_9col[:, cold_cols], axis=1)  # 冷

    return ans_load1, ans_load2, ans_load3


def reshape_thermal_power(thermal_3col: np.ndarray) -> np.ndarray:
    """
    将 3 列火电出力 reshape 为 1 列汇总。

    Args:
        thermal_3col: (8760, 3) 火电出力数据

    Returns:
        ans_ele: (8760,) 火电出力汇总
    """
    return np.sum(thermal_3col, axis=1)


def reshape_gas_source(gas_1col: np.ndarray) -> np.ndarray:
    """
    将气源出力 reshape 为 1 列。

    Args:
        gas_1col: (8760, 1) 或 (8760,) 气源出力数据

    Returns:
        ans_gas: (8760,) 气源出力
    """
    return gas_1col.flatten()


def reshape_plan18(plan18: np.ndarray) -> np.ndarray:
    """
    将 18 维规划向量 reshape 为 8760 维。

    Args:
        plan18: (18,) 规划向量

    Returns:
        ans_planning: (8760,) 规划向量，前 18 位为 plan18，其余为 0
    """
    ans_planning = np.zeros(HOURS_PER_YEAR)
    ans_planning[:N_PLAN_DEVICES] = plan18.flatten()[:N_PLAN_DEVICES]
    return ans_planning


# ============================================================================
# 主导出函数
# ============================================================================

def export_oj_csv(
    dispatch_result: DispatchResult8760,
    plan18: np.ndarray,
    output_path: Union[str, Path],
    validate: bool = True,
) -> OJData:
    """
    从调度结果生成 OJ CSV 文件。

    Args:
        dispatch_result: 全年调度结果
        plan18: 18 维规划向量
        output_path: 输出 CSV 路径
        validate: 是否执行格式校验

    Returns:
        OJData: reshape 后的 OJ 数据

    Raises:
        ValueError: 数据校验失败
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 校验输入
    if validate:
        issues = dispatch_result.validate()
        if issues:
            raise ValueError(f"调度结果校验失败: {issues}")

    # 执行 reshape
    ans_load1, ans_load2, ans_load3 = reshape_shed_load(dispatch_result.shed_load)
    ans_ele = reshape_thermal_power(dispatch_result.thermal_power)
    ans_gas = reshape_gas_source(dispatch_result.gas_source)
    ans_planning = reshape_plan18(plan18)

    # 构建 OJData
    oj_data = OJData(
        ans_load1=ans_load1,
        ans_load2=ans_load2,
        ans_load3=ans_load3,
        ans_ele=ans_ele,
        ans_gas=ans_gas,
        ans_planning=ans_planning,
    )

    # 校验输出
    if validate:
        issues = oj_data.validate()
        if issues:
            raise ValueError(f"OJ 数据校验失败: {issues}")

    # 写入 CSV
    _write_oj_csv(oj_data, output_path)

    return oj_data


def _write_oj_csv(oj_data: OJData, output_path: Path) -> None:
    """写入 OJ CSV 文件"""
    matrix = oj_data.to_matrix()

    if HAS_PANDAS:
        df = pd.DataFrame(matrix, columns=OJ_COLUMNS)
        df.to_csv(output_path, index=False)
    else:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(OJ_COLUMNS)
            for row in matrix:
                writer.writerow(row.tolist())


# ============================================================================
# 从平台导出文件转换
# ============================================================================

def export_oj_from_platform_excel(
    platform_excel_path: Union[str, Path],
    output_path: Union[str, Path],
    validate: bool = True,
) -> OJData:
    """
    从平台导出 Excel 文件生成 OJ CSV。

    Args:
        platform_excel_path: 平台导出 Excel 文件路径
        output_path: 输出 CSV 路径
        validate: 是否执行格式校验

    Returns:
        OJData: reshape 后的 OJ 数据
    """
    if not HAS_PANDAS:
        raise ImportError("需要 pandas 来读取 Excel 文件")

    platform_excel_path = Path(platform_excel_path)
    output_path = Path(output_path)

    # 读取 Excel（跳过表头）
    df = pd.read_excel(platform_excel_path, header=0)

    # 提取数据
    dispatch_result, plan18 = _parse_platform_excel(df)

    # 调用主导出函数
    return export_oj_csv(dispatch_result, plan18, output_path, validate)


def _parse_platform_excel(df: pd.DataFrame) -> Tuple[DispatchResult8760, np.ndarray]:
    """
    解析平台导出 Excel 数据。

    数据提取逻辑:
    - col2 (编码类型): 规划行 > 200
    - col3 (指标名): 逐字匹配
    - col5 (设备名称): 负荷类型关键词
    - col8-31: 24小时数据
    """
    # 列索引（0-based）
    COL_CODE = 1       # 编码类型
    COL_INDICATOR = 2  # 指标名
    COL_DEVICE = 4     # 设备名称
    COL_DATA_START = 7 # Hour1 起始列
    COL_DATA_END = 31  # Hour24 结束列

    # 提取切负荷行
    shed_mask = df.iloc[:, COL_INDICATOR] == "能量枢纽切负荷量（MW）"
    shed_rows = df[shed_mask]

    # 按负荷类型分组
    ele_mask = shed_rows.iloc[:, COL_DEVICE].str.contains("电负荷")
    heat_mask = shed_rows.iloc[:, COL_DEVICE].str.contains("热负荷")
    cold_mask = shed_rows.iloc[:, COL_DEVICE].str.contains("冷负荷")

    # 提取 24 小时数据并 reshape 为 8760
    ele_data = _reshape_daily_to_hourly(
        shed_rows[ele_mask].iloc[:, COL_DATA_START:COL_DATA_END+1].values
    )
    heat_data = _reshape_daily_to_hourly(
        shed_rows[heat_mask].iloc[:, COL_DATA_START:COL_DATA_END+1].values
    )
    cold_data = _reshape_daily_to_hourly(
        shed_rows[cold_mask].iloc[:, COL_DATA_START:COL_DATA_END+1].values
    )

    # 提取火电出力行
    thermal_mask = df.iloc[:, COL_INDICATOR] == "火电出力（MW）"
    thermal_rows = df[thermal_mask]
    thermal_data = _reshape_daily_to_hourly(
        thermal_rows.iloc[:, COL_DATA_START:COL_DATA_END+1].values
    )

    # 提取气源出力行
    gas_mask = df.iloc[:, COL_INDICATOR] == "气源出力（MW）"
    gas_rows = df[gas_mask]
    gas_data = _reshape_daily_to_hourly(
        gas_rows.iloc[:, COL_DATA_START:COL_DATA_END+1].values
    )

    # 提取规划行
    plan_mask = df.iloc[:, COL_CODE] > 200
    plan_rows = df[plan_mask]
    plan18 = plan_rows.iloc[:, COL_DATA_START].values.astype(float)

    # 构建 DispatchResult8760
    # 注意：需要按正确顺序组合 9 列切负荷
    shed_load = _combine_shed_load_columns(ele_data, heat_data, cold_data)

    dispatch_result = DispatchResult8760(
        shed_load=shed_load,
        thermal_power=thermal_data,
        gas_source=gas_data.reshape(-1, 1),
    )

    return dispatch_result, plan18


def _reshape_daily_to_hourly(daily_data: np.ndarray) -> np.ndarray:
    """
    将日×24小时数据 reshape 为 8760 小时数据。

    Args:
        daily_data: (n_objects * 365, 24) 日数据

    Returns:
        hourly_data: (8760, n_objects) 或 (8760,) 小时数据
    """
    n_rows = daily_data.shape[0]
    n_objects = n_rows // DAYS_PER_YEAR

    if n_objects == 1:
        # 单对象：直接 flatten
        return daily_data.flatten()
    else:
        # 多对象：reshape 并转置
        # 输入: (n_objects * 365, 24) -> 输出: (8760, n_objects)
        reshaped = daily_data.reshape(n_objects, DAYS_PER_YEAR, HOURS_PER_DAY)
        # (n_objects, 365, 24) -> (365, 24, n_objects) -> (8760, n_objects)
        transposed = reshaped.transpose(1, 2, 0)
        return transposed.reshape(HOURS_PER_YEAR, n_objects)


def _combine_shed_load_columns(
    ele_data: np.ndarray,
    heat_data: np.ndarray,
    cold_data: np.ndarray,
) -> np.ndarray:
    """
    组合切负荷数据为 9 列格式。
    
    列顺序:
    学生区(电/冷/热), 教工区(电/冷/热), 教学办公区(电/热/冷)
    """
    # 假设每类负荷有 3 个区域
    n_zones = N_ZONES

    # 确保数据形状正确
    if ele_data.ndim == 1:
        ele_data = ele_data.reshape(-1, 1)
    if heat_data.ndim == 1:
        heat_data = heat_data.reshape(-1, 1)
    if cold_data.ndim == 1:
        cold_data = cold_data.reshape(-1, 1)

    # 组合为 9 列
    # 顺序: 学生区(电0/冷1/热2), 教工区(电3/冷4/热5), 教学办公区(电6/热7/冷8)
    shed_load = np.zeros((HOURS_PER_YEAR, 9))

    for zone in range(n_zones):
        if zone < ele_data.shape[1]:
            shed_load[:, zone * 3] = ele_data[:, zone]  # 电

        if zone == 2:
            # 教学办公区顺序为 电/热/冷
            if zone < heat_data.shape[1]:
                shed_load[:, zone * 3 + 1] = heat_data[:, zone]
            if zone < cold_data.shape[1]:
                shed_load[:, zone * 3 + 2] = cold_data[:, zone]
        else:
            # 学生区、教工区顺序为 电/冷/热
            if zone < cold_data.shape[1]:
                shed_load[:, zone * 3 + 1] = cold_data[:, zone]
            if zone < heat_data.shape[1]:
                shed_load[:, zone * 3 + 2] = heat_data[:, zone]

    return shed_load


# ============================================================================
# 校验工具
# ============================================================================

def verify_oj_csv(csv_path: Union[str, Path]) -> Tuple[bool, List[str]]:
    """
    校验 OJ CSV 文件格式。

    Args:
        csv_path: CSV 文件路径

    Returns:
        (is_valid, issues): 是否有效及问题列表
    """
    csv_path = Path(csv_path)
    issues = []

    if not csv_path.exists():
        return False, ["文件不存在"]

    try:
        if HAS_PANDAS:
            df = pd.read_csv(csv_path)
            data = df.values
            columns = df.columns.tolist()
        else:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                columns = next(reader)
                data = np.array([list(map(float, row)) for row in reader])
    except Exception as e:
        return False, [f"读取文件失败: {e}"]

    # 检查行数
    if data.shape[0] != HOURS_PER_YEAR:
        issues.append(f"行数应为 8760，实际为 {data.shape[0]}")

    # 检查列数
    if data.shape[1] != len(OJ_COLUMNS):
        issues.append(f"列数应为 6，实际为 {data.shape[1]}")

    # 检查列名
    if columns != OJ_COLUMNS:
        issues.append(f"列名不匹配: 期望 {OJ_COLUMNS}，实际 {columns}")

    # 检查数值
    if np.any(np.isnan(data)):
        issues.append("存在 NaN 值")
    if np.any(np.isinf(data)):
        issues.append("存在 Inf 值")

    # 检查非负（前 5 列）
    for i, col_name in enumerate(OJ_COLUMNS[:5]):
        if np.any(data[:, i] < 0):
            issues.append(f"{col_name} 存在负值")

    # 检查 ans_planning 尾部
    if data.shape[0] >= HOURS_PER_YEAR and np.any(data[N_PLAN_DEVICES:, 5] != 0):
        issues.append("ans_planning(19:8760) 应全为 0")

    return len(issues) == 0, issues


def compare_with_matlab_reshape(
    python_csv: Union[str, Path],
    matlab_csv: Union[str, Path],
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> Tuple[bool, Dict[str, Any]]:
    """
    与 MATLAB reshape 结果对比校验。

    Args:
        python_csv: Python 生成的 CSV 路径
        matlab_csv: MATLAB 生成的 CSV 路径
        rtol: 相对容差
        atol: 绝对容差

    Returns:
        (is_match, details): 是否匹配及详细信息
    """
    if not HAS_PANDAS:
        raise ImportError("需要 pandas 进行对比")

    py_df = pd.read_csv(python_csv)
    mat_df = pd.read_csv(matlab_csv)

    details = {
        "python_shape": py_df.shape,
        "matlab_shape": mat_df.shape,
        "column_diffs": {},
        "max_abs_diff": {},
        "max_rel_diff": {},
    }

    if py_df.shape != mat_df.shape:
        return False, details

    is_match = True
    for col in OJ_COLUMNS:
        if col not in py_df.columns or col not in mat_df.columns:
            details["column_diffs"][col] = "列缺失"
            is_match = False
            continue

        py_col = py_df[col].values
        mat_col = mat_df[col].values

        abs_diff = np.abs(py_col - mat_col)
        rel_diff = abs_diff / (np.abs(mat_col) + 1e-15)

        details["max_abs_diff"][col] = float(np.max(abs_diff))
        details["max_rel_diff"][col] = float(np.max(rel_diff))

        if not np.allclose(py_col, mat_col, rtol=rtol, atol=atol):
            details["column_diffs"][col] = f"最大绝对差: {np.max(abs_diff):.2e}"
            is_match = False

    return is_match, details
