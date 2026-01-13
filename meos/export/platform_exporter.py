"""
平台导出格式实现

将单日/全年调度结果导出为平台 CSV/Excel 对齐格式。

Excel 格式契约：
- 1 行表头
- col2: 数值型，规划行 > 200
- col3: 指标名逐字匹配
- col5: 负荷类型关键词
- col8-31: 24小时数据
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ============================================================================
# 常量定义 - 对齐 MATLAB export_meos_excel.m
# ============================================================================

# 编码类型
CODE_SHED_LOAD = 16       # 切负荷
CODE_THERMAL_POWER = 43   # 火电出力
CODE_GAS_SOURCE = 44      # 气源出力
CODE_PLAN_BASE = 200      # 规划行编码基准 (201-218)

# 表头列名
HEADER_COLUMNS = [
    "序号",        # col1
    "编码类型",    # col2
    "指标名",      # col3
    "区域",        # col4
    "设备名称",    # col5
    "单位",        # col6
    "日期",        # col7
] + [f"Hour{h}" for h in range(1, 25)]  # col8-col31

# 指标名常量
INDICATOR_SHED_LOAD = "能量枢纽切负荷量（MW）"
INDICATOR_THERMAL_POWER = "火电出力（MW）"
INDICATOR_GAS_SOURCE = "气源出力（MW）"
INDICATOR_PLAN_CAPACITY = "规划容量"


# ============================================================================
# 切负荷对象定义 - 对齐 MATLAB build_shed_load_rows
# ============================================================================

# 9个切负荷对象，顺序与平台导出一致
SHED_LOAD_OBJECTS = [
    {"zone": "学生区", "device_name": "学生区-电负荷1", "col_idx": 0},
    {"zone": "学生区", "device_name": "学生区-冷负荷1", "col_idx": 1},
    {"zone": "学生区", "device_name": "学生区-热负荷1", "col_idx": 2},
    {"zone": "教工区", "device_name": "教工区-电负荷2", "col_idx": 3},
    {"zone": "教工区", "device_name": "教工区-冷负荷2", "col_idx": 4},
    {"zone": "教工区", "device_name": "教工区-热负荷2", "col_idx": 5},
    {"zone": "教学办公区", "device_name": "教学办公区-电负荷3", "col_idx": 6},
    {"zone": "教学办公区", "device_name": "教学办公区-热负荷3", "col_idx": 7},
    {"zone": "教学办公区", "device_name": "教学办公区-冷负荷3", "col_idx": 8},
]

# 火电机组定义
THERMAL_UNITS = [
    {"name": "火电机组1", "col_idx": 0},
    {"name": "火电机组2", "col_idx": 1},
    {"name": "火电机组3", "col_idx": 2},
]

# 气源定义
GAS_SOURCES = [
    {"name": "天然气源1", "col_idx": 0},
]

# Plan18 设备映射 - 对齐 spec/plan18_map.yaml
PLAN18_DEVICES = [
    {"idx": 1, "code": 201, "device_name": "热电联产A", "unit": "台"},
    {"idx": 2, "code": 202, "device_name": "热电联产B", "unit": "台"},
    {"idx": 3, "code": 203, "device_name": "内燃机", "unit": "台"},
    {"idx": 4, "code": 204, "device_name": "电锅炉", "unit": "台"},
    {"idx": 5, "code": 205, "device_name": "压缩式制冷机A", "unit": "台"},
    {"idx": 6, "code": 206, "device_name": "压缩式制冷机B", "unit": "台"},
    {"idx": 7, "code": 207, "device_name": "吸收式制冷机组", "unit": "台"},
    {"idx": 8, "code": 208, "device_name": "燃气蒸汽锅炉", "unit": "台"},
    {"idx": 9, "code": 209, "device_name": "地源热泵A", "unit": "台"},
    {"idx": 10, "code": 210, "device_name": "地源热泵B", "unit": "台"},
    {"idx": 11, "code": 211, "device_name": "电储能", "unit": "台"},
    {"idx": 12, "code": 212, "device_name": "热储能", "unit": "台"},
    {"idx": 13, "code": 213, "device_name": "冷储能", "unit": "台"},
    {"idx": 14, "code": 214, "device_name": "风电", "unit": "台"},
    {"idx": 15, "code": 215, "device_name": "光伏", "unit": "台"},
    {"idx": 16, "code": 216, "device_name": "电制气", "unit": "台"},
    {"idx": 17, "code": 217, "device_name": "燃气轮机", "unit": "台"},
    {"idx": 18, "code": 218, "device_name": "冷热电联供", "unit": "台"},
]


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class ExportConfig:
    """导出配置"""
    n_days: int = 365
    n_hours: int = 24
    n_shed_objects: int = 9
    n_thermal_units: int = 3
    n_gas_sources: int = 1
    n_plan18: int = 18


@dataclass
class DispatchResult:
    """调度结果数据结构"""
    shed_load: np.ndarray      # (8760, 9) 切负荷数据
    thermal_power: np.ndarray  # (8760, 3) 火电出力
    gas_source: np.ndarray     # (8760, 1) 气源出力


# ============================================================================
# 验证函数
# ============================================================================

def validate_dispatch_dimensions(
    dispatch: DispatchResult,
    config: ExportConfig = None,
) -> None:
    """验证调度结果维度"""
    config = config or ExportConfig()
    n_hours_total = config.n_days * config.n_hours

    # 切负荷：8760×9
    if dispatch.shed_load.shape != (n_hours_total, config.n_shed_objects):
        raise ValueError(
            f"shed_load 维度应为 ({n_hours_total}, {config.n_shed_objects})，"
            f"实际为 {dispatch.shed_load.shape}"
        )

    # 火电：8760×3
    if dispatch.thermal_power.shape != (n_hours_total, config.n_thermal_units):
        raise ValueError(
            f"thermal_power 维度应为 ({n_hours_total}, {config.n_thermal_units})，"
            f"实际为 {dispatch.thermal_power.shape}"
        )

    # 气源：8760×1
    if dispatch.gas_source.shape[0] != n_hours_total:
        raise ValueError(
            f"gas_source 行数应为 {n_hours_total}，"
            f"实际为 {dispatch.gas_source.shape[0]}"
        )


def validate_plan18(plan18: List[float], config: ExportConfig = None) -> None:
    """验证规划向量"""
    config = config or ExportConfig()
    if len(plan18) != config.n_plan18:
        raise ValueError(
            f"plan18 长度应为 {config.n_plan18}，实际为 {len(plan18)}"
        )


# ============================================================================
# 行构建函数
# ============================================================================

def build_shed_load_rows(
    shed_load_8760: np.ndarray,
    config: ExportConfig = None,
) -> List[List[Any]]:
    """
    构建切负荷数据行
    9个对象 × 365天 = 3285行
    """
    config = config or ExportConfig()
    rows = []
    row_idx = 1

    for obj in SHED_LOAD_OBJECTS:
        for day in range(1, config.n_days + 1):
            hour_start = (day - 1) * config.n_hours
            hour_end = day * config.n_hours
            hourly_data = shed_load_8760[hour_start:hour_end, obj["col_idx"]]

            row = [
                row_idx,                    # col1: 序号
                CODE_SHED_LOAD,             # col2: 编码类型
                INDICATOR_SHED_LOAD,        # col3: 指标名
                obj["zone"],                # col4: 区域
                obj["device_name"],         # col5: 设备名称
                "MW",                       # col6: 单位
                day,                        # col7: 日期
            ] + list(hourly_data)           # col8-31: 24小时数据

            rows.append(row)
            row_idx += 1

    return rows


def build_thermal_power_rows(
    thermal_power_8760: np.ndarray,
    config: ExportConfig = None,
) -> List[List[Any]]:
    """
    构建火电出力数据行
    3个机组 × 365天 = 1095行
    """
    config = config or ExportConfig()
    rows = []
    row_idx = 1

    for unit in THERMAL_UNITS:
        for day in range(1, config.n_days + 1):
            hour_start = (day - 1) * config.n_hours
            hour_end = day * config.n_hours
            hourly_data = thermal_power_8760[hour_start:hour_end, unit["col_idx"]]

            row = [
                row_idx,                    # col1: 序号
                CODE_THERMAL_POWER,         # col2: 编码类型
                INDICATOR_THERMAL_POWER,    # col3: 指标名
                "",                         # col4: 区域（空）
                unit["name"],               # col5: 设备名称
                "MW",                       # col6: 单位
                day,                        # col7: 日期
            ] + list(hourly_data)           # col8-31: 24小时数据

            rows.append(row)
            row_idx += 1

    return rows


def build_gas_source_rows(
    gas_source_8760: np.ndarray,
    config: ExportConfig = None,
) -> List[List[Any]]:
    """
    构建气源出力数据行
    1个气源 × 365天 = 365行
    """
    config = config or ExportConfig()
    rows = []

    # 确保是2D数组
    if gas_source_8760.ndim == 1:
        gas_source_8760 = gas_source_8760.reshape(-1, 1)

    for day in range(1, config.n_days + 1):
        hour_start = (day - 1) * config.n_hours
        hour_end = day * config.n_hours
        hourly_data = gas_source_8760[hour_start:hour_end, 0]

        row = [
            day,                        # col1: 序号
            CODE_GAS_SOURCE,            # col2: 编码类型
            INDICATOR_GAS_SOURCE,       # col3: 指标名
            "",                         # col4: 区域（空）
            GAS_SOURCES[0]["name"],     # col5: 设备名称
            "MW",                       # col6: 单位
            day,                        # col7: 日期
        ] + list(hourly_data)           # col8-31: 24小时数据

        rows.append(row)

    return rows


def build_plan18_rows(plan18: List[float]) -> List[List[Any]]:
    """
    构建规划载体行
    18行，col2 > 200，规划值写入 col8
    """
    rows = []

    for i, device in enumerate(PLAN18_DEVICES):
        row = [
            i + 1,                      # col1: 序号
            device["code"],             # col2: 编码类型 (201-218)
            INDICATOR_PLAN_CAPACITY,    # col3: 指标名
            "",                         # col4: 区域（空）
            device["device_name"],      # col5: 设备名称
            device["unit"],             # col6: 单位
            "",                         # col7: 日期（空）
            plan18[i],                  # col8: 规划值
        ] + [0] * 23                    # col9-31: 填充0

        rows.append(row)

    return rows


# ============================================================================
# 主导出类
# ============================================================================

class PlatformExporter:
    """平台导出器"""

    def __init__(self, config: ExportConfig = None):
        self.config = config or ExportConfig()

    def build_all_rows(
        self,
        dispatch: DispatchResult,
        plan18: List[float],
    ) -> List[List[Any]]:
        """构建所有数据行"""
        validate_dispatch_dimensions(dispatch, self.config)
        validate_plan18(plan18, self.config)

        shed_rows = build_shed_load_rows(dispatch.shed_load, self.config)
        thermal_rows = build_thermal_power_rows(dispatch.thermal_power, self.config)
        gas_rows = build_gas_source_rows(dispatch.gas_source, self.config)
        plan_rows = build_plan18_rows(plan18)

        return shed_rows + thermal_rows + gas_rows + plan_rows

    def export_csv(
        self,
        dispatch: DispatchResult,
        plan18: List[float],
        output_path: Union[str, Path],
    ) -> Path:
        """导出为 CSV 文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_rows = self.build_all_rows(dispatch, plan18)

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER_COLUMNS)
            writer.writerows(all_rows)

        return output_path

    def export_excel(
        self,
        dispatch: DispatchResult,
        plan18: List[float],
        output_path: Union[str, Path],
    ) -> Path:
        """导出为 Excel 文件"""
        if not HAS_PANDAS:
            raise ImportError("pandas is required for Excel export")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_rows = self.build_all_rows(dispatch, plan18)
        df = pd.DataFrame(all_rows, columns=HEADER_COLUMNS)
        df.to_excel(output_path, index=False, engine="openpyxl")

        return output_path

    def get_row_counts(self) -> Dict[str, int]:
        """获取各类数据行数统计"""
        n_days = self.config.n_days
        return {
            "shed_load": len(SHED_LOAD_OBJECTS) * n_days,
            "thermal_power": len(THERMAL_UNITS) * n_days,
            "gas_source": len(GAS_SOURCES) * n_days,
            "plan18": len(PLAN18_DEVICES),
            "total": (
                len(SHED_LOAD_OBJECTS) * n_days +
                len(THERMAL_UNITS) * n_days +
                len(GAS_SOURCES) * n_days +
                len(PLAN18_DEVICES)
            ),
        }


# ============================================================================
# 便捷函数
# ============================================================================

def export_platform_csv(
    dispatch: DispatchResult,
    plan18: List[float],
    output_path: Union[str, Path],
    config: ExportConfig = None,
) -> Path:
    """
    导出平台格式 CSV 文件

    Args:
        dispatch: 调度结果 (shed_load, thermal_power, gas_source)
        plan18: 18维规划向量
        output_path: 输出路径
        config: 导出配置

    Returns:
        输出文件路径
    """
    exporter = PlatformExporter(config)
    return exporter.export_csv(dispatch, plan18, output_path)


def export_platform_excel(
    dispatch: DispatchResult,
    plan18: List[float],
    output_path: Union[str, Path],
    config: ExportConfig = None,
) -> Path:
    """
    导出平台格式 Excel 文件

    Args:
        dispatch: 调度结果
        plan18: 18维规划向量
        output_path: 输出路径
        config: 导出配置

    Returns:
        输出文件路径
    """
    exporter = PlatformExporter(config)
    return exporter.export_excel(dispatch, plan18, output_path)
