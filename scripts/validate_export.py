#!/usr/bin/env python3
"""
CSV 导出校验工具 - Phase 4 输出校验
校验平台 CSV/OJ CSV 的列名、行数、字段完整性
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional


# OJ CSV 期望的列名和顺序
EXPECTED_OJ_COLUMNS = [
    "ans_load1",
    "ans_load2",
    "ans_load3",
    "ans_ele",
    "ans_gas",
    "ans_planning",
]

# 期望的行数 (8760 数据行 + 1 表头)
EXPECTED_ROW_COUNT = 8761


class ValidationReport:
    """校验报告类"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def add_error(self, msg: str):
        self.errors.append(f"[ERROR] {msg}")

    def add_warning(self, msg: str):
        self.warnings.append(f"[WARN]  {msg}")

    def add_info(self, msg: str):
        self.info.append(f"[INFO]  {msg}")

    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def print_report(self):
        print("=" * 60)
        print(f"校验报告: {self.file_path}")
        print("=" * 60)

        for msg in self.info:
            print(msg)

        if self.warnings:
            print("-" * 40)
            for msg in self.warnings:
                print(msg)

        if self.errors:
            print("-" * 40)
            for msg in self.errors:
                print(msg)

        print("-" * 40)
        status = "✓ 校验通过" if self.is_valid() else "✗ 校验失败"
        print(f"结果: {status}")
        print(f"错误: {len(self.errors)}, 警告: {len(self.warnings)}")
        print("=" * 60)


def read_csv_header(file_path: Path) -> tuple[list[str], int]:
    """读取 CSV 表头和行数"""
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, [])
        row_count = 1 + sum(1 for _ in reader)
    return header, row_count


def validate_columns(
    report: ValidationReport,
    actual_columns: list[str],
    expected_columns: list[str],
):
    """校验列名"""
    # 检查重复列
    seen = set()
    duplicates = []
    for col in actual_columns:
        if col in seen:
            duplicates.append(col)
        seen.add(col)

    if duplicates:
        report.add_error(f"重复列: {duplicates}")

    # 检查缺失列
    actual_set = set(actual_columns)
    expected_set = set(expected_columns)

    missing = expected_set - actual_set
    if missing:
        report.add_error(f"缺失列: {sorted(missing)}")

    # 检查多余列
    extra = actual_set - expected_set
    if extra:
        report.add_warning(f"多余列: {sorted(extra)}")

    # 检查列顺序
    if actual_columns != expected_columns and not missing:
        report.add_warning("列顺序与期望不一致")
        report.add_info(f"  期望: {expected_columns}")
        report.add_info(f"  实际: {actual_columns}")


def validate_row_count(
    report: ValidationReport,
    actual_count: int,
    expected_count: int,
):
    """校验行数"""
    report.add_info(f"行数: {actual_count} (含表头)")

    if actual_count != expected_count:
        diff = actual_count - expected_count
        sign = "+" if diff > 0 else ""
        report.add_error(
            f"行数不匹配: 期望 {expected_count}, 实际 {actual_count} ({sign}{diff})"
        )


def validate_data_completeness(report: ValidationReport, file_path: Path):
    """校验数据完整性（检查空值）"""
    empty_cells = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            for col, val in row.items():
                if val is None or val.strip() == '':
                    empty_cells.append((row_num, col))

    if empty_cells:
        if len(empty_cells) <= 5:
            for row_num, col in empty_cells:
                report.add_warning(f"空值: 行 {row_num}, 列 '{col}'")
        else:
            report.add_warning(f"发现 {len(empty_cells)} 个空值")
            for row_num, col in empty_cells[:3]:
                report.add_warning(f"  行 {row_num}, 列 '{col}'")
            report.add_warning(f"  ... 及其他 {len(empty_cells) - 3} 个")


def compare_with_reference(
    report: ValidationReport,
    actual_columns: list[str],
    reference_path: Path,
):
    """与参考文件（如 MATLAB 导出）对比列顺序"""
    if not reference_path.exists():
        report.add_warning(f"参考文件不存在: {reference_path}")
        return

    ref_columns, ref_count = read_csv_header(reference_path)

    report.add_info(f"参考文件: {reference_path}")
    report.add_info(f"参考文件行数: {ref_count}")

    if actual_columns == ref_columns:
        report.add_info("列顺序与参考文件一致 ✓")
    else:
        report.add_warning("列顺序与参考文件不一致")

        # 找出差异
        actual_set = set(actual_columns)
        ref_set = set(ref_columns)

        only_in_actual = actual_set - ref_set
        only_in_ref = ref_set - actual_set

        if only_in_actual:
            report.add_warning(f"  仅在当前文件: {sorted(only_in_actual)}")
        if only_in_ref:
            report.add_warning(f"  仅在参考文件: {sorted(only_in_ref)}")


def validate_csv(
    file_path: str,
    expected_columns: Optional[list[str]] = None,
    expected_rows: Optional[int] = None,
    reference_file: Optional[str] = None,
    check_completeness: bool = False,
) -> ValidationReport:
    """主校验函数"""
    path = Path(file_path)
    report = ValidationReport(file_path)

    # 检查文件存在
    if not path.exists():
        report.add_error(f"文件不存在: {file_path}")
        return report

    # 读取表头和行数
    try:
        actual_columns, row_count = read_csv_header(path)
    except Exception as e:
        report.add_error(f"读取文件失败: {e}")
        return report

    report.add_info(f"列数: {len(actual_columns)}")
    report.add_info(f"列名: {actual_columns}")

    # 校验列名
    if expected_columns:
        validate_columns(report, actual_columns, expected_columns)

    # 校验行数
    if expected_rows:
        validate_row_count(report, row_count, expected_rows)
    else:
        report.add_info(f"行数: {row_count} (含表头)")

    # 校验数据完整性
    if check_completeness:
        validate_data_completeness(report, path)

    # 与参考文件对比
    if reference_file:
        compare_with_reference(report, actual_columns, Path(reference_file))

    return report


def main():
    parser = argparse.ArgumentParser(
        description="CSV 导出校验工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 校验 OJ CSV (使用默认期望值)
  python validate_export.py HWdata_for_OJ.csv --oj

  # 校验任意 CSV 并检查完整性
  python validate_export.py data.csv --check-completeness

  # 与 MATLAB 导出对比
  python validate_export.py output.csv --reference matlab/output.csv

  # 自定义期望行数
  python validate_export.py data.csv --expected-rows 8761
        """,
    )

    parser.add_argument("file", help="要校验的 CSV 文件路径")
    parser.add_argument(
        "--oj",
        action="store_true",
        help="使用 OJ CSV 的默认期望值 (6列, 8761行)",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        help="期望的行数 (含表头)",
    )
    parser.add_argument(
        "--expected-columns",
        nargs="+",
        help="期望的列名列表",
    )
    parser.add_argument(
        "--reference",
        help="参考文件路径 (用于对比列顺序)",
    )
    parser.add_argument(
        "--check-completeness",
        action="store_true",
        help="检查数据完整性 (空值检测)",
    )

    args = parser.parse_args()

    # 确定期望值
    expected_columns = args.expected_columns
    expected_rows = args.expected_rows

    if args.oj:
        expected_columns = expected_columns or EXPECTED_OJ_COLUMNS
        expected_rows = expected_rows or EXPECTED_ROW_COUNT

    # 执行校验
    report = validate_csv(
        file_path=args.file,
        expected_columns=expected_columns,
        expected_rows=expected_rows,
        reference_file=args.reference,
        check_completeness=args.check_completeness,
    )

    report.print_report()

    # 返回退出码
    sys.exit(0 if report.is_valid() else 1)


if __name__ == "__main__":
    main()
