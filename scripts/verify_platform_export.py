#!/usr/bin/env python3
"""
Verify a MEOS platform export Excel file.

Outputs:
- Raw platform CSV (Excel -> CSV)
- OJ CSV (platform -> OJ format)
- Markdown report with score and key statistics
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
sys.path.insert(0, str(PYTHON_DIR))

from meos.export.oj_exporter import (  # noqa: E402
    export_oj_csv,
    _parse_platform_excel,
)
from meos.ga.evaluator import PLAN18_DEVICE_NAMES  # noqa: E402
from scripts.score_oj_csv import score_oj_csv  # noqa: E402


def _write_report(report_path: Path, payload: Dict[str, Any]) -> None:
    plan18 = payload["plan18"]
    score = payload["score"]
    totals = payload["totals"]
    files = payload["files"]
    timestamp = payload["timestamp"]
    neg = payload.get("negatives", {})
    clip_applied = payload.get("clip_applied", False)

    lines = [
        "# 平台导出文件验证报告",
        "",
        f"- 生成时间: {timestamp}",
        f"- 平台导出文件: `{payload['input_excel']}`",
        f"- 是否删除内燃机(ICE): {str(payload.get('drop_ice', False))}",
        "",
        "## 输出文件",
        f"- 平台 CSV: `{files['platform_csv']}`",
        f"- OJ CSV: `{files['oj_csv']}`",
        f"- 评分 JSON: `{files['score_json']}`",
        "",
        "## 校验与修正",
        f"- 是否执行负值裁剪: {str(clip_applied)}",
        f"- shed_load 负值数: {neg.get('shed_load', {}).get('count', 0)}",
        f"- thermal_power 负值数: {neg.get('thermal_power', {}).get('count', 0)}",
        f"- gas_source 负值数: {neg.get('gas_source', {}).get('count', 0)}",
        f"- gas_source 最小值: {neg.get('gas_source', {}).get('min', 0.0)}",
        "",
        "## 评分结果",
        f"- Score: {score['Score']:.11f}",
        f"- C_total: {score['C_total']:.6f}",
        f"- C_CAP: {score['C_CAP']:.6f}",
        f"- C_OP: {score['C_OP']:.6f}",
        f"- C_Carbon: {score['C_Carbon']:.6f}",
        f"- C_OP_ele: {score['C_OP_ele']:.6f}",
        f"- C_OP_gas: {score['C_OP_gas']:.6f}",
        f"- C_OP_penalty: {score['C_OP_penalty']:.6f}",
        f"- E_total: {score['E_total']:.6f}",
        f"- E_elec: {score['E_elec']:.6f}",
        f"- E_gas: {score['E_gas']:.6f}",
        "",
        "## 购能与切负荷统计",
        f"- 购电量 (MWh): {totals['ele_mwh']:.6f}",
        f"- 购气量 (MWh): {totals['gas_mwh']:.6f}",
        f"- 购气量 (m³): {totals['gas_m3']:.6f}",
        f"- 切负荷合计 (MWh): {totals['shed_mwh']:.6f}",
        "",
        "## 规划值 plan18",
    ]

    for idx, name in enumerate(PLAN18_DEVICE_NAMES, start=1):
        lines.append(f"- {idx:02d} {name}: {plan18[idx - 1]:.6f}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_platform_excel(input_excel: Path) -> pd.DataFrame:
    if input_excel.suffix.lower() == ".xls":
        try:
            import xlrd  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"无法读取平台导出文件: {input_excel}") from exc
        workbook = xlrd.open_workbook(str(input_excel))
        sheet = workbook.sheet_by_index(0)
        rows = [sheet.row_values(i) for i in range(sheet.nrows)]
        if not rows:
            raise RuntimeError(f"平台导出文件为空: {input_excel}")
        header, data = rows[0], rows[1:]
        return pd.DataFrame(data, columns=header)
    try:
        return pd.read_excel(input_excel, header=0)
    except Exception as exc:
        raise RuntimeError(f"无法读取平台导出文件: {input_excel}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MEOS platform export Excel")
    parser.add_argument("--input", default="meos/平台导出文件.xls", help="platform export Excel path")
    parser.add_argument("--output-dir", default="output/platform_verify", help="output directory")
    parser.add_argument("--data-dir", default="data/raw", help="scoring data dir")
    parser.add_argument("--spec", default="configs/oj_score.yaml", help="score spec path")
    parser.add_argument("--device-catalog", default="spec/device_catalog.yaml", help="device catalog path")
    parser.add_argument("--clip-negative", action="store_true", help="clip negative values to zero")
    parser.add_argument("--drop-ice", action="store_true", help="force ICE plan18 to 0 in OJ output")
    args = parser.parse_args()

    input_excel = Path(args.input)
    if not input_excel.exists():
        raise FileNotFoundError(f"input not found: {input_excel}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    platform_csv = out_dir / f"{input_excel.stem}_platform.csv"
    oj_csv = out_dir / f"{input_excel.stem}_oj.csv"
    score_json = out_dir / f"{input_excel.stem}_score.json"
    report_md = out_dir / f"{input_excel.stem}_report.md"

    # 1) Excel -> CSV
    df = _read_platform_excel(input_excel)
    df.to_csv(platform_csv, index=False, encoding="utf-8-sig")

    # 2) Excel -> OJ CSV (with optional clipping)
    dispatch_result, plan18 = _parse_platform_excel(df)
    negatives: Dict[str, Dict[str, float]] = {}

    def _clip_array(arr, name):
        arr = arr.astype(float)
        count = int((arr < 0).sum())
        min_val = float(arr.min()) if arr.size else 0.0
        negatives[name] = {"count": count, "min": min_val}
        if args.clip_negative and count > 0:
            arr = arr.copy()
            arr[arr < 0] = 0.0
        return arr

    shed_load = _clip_array(dispatch_result.shed_load, "shed_load")
    thermal_power = _clip_array(dispatch_result.thermal_power, "thermal_power")
    gas_source = _clip_array(dispatch_result.gas_source, "gas_source")

    dispatch_result = dispatch_result.__class__(
        shed_load=shed_load,
        thermal_power=thermal_power,
        gas_source=gas_source,
    )

    if args.drop_ice and len(plan18) >= 3:
        plan18 = plan18.copy()
        plan18[2] = 0.0
    export_oj_csv(dispatch_result, plan18, oj_csv, validate=True)

    # 3) Score
    score = score_oj_csv(oj_csv, Path(args.data_dir), Path(args.spec), Path(args.device_catalog))
    spec_path = Path(args.spec)
    score_spec = {}
    if spec_path.exists():
        try:
            import yaml  # type: ignore
            raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
            score_spec = raw.get("score_spec", {})
        except Exception:
            score_spec = {}
    score_payload = {"score": score, "score_spec": score_spec}
    score_json.write_text(json.dumps(score_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4) Totals and plan18
    oj_df = pd.read_csv(oj_csv)
    plan18 = oj_df["ans_planning"].to_numpy(dtype=float)[:18]
    gas_to_mwh = 0.01
    if score_spec:
        gas_to_mwh = float(score_spec.get("units", {}).get("gas_m3_to_MWh", 0.01))

    totals = {
        "ele_mwh": float(oj_df["ans_ele"].sum()),
        "gas_mwh": float(oj_df["ans_gas"].sum()),
        "gas_m3": float(oj_df["ans_gas"].sum() / gas_to_mwh) if gas_to_mwh else 0.0,
        "shed_mwh": float(oj_df[["ans_load1", "ans_load2", "ans_load3"]].sum().sum()),
    }

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_excel": str(input_excel),
        "files": {
            "platform_csv": str(platform_csv),
            "oj_csv": str(oj_csv),
            "score_json": str(score_json),
        },
        "score": score,
        "totals": totals,
        "plan18": plan18.tolist(),
        "negatives": negatives,
        "clip_applied": bool(args.clip_negative),
        "drop_ice": bool(args.drop_ice),
    }

    _write_report(report_md, payload)
    print(report_md)


if __name__ == "__main__":
    main()
