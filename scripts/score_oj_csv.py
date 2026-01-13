#!/usr/bin/env python3
"""
Score an OJ CSV file using the same formula as MATLAB oj_scorer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def _read_daily_matrix(path: Path) -> np.ndarray:
    if HAS_PANDAS:
        df = pd.read_csv(path)
        values = df.iloc[:, 1:25].to_numpy(dtype=float)
        return values
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                float(row[1])
            except Exception:
                continue
            rows.append([float(x) for x in row[1:25]])
    return np.array(rows, dtype=float)


def _read_column(path: Path, col_name: str) -> np.ndarray:
    if HAS_PANDAS:
        df = pd.read_csv(path)
        if col_name not in df.columns:
            return np.array([])
        return df[col_name].to_numpy(dtype=float)
    values = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if col_name not in row:
                continue
            try:
                values.append(float(row[col_name]))
            except Exception:
                values.append(0.0)
    return np.array(values, dtype=float)


def _to_hourly(series: np.ndarray, hours: int = 8760) -> np.ndarray:
    if series.size == 0:
        return np.zeros(hours)
    if series.ndim == 1:
        if series.size == hours:
            return series[:hours]
        if series.size == 365:
            return np.repeat(series, 24)[:hours]
        return np.resize(series, hours)
    # daily matrix: (365,24) -> 8760
    return series.reshape(-1)[:hours]


def _load_score_spec(path: Path) -> Dict[str, float]:
    if not HAS_YAML:
        raise ImportError("需要 PyYAML 解析 oj_score.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec = data.get("score_spec", {})
    return {
        "discount_rate": float(spec.get("capex", {}).get("discount_rate", 0.04)),
        "load_shed_penalty": float(spec.get("load_shedding_penalty", 500000.0)),
        "gas_to_MWh": float(spec.get("units", {}).get("gas_m3_to_MWh", 0.01)),
        "cost_unit": float(spec.get("units", {}).get("cost_unit", 10000.0)),
        "carbon_threshold": float(spec.get("carbon", {}).get("threshold", 100000.0)),
        "carbon_price": float(spec.get("carbon", {}).get("price", 600.0)),
        "gas_emission_factor": float(spec.get("carbon", {}).get("gas_emission_factor", 0.002)),
        "logistic_x0": float(spec.get("logistic", {}).get("x0", 100000.0)),
        "logistic_k": float(spec.get("logistic", {}).get("k", 15000.0)),
    }


def _load_device_catalog(path: Path) -> Dict[int, Dict[str, float]]:
    if not HAS_YAML:
        raise ImportError("需要 PyYAML 解析 device_catalog.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    devices = data.get("device_catalog", {}).get("devices", [])
    device_map: Dict[int, Dict[str, float]] = {}
    for item in devices:
        idx = int(item.get("idx", 0))
        device_map[idx] = {
            "device_id": str(item.get("device_id", "")),
            "base_capacity": float(item.get("base_capacity", 1.0)),
            "unit_cost": float(item.get("unit_cost", 0.0)),
            "lifespan": int(item.get("lifespan", item.get("lifespan_years", 20))),
        }
    return device_map


def _annuity(rate: float, years: int) -> float:
    if rate <= 0 or years <= 0:
        return 0.0
    factor = (1 + rate) ** years
    return rate * factor / (factor - 1)


def _load_scoring_prices(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    elec_path = data_dir / "电源价格_电.csv"
    gas_path = data_dir / "天然气价格.csv"
    summary_path = data_dir / "汇总_价格.csv"

    if elec_path.exists():
        elec_daily = _read_daily_matrix(elec_path)
        elec = _to_hourly(elec_daily)
    elif summary_path.exists():
        elec = _to_hourly(_read_column(summary_path, "电价"))
    else:
        elec = np.ones(8760) * 500.0

    if gas_path.exists():
        gas_daily = _read_daily_matrix(gas_path)
        gas = _to_hourly(gas_daily)
    elif summary_path.exists():
        gas = _to_hourly(_read_column(summary_path, "天然气价格"))
    else:
        gas = np.ones(8760) * 2.87

    return elec, gas


def _load_scoring_carbon(data_dir: Path) -> Tuple[np.ndarray, float]:
    elec_path = data_dir / "购电碳排放因子.csv"
    gas_path = data_dir / "购气碳排放因子.csv"
    summary_path = data_dir / "汇总_碳排放因子.csv"

    if elec_path.exists():
        elec_daily = _read_daily_matrix(elec_path)
        elec = _to_hourly(elec_daily)
    elif summary_path.exists():
        elec = _to_hourly(_read_column(summary_path, "购电碳排放因子"))
    else:
        elec = np.ones(8760) * 0.5

    if gas_path.exists():
        gas_daily = _read_daily_matrix(gas_path)
        gas_hourly = _to_hourly(gas_daily)
        gas_factor = float(np.mean(gas_hourly)) if gas_hourly.size else 0.002
    elif summary_path.exists():
        gas_series = _read_column(summary_path, "购气碳排放因子")
        gas_factor = float(np.mean(gas_series)) if gas_series.size else 0.002
    else:
        gas_factor = 0.002

    return elec, gas_factor


def score_oj_csv(csv_path: Path, data_dir: Path, spec_path: Path, catalog_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")
    if HAS_PANDAS:
        df = pd.read_csv(csv_path)
        ans_load1 = df["ans_load1"].to_numpy(dtype=float)
        ans_load2 = df["ans_load2"].to_numpy(dtype=float)
        ans_load3 = df["ans_load3"].to_numpy(dtype=float)
        ans_ele = df["ans_ele"].to_numpy(dtype=float)
        ans_gas = df["ans_gas"].to_numpy(dtype=float)
        ans_planning = df["ans_planning"].to_numpy(dtype=float)
    else:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        ans_load1 = np.array([float(r.get("ans_load1", 0.0)) for r in rows])
        ans_load2 = np.array([float(r.get("ans_load2", 0.0)) for r in rows])
        ans_load3 = np.array([float(r.get("ans_load3", 0.0)) for r in rows])
        ans_ele = np.array([float(r.get("ans_ele", 0.0)) for r in rows])
        ans_gas = np.array([float(r.get("ans_gas", 0.0)) for r in rows])
        ans_planning = np.array([float(r.get("ans_planning", 0.0)) for r in rows])

    plan18 = ans_planning[:18]

    spec = _load_score_spec(spec_path)
    device_map = _load_device_catalog(catalog_path)

    prices_e, prices_g = _load_scoring_prices(data_dir)
    carbon_e, gas_factor = _load_scoring_carbon(data_dir)

    # C_CAP
    C_CAP = 0.0
    for idx in range(1, 19):
        cap = float(plan18[idx - 1])
        if cap <= 0:
            continue
        dev = device_map.get(idx, {})
        unit_cost = float(dev.get("unit_cost", 0.0))
        lifespan = int(dev.get("lifespan", 20))
        base = float(dev.get("base_capacity", 1.0))
        if unit_cost <= 0 or lifespan <= 0:
            continue
        ann = _annuity(spec["discount_rate"], lifespan)
        device_id = str(dev.get("device_id", ""))
        if device_id in ("WindTurbine", "PV"):
            real_cap = cap
        else:
            real_cap = cap * base
        C_CAP += real_cap * unit_cost * ann

    # C_OP
    shed = ans_load1 + ans_load2 + ans_load3
    C_penalty = float(np.sum(shed) * spec["load_shed_penalty"])
    C_elec = float(np.sum(ans_ele * prices_e))
    gas_m3 = ans_gas / spec["gas_to_MWh"]
    C_gas = float(np.sum(gas_m3 * prices_g))
    C_OP = C_penalty + C_elec + C_gas

    # C_Carbon
    E_elec = float(np.sum(ans_ele * carbon_e))
    E_gas = float(np.sum(gas_m3) * gas_factor)
    E_total = E_elec + E_gas
    E_excess = max(0.0, E_total - spec["carbon_threshold"])
    C_Carbon = E_excess * spec["carbon_price"]

    C_total = C_CAP + C_OP + C_Carbon
    x = C_total / spec["cost_unit"]
    z = (x - spec["logistic_x0"]) / spec["logistic_k"]
    if z > 700:
        Score = 0.0
    elif z < -700:
        Score = 100.0
    else:
        Score = 100.0 / (1.0 + math.exp(z))

    return {
        "C_CAP": C_CAP,
        "C_OP": C_OP,
        "C_OP_ele": C_elec,
        "C_OP_gas": C_gas,
        "C_OP_penalty": C_penalty,
        "C_Carbon": C_Carbon,
        "C_total": C_total,
        "Score": Score,
        "E_elec": E_elec,
        "E_gas": E_gas,
        "E_total": E_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score OJ CSV file")
    parser.add_argument("csv_path", nargs="?", default="example/HWdata_for_OJ.csv", help="OJ csv path")
    parser.add_argument("--data-dir", default="data/raw", help="scoring data dir")
    parser.add_argument("--spec", default="configs/oj_score.yaml", help="score spec path")
    parser.add_argument("--device-catalog", default="spec/device_catalog.yaml", help="device catalog path")
    parser.add_argument("--json", action="store_true", help="output json")
    args = parser.parse_args()

    result = score_oj_csv(
        Path(args.csv_path),
        Path(args.data_dir),
        Path(args.spec),
        Path(args.device_catalog),
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("C_CAP=%.2f 万元" % (result["C_CAP"] / 10000))
        print("C_OP=%.2f 万元 (ele=%.2f gas=%.2f penalty=%.2f)" % (
            result["C_OP"] / 10000,
            result["C_OP_ele"] / 10000,
            result["C_OP_gas"] / 10000,
            result["C_OP_penalty"] / 10000,
        ))
        print("C_Carbon=%.2f 万元" % (result["C_Carbon"] / 10000))
        print("C_total=%.2f 万元" % (result["C_total"] / 10000))
        print("Score=%.11f" % result["Score"])


if __name__ == "__main__":
    main()
