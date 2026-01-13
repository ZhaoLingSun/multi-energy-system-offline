#!/usr/bin/env python3
"""
思考题2扩展：在新增跨季节冷热联储（固定容量）条件下重新优化 plan18。

实现方式：
- 复用 run_full_milp.py 的完整 MILP（8760 小时），
- 将热/冷储能容量固定为指定 MWh（对应 plan18 的固定倍数），
- 使用跨日 SOC（storage-cross-day）以体现跨季节作用，
- 通过修改储能充放效率近似“自损耗”（loss=0 时设为 1.0，小损耗设为 0.99/0.995 等）。

注意：该脚本仅用于思考题实验，不影响主任务代码与默认行为。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from meos.ga.codec import PLAN18_BOUNDS

import run_full_milp


def _fixed_storage_units(storage_mwh: float, base_capacity: float) -> int:
    units = storage_mwh / base_capacity
    return int(round(units))


def _patch_plan18_bounds(fixed_units: int):
    original = run_full_milp._build_plan18_bounds

    def wrapper(
        base_bounds: List[Tuple[int, int]],
        relax_zero_bounds: bool,
        zero_bound_max: int,
        upper_scale: float,
        upper_add: int,
        relax_from_plan18,
        device_max_units,
    ) -> List[Tuple[int, int]]:
        bounds = original(
            base_bounds,
            relax_zero_bounds,
            zero_bound_max,
            upper_scale,
            upper_add,
            relax_from_plan18,
            device_max_units,
        )
        # idx 11/12 (0-based) -> 热储能/冷储能
        for idx in (11, 12):
            if idx < len(bounds):
                bounds[idx] = (fixed_units, fixed_units)
        return bounds

    return original, wrapper


def _patch_device_catalog(charge_eff: float, discharge_eff: float):
    original = run_full_milp._load_device_catalog

    def wrapper(path: Path):
        catalog = original(path)
        updated = []
        for item in catalog:
            if item.get("device_id") in {"ThermalStorage", "ColdStorage"}:
                clone = copy.deepcopy(item)
                eff = clone.get("efficiency", {})
                eff["charge"] = float(charge_eff)
                eff["discharge"] = float(discharge_eff)
                clone["efficiency"] = eff
                updated.append(clone)
            else:
                updated.append(item)
        return updated

    return original, wrapper


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2 plan18 MILP with seasonal storage (fixed capacity)")
    parser.add_argument("--storage-mwh", type=float, default=2000.0)
    parser.add_argument("--charge-eff", type=float, default=1.0)
    parser.add_argument("--discharge-eff", type=float, default=1.0)
    parser.add_argument("--mip-gap", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--time-limit", type=float, default=0.0)
    parser.add_argument("--output-dir", default="runs/thought_questions")
    parser.add_argument("--session-tag", default="q2_plan18")
    args, extra = parser.parse_known_args()

    # base capacity for thermal/cold storage is 40 MWh
    fixed_units = _fixed_storage_units(args.storage_mwh, 40.0)

    orig_bounds, patched_bounds = _patch_plan18_bounds(fixed_units)
    orig_catalog, patched_catalog = _patch_device_catalog(args.charge_eff, args.discharge_eff)

    run_full_milp._build_plan18_bounds = patched_bounds
    run_full_milp._load_device_catalog = patched_catalog

    # build argv for run_full_milp
    argv = [
        "run_full_milp.py",
        "--days",
        "365",
        "--mip-gap",
        str(args.mip_gap),
        "--threads",
        str(args.threads),
        "--output-dir",
        str(args.output_dir),
        "--carbon-mode",
        "aligned",
        "--carbon-regime",
        "auto",
        "--objective",
        "cost",
        "--storage-cross-day",
    ]
    if args.time_limit and args.time_limit > 0:
        argv += ["--time-limit", str(args.time_limit)]
    argv += extra

    # attach metadata for traceability
    run_full_milp.EXTRA_TAGS = {
        "seasonal_storage_mwh": args.storage_mwh,
        "seasonal_storage_units": fixed_units,
        "charge_eff": args.charge_eff,
        "discharge_eff": args.discharge_eff,
        "session_tag": args.session_tag,
    }

    try:
        sys.argv = argv
        run_full_milp.main()
    finally:
        run_full_milp._build_plan18_bounds = orig_bounds
        run_full_milp._load_device_catalog = orig_catalog


if __name__ == "__main__":
    main()
