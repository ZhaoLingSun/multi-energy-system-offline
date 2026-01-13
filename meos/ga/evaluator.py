"""
GA 候选解评估器 - Phase 5

功能：
- 串联 Phase 2/3/4，完成候选解评估
- 支持结果缓存与重复评估跳过
- 统一输出：容量 YAML、引导价 CSV、平台 CSV、OJ CSV

用法:
    from meos.ga.evaluator import CandidateEvaluator
    evaluator = CandidateEvaluator(config)
    result = evaluator.evaluate(candidate)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from meos.export.platform_full_exporter import _parse_meos_inputs
from meos.ga.guidance_price import (
    generate_guided_price,
    generate_guided_price_192,
    generate_segment_map,
    _generate_mock_price,
    _load_price_csv,
)
from meos.simulate.annual_summary import (
    calculate_carbon_cost,
    calculate_score,
    summarize_annual,
)


# ============================================================================
# 常量定义
# ============================================================================

# 18 维规划向量设备名称
PLAN18_DEVICE_NAMES = [
    "CHP_A", "CHP_B", "ICE", "EB", "CC_A", "CC_B",
    "AC", "GB", "GSHP_A", "GSHP_B", "ES", "HS", "CS",
    "WT", "PV", "P2G", "GT", "CCHP",
]

# 默认输出目录
DEFAULT_OUTPUT_DIR = "output/ga_eval"
DEFAULT_CACHE_DIR = "output/ga_eval/cache"

# 价格口径换算：1 m^3 = 0.01 MWh
GAS_PRICE_MWH_TO_M3 = 0.01


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class Candidate:
    """候选解数据结构"""
    plan18: np.ndarray          # (18,) 规划向量
    price12: Optional[np.ndarray] = None  # (12|192,) 引导价倍率（可选）
    gas12: Optional[np.ndarray] = None    # (12|192,) 引导气价倍率（可选）
    generation: int = 0         # 所属代数
    index: int = 0              # 个体索引
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.plan18 = np.asarray(self.plan18, dtype=np.float64)
        if self.plan18.shape != (18,):
            raise ValueError(f"plan18 应为 (18,) 形状，实际为 {self.plan18.shape}")
        if self.price12 is not None:
            self.price12 = np.asarray(self.price12, dtype=np.float64)
            if self.price12.shape not in ((12,), (192,)):
                raise ValueError(f"price12 应为 (12,) 或 (192,) 形状，实际为 {self.price12.shape}")
        if self.gas12 is not None:
            self.gas12 = np.asarray(self.gas12, dtype=np.float64)
            if self.gas12.shape not in ((12,), (192,)):
                raise ValueError(f"gas12 应为 (12,) 或 (192,) 形状，实际为 {self.gas12.shape}")

    @property
    def cache_key(self) -> str:
        """生成缓存键（基于 plan18 内容的哈希）"""
        return compute_cache_key(self.plan18, self.price12, self.gas12)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan18": self.plan18.tolist(),
            "price12": self.price12.tolist() if self.price12 is not None else None,
            "gas12": self.gas12.tolist() if self.gas12 is not None else None,
            "generation": self.generation,
            "index": self.index,
            "metadata": self.metadata,
            "cache_key": self.cache_key,
        }


@dataclass
class EvalResult:
    """评估结果数据结构"""
    # 成本分项
    C_CAP: float = 0.0          # 年化投资成本
    C_OP: float = 0.0           # 运行成本
    C_Carbon: float = 0.0       # 碳成本
    C_total: float = 0.0        # 总成本
    Score: float = 0.0          # 最终分数

    # 状态
    status: str = "pending"     # pending/success/failed/cached
    error_msg: Optional[str] = None
    elapsed_sec: float = 0.0

    # 输出文件路径
    capacity_yaml: Optional[str] = None
    guide_price_csv: Optional[str] = None
    guide_gas_csv: Optional[str] = None
    platform_csv: Optional[str] = None
    oj_csv: Optional[str] = None

    # 元数据
    cache_key: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_valid(self) -> bool:
        return self.status in ("success", "cached")


@dataclass
class EvaluatorConfig:
    """评估器配置"""
    # 输出目录
    output_dir: str = DEFAULT_OUTPUT_DIR
    cache_dir: str = DEFAULT_CACHE_DIR
    export_outputs: bool = True

    # 缓存配置
    enable_cache: bool = True
    cache_format: str = "json"  # json / msgpack

    # 仿真配置
    n_days: int = 365
    n_hours: int = 24
    parallel_days: int = 1      # 日并行数
    annualize_days: Optional[int] = None  # 代表日评估时年化天数
    day_weights: Optional[List[float]] = None  # 代表日权重

    # 日志配置
    log_level: str = "INFO"
    verbose: bool = True

    # 输入数据路径
    data_dir: Optional[str] = None
    renewable_dir: Optional[str] = None
    device_catalog_path: Optional[str] = None
    score_spec_path: Optional[str] = None

    # 回调配置
    on_eval_start: Optional[Callable] = None
    on_eval_end: Optional[Callable] = None


# ============================================================================
# 缓存键计算
# ============================================================================

def compute_cache_key(
    plan18: np.ndarray,
    price12: Optional[np.ndarray] = None,
    gas12: Optional[np.ndarray] = None,
    precision: int = 6,
) -> str:
    """
    计算候选解的缓存键。

    缓存键设计：
    - 基于 plan18 的内容哈希
    - 使用固定精度避免浮点误差
    - SHA256 前 16 位作为键

    Args:
        plan18: 18 维规划向量
        precision: 浮点数精度

    Returns:
        16 字符的十六进制缓存键
    """
    # 四舍五入到固定精度
    rounded = np.round(plan18, precision)
    # 转为字节串
    data = rounded.tobytes()
    if price12 is not None:
        price12 = np.asarray(price12, dtype=float)
        data += np.round(price12, precision).tobytes()
    if gas12 is not None:
        gas12 = np.asarray(gas12, dtype=float)
        data += np.round(gas12, precision).tobytes()
    # 计算 SHA256
    hash_obj = hashlib.sha256(data)
    # 取前 16 位
    return hash_obj.hexdigest()[:16]


# ============================================================================
# 缓存管理器
# ============================================================================

class EvalCache:
    """评估结果缓存管理器"""

    def __init__(self, cache_dir: str, cache_format: str = "json"):
        self.cache_dir = Path(cache_dir)
        self.cache_format = cache_format
        self._ensure_dir()
        self._index: Dict[str, str] = {}  # cache_key -> file_path
        self._load_index()

    def _ensure_dir(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _index_path(self) -> Path:
        return self.cache_dir / "cache_index.json"

    def _load_index(self) -> None:
        """加载缓存索引"""
        index_path = self._index_path()
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                self._index = json.load(f)

    def _save_index(self) -> None:
        """保存缓存索引"""
        with open(self._index_path(), "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2)

    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        ext = "json" if self.cache_format == "json" else "msgpack"
        return self.cache_dir / f"{cache_key}.{ext}"

    def exists(self, cache_key: str) -> bool:
        """检查缓存是否存在"""
        return cache_key in self._index and self._get_cache_path(cache_key).exists()

    def get(self, cache_key: str) -> Optional[EvalResult]:
        """获取缓存的评估结果"""
        if not self.exists(cache_key):
            return None

        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            result = EvalResult(**data)
            result.status = "cached"
            return result
        except Exception:
            return None

    def put(self, cache_key: str, result: EvalResult) -> None:
        """存储评估结果到缓存"""
        cache_path = self._get_cache_path(cache_key)
        data = result.to_dict()

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self._index[cache_key] = str(cache_path)
        self._save_index()

    def clear(self) -> int:
        """清空缓存，返回清除的条目数"""
        count = len(self._index)
        for cache_key in list(self._index.keys()):
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
        self._index.clear()
        self._save_index()
        return count

    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_size = sum(
            self._get_cache_path(k).stat().st_size
            for k in self._index
            if self._get_cache_path(k).exists()
        )
        return {
            "count": len(self._index),
            "total_size_bytes": total_size,
            "cache_dir": str(self.cache_dir),
        }


# ============================================================================
# 输出生成器
# ============================================================================

class OutputGenerator:
    """统一输出生成器"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_output_path(self, cache_key: str, suffix: str) -> Path:
        """获取输出文件路径"""
        subdir = self.output_dir / cache_key[:4]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{cache_key}_{suffix}"

    def export_capacity_yaml(
        self,
        plan18: np.ndarray,
        cache_key: str,
    ) -> str:
        """导出容量 YAML 文件"""
        output_path = self._get_output_path(cache_key, "capacity.yaml")

        capacity_dict = {
            "plan18": {
                name: float(plan18[i])
                for i, name in enumerate(PLAN18_DEVICE_NAMES)
            },
            "metadata": {
                "cache_key": cache_key,
                "timestamp": datetime.now().isoformat(),
            },
        }

        if HAS_YAML:
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(capacity_dict, f, allow_unicode=True, default_flow_style=False)
        else:
            # 回退到 JSON 格式
            output_path = output_path.with_suffix(".json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(capacity_dict, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def export_guide_price_csv(
        self,
        guide_prices: Dict[str, List[float]],
        cache_key: str,
    ) -> str:
        """导出引导价 CSV 文件"""
        import csv
        output_path = self._get_output_path(cache_key, "guide_price.csv")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 写入表头
            headers = ["hour"] + list(guide_prices.keys())
            writer.writerow(headers)
            # 写入数据
            n_hours = len(next(iter(guide_prices.values())))
            for h in range(n_hours):
                row = [h + 1]
                for k in guide_prices:
                    val = guide_prices[k][h]
                    if str(k).lower() == "gas":
                        # 导出为元/m^3
                        val = val * GAS_PRICE_MWH_TO_M3
                    row.append(val)
                writer.writerow(row)

        return str(output_path)

    def export_guide_gas_csv(
        self,
        gas_prices: List[float],
        cache_key: str,
    ) -> str:
        """导出引导气价 CSV 文件（元/m^3）"""
        import csv
        output_path = self._get_output_path(cache_key, "guide_gas_price.csv")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["hour", "gas"])
            for h, val in enumerate(gas_prices):
                writer.writerow([h + 1, val * GAS_PRICE_MWH_TO_M3])

        return str(output_path)

    def export_renewable_curve_csv(
        self,
        hourly_values: np.ndarray,
        cache_key: str,
        name: str,
        dates: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """导出风光出力曲线（MW），返回 CSV/XLSX 路径"""
        import pandas as pd

        values = np.asarray(hourly_values, dtype=float).reshape(-1)
        if values.size != 8760:
            raise ValueError(f"renewable curve requires 8760 values, got {values.size}")

        data = values.reshape(365, 24)
        cols = [f"{i}点" for i in range(1, 25)]
        df = pd.DataFrame(data, columns=cols)
        if dates is None:
            dates = pd.date_range("2022-01-01", periods=365, freq="D").strftime("%Y%m%d").tolist()
        df.insert(0, "日期", dates)

        csv_path = self._get_output_path(cache_key, f"{name}_mw.csv")
        xlsx_path = self._get_output_path(cache_key, f"{name}_mw.xlsx")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df.to_excel(xlsx_path, index=False, sheet_name="Sheet1")
        return {"csv": str(csv_path), "xlsx": str(xlsx_path)}

    def export_platform_csv(
        self,
        dispatch_data: Dict[str, np.ndarray],
        plan18: np.ndarray,
        cache_key: str,
        options: Optional["PlatformExportOptions"] = None,
    ) -> str:
        """导出平台 CSV 文件（全量模板对齐）"""
        from meos.export.platform_full_exporter import (
            export_platform_csv_full,
            PlatformExportOptions,
        )

        output_path = self._get_output_path(cache_key, "platform.csv")
        export_platform_csv_full(
            dispatch_data,
            plan18,
            output_path,
            options or PlatformExportOptions(),
        )
        return str(output_path)

    def export_oj_csv(
        self,
        oj_data: Dict[str, np.ndarray],
        cache_key: str,
    ) -> str:
        """导出 OJ CSV 文件"""
        import csv
        output_path = self._get_output_path(cache_key, "oj.csv")

        columns = ["ans_load1", "ans_load2", "ans_load3", "ans_ele", "ans_gas", "ans_planning"]
        n_rows = 8760

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)

            for i in range(n_rows):
                row = [oj_data[col][i] if col in oj_data else 0.0 for col in columns]
                writer.writerow(row)

        return str(output_path)


# ============================================================================
# 主评估器类
# ============================================================================

class CandidateEvaluator:
    """
    候选解评估器。

    串联 Phase 2/3/4，完成候选解评估流水线：
    1. 检查缓存，跳过重复评估
    2. 运行年度仿真 (Phase 2)
    3. 汇总成本与评分 (Phase 3)
    4. 导出统一输出文件 (Phase 4)
    5. 缓存评估结果
    """

    def __init__(
        self,
        config: Optional[EvaluatorConfig] = None,
        solve_day_func: Optional[Callable] = None,
    ):
        self.config = config or EvaluatorConfig()
        self.solve_day_func = solve_day_func
        self.logger = self._setup_logger()

        # 初始化组件
        self.cache = EvalCache(
            self.config.cache_dir,
            self.config.cache_format,
        ) if self.config.enable_cache else None

        self.output_gen = OutputGenerator(self.config.output_dir)

        # 统计
        self._eval_count = 0
        self._cache_hit_count = 0

        # 运行期缓存
        self._device_catalog: Optional[List[Dict[str, Any]]] = None
        self._base_caps: Optional[np.ndarray] = None
        self._score_config: Optional[Dict[str, Any]] = None
        self._carbon_factors: Optional[np.ndarray] = None
        self._gas_emission_factor: Optional[float] = None
        self._guidance_base: Optional[Dict[str, Any]] = None

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ga_evaluator")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S"
            ))
            logger.addHandler(handler)
        return logger

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _resolve_data_dir(self) -> Path:
        if self.config.data_dir:
            return Path(self.config.data_dir)
        return self._project_root() / "matlab" / "data" / "raw"

    def _resolve_renewable_dir(self, data_dir: Path) -> Path:
        if self.config.renewable_dir:
            return Path(self.config.renewable_dir)
        return data_dir

    def _load_guidance_base(self) -> Optional[Dict[str, Any]]:
        if self._guidance_base is not None:
            return self._guidance_base

        data_dir = self._resolve_data_dir()
        gas_to_MWh = float(self._load_score_config().get("gas_to_MWh", 0.01) or 0.01)
        price_path_e = data_dir / "电源价格_电.csv"
        price_path_g = data_dir / "天然气价格.csv"

        try:
            if price_path_e.exists():
                p_base_e, dates_e = _load_price_csv(price_path_e)
            else:
                p_base_e, dates_e = _generate_mock_price()
        except Exception as exc:
            self.logger.warning("加载电价失败: %s", exc)
            return None

        try:
            if price_path_g.exists():
                p_base_g, dates_g = _load_price_csv(price_path_g)
            else:
                p_base_g, dates_g = _generate_mock_price()
            p_base_g = p_base_g / gas_to_MWh
        except Exception as exc:
            self.logger.warning("加载气价失败: %s", exc)
            return None

        self._guidance_base = {
            "p_base_e": p_base_e,
            "dates_e": dates_e,
            "segment_map_e": generate_segment_map(p_base_e, dates_e),
            "p_base_g": p_base_g,
            "dates_g": dates_g,
            "segment_map_g": generate_segment_map(p_base_g, dates_g),
        }
        return self._guidance_base

    def _build_guidance_prices(
        self,
        price12: Optional[np.ndarray],
        gas12: Optional[np.ndarray],
    ) -> Optional[Dict[str, np.ndarray]]:
        base = self._load_guidance_base()
        if base is None:
            return None

        def build_matrix(
            multipliers: Optional[np.ndarray],
            p_base: np.ndarray,
            segment_map: Any,
            dates: np.ndarray,
        ) -> np.ndarray:
            if multipliers is None:
                return p_base
            mults = np.asarray(multipliers, dtype=float).reshape(-1)
            if mults.size == 192:
                return generate_guided_price_192(p_base, mults, dates)
            return generate_guided_price(p_base, mults, segment_map, dates)

        p_guided = build_matrix(
            price12,
            base["p_base_e"],
            base["segment_map_e"],
            base["dates_e"],
        )
        g_guided = build_matrix(
            gas12,
            base["p_base_g"],
            base["segment_map_g"],
            base["dates_g"],
        )
        return {"electricity": p_guided, "gas": g_guided}

    def _load_device_catalog(self) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        if self._device_catalog is not None and self._base_caps is not None:
            return self._device_catalog, self._base_caps

        catalog_path = self.config.device_catalog_path
        if catalog_path is None:
            catalog_path = self._project_root() / "spec" / "device_catalog.yaml"
        else:
            catalog_path = Path(catalog_path)

        if not catalog_path.exists():
            raise FileNotFoundError(f"设备目录不存在: {catalog_path}")
        if not HAS_YAML:
            raise ImportError("需要 PyYAML 解析 device_catalog.yaml")

        data = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
        devices = data.get("device_catalog", {}).get("devices", [])
        devices_sorted = sorted(devices, key=lambda d: d.get("idx", 0))

        device_catalog: List[Dict[str, Any]] = []
        base_caps: List[float] = []
        for item in devices_sorted:
            base_caps.append(float(item.get("base_capacity", 0.0)))
            device_catalog.append({
                "unit_cost": float(item.get("unit_cost", 0.0)),
                "lifespan_years": int(item.get("lifespan", item.get("lifespan_years", 20))),
            })

        self._device_catalog = device_catalog
        self._base_caps = np.asarray(base_caps, dtype=float)
        return device_catalog, self._base_caps

    def _load_score_config(self) -> Dict[str, Any]:
        if self._score_config is not None:
            return self._score_config

        score_path = self.config.score_spec_path
        if score_path is None:
            score_path = self._project_root() / "configs" / "oj_score.yaml"
        else:
            score_path = Path(score_path)

        if score_path.exists() and HAS_YAML:
            data = yaml.safe_load(score_path.read_text(encoding="utf-8"))
            score = data.get("score_spec", {})
            self._score_config = {
                "discount_rate": score.get("capex", {}).get("discount_rate", 0.04),
                "carbon_threshold": score.get("carbon", {}).get("threshold", 100000.0),
                "carbon_price": score.get("carbon", {}).get("price", 600.0),
                "gas_emission_factor": score.get("carbon", {}).get("gas_emission_factor", 0.002),
                "gas_to_MWh": score.get("units", {}).get("gas_m3_to_MWh", 0.01),
            }
        else:
            self._score_config = {
                "discount_rate": 0.04,
                "carbon_threshold": 100000.0,
                "carbon_price": 600.0,
                "gas_emission_factor": 0.002,
                "gas_to_MWh": 0.01,
            }
        if self._gas_emission_factor:
            self._score_config["gas_emission_factor"] = self._gas_emission_factor
        return self._score_config

    def _load_carbon_factors(self) -> Optional[np.ndarray]:
        if self._carbon_factors is not None:
            return self._carbon_factors
        data_dir = self._resolve_data_dir()
        renewable_dir = self._resolve_renewable_dir(data_dir)
        parsed = _parse_meos_inputs(data_dir, renewable_dir)
        self._carbon_factors = np.asarray(parsed.carbon_electricity, dtype=float)
        gas_factors = np.asarray(parsed.carbon_gas, dtype=float).reshape(-1)
        if gas_factors.size:
            mean_factor = float(np.mean(gas_factors))
            if mean_factor > 0:
                self._gas_emission_factor = mean_factor
        return self._carbon_factors

    def _normalize_day_result(self, day_result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(day_result, dict):
            return {"status": "failed", "error": "invalid_day_result"}

        result = dict(day_result)
        if "cost_original" in result:
            if "cost" in result:
                result["cost_guidance"] = result.get("cost", {})
            cost_src = result.get("cost_original", {})
            result["cost"] = {
                "electricity": cost_src.get("electricity", 0.0),
                "gas": cost_src.get("gas", 0.0),
                "penalty": cost_src.get("penalty", 0.0),
                "total": cost_src.get("total", 0.0),
            }

        if "P_grid" not in result and "P_thermal" in result:
            thermal = np.asarray(result.get("P_thermal", 0.0), dtype=float)
            result["P_grid"] = np.sum(thermal, axis=1) if thermal.ndim == 2 else thermal

        if "P_grid" in result:
            result["P_grid"] = np.asarray(result.get("P_grid", 0.0), dtype=float).reshape(-1).tolist()

        if "L_shed" in result:
            shed = np.asarray(result.get("L_shed", 0.0), dtype=float)
            if shed.ndim == 2:
                shed = np.sum(shed, axis=1)
            result["L_shed"] = shed.reshape(-1).tolist()

        if "G_buy" in result:
            result["G_buy"] = np.asarray(result.get("G_buy", 0.0), dtype=float).reshape(-1).tolist()

        if "shed_load" not in result and "L_shed" in result and "L_shed" in day_result:
            raw_shed = np.asarray(day_result.get("L_shed", 0.0), dtype=float)
            if raw_shed.ndim == 2 and raw_shed.shape[1] == 9:
                result["shed_load"] = raw_shed

        if "thermal_power" not in result and "P_thermal" in result:
            result["thermal_power"] = result.get("P_thermal", 0.0)
        if "gas_source" not in result and "G_buy" in result:
            result["gas_source"] = result.get("G_buy", 0.0)

        if "gas_source" in result:
            gas = np.asarray(result.get("gas_source", 0.0), dtype=float)
            if gas.ndim == 1:
                result["gas_source"] = gas.reshape(-1, 1)

        return result

    def evaluate(self, candidate: Candidate) -> EvalResult:
        """评估单个候选解"""
        cache_key = candidate.cache_key
        self._eval_count += 1

        # 回调：评估开始
        if self.config.on_eval_start:
            self.config.on_eval_start(candidate)

        # 1. 检查缓存
        if self.cache and self.cache.exists(cache_key):
            self._cache_hit_count += 1
            if self.config.verbose:
                self.logger.info(f"缓存命中: {cache_key}")
            result = self.cache.get(cache_key)
            if self.config.on_eval_end:
                self.config.on_eval_end(candidate, result)
            return result

        # 2. 执行评估
        start_time = time.time()
        try:
            result = self._run_evaluation(candidate)
            result.status = "success"
        except Exception as e:
            result = EvalResult(
                status="failed",
                error_msg=str(e),
                cache_key=cache_key,
            )
            self.logger.error(f"评估失败 [{cache_key}]: {e}")

        result.elapsed_sec = time.time() - start_time
        result.cache_key = cache_key
        result.timestamp = datetime.now().isoformat()

        # 3. 缓存结果
        if self.cache and result.status == "success":
            self.cache.put(cache_key, result)

        # 回调：评估结束
        if self.config.on_eval_end:
            self.config.on_eval_end(candidate, result)

        if self.config.verbose:
            self.logger.info(
                f"评估完成 [{cache_key}]: "
                f"Score={result.Score:.2f}, 耗时={result.elapsed_sec:.2f}s"
            )

        return result

    def _run_evaluation(self, candidate: Candidate) -> EvalResult:
        """执行实际评估流程"""
        plan18 = candidate.plan18
        cache_key = candidate.cache_key

        # Phase 2: 年度仿真
        daily_results, dispatch_data = self._run_annual_simulation(plan18)

        # Phase 3: 成本汇总与评分
        costs = self._calculate_costs(daily_results, plan18)

        # Phase 4: 导出统一输出文件
        output_paths = {}
        if self.config.export_outputs:
            output_paths = self._export_outputs(
                plan18,
                dispatch_data,
                cache_key,
                price12=candidate.price12,
                gas12=candidate.gas12,
            )

        return EvalResult(
            C_CAP=costs["C_CAP"],
            C_OP=costs["C_OP"],
            C_Carbon=costs["C_Carbon"],
            C_total=costs["C_total"],
            Score=costs["Score"],
            capacity_yaml=output_paths.get("capacity_yaml"),
            guide_price_csv=output_paths.get("guide_price_csv"),
            guide_gas_csv=output_paths.get("guide_gas_csv"),
            platform_csv=output_paths.get("platform_csv"),
            oj_csv=output_paths.get("oj_csv"),
        )

    def _run_annual_simulation(
        self, plan18: np.ndarray
    ) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
        """运行年度仿真（Phase 2）"""
        n_days = self.config.n_days
        n_hours = self.config.n_hours

        daily_results = []
        # 初始化调度数据
        dispatch_data = {
            "shed_load": np.zeros((n_days * n_hours, 9)),
            "thermal_power": np.zeros((n_days * n_hours, 3)),
            "gas_source": np.zeros((n_days * n_hours, 1)),
        }

        for day in range(1, n_days + 1):
            if self.solve_day_func:
                # 使用实际求解函数
                raw_result = self.solve_day_func(day, plan18)
            else:
                # 使用 mock 数据
                raw_result = self._mock_solve_day(day, plan18)

            day_result = self._normalize_day_result(raw_result)
            daily_results.append(day_result)

            # 填充调度数据
            h_start = (day - 1) * n_hours
            h_end = day * n_hours
            if "shed_load" in day_result:
                dispatch_data["shed_load"][h_start:h_end] = day_result["shed_load"]
            if "thermal_power" in day_result:
                dispatch_data["thermal_power"][h_start:h_end] = day_result["thermal_power"]
            if "gas_source" in day_result:
                dispatch_data["gas_source"][h_start:h_end] = day_result["gas_source"]

        return daily_results, dispatch_data

    def _mock_solve_day(self, day: int, plan18: np.ndarray) -> Dict[str, Any]:
        """Mock 单日求解（用于测试）"""
        import math
        n_hours = self.config.n_hours
        cap_sum = float(np.sum(plan18))

        # 基于容量生成 mock 成本
        base_cost = 1000 + cap_sum * 10
        daily_var = math.sin(day * 0.1) * 100

        return {
            "cost": {
                "electricity": base_cost * 0.6 + daily_var,
                "gas": base_cost * 0.3,
                "penalty": max(0, 50 - cap_sum) * 100,
                "total": base_cost + daily_var,
            },
            "P_grid": [5.0 + cap_sum * 0.01] * n_hours,
            "G_buy": [2.0 + cap_sum * 0.005] * n_hours,
            "L_shed": [0.0] * n_hours,
            "shed_load": np.zeros((n_hours, 9)),
            "thermal_power": np.ones((n_hours, 3)) * (cap_sum / 30),
            "gas_source": np.ones((n_hours, 1)) * (cap_sum / 50),
        }

    def _calculate_costs(
        self, daily_results: List[Dict], plan18: np.ndarray
    ) -> Dict[str, float]:
        """计算成本与评分（Phase 3）"""
        try:
            device_catalog, base_caps = self._load_device_catalog()
            carbon_factors = self._load_carbon_factors()
            score_config = self._load_score_config()

            plan_capacity = plan18.astype(float) * base_caps
            if plan_capacity.size >= 15:
                # 风电/光伏的 plan18 在 OJ 中按 MW 计，不再乘 base_capacity
                plan_capacity[13] = plan18[13]
                plan_capacity[14] = plan18[14]
            summary = summarize_annual(
                daily_results=daily_results,
                plan18=plan_capacity.tolist(),
                device_catalog=device_catalog,
                carbon_factors=carbon_factors.tolist() if carbon_factors is not None else None,
                config=score_config,
                day_weights=self.config.day_weights,
            )
            if self.config.day_weights is None:
                annualize_days = self.config.annualize_days
                if annualize_days and summary.n_days and summary.n_days != annualize_days:
                    scale = float(annualize_days) / float(summary.n_days)
                    C_OP_ele = summary.C_OP_ele * scale
                    C_OP_gas = summary.C_OP_gas * scale
                    C_OP_penalty = summary.C_OP_penalty * scale
                    C_OP_total = C_OP_ele + C_OP_gas + C_OP_penalty

                    E_elec = summary.carbon.emission_electricity * scale
                    E_gas = summary.carbon.emission_gas * scale
                    C_Carbon, _ = calculate_carbon_cost(
                        E_elec,
                        E_gas,
                        score_config.get("carbon_threshold", 100000.0),
                        score_config.get("carbon_price", 600.0),
                    )

                    C_total = summary.C_CAP + C_OP_total + C_Carbon
                    Score = calculate_score(C_total)
                    return {
                        "C_CAP": summary.C_CAP,
                        "C_OP": C_OP_total,
                        "C_Carbon": C_Carbon,
                        "C_total": C_total,
                        "Score": Score,
                    }
            return {
                "C_CAP": summary.C_CAP,
                "C_OP": summary.C_OP_total,
                "C_Carbon": summary.C_Carbon,
                "C_total": summary.C_total,
                "Score": summary.Score,
            }
        except Exception as exc:
            self.logger.warning(f"成本核算回退到简化模式: {exc}")

            C_OP_ele = sum(d["cost"]["electricity"] for d in daily_results)
            C_OP_gas = sum(d["cost"]["gas"] for d in daily_results)
            C_OP_penalty = sum(d["cost"]["penalty"] for d in daily_results)
            C_OP = C_OP_ele + C_OP_gas + C_OP_penalty

            C_CAP = float(np.sum(plan18)) * 5000
            C_Carbon = max(0, C_OP * 0.01 - 10000)
            C_total = C_CAP + C_OP + C_Carbon
            z = (C_total / 10000 - 100000) / 15000
            if z > 700:
                Score = 0.0
            elif z < -700:
                Score = 100.0
            else:
                Score = 100.0 / (1.0 + math.exp(z))

            return {
                "C_CAP": C_CAP,
                "C_OP": C_OP,
                "C_Carbon": C_Carbon,
                "C_total": C_total,
                "Score": Score,
            }

    def _export_outputs(
        self,
        plan18: np.ndarray,
        dispatch_data: Dict[str, np.ndarray],
        cache_key: str,
        price12: Optional[np.ndarray] = None,
        gas12: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """导出统一输出文件（Phase 4）"""
        paths = {}

        # 1. 容量 YAML
        paths["capacity_yaml"] = self.output_gen.export_capacity_yaml(
            plan18, cache_key
        )

        # 2. 引导价 CSV（电/气）
        guided = self._build_guidance_prices(price12, gas12)
        if guided is None:
            guide_prices = {
                "electricity": [0.0] * 8760,
                "gas": [0.0] * 8760,
            }
        else:
            guide_prices = {
                "electricity": guided["electricity"].reshape(-1).tolist(),
                "gas": guided["gas"].reshape(-1).tolist(),
            }
        paths["guide_price_csv"] = self.output_gen.export_guide_price_csv(guide_prices, cache_key)
        paths["guide_gas_csv"] = self.output_gen.export_guide_gas_csv(guide_prices["gas"], cache_key)

        # 3. 平台 CSV
        paths["platform_csv"] = self.output_gen.export_platform_csv(
            dispatch_data, plan18, cache_key
        )

        # 4. OJ CSV
        oj_data = self._build_oj_data(dispatch_data, plan18)
        paths["oj_csv"] = self.output_gen.export_oj_csv(oj_data, cache_key)

        return paths

    def _build_oj_data(
        self,
        dispatch_data: Dict[str, np.ndarray],
        plan18: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """构建 OJ 数据"""
        n_hours = self.config.n_days * self.config.n_hours
        n_oj_hours = 8760

        shed = dispatch_data.get("shed_load", np.zeros((n_hours, 9)))
        thermal = dispatch_data.get("thermal_power", np.zeros((n_hours, 3)))
        gas = dispatch_data.get("gas_source", np.zeros((n_hours, 1)))

        # 扩展到 8760 小时（如果仿真天数不足365天）
        def pad_to_8760(arr: np.ndarray) -> np.ndarray:
            if arr.shape[0] >= n_oj_hours:
                return arr[:n_oj_hours]
            padded = np.zeros((n_oj_hours,) + arr.shape[1:])
            padded[:arr.shape[0]] = arr
            return padded

        shed = pad_to_8760(shed)
        thermal = pad_to_8760(thermal)
        gas = pad_to_8760(gas)

        # 汇总切负荷
        ans_load1 = np.sum(shed[:, [0, 3, 6]], axis=1)  # 电
        ans_load2 = np.sum(shed[:, [2, 5, 7]], axis=1)  # 热
        ans_load3 = np.sum(shed[:, [1, 4, 8]], axis=1)  # 冷

        # 汇总火电
        ans_ele = np.sum(thermal, axis=1)

        # 气源
        ans_gas = gas.flatten()

        # 规划向量
        ans_planning = np.zeros(n_oj_hours)
        ans_planning[:18] = plan18

        return {
            "ans_load1": ans_load1,
            "ans_load2": ans_load2,
            "ans_load3": ans_load3,
            "ans_ele": ans_ele,
            "ans_gas": ans_gas,
            "ans_planning": ans_planning,
        }

    def evaluate_batch(
        self, candidates: List[Candidate]
    ) -> List[EvalResult]:
        """批量评估候选解"""
        results = []
        for i, cand in enumerate(candidates):
            if self.config.verbose:
                self.logger.info(f"评估 {i+1}/{len(candidates)}")
            results.append(self.evaluate(cand))
        return results

    def stats(self) -> Dict[str, Any]:
        """获取评估器统计信息"""
        cache_stats = self.cache.stats() if self.cache else {}
        return {
            "eval_count": self._eval_count,
            "cache_hit_count": self._cache_hit_count,
            "cache_hit_rate": (
                self._cache_hit_count / self._eval_count
                if self._eval_count > 0 else 0.0
            ),
            "cache": cache_stats,
        }

    def clear_cache(self) -> int:
        """清空缓存"""
        if self.cache:
            return self.cache.clear()
        return 0


# ============================================================================
# 便捷函数
# ============================================================================

def evaluate_candidate(
    plan18: Union[List[float], np.ndarray],
    config: Optional[EvaluatorConfig] = None,
) -> EvalResult:
    """便捷函数：评估单个候选解"""
    evaluator = CandidateEvaluator(config)
    candidate = Candidate(plan18=np.asarray(plan18))
    return evaluator.evaluate(candidate)


def evaluate_candidates(
    plan18_list: List[Union[List[float], np.ndarray]],
    config: Optional[EvaluatorConfig] = None,
) -> List[EvalResult]:
    """便捷函数：批量评估候选解"""
    evaluator = CandidateEvaluator(config)
    candidates = [
        Candidate(plan18=np.asarray(p), index=i)
        for i, p in enumerate(plan18_list)
    ]
    return evaluator.evaluate_batch(candidates)
