"""
MEOS 属性解析器
将 attributes.json 规范化为 schema_spec.md 定义的 Attributes 结构
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


PARAMETER_FIELDS = {
    "max_output_MW",
    "min_output_MW",
    "max_input_MW",
    "capacity_MW",
    "efficiency",
    "COP",
    "eta_e",
    "eta_h",
    "eta_c",
    "base_injection_MW",
}

CONSTRAINT_FIELDS = {
    "gas_to_MWh",
    "load_shedding_penalty",
}


class AttributesLoader:
    """属性文件加载与规范化解析器"""

    def __init__(self, file_path: Optional[str] = None):
        self.file_path = Path(file_path) if file_path else None
        self.raw_data: Dict[str, Any] = {}
        self.normalized: Optional[Dict[str, Any]] = None

    def load(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """加载原始 JSON 文件"""
        path = Path(file_path) if file_path else self.file_path
        if not path:
            raise ValueError("未指定文件路径")
        with open(path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        return self.raw_data

    def _extract_parameters(self, raw_params: Dict[str, Any]) -> Dict[str, Any]:
        """提取设备参数字段"""
        params = {}
        for key in PARAMETER_FIELDS:
            if key in raw_params:
                params[key] = raw_params[key]
        return params

    def _normalize_sources(self, top_level: Dict[str, Any]) -> List[Dict[str, Any]]:
        """规范化顶层源设备"""
        sources = []
        for raw_source in top_level.get("sources", []):
            params = self._extract_parameters(raw_source.get("parameters", {}))
            source = {
                "id": raw_source["id"],
                "name": raw_source["name"],
                "type": raw_source["type"],
                "node_ref": raw_source.get("node_ref"),
            }
            if params:
                source["parameters"] = params
            sources.append(source)
        return sources

    def _normalize_zones(self, zones_raw: Dict[str, Any]) -> Dict[str, Any]:
        """规范化各区域设备"""
        zones: Dict[str, Any] = {}
        for zone_id, zone_data in zones_raw.items():
            devices = []
            for raw_device in zone_data.get("devices", []):
                params = self._extract_parameters(raw_device)
                device = {
                    "id": raw_device["id"],
                    "name": raw_device["name"],
                    "type": raw_device["type"],
                }
                if params:
                    device["parameters"] = params
                devices.append(device)

            zone = {
                "id": zone_id,
                "name": zone_data.get("name", zone_id),
                "devices": devices,
            }
            zones[zone_id] = zone
        return zones

    def _normalize_constraints(self, raw_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """规范化约束字段"""
        constraints = {}
        for key in CONSTRAINT_FIELDS:
            if key in raw_constraints:
                constraints[key] = raw_constraints[key]
        return constraints

    def normalize(self) -> Dict[str, Any]:
        """执行规范化解析"""
        if not self.raw_data:
            raise ValueError("请先调用 load() 加载数据")

        metadata = self.raw_data.get("metadata", {})
        top_level = self.raw_data.get("top_level", {})
        zones = self.raw_data.get("zones", {})
        constraints = self.raw_data.get("constraints", {})

        normalized: Dict[str, Any] = {
            "version": metadata.get("version", 1),
            "sources": self._normalize_sources(top_level),
            "zones": self._normalize_zones(zones),
        }

        description = metadata.get("description")
        if description:
            normalized["description"] = description

        normalized_constraints = self._normalize_constraints(constraints)
        if normalized_constraints:
            normalized["constraints"] = normalized_constraints

        self.normalized = normalized
        return normalized

    def to_dict(self) -> Dict[str, Any]:
        """将规范化结果转为字典"""
        if not self.normalized:
            raise ValueError("请先调用 normalize() 进行规范化")
        return dict(self.normalized)

    def export_json(self, output_path: str) -> None:
        """导出规范化结果为 JSON 文件"""
        data = self.to_dict()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ============ 便捷函数 ============

def load_attributes(file_path: str) -> Dict[str, Any]:
    """加载原始属性文件"""
    loader = AttributesLoader(file_path)
    return loader.load()


def normalize_attributes(file_path: str) -> Dict[str, Any]:
    """加载并规范化属性文件"""
    loader = AttributesLoader(file_path)
    loader.load()
    return loader.normalize()
