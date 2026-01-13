"""
拓扑解析器：将 topology.json 规范化为 schema_spec.md 定义的拓扑结构。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class TopologyLoader:
    """拓扑解析器：规范化端口字段并补齐拓扑信息。"""

    def __init__(self, topology_path: str | Path):
        self.topology_path = Path(topology_path)
        self.raw_data: List[Dict[str, Any]] = []
        self.systems: List[Dict[str, Any]] = []

    def load(self) -> "TopologyLoader":
        """加载并解析拓扑文件"""
        with open(self.topology_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("topology.json 需要是系统列表结构")
        self.raw_data = data
        self.systems = [self._normalize_system(system) for system in data]
        return self

    def _normalize_system(self, system: Dict[str, Any]) -> Dict[str, Any]:
        """规范化单个系统"""
        normalized: Dict[str, Any] = {}

        system_info = dict(system.get("system_info", {}))
        if "topology_type" not in system_info:
            if system_info.get("parent_id"):
                system_info["topology_type"] = "Subsystem"
            else:
                system_info["topology_type"] = "Top_Level_Network"
        normalized["system_info"] = system_info

        if "network_params" in system:
            normalized["network_params"] = system["network_params"]

        nodes = system.get("nodes", [])
        normalized["nodes"] = [self._normalize_node(node) for node in nodes]

        links = system.get("links", [])
        normalized["links"] = [self._normalize_link(link) for link in links]

        return normalized

    def _normalize_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """规范化节点"""
        normalized = {
            "id": node["id"],
            "name": node.get("name", node["id"]),
            "type": node.get("type", "Unknown"),
        }
        if "category" in node:
            normalized["category"] = node["category"]
        if "description" in node:
            normalized["description"] = node["description"]
        if "ports" in node:
            normalized["ports"] = [self._normalize_port(port) for port in node.get("ports", [])]
        return normalized

    def _normalize_port(self, port: Dict[str, Any]) -> Dict[str, Any]:
        """规范化端口字段"""
        direction = port.get("direction", port.get("type"))
        normalized = {
            "id": port["id"],
            "medium": port["medium"],
            "direction": direction,
            "node_ref": port.get("node_ref"),
            "internal_ref": port.get("internal_ref"),
        }
        if "name" in port:
            normalized["name"] = port["name"]
        return normalized

    def _normalize_link(self, link: Dict[str, Any]) -> Dict[str, Any]:
        """规范化连接"""
        normalized = {
            "source": link["source"],
            "target": link["target"],
            "medium": link["medium"],
            "flow_type": link.get("flow_type", "Unidirectional"),
        }
        if "category" in link:
            normalized["category"] = link["category"]
        if "note" in link:
            normalized["note"] = link["note"]
        return normalized

    def to_schema_list(self) -> List[Dict[str, Any]]:
        """导出规范化拓扑结构（系统列表）"""
        return list(self.systems)

    def to_normalized_dict(self) -> List[Dict[str, Any]]:
        """兼容旧命名：返回规范化系统列表"""
        return self.to_schema_list()

    def get_statistics(self) -> Dict[str, Any]:
        """获取拓扑统计信息"""
        total_nodes = 0
        total_links = 0
        by_medium: Dict[str, int] = {}
        by_subsystem: Dict[str, int] = {}

        for system in self.systems:
            system_info = system.get("system_info", {})
            subsystem_key = system_info.get("parent_id") or "top_level"

            nodes = system.get("nodes", [])
            links = system.get("links", [])
            total_nodes += len(nodes)
            total_links += len(links)

            for link in links:
                medium = link.get("medium", "Unknown")
                by_medium[medium] = by_medium.get(medium, 0) + 1
                by_subsystem[subsystem_key] = by_subsystem.get(subsystem_key, 0) + 1

        return {
            "total_branches": total_links,
            "total_nodes": total_nodes,
            "branches_by_medium": by_medium,
            "branches_by_subsystem": by_subsystem,
        }
