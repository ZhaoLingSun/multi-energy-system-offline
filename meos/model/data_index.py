"""
MEOS Model - 数据索引构建模块

从 normalized_topology.json 和 normalized_attributes.json 构建可建模索引。
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from pathlib import Path


# ============================================================================
# 时间序列输入接口（与 MATLAB preprocess 对齐）
# ============================================================================

@dataclass
class LoadProfile:
    """负荷时序数据"""
    electric: Dict[str, List[float]] = field(default_factory=dict)  # zone_id -> [T]
    heat: Dict[str, List[float]] = field(default_factory=dict)
    cool: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class RenewableProfile:
    """可再生能源出力时序"""
    pv: Dict[str, List[float]] = field(default_factory=dict)   # device_id -> [T]
    wind: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class PriceProfile:
    """价格时序数据"""
    electricity: List[float] = field(default_factory=list)  # [T]
    gas: List[float] = field(default_factory=list)


@dataclass
class CarbonProfile:
    """碳排放因子时序"""
    electricity: List[float] = field(default_factory=list)  # kg/MWh [T]
    gas: List[float] = field(default_factory=list)          # kg/MWh [T]


@dataclass
class TimeSeriesInputs:
    """时序输入数据容器"""
    hours: int = 8760
    load: LoadProfile = field(default_factory=LoadProfile)
    renewable: RenewableProfile = field(default_factory=RenewableProfile)
    price: PriceProfile = field(default_factory=PriceProfile)
    carbon: CarbonProfile = field(default_factory=CarbonProfile)


# ============================================================================
# 拓扑索引结构
# ============================================================================

@dataclass
class NodeIndex:
    """节点索引"""
    all_nodes: Dict[str, dict] = field(default_factory=dict)
    by_medium: Dict[str, List[str]] = field(default_factory=dict)
    by_type: Dict[str, List[str]] = field(default_factory=dict)
    by_subsystem: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class DeviceIndex:
    """设备索引"""
    all_devices: Dict[str, dict] = field(default_factory=dict)
    by_type: Dict[str, List[str]] = field(default_factory=dict)
    by_zone: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class LinkIndex:
    """连接索引"""
    all_links: List[dict] = field(default_factory=list)
    by_medium: Dict[str, List[dict]] = field(default_factory=dict)
    by_source: Dict[str, List[dict]] = field(default_factory=dict)
    by_target: Dict[str, List[dict]] = field(default_factory=dict)


@dataclass
class PortIndex:
    """端口索引"""
    all_ports: Dict[str, dict] = field(default_factory=dict)
    by_subsystem: Dict[str, List[str]] = field(default_factory=dict)
    by_medium: Dict[str, List[str]] = field(default_factory=dict)
    by_direction: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class NetworkParams:
    """网络参数"""
    electric: Optional[dict] = None
    gas: Optional[dict] = None
    heat: Optional[dict] = None


@dataclass
class ModelDataIndex:
    """模型数据索引 - 主入口"""
    nodes: NodeIndex = field(default_factory=NodeIndex)
    devices: DeviceIndex = field(default_factory=DeviceIndex)
    links: LinkIndex = field(default_factory=LinkIndex)
    ports: PortIndex = field(default_factory=PortIndex)
    network_params: NetworkParams = field(default_factory=NetworkParams)
    constraints: dict = field(default_factory=dict)
    subsystem_ids: List[str] = field(default_factory=list)
    zone_ids: List[str] = field(default_factory=list)


# ============================================================================
# 索引构建器
# ============================================================================

class DataIndexBuilder:
    """从 normalized JSON 构建索引"""

    def __init__(self):
        self.index = ModelDataIndex()
        self.time_series = TimeSeriesInputs()

    def load_topology(self, topology_path: str) -> "DataIndexBuilder":
        """加载拓扑文件并构建索引"""
        with open(topology_path, 'r', encoding='utf-8') as f:
            topology_list = json.load(f)

        for system in topology_list:
            sys_info = system.get("system_info", {})
            topo_type = sys_info.get("topology_type")

            if topo_type == "Top_Level_Network":
                self._index_top_level(system)
            else:
                self._index_subsystem(system)

        return self

    def load_attributes(self, attributes_path: str) -> "DataIndexBuilder":
        """加载属性文件并构建设备索引"""
        with open(attributes_path, 'r', encoding='utf-8') as f:
            attrs = json.load(f)

        # 索引顶层源设备
        for source in attrs.get("sources", []):
            self._index_device(source, zone_id=None)

        # 索引各区域设备
        for zone_id, zone_data in attrs.get("zones", {}).items():
            self.index.zone_ids.append(zone_id)
            for device in zone_data.get("devices", []):
                self._index_device(device, zone_id=zone_id)

        # 约束参数
        self.index.constraints = attrs.get("constraints", {})

        return self

    def _index_top_level(self, system: dict) -> None:
        """索引顶层网络"""
        # 网络参数
        net_params = system.get("network_params", {})
        self.index.network_params.electric = net_params.get("electric")
        self.index.network_params.gas = net_params.get("gas")
        self.index.network_params.heat = net_params.get("heat")

        # 节点索引
        for node in system.get("nodes", []):
            self._index_node(node, subsystem_id=None)

        # 连接索引
        for link in system.get("links", []):
            self._index_link(link)

    def _index_subsystem(self, system: dict) -> None:
        """索引子系统"""
        sys_info = system.get("system_info", {})
        subsystem_id = sys_info.get("parent_id")
        self.index.subsystem_ids.append(subsystem_id)

        for node in system.get("nodes", []):
            self._index_node(node, subsystem_id=subsystem_id)

        for link in system.get("links", []):
            self._index_link(link)

    def _index_node(self, node: dict, subsystem_id: Optional[str]) -> None:
        """索引单个节点"""
        node_id = node["id"]
        node_type = node.get("type")

        # 存储完整节点
        self.index.nodes.all_nodes[node_id] = node

        # 按类型索引
        if node_type not in self.index.nodes.by_type:
            self.index.nodes.by_type[node_type] = []
        self.index.nodes.by_type[node_type].append(node_id)

        # 按子系统索引
        sub_key = subsystem_id or "top_level"
        if sub_key not in self.index.nodes.by_subsystem:
            self.index.nodes.by_subsystem[sub_key] = []
        self.index.nodes.by_subsystem[sub_key].append(node_id)

        # 处理端口（Subsystem 节点）
        if node_type == "Subsystem" and "ports" in node:
            for port in node["ports"]:
                self._index_port(port, subsystem_id=node_id)

    def _index_port(self, port: dict, subsystem_id: str) -> None:
        """索引端口"""
        port_id = port["id"]
        medium = port.get("medium")
        direction = port.get("direction")

        self.index.ports.all_ports[port_id] = port

        # 按子系统
        if subsystem_id not in self.index.ports.by_subsystem:
            self.index.ports.by_subsystem[subsystem_id] = []
        self.index.ports.by_subsystem[subsystem_id].append(port_id)

        # 按介质
        if medium not in self.index.ports.by_medium:
            self.index.ports.by_medium[medium] = []
        self.index.ports.by_medium[medium].append(port_id)

        # 按方向
        if direction not in self.index.ports.by_direction:
            self.index.ports.by_direction[direction] = []
        self.index.ports.by_direction[direction].append(port_id)

    def _index_link(self, link: dict) -> None:
        """索引连接"""
        medium = link.get("medium")
        source = link.get("source")
        target = link.get("target")

        self.index.links.all_links.append(link)

        # 按介质
        if medium not in self.index.links.by_medium:
            self.index.links.by_medium[medium] = []
        self.index.links.by_medium[medium].append(link)

        # 按源节点
        if source not in self.index.links.by_source:
            self.index.links.by_source[source] = []
        self.index.links.by_source[source].append(link)

        # 按目标节点
        if target not in self.index.links.by_target:
            self.index.links.by_target[target] = []
        self.index.links.by_target[target].append(link)

    def _index_device(self, device: dict, zone_id: Optional[str]) -> None:
        """索引设备"""
        device_id = device["id"]
        device_type = device.get("type")

        self.index.devices.all_devices[device_id] = device

        # 按类型
        if device_type not in self.index.devices.by_type:
            self.index.devices.by_type[device_type] = []
        self.index.devices.by_type[device_type].append(device_id)

        # 按区域
        zone_key = zone_id or "top_level"
        if zone_key not in self.index.devices.by_zone:
            self.index.devices.by_zone[zone_key] = []
        self.index.devices.by_zone[zone_key].append(device_id)

    def build(self) -> ModelDataIndex:
        """返回构建完成的索引"""
        return self.index

    def get_time_series(self) -> TimeSeriesInputs:
        """返回时序输入容器"""
        return self.time_series


# ============================================================================
# 查询辅助函数
# ============================================================================

def get_device_param(index: ModelDataIndex, device_id: str, param: str) -> Any:
    """获取设备参数"""
    device = index.devices.all_devices.get(device_id)
    if device:
        return device.get("parameters", {}).get(param)
    return None


def get_loads_in_zone(index: ModelDataIndex, zone_id: str) -> List[str]:
    """获取区域内的负荷节点"""
    nodes = index.nodes.by_subsystem.get(zone_id, [])
    return [n for n in nodes if index.nodes.all_nodes[n].get("type") == "Load"]


def get_converters_in_zone(index: ModelDataIndex, zone_id: str) -> List[str]:
    """获取区域内的转换设备"""
    nodes = index.nodes.by_subsystem.get(zone_id, [])
    return [n for n in nodes if index.nodes.all_nodes[n].get("type") == "Converter"]


# ============================================================================
# 便捷工厂函数
# ============================================================================

def build_index_from_files(
    topology_path: str,
    attributes_path: str
) -> ModelDataIndex:
    """从文件构建索引"""
    builder = DataIndexBuilder()
    builder.load_topology(topology_path)
    builder.load_attributes(attributes_path)
    return builder.build()
