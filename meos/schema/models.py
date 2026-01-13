"""
MEOS Schema - 核心数据模型
定义多能源系统的统一数据结构
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from .enums import (
    Medium, NodeType, NodeCategory, PortDirection,
    FlowType, DeviceType, TopologyType
)


@dataclass
class Port:
    """
    端口：子系统与外部网络的连接接口
    """
    id: str                          # 端口唯一标识
    medium: Medium                   # 能源介质类型
    direction: PortDirection         # 端口方向 (Input/Output)
    node_ref: str                    # 外部网络节点引用
    internal_ref: str                # 内部节点引用
    name: Optional[str] = None       # 端口名称（可选）


@dataclass
class Node:
    """
    节点：网络中的基本单元
    """
    id: str                          # 节点唯一标识
    name: str                        # 节点名称
    type: NodeType                   # 节点类型
    category: Optional[NodeCategory] = None  # 节点细分类别
    description: Optional[str] = None        # 描述信息
    ports: List[Port] = field(default_factory=list)  # 端口列表（仅Subsystem）


@dataclass
class Link:
    """
    连接：节点之间的能量传输路径
    """
    source: str                      # 源节点ID
    target: str                      # 目标节点ID
    medium: Medium                   # 能源介质类型
    flow_type: FlowType              # 流向类型
    category: Optional[NodeCategory] = None  # 连接类别
    note: Optional[str] = None       # 备注信息


@dataclass
class DeviceParameters:
    """
    设备参数：设备运行特性参数
    """
    max_output_MW: Optional[float] = None    # 最大输出功率
    min_output_MW: Optional[float] = None    # 最小输出功率
    max_input_MW: Optional[float] = None     # 最大输入功率
    capacity_MW: Optional[float] = None      # 装机容量
    efficiency: Optional[float] = None       # 效率
    COP: Optional[float] = None              # 性能系数
    eta_e: Optional[float] = None            # 电效率（CCHP）
    eta_h: Optional[float] = None            # 热效率（CCHP）
    eta_c: Optional[float] = None            # 冷效率（CCHP）
    base_injection_MW: Optional[float] = None  # 基础注入功率


@dataclass
class Device:
    """
    设备：能源转换或生产设备
    """
    id: str                          # 设备唯一标识
    name: str                        # 设备名称
    type: DeviceType                 # 设备类型
    node_ref: Optional[str] = None   # 关联节点引用
    parameters: DeviceParameters = field(default_factory=DeviceParameters)


@dataclass
class ElectricNetworkParams:
    """电网参数"""
    line_reactance: float            # 线路电抗
    line_capacity: float             # 线路容量
    theta_min: float                 # 最小相角
    theta_max: float                 # 最大相角


@dataclass
class GasNetworkParams:
    """气网参数"""
    pressure_min_MPa: float          # 最小压力
    pressure_max_MPa: float          # 最大压力
    pressure_base_MPa: float         # 基准压力


@dataclass
class HeatNetworkParams:
    """热网参数"""
    capacity_MW: Optional[float] = None  # 容量


@dataclass
class NetworkParams:
    """网络参数集合"""
    electric: Optional[ElectricNetworkParams] = None
    gas: Optional[GasNetworkParams] = None
    heat: Optional[HeatNetworkParams] = None


@dataclass
class SystemInfo:
    """系统元信息"""
    system_name: str                 # 系统名称
    parent_id: Optional[str] = None  # 父系统ID
    description: Optional[str] = None
    topology_type: Optional[TopologyType] = None


@dataclass
class Subsystem:
    """子系统拓扑"""
    system_info: SystemInfo
    nodes: List[Node] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)


@dataclass
class TopLevelNetwork:
    """顶层网络拓扑"""
    system_info: SystemInfo
    network_params: Optional[NetworkParams] = None
    nodes: List[Node] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)


@dataclass
class Zone:
    """区域：包含设备的逻辑分组"""
    id: str                          # 区域ID
    name: str                        # 区域名称
    devices: List[Device] = field(default_factory=list)


@dataclass
class Constraints:
    """系统约束参数"""
    gas_to_MWh: Optional[float] = None
    load_shedding_penalty: Optional[float] = None


@dataclass
class Attributes:
    """设备属性集合（对应 attributes.json）"""
    version: int = 1
    description: Optional[str] = None
    sources: List[Device] = field(default_factory=list)
    zones: Dict[str, Zone] = field(default_factory=dict)
    constraints: Optional[Constraints] = None


@dataclass
class MultiEnergySystem:
    """
    完整的多能源系统模型
    整合拓扑与属性信息
    """
    top_level: TopLevelNetwork
    subsystems: List[Subsystem] = field(default_factory=list)
    attributes: Optional[Attributes] = None
