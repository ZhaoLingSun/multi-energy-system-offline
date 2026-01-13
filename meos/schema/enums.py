"""
MEOS Schema - 枚举类型定义
定义多能源系统中使用的所有枚举类型
"""

from enum import Enum


class Medium(str, Enum):
    """能源介质类型"""
    ELECTRICITY = "Electricity"
    GAS = "Gas"
    HEAT = "Heat"
    COOLING = "Cooling"


class NodeType(str, Enum):
    """节点类型"""
    GRID_NODE = "Grid_Node"      # 网络节点（母线/Hub）
    SOURCE = "Source"            # 源节点
    SINK = "Sink"                # 汇节点
    CONVERTER = "Converter"      # 转换设备
    LOAD = "Load"                # 负荷节点
    SUBSYSTEM = "Subsystem"      # 子系统


class NodeCategory(str, Enum):
    """节点细分类别"""
    # 电力类
    ELECTRICITY_BUS = "Electricity_Bus"
    THERMAL_GENERATOR = "Thermal_Generator"
    # 气网类
    GAS_HUB = "Gas_Hub"
    GAS_SOURCE = "Gas_Source"
    # 热网类
    HEAT_HUB = "Heat_Hub"
    # 端口类
    INPUT_PORT = "Input_Port"
    OUTPUT_PORT = "Output_Port"
    # 设备类
    DEVICE = "Device"
    RENEWABLE = "Renewable"
    # 传输类
    TRANSMISSION_LINE = "Transmission_Line"


class PortDirection(str, Enum):
    """端口方向"""
    INPUT = "Input"
    OUTPUT = "Output"


class FlowType(str, Enum):
    """流向类型"""
    UNIDIRECTIONAL = "Unidirectional"
    BIDIRECTIONAL = "Bidirectional"


class DeviceType(str, Enum):
    """设备类型"""
    # 发电设备
    THERMAL_GENERATOR = "Thermal_Generator"
    NATURAL_GAS_SOURCE = "Natural_Gas_Source"
    RENEWABLE = "Renewable"
    # 转换设备
    P2G = "P2G"
    E_BOILER = "E_Boiler"
    GAS_TURBINE = "GasTurbine"
    GAS_BOILER = "GasBoiler"
    HEAT_PUMP = "HeatPump"
    CHILLER = "Chiller"
    ABSORPTION = "Absorption"
    CCHP = "CCHP"


class TopologyType(str, Enum):
    """拓扑类型"""
    TOP_LEVEL_NETWORK = "Top_Level_Network"
    SUBSYSTEM = "Subsystem"
