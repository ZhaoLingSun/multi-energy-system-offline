# MEOS Schema 规范文档

## 概述

本文档定义多能源系统（Multi-Energy System）的统一数据模型规范，用于描述电、气、热、冷多种能源形式的网络拓扑与设备属性。

## 设计原则

1. **ID 保留原则**：所有实体保留原始 `id`，确保可追溯映射
2. **层次化结构**：顶层网络 → 子系统 → 设备/节点
3. **类型安全**：使用枚举约束关键字段取值
4. **可选字段**：非必填字段使用 `Optional` 标注

---

## 枚举类型

### Medium（能源介质）
| 值 | 说明 |
|---|---|
| `Electricity` | 电力 |
| `Gas` | 天然气 |
| `Heat` | 热能 |
| `Cooling` | 冷能 |

### NodeType（节点类型）
| 值 | 说明 |
|---|---|
| `Grid_Node` | 网络节点（母线/Hub） |
| `Source` | 源节点 |
| `Sink` | 汇节点 |
| `Converter` | 转换设备 |
| `Load` | 负荷节点 |
| `Subsystem` | 子系统 |

### FlowType（流向类型）
| 值 | 说明 |
|---|---|
| `Unidirectional` | 单向流动 |
| `Bidirectional` | 双向流动 |

### PortDirection（端口方向）
| 值 | 说明 |
|---|---|
| `Input` | 输入端口 |
| `Output` | 输出端口 |

### DeviceType（设备类型）
| 值 | 说明 |
|---|---|
| `Thermal_Generator` | 火电机组 |
| `Natural_Gas_Source` | 天然气源 |
| `Renewable` | 可再生能源（光伏/风电） |
| `P2G` | 电制气 |
| `E_Boiler` | 电锅炉 |
| `GasTurbine` | 燃气轮机 |
| `GasBoiler` | 燃气锅炉 |
| `HeatPump` | 热泵 |
| `Chiller` | 压缩式制冷机 |
| `Absorption` | 吸收式制冷机 |
| `CCHP` | 冷热电联供 |

---

## 核心数据结构

### Port（端口）

子系统与外部网络的连接接口。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | string | ✓ | 端口唯一标识 |
| `medium` | Medium | ✓ | 能源介质类型 |
| `direction` | PortDirection | ✓ | 端口方向 |
| `node_ref` | string | ✓ | 外部网络节点引用 |
| `internal_ref` | string | ✓ | 内部节点引用 |
| `name` | string | | 端口名称 |

### Node（节点）

网络中的基本单元。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | string | ✓ | 节点唯一标识 |
| `name` | string | ✓ | 节点名称 |
| `type` | NodeType | ✓ | 节点类型 |
| `category` | NodeCategory | | 节点细分类别 |
| `description` | string | | 描述信息 |
| `ports` | Port[] | | 端口列表（仅Subsystem类型） |

### Link（连接）

节点之间的能量传输路径。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `source` | string | ✓ | 源节点ID |
| `target` | string | ✓ | 目标节点ID |
| `medium` | Medium | ✓ | 能源介质类型 |
| `flow_type` | FlowType | ✓ | 流向类型 |
| `category` | NodeCategory | | 连接类别 |
| `note` | string | | 备注信息 |

### DeviceParameters（设备参数）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `max_output_MW` | float | | 最大输出功率 (MW) |
| `min_output_MW` | float | | 最小输出功率 (MW) |
| `max_input_MW` | float | | 最大输入功率 (MW) |
| `capacity_MW` | float | | 装机容量 (MW) |
| `efficiency` | float | | 效率 (0-1) |
| `COP` | float | | 性能系数 |
| `eta_e` | float | | 电效率（CCHP） |
| `eta_h` | float | | 热效率（CCHP） |
| `eta_c` | float | | 冷效率（CCHP） |
| `base_injection_MW` | float | | 基础注入功率 |

### Device（设备）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | string | ✓ | 设备唯一标识 |
| `name` | string | ✓ | 设备名称 |
| `type` | DeviceType | ✓ | 设备类型 |
| `node_ref` | string | | 关联节点引用 |
| `parameters` | DeviceParameters | | 设备参数 |

---

## 网络参数

### ElectricNetworkParams（电网参数）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `line_reactance` | float | ✓ | 线路电抗 |
| `line_capacity` | float | ✓ | 线路容量 (MW) |
| `theta_min` | float | ✓ | 最小相角 (rad) |
| `theta_max` | float | ✓ | 最大相角 (rad) |

### GasNetworkParams（气网参数）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `pressure_min_MPa` | float | ✓ | 最小压力 (MPa) |
| `pressure_max_MPa` | float | ✓ | 最大压力 (MPa) |
| `pressure_base_MPa` | float | ✓ | 基准压力 (MPa) |

### HeatNetworkParams（热网参数）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `capacity_MW` | float | | 容量 (MW) |

---

## 顶层结构

### SystemInfo（系统信息）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `system_name` | string | ✓ | 系统名称 |
| `parent_id` | string | | 父系统ID |
| `description` | string | | 描述 |
| `topology_type` | TopologyType | | 拓扑类型 |

### TopLevelNetwork（顶层网络）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `system_info` | SystemInfo | ✓ | 系统元信息 |
| `network_params` | NetworkParams | | 网络参数 |
| `nodes` | Node[] | ✓ | 节点列表 |
| `links` | Link[] | ✓ | 连接列表 |

### Subsystem（子系统）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `system_info` | SystemInfo | ✓ | 系统元信息 |
| `nodes` | Node[] | ✓ | 节点列表 |
| `links` | Link[] | ✓ | 连接列表 |

### Zone（区域）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | string | ✓ | 区域ID |
| `name` | string | ✓ | 区域名称 |
| `devices` | Device[] | | 设备列表 |

### Attributes（属性集合）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `version` | int | ✓ | 版本号 |
| `description` | string | | 描述 |
| `sources` | Device[] | | 顶层源设备 |
| `zones` | Dict[str, Zone] | | 区域映射 |
| `constraints` | Constraints | | 约束参数 |

### MultiEnergySystem（完整系统）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `top_level` | TopLevelNetwork | ✓ | 顶层网络 |
| `subsystems` | Subsystem[] | | 子系统列表 |
| `attributes` | Attributes | | 设备属性 |

---

## ID 映射与追溯

### 映射关系

1. **设备 ↔ 节点**：`Device.id` 对应 `Node.id`
2. **端口 ↔ 节点**：`Port.internal_ref` 对应子系统内部 `Node.id`
3. **端口 ↔ 外部节点**：`Port.node_ref` 对应顶层网络 `Node.id`
4. **子系统 ↔ 区域**：`Subsystem.system_info.parent_id` 对应 `Zone.id`

### 示例

```
topology.json 中:
  Node(id="pv_1", type="Source")
    ↓ 映射
attributes.json 中:
  Device(id="pv_1", type="Renewable")
```
