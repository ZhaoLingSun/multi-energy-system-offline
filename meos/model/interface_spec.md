# MEOS Model 数据接口规范

## 概述

本文档定义模型层的数据接口规范，包括索引结构和时序输入接口。

---

## 索引结构

### 1. NodeIndex（节点索引）

| 字段 | 类型 | 说明 |
|------|------|------|
| `all_nodes` | Dict[str, dict] | 所有节点，key=node_id |
| `by_medium` | Dict[str, List[str]] | 按介质分组 |
| `by_type` | Dict[str, List[str]] | 按类型分组 |
| `by_subsystem` | Dict[str, List[str]] | 按子系统分组 |

**类型枚举**: `Grid_Node`, `Source`, `Sink`, `Converter`, `Load`, `Subsystem`

### 2. DeviceIndex（设备索引）

| 字段 | 类型 | 说明 |
|------|------|------|
| `all_devices` | Dict[str, dict] | 所有设备，key=device_id |
| `by_type` | Dict[str, List[str]] | 按设备类型分组 |
| `by_zone` | Dict[str, List[str]] | 按区域分组 |

**设备类型**: `Thermal_Generator`, `Natural_Gas_Source`, `Renewable`, `P2G`, `E_Boiler`, `GasTurbine`, `GasBoiler`, `HeatPump`, `Chiller`, `Absorption`, `CCHP`

### 3. LinkIndex（连接索引）

| 字段 | 类型 | 说明 |
|------|------|------|
| `all_links` | List[dict] | 所有连接 |
| `by_medium` | Dict[str, List[dict]] | 按介质分组 |
| `by_source` | Dict[str, List[dict]] | 按源节点分组 |
| `by_target` | Dict[str, List[dict]] | 按目标节点分组 |

**介质枚举**: `Electricity`, `Gas`, `Heat`, `Cooling`

### 4. PortIndex（端口索引）

| 字段 | 类型 | 说明 |
|------|------|------|
| `all_ports` | Dict[str, dict] | 所有端口 |
| `by_subsystem` | Dict[str, List[str]] | 按子系统分组 |
| `by_medium` | Dict[str, List[str]] | 按介质分组 |
| `by_direction` | Dict[str, List[str]] | 按方向分组 |

**方向枚举**: `Input`, `Output`

---

## 时序输入接口（与 MATLAB preprocess 对齐）

### LoadProfile（负荷时序）

| 字段 | 类型 | 维度 | 说明 |
|------|------|------|------|
| `electric` | Dict[str, List[float]] | zone_id → [T] | 各区域电负荷 (MW) |
| `heat` | Dict[str, List[float]] | zone_id → [T] | 各区域热负荷 (MW) |
| `cool` | Dict[str, List[float]] | zone_id → [T] | 各区域冷负荷 (MW) |

### RenewableProfile（可再生能源出力）

| 字段 | 类型 | 维度 | 说明 |
|------|------|------|------|
| `pv` | Dict[str, List[float]] | device_id → [T] | 光伏出力 (MW) |
| `wind` | Dict[str, List[float]] | device_id → [T] | 风电出力 (MW) |

### PriceProfile（价格时序）

| 字段 | 类型 | 维度 | 说明 |
|------|------|------|------|
| `electricity` | List[float] | [T] | 电价 (元/MWh) |
| `gas` | List[float] | [T] | 气价 (元/MWh) |

### CarbonProfile（碳排放因子）

| 字段 | 类型 | 维度 | 说明 |
|------|------|------|------|
| `electricity` | List[float] | [T] | 电网碳因子 (kg/MWh) |
| `gas` | List[float] | [T] | 天然气碳因子 (kg/MWh) |

---

## 模型调用示例

### 构建索引

```python
from meos.model.data_index import build_index_from_files

index = build_index_from_files(
    "schema/normalized_topology.json",
    "schema/normalized_attributes.json"
)
```

### 查询索引

```python
# 获取所有电力连接
elec_links = index.links.by_medium.get("Electricity", [])

# 获取某区域的设备
devices = index.devices.by_zone.get("student_zone", [])

# 获取设备参数
from meos.model.data_index import get_device_param
max_input = get_device_param(index, "p2g_1", "max_input_MW")
```

---

## 索引被模型调用的场景

| 场景 | 使用的索引 | 说明 |
|------|-----------|------|
| 构建电力平衡约束 | `links.by_medium["Electricity"]` | 遍历电力连接建立功率平衡 |
| 构建设备容量约束 | `devices.by_type["CCHP"]` | 按类型获取设备并读取参数 |
| 构建子系统边界约束 | `ports.by_subsystem[zone_id]` | 获取子系统端口建立边界条件 |
| 构建负荷满足约束 | `nodes.by_type["Load"]` | 获取所有负荷节点 |
| 读取网络参数 | `network_params.electric` | 获取线路电抗、容量等 |
