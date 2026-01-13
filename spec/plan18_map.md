# Plan18Map 说明文档

## 概述

Plan18Map 定义了 OJ 提交 CSV 中 `ans_planning(1:18)` 的 18 维规划向量顺序映射。

## 映射来源

- **来源文件**：平台导出文件（方案号 7940）
- **确定方法**：从 MEOS 平台导出 Excel，筛选 `col2 > 200` 的行，按出现顺序确定映射
- **冻结状态**：待冻结

## 18 维规划向量

| idx | 编码类型 | 设备名称 | 英文名 | 单位 | 容量单位 |
|-----|----------|----------|--------|------|----------|
| 1 | 201 | 热电联产A | CHP_A | 台 | MW |
| 2 | 202 | 热电联产B | CHP_B | 台 | MW |
| 3 | 203 | 内燃机 | ICE | 台 | MW |
| 4 | 204 | 电锅炉 | ElectricBoiler | 台 | MW |
| 5 | 205 | 压缩式制冷机A | Chiller_A | 台 | MW |
| 6 | 206 | 压缩式制冷机B | Chiller_B | 台 | MW |
| 7 | 207 | 吸收式制冷机组 | AbsorptionChiller | 台 | MW |
| 8 | 208 | 燃气蒸汽锅炉 | GasBoiler | 台 | MW |
| 9 | 209 | 地源热泵A | GSHP_A | 台 | MW |
| 10 | 210 | 地源热泵B | GSHP_B | 台 | MW |
| 11 | 211 | 电储能 | BatteryStorage | 台 | MWh |
| 12 | 212 | 热储能 | ThermalStorage | 台 | MWh |
| 13 | 213 | 冷储能 | ColdStorage | 台 | MWh |
| 14 | 214 | 风电 | WindTurbine | 台 | MW |
| 15 | 215 | 光伏 | PV | 台 | MW |
| 16 | 216 | 电制气 | P2G | 台 | MW |
| 17 | 217 | 燃气轮机 | GasTurbine | 台 | MW |
| 18 | 218 | 冷热电联供 | CCHP | 台 | MW |

## 重要说明

### 单位说明

虽然平台显示单位为"台"，但实际承载的是**容量/规模的离散化载体**。在阶段3做规划搜索时，必须按 `DeviceCatalog` 中的基准容量解释这些数值，而非字面理解为"设备台数"。

### 行选择器

- **列号**：col2（第2列）
- **条件**：`col2 > 200`
- **规划值列**：col8（第8列）

### 校验规则

1. 规划行数量必须等于 18
2. 编码类型范围：201-218
3. 编码类型必须唯一

## 使用方法

### 导出时

```matlab
% 确保规划行按 Plan18Map 顺序排列
for i = 1:18
    plan_row = create_plan_row(plan18_map.devices(i).code, plan_values(i));
    write_to_excel(plan_row);
end
```

### 验证时

```matlab
% 验证导出的规划行顺序
plan_rows = excel_data(excel_data(:,2) > 200, :);
assert(size(plan_rows, 1) == 18, '规划行数量应为 18');
for i = 1:18
    assert(plan_rows(i, 2) == plan18_map.devices(i).code, ...
           sprintf('第 %d 行编码类型不匹配', i));
end
```

## 变更历史

| 日期 | 版本 | 变更内容 | 变更人 |
|------|------|----------|--------|
| 2024-12-20 | 1.0 | 初始创建 | Kiro |

## 相关文件

- `spec/plan18_map.yaml` - 机读映射文件
- `docs/02_meos_format.md` - MEOS 导出格式说明
- `matlab/src/tests/test_plan18_order.m` - Plan18 顺序验证测试
