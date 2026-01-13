# 官方 reshape 脚本复现指南

## 概述

本文档记录如何复现官方 `MEHW_data_reshape.m` 脚本的行为，确保本地实现与平台一致。

## 环境要求

### MATLAB 版本
- MATLAB R2020a 或更高版本
- 需要 Statistics and Machine Learning Toolbox（用于 `readtable`）

### 文件依赖
- 输入文件：`平台导出文件.xls`（必须为 `.xls` 格式，不支持 `.xlsx`）
- 输出文件：`HWdata_for_OJ.csv`

## 官方脚本位置

```
matlab/scripts/MEHW_data_reshape.m
```

## 运行命令

```matlab
cd matlab/scripts
MEHW_data_reshape
```

## 输入文件要求

### 文件名
- 必须为 `平台导出文件.xls`（固定文件名）
- 必须为 Excel 97-2003 格式（`.xls`），不支持 `.xlsx`

### 文件结构
- 第 1 行：表头
- col2：编码类型（数值）
  - 16: 切负荷
  - 43: 火电出力
  - 44: 气源出力
  - 201-218: 规划行
- col3：指标名（字符串）
- col8-col31：24 小时数据

### 行数要求
- 切负荷：9 对象 × 365 天 = 3285 行
- 火电出力：3 机组 × 365 天 = 1095 行
- 气源出力：1 对象 × 365 天 = 365 行
- 规划行：18 行
- 总计：4763 行（不含表头）

## 输出文件格式

### 文件名
- `HWdata_for_OJ.csv`

### 列结构
| 列号 | 列名 | 说明 |
|------|------|------|
| 1 | ans_load1 | 切负荷1（电负荷） |
| 2 | ans_load2 | 切负荷2（热负荷） |
| 3 | ans_load3 | 切负荷3（冷负荷） |
| 4 | ans_ele | 购电功率 |
| 5 | ans_gas | 购气量 |
| 6 | ans_planning | 规划容量 |

### 行数
- 8760 行（365 天 × 24 小时）

### 数据分布
- 行 1-18：规划容量（ans_planning 有值，其他列为 0）
- 行 19-8760：调度数据（ans_planning 为 0）

## 复现验证

### 验证命令

```matlab
% 运行官方脚本
cd matlab/scripts
MEHW_data_reshape

% 验证输出
T = readtable('HWdata_for_OJ.csv');
assert(height(T) == 8760, '行数错误');
assert(width(T) == 6, '列数错误');
assert(all(T.ans_planning(19:end) == 0), 'ans_planning(19:8760) 应为 0');
```

### 哈希验证

```matlab
% 计算输出文件哈希
hash = compute_file_md5('HWdata_for_OJ.csv');
fprintf('输出文件 MD5: %s\n', hash);
```

## 常见问题

### Q1: 提示找不到文件
- 确保 `平台导出文件.xls` 在当前目录
- 确保文件名完全匹配（包括中文）

### Q2: 读取 .xlsx 失败
- 官方脚本只支持 `.xls` 格式
- 使用 LibreOffice 或 Excel 转换：`soffice --headless --convert-to xls 平台导出文件.xlsx`

### Q3: 输出行数不对
- 检查输入文件是否完整
- 检查 col2 编码是否正确

## 与本地实现对比

### 本地实现
- `matlab/src/oj_pipeline/reshape/reshape_for_oj.m`

### 对比方法
```matlab
% 运行官方脚本
cd matlab/scripts
MEHW_data_reshape
official_output = readtable('HWdata_for_OJ.csv');

% 运行本地实现
cd ../src/oj_pipeline/reshape
local_output = reshape_for_oj('平台导出文件.xls');

% 对比
assert(isequal(official_output, local_output), '输出不一致');
```

## 变更历史

| 日期 | 变更 | 原因 |
|------|------|------|
| 2025-12-20 | 初始创建 | 阶段2交付物 |

## 相关文件

- `matlab/scripts/MEHW_data_reshape.m` - 官方脚本
- `matlab/src/oj_pipeline/reshape/reshape_for_oj.m` - 本地实现
- `spec/excel_compat_policy.md` - Excel 兼容性策略
