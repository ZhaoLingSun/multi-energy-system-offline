# 原始数据目录
此目录用于存放原始输入数据，**只读不改**。

## 子目录说明

- `meos_inputs/` - MEOS 平台输入数据包（Excel/CSV）
- `device_tables/` - 设备参数表（从 PDF/PPT 整理）
- `scenarios/` - 多场景输入数据（不同负荷/风光/价格）

## 注意事项

1. 原始数据永远不要直接修改
2. 任何清洗/转换后的数据应输出到 `../interim/` 或 `../processed/`
3. 建议使用 Git LFS 管理大型数据文件
