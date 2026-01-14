# MEOS 平台提交摘要（full_milp_20260113_134017）

## 1. 规划设备容量（plan18）

| 序号 | 设备 | 规划值(plan18) | 基准容量 | 容量（最大出力/能量） | 备注 |
| --- | --- | --- | --- | --- | --- |
| 1 | 热电联产A | 0.000 | 2.000 MW | 0.000 MW |  |
| 2 | 热电联产B | 0.000 | 8.000 MW | 0.000 MW |  |
| 3 | 内燃机 | 0.000 | 10.000 MW | 0.000 MW |  |
| 4 | 电锅炉 | 0.000 | 2.000 MW | 0.000 MW |  |
| 5 | 压缩式制冷机A | 69.000 | 0.500 MW | 34.500 MW |  |
| 6 | 压缩式制冷机B | 2.000 | 12.000 MW | 24.000 MW |  |
| 7 | 吸收式制冷机组 | 6.000 | 12.000 MW | 72.000 MW |  |
| 8 | 燃气蒸汽锅炉 | 35.000 | 2.000 MW | 70.000 MW |  |
| 9 | 地源热泵A | 9.000 | 2.000 MW | 18.000 MW |  |
| 10 | 地源热泵B | 2.000 | 10.000 MW | 20.000 MW |  |
| 11 | 电储能 | 0.000 | 2.000 MWh | 0.000 MWh | 储能功率上限见分区表 |
| 12 | 热储能 | 22.000 | 40.000 MWh | 880.000 MWh | 储能功率上限见分区表 |
| 13 | 冷储能 | 22.000 | 40.000 MWh | 880.000 MWh | 储能功率上限见分区表 |
| 14 | 风电 | 359.000 | 0.482 (风/光按 MW 直填) | 359.000 MW | 风/光容量按 MW 直接填写 |
| 15 | 光伏 | 373.000 | 0.356 (风/光按 MW 直填) | 373.000 MW | 风/光容量按 MW 直接填写 |
| 16 | 电制气 | 315.000 | 0.500 MW | 157.500 MW |  |
| 17 | 燃气轮机 | 29.000 | 5.000 MW | 145.000 MW |  |
| 18 | 冷热电联供 | 10.000 | 3.000 MW | 30.000 MW |  |

## 2. 储能分区配置（来自 summary.json）

| 区域 | 电储能（MWh / MW） | 热储能（MWh / MW） | 冷储能（MWh / MW） |
| --- | --- | --- | --- |
| 学生区 | 0.000 / 0.000 | 400.000 / 80.000 | 280.000 / 56.000 |
| 教工区 | 0.000 / 0.000 | 360.000 / 72.000 | -0.000 / -0.000 |
| 教学办公区 | 0.000 / 0.000 | 120.000 / 24.000 | 600.000 / 120.000 |

## 3. 结果总结

- C_total = 687945931.417
- C_CAP = 343729195.569
- C_OP = 298020095.893
- C_Carbon = 46196639.955
- Score = 88.897961
- 年排放 = 176994.400 tCO2（电 80915.552，气 96078.848）
- 超额排放 = 76994.400 tCO2

## 4. MEOS 平台需要上传的文件

- output/final_submission/full_milp_20260113_134017/2c5c06fab09754cf_capacity.yaml
- output/final_submission/full_milp_20260113_134017/2c5c06fab09754cf_guide_price.csv
- output/final_submission/full_milp_20260113_134017/guide_price.xlsx
- output/final_submission/full_milp_20260113_134017/2c5c06fab09754cf_guide_gas_price.csv
- output/final_submission/full_milp_20260113_134017/guide_gas_price.xlsx
- output/final_submission/full_milp_20260113_134017/2c5c06fab09754cf_platform.csv
- output/final_submission/full_milp_20260113_134017/2c5c06fab09754cf_oj.csv
- output/final_submission/full_milp_20260113_134017/2c5c06fab09754cf_pv_curve_mw.csv
- output/final_submission/full_milp_20260113_134017/2c5c06fab09754cf_pv_curve_mw.xlsx
- output/final_submission/full_milp_20260113_134017/2c5c06fab09754cf_wind_curve_mw.csv
- output/final_submission/full_milp_20260113_134017/2c5c06fab09754cf_wind_curve_mw.xlsx
