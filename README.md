# Multi-Energy-Offline (Python MILP)

> 综合能源系统离线优化工具链：MEOS 平台复刻、OJ 评分一致性校验与全年的 MILP 精确求解

## 项目概览

本项目面向校园综合能源系统规划与运行优化，基于 **Python + Gurobi** 实现：

- **全年 8760 小时调度** 的单层 MILP（对齐 OJ 评分口径）
- **MEOS 平台导出/导入格式** 的全流程复刻与一致性验证
- **设备选型 Plan18 优化** 与关键敏感性分析
- **四个思考题** 的可复现实验与报告产出

核心特点：
- **目标对齐**：将碳税并入电/气价，实现 OJ 评分口径与日内调度目标一致
- **可验证**：本地评分与平台输出对齐，可一键生成 OJ 提交文件
- **可复现**：统一的配置与数据目录结构，脚本化运行

---

## 运行环境

- Python 3.10+
- Gurobi 10+（需有效许可）
- 依赖安装：

```bash
pip install -r requirements.txt
# 另外安装 gurobipy（需授权）
```

---

## 快速开始

### 1) 全年 MILP（主问题）

```bash
python scripts/run_full_milp.py \
  --data-dir data/raw \
  --renewable-dir data/raw \
  --output-dir runs/full_milp \
  --mip-gap 1e-3 \
  --threads 64
```

输出：
- `runs/full_milp/*/full_milp_summary.json`
- `*_platform.csv`、`*_oj.csv`、引导电/气价、风光曲线等

### 一键运行全部仿真

```bash
python run_all.py --mip-gap 1e-3 --threads 64
```

### Docker（intranet）

```bash
docker build -t meos-offline .
docker run --rm --name meos-offline \\
  --mount type=bind,src=/home/ace/gurobi.lic,dst=/app/gurobi.lic,ro \\
  -e GRB_LICENSE_FILE=/app/gurobi.lic \\
  -v $(pwd):/app \\
  -w /app \\
  meos-offline \\
  python run_all.py --mip-gap 1e-3 --threads 64
```

### 2) OJ 评分复算

```bash
python scripts/score_oj_csv.py \
  output/xxx_oj.csv \
  --data-dir data/raw \
  --spec configs/oj_score.yaml \
  --device-catalog spec/device_catalog.yaml
```

### 3) 平台导出验证

```bash
python scripts/verify_platform_export.py \
  meos/平台导出文件_最终.xls \
  --data-dir data/raw
```

---

## 思考题（四题）

- Q1 建筑节能改造：`python thought_questions/q1_retrofit/run_q1_retrofit.py`
- Q2 跨季节冷热联储：`python thought_questions/q2_seasonal_storage/run_q2_seasonal_storage.py`
- Q3 管道线包等效储能：`python thought_questions/q3_linepack/run_milp_linepack.py`
- Q4 分区规划与线路扩容：`python thought_questions/q4_line_capacity/run_milp_line_capacity.py`

各脚本都会输出 `summary.json` 及对应报告所需的中间结果。

---

## 目录结构（核心）

```
meos/                   # MEOS 仿真器、模型与导出器
scripts/                # 全年 MILP + 评分工具
thought_questions/      # 四个思考题
spec/
  device_catalog.yaml   # 设备目录与上限
  plan18_map.yaml       # Plan18 映射
data/raw/               # 平台原始输入（负荷/电价/气价/碳因子/风光）
configs/oj_score.yaml   # OJ 评分规范（唯一真值）
example/                # attributes.json / topology.json / 平台导出样例
runs/                   # 运行输出（可清理再生）
```

---

## 数学目标与评分口径

- 年化投资成本：
  $$C_{CAP}=\sum_k Cap_k\cdot UnitCost_k\cdot \alpha(i,n)$$
- 运行成本：
  $$C_{OP}=\sum_t(\pi_e P_e + \pi_g G + 500000\cdot L_{shed})$$
- 碳惩罚：
  $$C_{Carbon}=600\cdot \max(0, E_{total}-100000)$$

当排放必然超阈值时，碳惩罚等价于线性碳税，可并入价格系数，形成单层 MILP。

---

## 说明

- `data/` 与 `configs/` 仅保留 **数据与评分规范**（不包含 MATLAB 脚本）。
- `runs/` 与 `output/` 可随时清理，均可复现。
- 若需生成平台 Excel 或 OJ 提交 CSV，可使用 `meos/export` 与 `scripts/` 中的工具。

---

## License

MIT License
