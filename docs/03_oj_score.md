# OJ 评分标准解读与实现

## 1. 评分公式

### 1.1 总成本计算

$$C^{total} = C^{CAP} + C^{OP} + C^{Carbon}$$

其中：
- $C^{CAP}$ - 等年值建设成本
- $C^{OP}$ - 年运行成本
- $C^{Carbon}$ - 碳排放成本

### 1.2 等年值建设成本

$$C^{CAP} = \sum_{type} \frac{i}{1 - (1+i)^{-Y_{type}}} \times C^{INV,type}$$

参数：
- $i = 0.04$ (4% 贴现率)
- $Y_{type}$ - 设备寿命（年）
- $C^{INV,type}$ - 设备投资成本

### 1.3 运行成本

$$C^{OP} = \sum_{t=1}^{8760} \left( P_t^{grid} \times \pi_t^{elec} + P_t^{gas} \times \pi_t^{gas} + P_t^{shed} \times 500000 \right)$$

其中：
- $P_t^{grid}$ - 电网购电功率 (MW)
- $\pi_t^{elec}$ - 电价 (元/MWh)
- $P_t^{gas}$ - 燃气消耗 (MW)
- $\pi_t^{gas}$ - 气价 (元/MW)
- $P_t^{shed}$ - 失负荷量 (MW)
- 500000 - 失负荷惩罚系数 (元/MWh)

### 1.4 碳排放成本

$$C^{Carbon} = \begin{cases}
0 & \text{if } E^{total} \leq 100000 \\
(E^{total} - 100000) \times 600 & \text{if } E^{total} > 100000
\end{cases}$$

其中 $E^{total}$ 为年碳排放量（吨 CO₂）。

## 2. 评分函数

### 2.1 Logistic 评分函数

$$f(x) = \frac{100}{1 + e^{(x - 100000)/15000}}$$

其中 $x$ 为总成本（**万元**）。

### 2.2 函数特性

| 成本 (万元) | 评分 |
|------------|------|
| 50,000 | ~96.8 |
| 70,000 | ~87.9 |
| 85,000 | ~73.1 |
| 100,000 | 50.0 |
| 115,000 | ~26.9 |
| 130,000 | ~12.1 |
| 150,000 | ~3.2 |

### 2.3 边际效应

评分对成本的导数：

$$\frac{df}{dx} = -\frac{100 \times e^{(x-100000)/15000}}{15000 \times (1 + e^{(x-100000)/15000})^2}$$

在中心点 $x = 100000$ 万元处，边际效应约为 -0.00167 分/万元。

## 3. 实现代码

### 3.1 评分计算

```matlab
function score = calculate_score(totalCost)
    % totalCost: 总成本（元）
    center = 100000;  % 万元
    scale = 15000;    % 万元
    
    totalCostWan = totalCost / 10000;  % 转换为万元
    score = 100 / (1 + exp((totalCostWan - center) / scale));
end
```

### 3.2 碳成本计算

```matlab
function carbonCost = calculate_carbon_cost(emission)
    % emission: 碳排放量（吨）
    threshold = 100000;  % 吨
    price = 600;         % 元/吨
    
    if emission <= threshold
        carbonCost = 0;
    else
        carbonCost = (emission - threshold) * price;
    end
end
```

### 3.3 年化成本计算

```matlab
function annualCost = calculate_annual_cost(investment, life, discountRate)
    % investment: 投资成本（元）
    % life: 寿命（年）
    % discountRate: 贴现率（默认 0.04）
    
    if nargin < 3
        discountRate = 0.04;
    end
    
    annualCost = investment * discountRate / (1 - (1 + discountRate)^(-life));
end
```

## 4. 优化策略

### 4.1 成本与碳排放权衡

由于评分公式只考虑总成本，而总成本包含碳成本，因此：

1. 当碳排放 ≤ 100000 吨时，碳成本为 0，应专注降低投资和运行成本
2. 当碳排放 > 100000 吨时，每超 1 吨需付 600 元，需权衡减排投入

### 4.2 边际分析

假设当前成本为 80000 万元：
- 评分约为 79.4 分
- 边际效应约为 -0.0011 分/万元
- 降低 1000 万元成本可提升约 1.1 分

### 4.3 碳阈值利用

最优策略通常是使碳排放接近但不超过 100000 吨：
- 这样可以获得满额的"免费"碳排放空间
- 同时避免过度投资低碳技术导致成本上升

### 4.4 价格调整法

根据 FAQ 建议，可通过调整能源价格来间接影响优化结果：
- 提高燃气价格 → 减少燃气使用 → 降低碳排放
- 提高电价 → 增加本地发电 → 可能增加投资
- 需找到平衡点使总评分最高

## 5. 敏感性分析

### 5.1 关键参数

| 参数 | 影响 |
|------|------|
| 电价 | 影响购电成本，进而影响运行成本 |
| 气价 | 影响燃气成本，同时影响碳排放 |
| 碳阈值 | 决定碳成本是否触发 |
| 设备投资成本 | 影响年化投资 |

### 5.2 建议实验

1. 电价因子扫描：[0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
2. 气价因子扫描：[0.5, 1.0, 2.0, 3.0, 5.0]
3. 碳权重扫描：[0, 0.2, 0.4, 0.6, 0.8, 1.0]
