# Excel 文件冒烟测试

## 概述

本文档记录 Excel 文件的冒烟测试方法，用于验证导出文件与官方 reshape 脚本的兼容性。

## 测试目的

1. 验证导出的 Excel 文件格式正确
2. 验证官方 reshape 脚本可以正常处理
3. 验证输出 CSV 符合 OJ 平台要求

## 测试用例

### TC1: 基本格式验证

**输入**：`平台导出文件.xls`

**验证项**：
| 项目 | 预期值 | 验证方法 |
|------|--------|----------|
| 文件格式 | Excel 97-2003 (.xls) | 文件扩展名 |
| 表头行数 | 1 | `readtable(..., 'NumHeaderLines', 1)` |
| 数据行数 | 4763 | `height(T)` |
| 列数 | ≥31 | `width(T)` |

**验证命令**：
```matlab
T = readtable('平台导出文件.xls', 'NumHeaderLines', 1);
assert(height(T) == 4763, '数据行数错误');
assert(width(T) >= 31, '列数不足');
```

### TC2: 编码类型验证

**输入**：`平台导出文件.xls`

**验证项**：
| 编码类型 | 预期行数 | 说明 |
|----------|----------|------|
| 16 | 3285 | 切负荷 (9×365) |
| 43 | 1095 | 火电出力 (3×365) |
| 44 | 365 | 气源出力 (1×365) |
| 201-218 | 18 | 规划行 |

**验证命令**：
```matlab
T = readtable('平台导出文件.xls', 'NumHeaderLines', 1);
col2 = T{:, 2};
assert(sum(col2 == 16) == 3285, '切负荷行数错误');
assert(sum(col2 == 43) == 1095, '火电行数错误');
assert(sum(col2 == 44) == 365, '气源行数错误');
assert(sum(col2 > 200) == 18, '规划行数错误');
```

### TC3: reshape 兼容性验证

**输入**：`平台导出文件.xls`

**步骤**：
1. 运行官方 reshape 脚本
2. 验证输出文件格式

**验证命令**：
```matlab
cd matlab/scripts
MEHW_data_reshape
T = readtable('HWdata_for_OJ.csv');
assert(height(T) == 8760, '输出行数错误');
assert(width(T) == 6, '输出列数错误');
```

### TC4: Plan18 顺序验证

**输入**：`平台导出文件.xls`

**验证项**：
- 规划行按 col2 升序排列（201, 202, ..., 218）
- ans_planning(1:18) 与 Excel col8 一致

**验证命令**：
```matlab
T = readtable('平台导出文件.xls', 'NumHeaderLines', 1);
plan_rows = T(T{:, 2} > 200, :);
codes = plan_rows{:, 2};
assert(isequal(codes, (201:218)'), 'Plan18 顺序错误');
```

### TC5: 数值非负性验证

**输入**：`平台导出文件.xls`

**验证项**：
- col8-col31 数值非负

**验证命令**：
```matlab
T = readtable('平台导出文件.xls', 'NumHeaderLines', 1);
data_cols = T{:, 8:31};
assert(all(data_cols(:) >= 0), '存在负值');
```

## 自动化测试

### 测试脚本

```matlab
function results = run_excel_smoke_test(excel_path)
    results = struct();
    results.passed = 0;
    results.failed = 0;
    
    % TC1: 基本格式
    try
        T = readtable(excel_path, 'NumHeaderLines', 1);
        assert(height(T) == 4763);
        assert(width(T) >= 31);
        results.passed = results.passed + 1;
        fprintf('TC1: ✓ 通过\n');
    catch
        results.failed = results.failed + 1;
        fprintf('TC1: ✗ 失败\n');
    end
    
    % TC2: 编码类型
    try
        col2 = T{:, 2};
        assert(sum(col2 == 16) == 3285);
        assert(sum(col2 == 43) == 1095);
        assert(sum(col2 == 44) == 365);
        assert(sum(col2 > 200) == 18);
        results.passed = results.passed + 1;
        fprintf('TC2: ✓ 通过\n');
    catch
        results.failed = results.failed + 1;
        fprintf('TC2: ✗ 失败\n');
    end
    
    % TC4: Plan18 顺序
    try
        plan_rows = T(T{:, 2} > 200, :);
        codes = plan_rows{:, 2};
        assert(isequal(codes, (201:218)'));
        results.passed = results.passed + 1;
        fprintf('TC4: ✓ 通过\n');
    catch
        results.failed = results.failed + 1;
        fprintf('TC4: ✗ 失败\n');
    end
    
    % TC5: 数值非负
    try
        data_cols = T{:, 8:31};
        assert(all(data_cols(:) >= 0));
        results.passed = results.passed + 1;
        fprintf('TC5: ✓ 通过\n');
    catch
        results.failed = results.failed + 1;
        fprintf('TC5: ✗ 失败\n');
    end
    
    fprintf('\n总计: %d 通过, %d 失败\n', results.passed, results.failed);
end
```

## 测试结果记录

| 日期 | 测试文件 | TC1 | TC2 | TC3 | TC4 | TC5 | 结论 |
|------|----------|-----|-----|-----|-----|-----|------|
| 2025-12-20 | example/平台导出文件.xls | ✓ | ✓ | ✓ | ✓ | ✓ | 通过 |

## 变更历史

| 日期 | 变更 | 原因 |
|------|------|------|
| 2025-12-20 | 初始创建 | 阶段2交付物 |

## 相关文件

- `spec/excel_compat_policy.md` - Excel 兼容性策略
- `spec/reproduce_official_reshape.md` - 官方脚本复现指南
- `matlab/src/tests/test_excel_format_contract.m` - 格式契约测试
