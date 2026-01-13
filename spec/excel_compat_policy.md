# Excel Compatibility Policy (ExcelCompatPolicy)

## 概述

本文档定义了 MEOS Emulator 项目中 Excel 文件格式的兼容性策略。

## 背景

官方 `MEHW_data_reshape.m` 脚本使用硬编码的文件名 `平台导出文件.xls`，且只能读取 `.xls` 格式（Excel 97-2003）。为了满足"官方脚本不改一行"的硬规则，我们需要确保输入文件符合这一要求。

## 策略

### 1. 文件格式要求

- **必须格式**: `.xls` (Excel 97-2003)
- **不支持**: `.xlsx` (Excel 2007+) 直接输入
- **文件名**: 必须为 `平台导出文件.xls`

### 2. 格式转换策略

当输入文件为 `.xlsx` 格式时，按以下优先级进行转换：

1. **LibreOffice** (推荐)
   ```bash
   soffice --headless --convert-to xls --outdir <output_dir> <input.xlsx>
   ```

2. **Gnumeric**
   ```bash
   ssconvert <input.xlsx> <output.xls>
   ```

3. **MATLAB 内置功能** (fallback)
   - 使用 `readcell` 读取 xlsx
   - 使用 `writecell` 写入 xls

### 3. 不使用的方法

- **actxserver** (Windows COM 自动化): 仅限 Windows，不跨平台
- **xlswrite**: 已弃用，且依赖 Excel 安装

## 实现位置

- `matlab/src/oj_pipeline/reshape/reshape_for_oj.m`
  - `run_official_reshape()`: 检测格式并调用转换
  - `convert_xlsx_to_xls()`: 实现转换逻辑

## 验证方法

```matlab
% 测试 xlsx 转换
options.use_official_script = true;
[csv_path, oj_data] = reshape_for_oj('test.xlsx', 'output_dir', options);

% 验证输出
assert(isfile(fullfile('output_dir', '平台导出文件.xls')));
assert(isfile(csv_path));
```

## 环境要求

### Linux (推荐)

```bash
# 安装 LibreOffice
sudo apt-get install libreoffice

# 或安装 Gnumeric
sudo apt-get install gnumeric
```

### macOS

```bash
# 使用 Homebrew 安装 LibreOffice
brew install --cask libreoffice
```

### Windows

- 安装 LibreOffice: https://www.libreoffice.org/download/
- 确保 `soffice` 在 PATH 中

## 相关文档

- `docs/阶段2技术路线.md`: 阶段2硬规则定义
- `matlab/scripts/MEHW_data_reshape.m`: 官方 reshape 脚本
- `.kiro/specs/meos-emulator/requirements.md`: Requirement 13

## 版本历史

| 日期 | 版本 | 变更 |
|------|------|------|
| 2025-12-20 | 1.0 | 初始版本 |
