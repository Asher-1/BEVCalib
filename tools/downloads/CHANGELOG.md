# 更新日志

## 2026-02-28 - 最新更新

### 🐛 修复：目录结构重复问题 (15:30)

**问题**：下载后出现重复的目录层级
- ❌ 错误：`configs/configs/`、`bags/important/important/`
- ✅ 正确：`configs/`、`bags/important/`

**根本原因**：
- drfile 命令 `drfile download trip:/xxx/DIR PATH` 会在 `PATH` 下自动创建 `DIR/` 子目录
- 之前的实现下载到目标目录本身，导致重复

**解决方案**：
```python
# 修复前（错误）
download_map = [
    ('configs', 'configs', False),       # ❌ 会创建 configs/configs/
    ('bags/important', 'bags', False),   # ✅ 这个是对的
]

# 修复后（正确）
download_map = [
    ('configs', '.', False),              # ✅ 下载到根目录，创建 configs/
    ('bags/important', 'bags', False),    # ✅ 下载到 bags/，创建 bags/important/
]
```

**影响**：
- 修复后的目录结构与预期完全一致
- 与 drfile 的原生下载行为保持一致
- 详见 [验证说明.md](./验证说明.md)

---

## 2026-02-28 - 选择性下载功能

### 🎉 新增功能

#### 选择性下载模式（默认启用）
- **功能**：默认只下载每个 trip 中的关键目录（按顺序）
  - 1️⃣ `configs/` - 配置文件（先下载，小文件快速）
  - 2️⃣ `bags/important/` - 重要的 bag 文件（后下载）
  - ❌ 其他目录（不下载）

- **优点**：
  - 大幅减少下载时间
  - 节省磁盘空间（通常节省 70-90%）
  - 满足大多数标定和测试场景需求

#### 完整下载模式
- **参数**：`--full`
- **功能**：下载整个 trip 的所有数据
- **使用场景**：需要完整数据分析时使用

### 📝 使用示例

```bash
# 选择性下载（默认）- 只下载 bags/important 和 configs
python3 download_trips.py YR-P789-19_20260213_012754

# 完整下载 - 下载所有数据
python3 download_trips.py --full YR-P789-19_20260213_012754

# 批量选择性下载
python3 download_trips.py --file trips.txt

# 批量完整下载
python3 download_trips.py --full --file trips.txt
```

### 🔧 技术实现

1. **新增类初始化参数**：
   - `selective_download`: bool - 控制是否启用选择性下载

2. **新增方法**：
   - `download_subdirectory()` - 下载单个子目录

3. **修改方法**：
   - `download_trip()` - 支持选择性下载和完整下载两种模式

4. **新增命令行参数**：
   - `--full` - 启用完整下载模式

### 📚 文档更新

- ✅ 更新 `download_trips.py` 脚本
- ✅ 更新 `QUICKSTART.md` - 添加选择性下载说明
- ✅ 创建 `README.md` - 完整使用文档
- ✅ 创建 `example_usage.sh` - 使用示例脚本
- ✅ 创建 `CHANGELOG.md` - 本文档

### ⚠️ 重要变更

**默认行为变更**：
- **之前**：默认下载整个 trip 的所有数据
- **现在**：默认只下载 `bags/important` 和 `configs` 目录
- **恢复旧行为**：使用 `--full` 参数

### 🧪 测试

```bash
# 测试检查功能
python3 download_trips.py --check-only YR-P789-19_20260213_012754

# 测试选择性下载
python3 download_trips.py YR-P789-19_20260213_012754

# 测试完整下载
python3 download_trips.py --full YR-P789-19_20260213_012754
```

### 📊 预期效果

以典型的 trip 为例（YR-P789-19_20260213_012754）：

| 模式 | 下载大小 | 下载时间 | 文件数 |
|------|---------|---------|--------|
| 选择性下载 | ~2-5 GB | ~5-10 分钟 | ~20-50 |
| 完整下载 | ~20-50 GB | ~30-60 分钟 | ~200-500 |

*实际数据取决于具体 trip 的大小和内容*

### 🔄 迁移指南

如果你的脚本或工作流依赖旧的默认行为（下载完整 trip）：

1. **方法 1**：添加 `--full` 参数
```bash
# 旧命令
python3 download_trips.py YR-P789-19_20260213_012754

# 新命令（保持旧行为）
python3 download_trips.py --full YR-P789-19_20260213_012754
```

2. **方法 2**：修改代码
```python
# 创建下载器时禁用选择性下载
downloader = TripDownloader(
    output_dir=args.output,
    selective_download=False  # 禁用选择性下载
)
```

### 🐛 已知问题

无

### 📋 待办事项

- [ ] 支持自定义下载目录列表（通过配置文件）
- [ ] 添加下载进度条
- [ ] 支持断点续传
- [ ] 添加并行下载支持

---

## 初始版本

### 基础功能

- ✅ 下载单个或多个 trips
- ✅ 从文件读取 trip 列表
- ✅ 交互模式输入
- ✅ 检查 trip 是否存在
- ✅ 详细日志记录
- ✅ 统计下载结果
