# validate_dataset.py 使用说明

## ⚠️ 重要区别

`validate_dataset.py` 有两种验证模式：

### 1. 快速验证模式（默认）

```bash
python tools/validate_dataset.py full dataset/ --output-dir validation/
```

**特点**:
- ⚡ 快速完成（~17秒）
- 验证前3个序列的格式
- 每个序列只测试第0帧投影
- 适合：日常快速检查

**输出**:
- `sample_projections/` - 3张投影图（seq00-02的第0帧）

---

### 2. 完整验证模式（推荐首次验证）

```bash
python tools/validate_dataset.py full dataset/ --output-dir validation/ --full
```

**特点**:
- 📊 全面验证（~10-15分钟）
- 验证所有12个序列的格式
- **完整投影验证**：每个序列采样5帧（开始、1/4、中间、3/4、结束）
- 适合：首次验证、发布前验证

**输出**:
- `projection_validation/` - 包含完整投影验证
  - `sequence_00/` - 5张投影图 + statistics.json
  - `sequence_01/` - 5张投影图 + statistics.json
  - ... (共12个序列)
  - `PROJECTION_VALIDATION_REPORT.md` - 详细报告

---

### 3. 仅投影验证（单独运行）

如果只需要投影验证：

```bash
python tools/validate_dataset.py projection-full dataset/ \
    --output-dir validation/projection_validation
```

**特点**:
- 🎯 专注于投影验证
- 每个序列采样5帧
- 生成详细的投影报告

---

## 📋 命令对比

| 命令 | 耗时 | 格式验证 | 投影验证 | 适用场景 |
|------|------|---------|---------|---------|
| `full` | ~17秒 | 前3个序列 | 3张图（采样） | 日常快速检查 |
| `full --full` | ~15分钟 | 所有序列 | 120张图（10×12） | 首次验证、发布前 |
| `projection-full` | ~10分钟 | 无 | 120张图（10×12） | 仅需投影验证 |

---

## 🎯 推荐工作流

### 首次验证新数据集

```bash
# 1. 快速摘要（5秒）
python tools/validate_dataset.py summary dataset/

# 2. 完整验证（15分钟）
python tools/validate_dataset.py full dataset/ \
    --output-dir validation/ --full

# 3. 查看报告
cat validation/VALIDATION_SUMMARY.md
cat validation/projection_validation/PROJECTION_VALIDATION_REPORT.md
```

### 日常验证

```bash
# 快速验证（17秒）
python tools/validate_dataset.py full dataset/ --output-dir quick_check/
```

### 仅需投影验证

```bash
# 完整投影验证（10分钟）
python tools/validate_dataset.py projection-full dataset/ \
    --output-dir projection_results/
```

---

## 💡 关键点

1. **`full` 默认是快速模式** - 适合日常使用
2. **`full --full` 是完整模式** - 包含完整投影验证
3. **`projection-full` 专注于投影** - 当你只需要投影验证时

---

## 📚 示例

### 示例1: 首次验证（推荐）

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib

# 完整验证（包含每序列5帧投影）
python tools/validate_dataset.py full \
    /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --output-dir validation_complete \
    --full

# 结果
# validation_complete/
# ├── VALIDATION_SUMMARY.md
# ├── projection_validation/
# │   ├── PROJECTION_VALIDATION_REPORT.md
# │   ├── sequence_00/ (10张图)
# │   ├── sequence_01/ (10张图)
# │   └── ... (12个序列)
```

### 示例2: 快速检查

```bash
# 快速验证（17秒）
python tools/validate_dataset.py full \
    /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --output-dir quick_check

# 结果
# quick_check/
# ├── VALIDATION_SUMMARY.md
# └── sample_projections/ (3张图)
```

### 示例3: 仅投影验证

```bash
# 只运行投影验证（10分钟）
python tools/validate_dataset.py projection-full \
    /mnt/drtraining/user/dahailu/data/bevcalib/all_training_data \
    --output-dir projection_only

# 结果
# projection_only/
# ├── PROJECTION_VALIDATION_REPORT.md
# ├── sequence_00/ (5张图)
# └── ... (12个序列)
```

---

## 🔧 解决你的问题

你遇到的问题：
- ❌ 使用了 `full`（快速模式）
- ❌ 只得到3个序列的1帧投影

解决方案（三选一）：

**方案1**: 使用完整模式
```bash
python tools/validate_dataset.py full dataset/ --output-dir validation/ --full
```

**方案2**: 单独运行投影验证
```bash
python tools/validate_dataset.py projection-full dataset/ --output-dir projections/
```

**方案3**: 直接使用基础工具
```bash
python tools/comprehensive_projection_validation.py \
    --dataset_root dataset/ \
    --output_dir projections/
```

---

## 📖 更多帮助

```bash
# 查看帮助
python tools/validate_dataset.py --help
python tools/validate_dataset.py full --help
python tools/validate_dataset.py projection-full --help
```
