# 分析工具 (Analysis Tools)

数据分析和统计工具集。

---

## 📋 工具列表

### `analyze_perturbation_training.py` - 扰动训练分析

分析扰动训练实验的效果和统计数据。

**功能**:
- 扰动参数影响分析
- 训练收敛性统计
- 性能指标对比
- 生成可视化报告

**使用方法**:
```bash
python tools/analysis/analyze_perturbation_training.py \
    --experiment_dir /path/to/experiments \
    --output analysis_report.pdf
```

**分析内容**:
- 不同扰动级别的性能对比
- Loss曲线和收敛速度
- 精度指标统计
- 最优参数推荐

**输出报告**:
```
analysis_report/
├── summary.txt               # 文本摘要
├── loss_curves.png          # Loss曲线图
├── accuracy_comparison.png  # 精度对比图
├── parameter_heatmap.png    # 参数影响热图
└── statistics.json          # 详细统计数据
```

---

## 🎯 使用场景

### 场景1: 实验结果分析

```bash
# 分析实验结果
python tools/analysis/analyze_perturbation_training.py \
    --experiment_dir experiments/perturbation_sweep/ \
    --output analysis/

# 查看报告
cat analysis/summary.txt
```

### 场景2: 参数调优

```bash
# 对比不同参数配置
python tools/analysis/analyze_perturbation_training.py \
    --experiment_dir experiments/ \
    --compare_configs config1.yaml config2.yaml config3.yaml \
    --output param_comparison.pdf
```

### 场景3: 生成论文图表

```bash
# 生成高质量图表
python tools/analysis/analyze_perturbation_training.py \
    --experiment_dir experiments/ \
    --output publication_figures/ \
    --high_quality \
    --dpi 300
```

---

## 📊 分析指标

### 训练指标

- **Loss曲线**: 训练和验证Loss随epoch变化
- **收敛速度**: 达到目标精度所需epoch数
- **稳定性**: Loss波动标准差

### 性能指标

- **Translation Error**: 位移误差（米）
- **Rotation Error**: 旋转误差（度）
- **Time per Epoch**: 每轮训练时间
- **GPU Memory**: GPU内存使用

### 扰动影响分析

- **扰动级别 vs 精度**: 不同扰动强度对最终精度的影响
- **鲁棒性评估**: 模型对扰动的容忍度
- **最优扰动范围**: 推荐的扰动参数范围

---

## 📈 可视化输出

### 1. Loss曲线图

展示训练过程中Loss的变化：
- 训练Loss vs 验证Loss
- 多实验对比
- 平滑趋势线

### 2. 精度对比图

不同配置的精度对比：
- 条形图或箱线图
- 包含误差范围
- 突出最佳配置

### 3. 参数影响热图

参数组合的影响分析：
- X轴: 扰动参数1
- Y轴: 扰动参数2
- 颜色: 性能指标

---

## ⚠️ 注意事项

### 1. 实验数据格式

工具期望的实验目录结构：
```
experiments/
├── perturbation_0.0/
│   ├── config.yaml
│   ├── train_log.txt
│   ├── metrics.json
│   └── checkpoints/
├── perturbation_0.1/
└── ...
```

### 2. 依赖项

需要安装：
```bash
pip install matplotlib seaborn pandas numpy scipy
```

### 3. 内存使用

大规模实验分析可能需要较多内存：
- 单个实验日志可能很大
- 图表渲染占用内存
- 建议: 分批次分析

---

## 💡 分析技巧

### 1. 快速筛选有效实验

```bash
# 先生成摘要
python tools/analysis/analyze_perturbation_training.py \
    --experiment_dir experiments/ \
    --quick_summary

# 根据摘要选择重点实验进行详细分析
```

### 2. 自动化报告生成

```bash
# 在训练脚本中自动调用
python train.py --config config.yaml
python tools/analysis/analyze_perturbation_training.py \
    --experiment_dir experiments/latest/ \
    --output reports/$(date +%Y%m%d)/
```

### 3. 对比基线

```bash
# 指定基线实验
python tools/analysis/analyze_perturbation_training.py \
    --experiment_dir experiments/ \
    --baseline perturbation_0.0 \
    --output comparison_with_baseline/
```

---

## 🔗 相关文档

- [主文档](../README.md)
- [验证工具文档](../validation/README.md)
- [可视化工具文档](../visualization/README.md)

---

**最后更新**: 2026-03-01
