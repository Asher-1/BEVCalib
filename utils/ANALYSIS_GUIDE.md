# BEVCalib 模型性能分析工具完整指南

通用的BEVCalib训练模型性能分析和泛化能力评估工具集。

## 📚 目录

- [快速开始](#快速开始)
- [工具概述](#工具概述)
- [配置文件系统](#配置文件系统)
- [使用示例](#使用示例)
- [输出结果](#输出结果)
- [高级用法](#高级用法)
- [常见问题](#常见问题)
- [最佳实践](#最佳实践)
- [模块API](#模块api)

## 快速开始

### 最简单的方式

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib

# 使用快捷脚本 - 分析5deg实验
bash utils/scripts/quick_analyze.sh 5deg --skip-test

# 分析10deg rotation实验  
bash utils/scripts/quick_analyze.sh 10deg --only-train
```

### 完整分析流程

```bash
# 1. 分析训练性能 + 运行测试评估（耗时20-30分钟）
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config.yaml

# 2. 只分析训练数据（快速预览，2-3秒）
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config.yaml --only-train

# 3. 跳过测试评估，读取已有结果（2-3秒）
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config.yaml --skip-test
```

## 工具概述

### 主要功能

- ✅ **训练日志自动解析** - 支持完整标定和rotation-only模式
- ✅ **测试集泛化评估** - 自动运行checkpoint评估
- ✅ **多组实验对比** - 可视化对比任意多组实验
- ✅ **Feishu兼容报告** - 生成详细的Markdown报告
- ✅ **配置文件驱动** - 快速切换实验组

### 核心文件

| 文件 | 说明 |
| --- | --- |
| `utils/scripts/analyze_experiments.py` | 🌟 统一入口脚本 |
| `utils/configs/experiment_config.yaml` | 配置文件（5deg实验） |
| `utils/configs/experiment_config_rotation.yaml` | 配置文件（rotation实验） |
| `utils/scripts/quick_analyze.sh` | 快捷脚本 |
| `utils/analysis/` | 分析工具模块 |
| `utils/analysis/README.md` | 模块API文档 |

### 输出结果

分析完成后会在配置的 `output_dir` 生成：

**报告**
- `ANALYSIS_REPORT.md` - 完整的Feishu兼容Markdown报告

**可视化**
- `convergence_curves.png` - 收敛曲线对比（4子图）
- `component_breakdown.png` - 误差分量分解
- `generalization_comparison.png` - 泛化能力对比

**测试评估**
- `logs/B26A/{model_dir}/test_data_eval/` - 详细评估结果

## 配置文件系统

### 配置文件快速切换

修改 `utils/configs/experiment_config.yaml` 中的 `experiments` 部分即可切换分析不同实验组：

```yaml
experiments:
  - name: "z=1"
    model_dir: "model_small_5deg_v4-z1"
    zbound_step: 20.0
    checkpoint: "ckpt_400.pth"
  
  - name: "z=5"
    model_dir: "model_small_5deg_v4-z5"
    zbound_step: 4.0
    checkpoint: "ckpt_400.pth"
```

### 预配置场景

| 配置文件 | 场景 | 实验组 |
| --- | --- | --- |
| `experiment_config.yaml` | 5deg完整标定 | z=1, z=5, z=10 |
| `experiment_config_rotation.yaml` | 10deg rotation-only | z=1, z=5, z=10 |

### 完整配置结构

```yaml
# 实验基本信息
experiment:
  name: "实验名称"
  description: "实验描述"
  dataset: "训练数据集名称"
  test_dataset: "测试数据集名称"

# 路径配置
paths:
  bevcalib_root: "/path/to/BEVCalib"          # BEVCalib项目根目录
  logs_base: "/path/to/logs/B26A"             # 日志基础目录
  test_data: "/path/to/test_data"             # 测试数据目录
  output_dir: "/path/to/analysis_results"     # 分析输出目录

# 实验组配置（关键！）
experiments:
  - name: "实验1"              # 显示名称（图表和报告中）
    model_dir: "model_xxx"     # 模型目录名（相对于logs_base）
    zbound_step: 2.0           # BEV Z步长（用于测试评估）
    checkpoint: "ckpt_400.pth" # checkpoint文件名
  
  - name: "实验2"
    model_dir: "model_yyy"
    zbound_step: 4.0
    checkpoint: "ckpt_400.pth"

# 评估配置
evaluation:
  angle_range_deg: 5.0        # 测试时的角度扰动范围
  trans_range: 0.3            # 测试时的平移扰动范围
  batch_size: 8               # 测试批大小
  rotation_only: true         # 是否仅评估旋转
  vis_interval: 100           # 可视化间隔（每N个样本保存一个）
  use_conda: true             # 是否使用conda环境
  conda_env: "bevcalib"       # conda环境名
  run_test_eval: true         # 是否运行测试评估
  parallel: false             # 是否并行评估（暂不支持）

# 可视化配置
visualization:
  generate_plots: true        # 是否生成图表
  plot_style: "default"       # 绘图风格
  figsize:
    convergence: [16, 12]     # 收敛曲线图大小
    generalization: [16, 6]   # 泛化对比图大小
    component: [16, 6]        # 分量分解图大小

# 报告配置
report:
  generate_report: true       # 是否生成报告
  output_filename: "ANALYSIS_REPORT.md"
  include_convergence_milestones: [20, 40, 80, 120, 200, 400]
```

### 关键配置项说明

#### 1. `experiments` - 实验组配置

这是最重要的配置项，定义要分析的所有实验：

- `name`: 实验显示名称，会出现在图表legend和报告表格中
- `model_dir`: 模型日志目录名，相对于 `logs_base` 路径
- `zbound_step`: BEV Z方向的步长，**必须与训练时一致**
  - z=1: 20.0
  - z=5: 4.0
  - z=8: 2.5
  - z=10: 2.0
- `checkpoint`: checkpoint文件名，通常为 `ckpt_400.pth`

#### 2. `evaluation.run_test_eval` - 测试评估开关

- `true`: 运行测试集评估（耗时，适合首次分析）
- `false`: 不运行测试评估，但会尝试读取已有结果（适合重复分析）

使用 `--only-train` 参数可以临时覆盖此配置。

## 使用示例

### 示例1: 完整分析流程

```bash
# 1. 首次分析，运行测试评估
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config.yaml

# 输出:
#   - 解析训练日志
#   - 运行测试集评估（耗时约20-30分钟）
#   - 生成可视化图表
#   - 生成分析报告

# 2. 查看结果
cat analysis_results/ANALYSIS_REPORT.md
```

### 示例2: 快速重新分析

如果已经运行过测试评估，想要重新生成报告和图表：

```bash
# 跳过测试评估，只读取已有结果
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config.yaml --skip-test

# 或者只分析训练数据
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config.yaml --only-train
```

### 示例3: 对比不同超参数实验

假设你训练了多组学习率实验，想对比它们的性能：

创建 `utils/configs/experiment_config_lr.yaml`:

```yaml
experiment:
  name: "学习率消融实验"
  description: "对比不同初始学习率的影响"

paths:
  bevcalib_root: "/mnt/drtraining/user/dahailu/code/BEVCalib"
  logs_base: "/mnt/drtraining/user/dahailu/code/BEVCalib/logs/B26A"
  output_dir: "/mnt/drtraining/user/dahailu/code/BEVCalib/analysis_lr"

experiments:
  - name: "lr=1e-4"
    model_dir: "model_small_5deg_v5-lr1e4"
    zbound_step: 2.0
  
  - name: "lr=2e-4"
    model_dir: "model_small_5deg_v5-lr2e4"
    zbound_step: 2.0
  
  - name: "lr=5e-4"
    model_dir: "model_small_5deg_v5-lr5e4"
    zbound_step: 2.0

evaluation:
  run_test_eval: false  # 假设不需要测试评估
```

运行:
```bash
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config_lr.yaml --only-train
```

### 示例4: 单个模型深度分析

如果只想分析单个模型：

```yaml
experiments:
  - name: "最佳模型"
    model_dir: "model_small_5deg_v4-z10"
    zbound_step: 2.0
```

### 示例5: 快速切换到rotation实验

```bash
# 方式1: 使用快捷脚本
bash utils/scripts/quick_analyze.sh 10deg --only-train

# 方式2: 直接使用入口脚本
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config_rotation.yaml --only-train
```

## 输出结果

### 报告示例

生成的报告包含：

#### 一、训练性能总结
- 最终训练精度表格（所有误差指标）
- 关键发现（最佳性能、性能提升百分比）
- 收敛速度分析（关键里程碑对比）

#### 二、泛化能力评估
- 测试集性能总结表格
- 详细统计（mean, std, percentiles）
- 泛化能力分析（衰退倍数评级）

#### 三、可视化图表
- 收敛曲线对比
- 误差分量分解
- 泛化能力对比

#### 四、结论与建议
- 核心结论
- 应用场景建议

### 可视化图表

#### 1. convergence_curves.png
- 训练平移误差收敛曲线
- 训练旋转误差收敛曲线
- Lateral(Y)误差收敛（关键指标）
- 所有误差分量最终对比

#### 2. component_breakdown.png
- 平移分量: Forward/Lateral/Height
- 旋转分量: Roll/Pitch/Yaw

#### 3. generalization_comparison.png
- 训练vs测试误差对比
- 泛化衰退倍数柱状图

## 高级用法

### 使用Python模块进行自定义分析

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, 'utils')
from analysis import TrainingAnalyzer, Visualizer, ReportGenerator

# 分析单个实验
analyzer = TrainingAnalyzer("logs/B26A/model_xxx/train.log")
data = analyzer.parse_log()
final_metrics = analyzer.get_final_metrics()

print(f"Final Trans Error: {final_metrics['trans_error']:.4f}m")
print(f"Final Rot Error: {final_metrics['rot_error']:.2f}°")

# 生成自定义可视化
visualizer = Visualizer("custom_output")
visualizer.plot_convergence_curves({"exp1": {"data": data, "final_metrics": final_metrics}})
```

### 批量评估多组checkpoint

```python
from analysis import TestEvaluator

evaluator = TestEvaluator(
    bevcalib_root="/path/to/BEVCalib",
    test_data_root="/path/to/test_data"
)

checkpoints = [
    ("ckpt_200.pth", 2.0),
    ("ckpt_400.pth", 2.0),
]

for ckpt_name, zbound in checkpoints:
    evaluator.evaluate_checkpoint(
        ckpt_path=f"logs/model/checkpoint/{ckpt_name}",
        output_dir=f"test_eval_{ckpt_name}",
        zbound_step=zbound
    )
```

## 常见问题

### Q1: 如何添加新的实验组？

在配置文件的 `experiments` 部分添加新条目：

```yaml
experiments:
  - name: "新实验"
    model_dir: "model_new_v1"
    zbound_step: 2.0
    checkpoint: "ckpt_400.pth"
```

### Q2: 训练日志解析失败怎么办？

检查：
1. 日志文件路径是否正确
2. 日志文件是否完整（至少包含几个epoch的数据）
3. 日志格式是否与工具兼容

如果是新的日志格式，可能需要修改 `training_analyzer.py` 中的正则表达式。

### Q3: 测试评估时间太长怎么办？

- 使用 `--skip-test` 或 `--only-train` 跳过评估
- 减小 `evaluation.batch_size`
- 增大 `evaluation.vis_interval`（减少可视化样本数）

### Q4: 如何只评估某些模型？

在配置文件中只列出要评估的实验，注释掉或删除其他实验。

### Q5: 内存不足怎么办？

- 减小 `evaluation.batch_size` (8 → 4 → 2)
- 串行评估而非并行（默认已是串行）

### Q6: 如何自定义图表样式？

修改配置文件的 `visualization.plot_style`：

```yaml
visualization:
  plot_style: "seaborn"  # 可选: seaborn, ggplot, bmh等
```

## 最佳实践

### 1. 首次分析新实验组

```bash
# 步骤1: 创建配置文件
cp utils/configs/experiment_config.yaml utils/configs/my_experiment.yaml
# 编辑my_experiment.yaml，修改experiments列表

# 步骤2: 先只分析训练数据（快速预览）
python utils/scripts/analyze_experiments.py --config utils/configs/my_experiment.yaml --only-train

# 步骤3: 如果需要，运行完整评估
python utils/scripts/analyze_experiments.py --config utils/configs/my_experiment.yaml
```

### 2. 重复分析优化

如果测试评估已经完成，想要调整可视化或报告：

```bash
# 修改配置文件中的visualization或report部分
# 然后运行（会读取已有的测试结果）
python utils/scripts/analyze_experiments.py --config utils/configs/my_experiment.yaml --skip-test
```

### 3. 对比不同实验组

保持多个配置文件，快速切换：

```bash
# 分析5deg实验
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config.yaml

# 切换到10deg实验
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config_rotation.yaml

# 切换到学习率实验
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config_lr.yaml
```

## 模块API

### TrainingAnalyzer - 训练日志分析器

解析训练日志提取性能指标。

```python
from analysis import TrainingAnalyzer

analyzer = TrainingAnalyzer("path/to/train.log")
data = analyzer.parse_log()
final_metrics = analyzer.get_final_metrics()
convergence = analyzer.get_convergence_data()
```

### TestEvaluator - 测试数据评估器

运行checkpoint在测试数据上的评估。

```python
from analysis import TestEvaluator

evaluator = TestEvaluator(
    bevcalib_root="/path/to/BEVCalib",
    test_data_root="/path/to/test_data"
)

# 运行评估
evaluator.evaluate_checkpoint(
    ckpt_path="path/to/ckpt_400.pth",
    output_dir="output/dir",
    zbound_step=2.0,
    angle_range=5.0,
    trans_range=0.3
)

# 解析结果
results = evaluator.parse_evaluation_results("output/dir")
```

### Visualizer - 可视化工具

生成对比图表。

```python
from analysis import Visualizer

visualizer = Visualizer(output_dir="output/dir")

# 绘制收敛曲线
visualizer.plot_convergence_curves(experiments_data)

# 绘制泛化对比
visualizer.plot_generalization_comparison(
    experiments_data,
    test_results
)

# 绘制误差分量
visualizer.plot_component_breakdown(experiments_data)
```

### ReportGenerator - 报告生成器

生成Feishu兼容的Markdown报告。

```python
from analysis import ReportGenerator

generator = ReportGenerator(output_dir="output/dir")

generator.generate_complete_report(
    experiment_config=config,
    training_results=training_results,
    test_results=test_results
)
```

详细API文档见 `utils/analysis/README.md`

## 目录结构

```
code/BEVCalib/
├── utils/
│   ├── scripts/
│   │   ├── analyze_experiments.py     # 统一入口脚本
│   │   └── quick_analyze.sh           # 快捷脚本
│   │
│   ├── configs/
│   │   ├── experiment_config.yaml     # 5deg实验配置
│   │   └── experiment_config_rotation.yaml  # rotation实验配置
│   │
│   ├── analysis/                       # 分析工具模块
│   │   ├── __init__.py
│   │   ├── training_analyzer.py       # 训练日志分析
│   │   ├── test_evaluator.py          # 测试数据评估
│   │   ├── report_generator.py        # 报告生成
│   │   ├── visualizer.py              # 可视化
│   │   ├── README.md                  # 模块文档
│   │   └── legacy/                    # 旧版脚本
│   │
│   └── ANALYSIS_GUIDE.md              # 本指南
│
├── analysis_results/                   # 默认输出目录（5deg）
│   ├── ANALYSIS_REPORT.md
│   ├── convergence_curves.png
│   ├── component_breakdown.png
│   └── generalization_comparison.png
│
└── analysis_results_rotation/          # rotation实验输出
```

## 报告Feishu兼容说明

生成的Markdown报告遵循Feishu最佳实践：

- ✅ 使用标准Markdown表格格式（`| --- |`分隔符）
- ✅ 数字列右对齐（`---:`）
- ✅ 表格内避免使用bold标记
- ✅ 使用中文编号而非 `#` 标题（一、二、三）
- ✅ 图表使用标准 `![](image.png)` 引用

可以直接复制报告内容粘贴到Feishu文档。

## 快速命令备忘

```bash
# ============ 快捷分析 ============
# 5deg实验（快速）
bash utils/scripts/quick_analyze.sh 5deg --skip-test

# 10deg rotation实验（只看训练）
bash utils/scripts/quick_analyze.sh 10deg --only-train

# ============ 完整分析 ============
# 使用默认配置
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config.yaml

# 只分析训练
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config.yaml --only-train

# 跳过测试评估
python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config.yaml --skip-test

# ============ 自定义配置 ============
python utils/scripts/analyze_experiments.py --config utils/configs/my_config.yaml
```

## 更新日志

### v1.0.0 (2026-03-09)

- ✅ 初始版本发布
- ✅ 支持训练日志分析
- ✅ 支持测试集评估
- ✅ 支持可视化生成
- ✅ 支持报告生成
- ✅ 配置文件系统
- ✅ 支持rotation-only模式
- ✅ 文件整理到utils/目录

---

**创建日期**: 2026-03-09  
**推荐使用**: `quick_analyze.sh` 或 `analyze_experiments.py`  
**文档位置**: `utils/ANALYSIS_GUIDE.md`
