# BEVCalib 训练分析工具包

通用的BEVCalib训练模型性能分析和泛化能力评估工具集。

## 功能特性

- ✅ **训练日志分析**: 自动解析训练日志，提取性能指标
- ✅ **测试集评估**: 自动运行checkpoint在测试数据上的评估
- ✅ **可视化对比**: 生成收敛曲线、误差分量、泛化能力等对比图表
- ✅ **报告生成**: 自动生成Feishu兼容的Markdown分析报告
- ✅ **配置化**: 通过YAML配置文件快速切换分析不同实验组
- ✅ **模块化**: 可单独使用各个模块进行定制分析

## 快速开始

### 1. 基本用法

```bash
# 使用默认配置分析实验
cd /mnt/drtraining/user/dahailu/code/BEVCalib
python analyze_experiments.py

# 使用自定义配置
python analyze_experiments.py --config experiment_config_rotation.yaml

# 只分析训练数据，不运行测试评估
python analyze_experiments.py --only-train

# 跳过测试评估但读取已有结果
python analyze_experiments.py --skip-test
```

### 2. 配置文件

编辑 `experiment_config.yaml` 配置要分析的实验：

```yaml
experiments:
  - name: "实验1"
    model_dir: "model_small_5deg_v4-z1"  # 模型目录名
    zbound_step: 20.0                     # BEV Z步长
    checkpoint: "ckpt_400.pth"            # checkpoint文件名
  
  - name: "实验2"
    model_dir: "model_small_5deg_v4-z5"
    zbound_step: 4.0
    checkpoint: "ckpt_400.pth"
```

### 3. 输出结果

分析完成后会在 `analysis_results/` 目录生成：

- `ANALYSIS_REPORT.md` - 完整分析报告
- `convergence_curves.png` - 训练收敛曲线对比
- `component_breakdown.png` - 误差分量详细分解
- `generalization_comparison.png` - 泛化能力对比（如果运行了测试评估）

## 模块说明

### TrainingAnalyzer

训练日志分析器，解析训练日志提取性能指标。

```python
from analysis import TrainingAnalyzer

analyzer = TrainingAnalyzer("path/to/train.log")
data = analyzer.parse_log()
final_metrics = analyzer.get_final_metrics()
convergence = analyzer.get_convergence_data()
```

### TestEvaluator

测试数据评估器，运行checkpoint在测试数据上的评估。

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

### Visualizer

可视化工具，生成对比图表。

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

### ReportGenerator

报告生成器，生成Feishu兼容的Markdown报告。

```python
from analysis import ReportGenerator

generator = ReportGenerator(output_dir="output/dir")

generator.generate_complete_report(
    experiment_config=config,
    training_results=training_results,
    test_results=test_results
)
```

## 配置文件详解

### 基本信息

```yaml
experiment:
  name: "实验名称"
  description: "实验描述"
  dataset: "训练数据集"
  test_dataset: "测试数据集"
```

### 路径配置

```yaml
paths:
  bevcalib_root: "/path/to/BEVCalib"        # BEVCalib根目录
  logs_base: "/path/to/logs/B26A"           # 日志基础目录
  test_data: "/path/to/test_data"           # 测试数据目录
  output_dir: "/path/to/analysis_results"   # 输出目录
```

### 实验配置

```yaml
experiments:
  - name: "z=1"              # 实验名称（显示在图表和报告中）
    model_dir: "model_xxx"   # 模型目录（相对于logs_base）
    zbound_step: 20.0        # BEV Z步长（用于评估）
    checkpoint: "ckpt_400.pth"  # checkpoint文件名
```

### 评估配置

```yaml
evaluation:
  angle_range_deg: 5.0      # 角度扰动范围
  trans_range: 0.3          # 平移扰动范围
  batch_size: 8             # 批大小
  rotation_only: true       # 是否仅评估旋转
  vis_interval: 100         # 可视化间隔
  use_conda: true           # 是否使用conda环境
  conda_env: "bevcalib"     # conda环境名
  run_test_eval: true       # 是否运行测试评估
```

### 可视化配置

```yaml
visualization:
  generate_plots: true      # 是否生成图表
  plot_style: "default"     # 绘图风格
  figsize:
    convergence: [16, 12]   # 收敛曲线图大小
    generalization: [16, 6] # 泛化对比图大小
    component: [16, 6]      # 分量分解图大小
```

## 使用示例

### 示例1: 分析Z-Ablation实验

```bash
# 1. 编辑配置文件（experiment_config.yaml已配置好）
# 2. 运行分析
python analyze_experiments.py

# 结果将保存在 analysis_results/
```

### 示例2: 快速切换到Rotation实验

```bash
# 使用预配置的rotation配置文件
python analyze_experiments.py --config experiment_config_rotation.yaml

# 结果将保存在 analysis_results_rotation/
```

### 示例3: 只分析训练数据

```bash
# 适用于测试集评估时间较长的情况
python analyze_experiments.py --only-train
```

### 示例4: 自定义分析脚本

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, 'utils')
from analysis import TrainingAnalyzer, Visualizer

# 分析多个实验
experiments = {
    "exp1": "logs/B26A/model_xxx/train.log",
    "exp2": "logs/B26A/model_yyy/train.log",
}

results = {}
for name, log_path in experiments.items():
    analyzer = TrainingAnalyzer(log_path)
    results[name] = {
        'data': analyzer.parse_log(),
        'final_metrics': analyzer.get_final_metrics()
    }

# 生成可视化
visualizer = Visualizer("output")
visualizer.plot_convergence_curves(results)
```

## 常见问题

### Q: 如何添加新的实验组？

A: 在配置文件的 `experiments` 部分添加新条目：

```yaml
experiments:
  - name: "新实验"
    model_dir: "model_new"
    zbound_step: 2.0
    checkpoint: "ckpt_400.pth"
```

### Q: 如何修改测试集评估参数？

A: 修改配置文件的 `evaluation` 部分：

```yaml
evaluation:
  angle_range_deg: 10.0  # 改为10度扰动
  trans_range: 0.5       # 改为0.5m扰动
```

### Q: 如何跳过已完成的测试评估？

A: 使用 `--skip-test` 参数，工具会自动读取已有的评估结果：

```bash
python analyze_experiments.py --skip-test
```

### Q: 如何自定义图表样式？

A: 修改配置文件的 `visualization.plot_style`：

```yaml
visualization:
  plot_style: "seaborn"  # 可选: seaborn, ggplot, bmh等
```

### Q: 评估时内存不足怎么办？

A: 减小batch_size：

```yaml
evaluation:
  batch_size: 4  # 从8减到4或更小
```

## 依赖要求

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- PyYAML
- BEVCalib环境

## 更新日志

### v1.0.0 (2026-03-09)

- ✅ 初始版本发布
- ✅ 支持训练日志分析
- ✅ 支持测试集评估
- ✅ 支持可视化生成
- ✅ 支持报告生成
- ✅ 配置文件系统

## 作者

dahailu @ BEVCalib Project

## 许可

遵循BEVCalib项目许可
