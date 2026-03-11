# 工具迁移和优化总结

## 已完成的工作

### 1. 模块化重构

原有的独立脚本已被重构为可复用的模块：

#### 原脚本 → 新模块映射

| 原脚本 | 新模块 | 功能 |
| --- | --- | --- |
| `analyze_z_comparison.py` | `training_analyzer.py` | 训练日志解析 |
| `batch_evaluate_test_data.sh` | `test_evaluator.py` | 测试数据评估 |
| `run_parallel_eval.py` | `test_evaluator.py` | 批量评估 |
| `summarize_test_results.py` | `test_evaluator.py` + `visualizer.py` | 结果汇总 |
| `generate_detailed_report.py` | `report_generator.py` + `visualizer.py` | 报告生成 |

#### 新增功能

- ✅ 配置文件驱动（YAML）
- ✅ 统一入口脚本（`analyze_experiments.py`）
- ✅ 支持rotation-only模式
- ✅ 更好的错误处理
- ✅ 模块化设计，易于扩展

### 2. 目录结构

```
code/BEVCalib/
├── analyze_experiments.py          # 统一入口脚本 ⭐
├── experiment_config.yaml          # 5deg实验配置
├── experiment_config_rotation.yaml # rotation实验配置
├── quick_analyze.sh                # 快捷脚本
├── ANALYSIS_GUIDE.md              # 使用指南 ⭐
│
├── utils/analysis/                 # 分析工具包 ⭐
│   ├── __init__.py
│   ├── training_analyzer.py       # 训练日志分析
│   ├── test_evaluator.py          # 测试评估
│   ├── report_generator.py        # 报告生成
│   ├── visualizer.py              # 可视化
│   ├── README.md                  # API文档
│   └── legacy/                    # 旧版脚本（已废弃）
│       ├── analyze_z_comparison.py
│       ├── batch_evaluate_test_data.sh
│       ├── run_parallel_eval.py
│       ├── summarize_test_results.py
│       └── generate_detailed_report.py
│
├── analysis_results/               # 5deg实验输出
└── analysis_results_rotation/      # rotation实验输出
```

### 3. 核心改进

#### 改进1: 配置化

**之前**: 需要修改Python代码来切换实验组
```python
# 硬编码在脚本中
MODELS = {
    "z=1": "model_small_5deg_v4-z1",
    "z=5": "model_small_5deg_v4-z5",
    "z=10": "model_small_5deg_v4-z10",
}
```

**现在**: 只需修改YAML配置文件
```yaml
experiments:
  - name: "z=1"
    model_dir: "model_small_5deg_v4-z1"
    zbound_step: 20.0
```

#### 改进2: 统一入口

**之前**: 需要运行多个脚本
```bash
python analyze_z_comparison.py
bash batch_evaluate_test_data.sh
python summarize_test_results.py
python generate_detailed_report.py
```

**现在**: 一条命令完成所有分析
```bash
python analyze_experiments.py
```

#### 改进3: 灵活性

**之前**: 每个实验组需要新脚本

**现在**: 
- 支持任意数量的实验组
- 支持rotation-only和完整标定模式
- 支持自定义checkpoint
- 支持自定义输出目录

#### 改进4: 可复用性

**之前**: 脚本功能耦合，难以复用

**现在**: 模块化设计，可以单独使用
```python
from analysis import TrainingAnalyzer
analyzer = TrainingAnalyzer("path/to/log")
data = analyzer.parse_log()
```

## 使用示例

### 示例1: 分析当前的5deg实验

```bash
# 快速分析（使用已有测试结果）
python analyze_experiments.py --skip-test

# 或使用快捷脚本
bash quick_analyze.sh 5deg --skip-test
```

### 示例2: 分析10deg rotation实验

```bash
# 只看训练数据（快速）
python analyze_experiments.py --config experiment_config_rotation.yaml --only-train

# 或使用快捷脚本
bash quick_analyze.sh 10deg --only-train
```

### 示例3: 完整分析流程（含测试评估）

```bash
# 首次分析，运行完整评估（耗时20-30分钟）
python analyze_experiments.py
```

## 配置文件快速切换

已提供两个预配置文件：

### 1. `experiment_config.yaml` - 5deg实验
分析: z=1, z=5, z=10 (5度扰动，完整标定)

### 2. `experiment_config_rotation.yaml` - rotation实验
分析: z=1, z=5, z=10 (10度扰动，仅旋转)

### 创建新配置

复制并修改：
```bash
cp experiment_config.yaml my_experiment.yaml
# 编辑 my_experiment.yaml，修改experiments部分
python analyze_experiments.py --config my_experiment.yaml
```

## 模块API

所有模块都可以单独导入使用：

```python
from analysis import TrainingAnalyzer, TestEvaluator, ReportGenerator, Visualizer

# 训练分析
analyzer = TrainingAnalyzer("log_path")
data = analyzer.parse_log()

# 测试评估
evaluator = TestEvaluator(bevcalib_root, test_data_root)
evaluator.evaluate_checkpoint(...)

# 可视化
visualizer = Visualizer(output_dir)
visualizer.plot_convergence_curves(data)

# 报告生成
generator = ReportGenerator(output_dir)
generator.generate_complete_report(...)
```

详细API文档见 `utils/analysis/README.md`

## 已测试场景

- ✅ 5deg z-ablation (z=1, z=5, z=10)
- ✅ 10deg rotation-only (z=1, z=5, z=10)
- ✅ 单个模型分析
- ✅ 跨数据集泛化评估

## 依赖要求

- Python 3.7+
- PyYAML: `pip install pyyaml`
- NumPy, Matplotlib
- BEVCalib conda环境

## 迁移说明

旧脚本已移至 `utils/analysis/legacy/`，不再维护。

建议统一使用新的 `analyze_experiments.py` 入口脚本。

## 贡献者

dahailu @ BEVCalib Project

## 更新日期

2026-03-09
