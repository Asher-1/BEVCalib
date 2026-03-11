# BEVCalib 文档结构说明

## 📚 文档组织

### 主要文档（项目根目录）

| 文档 | 用途 | 优先级 |
| --- | --- | --- |
| **README.md** | 项目总览、快速入口 | ⭐⭐⭐ |
| **TRAINING_GUIDE.md** | 完整训练指南 | ⭐⭐⭐ |
| **PREPARE_CUSTOM_DATASET.md** | 数据集准备 | ⭐⭐ |
| **CUSTOM_DATASET_TRAINING.md** | 自定义数据集训练 | ⭐⭐ |
| **CHANGELOG_DOCS.md** | 文档更新日志 | ⭐ |

### 分析工具文档（utils/）

| 文档 | 用途 | 优先级 |
| --- | --- | --- |
| **utils/ANALYSIS_GUIDE.md** | 完整分析指南 | ⭐⭐⭐ |
| **utils/analysis/README.md** | 模块API文档 | ⭐⭐ |
| **utils/analysis/MIGRATION_SUMMARY.md** | 工具迁移说明 | ⭐ |

## 🗂️ 目录结构

```
BEVCalib/
├── README.md                         # 项目入口
├── TRAINING_GUIDE.md                 # 🌟 完整训练指南
├── CHANGELOG_DOCS.md                 # 文档更新日志
├── DOCS_STRUCTURE.md                 # 本文件
│
├── evaluate_checkpoint.py            # Checkpoint评估工具
│
├── start_training.sh                 # 🚀 快速训练脚本
├── train_universal.sh                # 🔧 通用训练脚本
├── batch_train.sh                    # 📊 批量训练脚本
├── stop_training.sh
│
├── kitti-bev-calib/                  # 核心训练代码
│   ├── train_kitti.py
│   ├── inference_kitti.py
│   └── ...
│
├── tools/                            # 数据集工具
│   ├── prepare_custom_dataset.py
│   ├── validate_kitti_odometry.py
│   └── view_pointcloud.py
│
├── utils/                            # 工具和分析
│   ├── scripts/
│   │   ├── analyze_experiments.py  # 分析入口脚本
│   │   └── quick_analyze.sh        # 快捷脚本
│   │
│   ├── configs/                    # 配置文件
│   │   ├── experiment_config.yaml
│   │   └── experiment_config_rotation.yaml
│   │
│   ├── analysis/                   # 分析模块
│   │   ├── training_analyzer.py
│   │   ├── test_evaluator.py
│   │   ├── report_generator.py
│   │   ├── visualizer.py
│   │   ├── README.md
│   │   └── legacy/                 # 旧版脚本
│   │
│   └── ANALYSIS_GUIDE.md           # 🌟 完整分析指南
│
├── logs/                             # 训练日志（自动生成）
│   ├── B26A/
│   └── all_training_data/
│
└── analysis_results/                 # 分析结果（自动生成）
```

## 🎯 快速导航

### 我想训练模型
1. 阅读: **TRAINING_GUIDE.md**
2. 运行: `bash start_training.sh B26A v1`
3. 监控: `tail -f logs/B26A/model_small_10deg_v1/train.log`

### 我想分析性能
1. 阅读: **utils/ANALYSIS_GUIDE.md**
2. 运行: `bash utils/scripts/quick_analyze.sh 5deg --skip-test`
3. 查看: `cat analysis_results/ANALYSIS_REPORT.md`

### 我想评估模型
1. 阅读: **README.md** → Evaluation 章节
2. 运行: `python evaluate_checkpoint.py --ckpt_path ... --dataset_root ...`
3. 查看: `logs/.../ckpt_XXX_eval/extrinsics_and_errors.txt`

### 我想准备自定义数据集
1. 阅读: **PREPARE_CUSTOM_DATASET.md**
2. 运行: `python tools/prepare_custom_dataset.py ...`
3. 训练: **CUSTOM_DATASET_TRAINING.md**

## 📖 推荐阅读顺序

### 新用户
1. README.md (项目概览)
2. TRAINING_GUIDE.md (快速开始 → 基础训练)
3. utils/ANALYSIS_GUIDE.md (快速开始 → 基础分析)

### 进阶用户
1. TRAINING_GUIDE.md (高级用法 → 最佳实践)
2. utils/ANALYSIS_GUIDE.md (模块API → 自定义分析)
3. utils/analysis/README.md (深入模块使用)

### 开发者
1. 所有主要文档
2. utils/analysis/MIGRATION_SUMMARY.md (了解工具演化)
3. CHANGELOG_DOCS.md (了解文档变更)

## 🔄 主要变更（2026-03-09）

### 文档合并
- ✅ `README_TRAINING_SCRIPTS.md` + `QUICK_START_TRAINING.md` → **TRAINING_GUIDE.md**
- ✅ `README_ANALYSIS.md` + `ANALYSIS_GUIDE.md` → **utils/ANALYSIS_GUIDE.md**

### 文件迁移
- ✅ 分析脚本 → `utils/scripts/`
- ✅ 配置文件 → `utils/configs/`
- ✅ 分析文档 → `utils/`

### 删除重复
- ❌ 删除 `README_TRAINING_SCRIPTS.md`
- ❌ 删除 `QUICK_START_TRAINING.md`
- ❌ 删除 `README_ANALYSIS.md`
- ❌ 删除 `ANALYSIS_GUIDE.md`

详细变更见: **CHANGELOG_DOCS.md**

## 💡 文档编写原则

### 1. 单一信息源（SST - Single Source of Truth）
- 每个主题只在一个主要文档中详细说明
- 其他文档通过链接引用
- 避免在多处维护相同内容

### 2. 分层组织
- **主文档**: 完整、详细
- **README**: 概览、快速链接
- **API文档**: 技术细节

### 3. 实用优先
- 提供可直接运行的命令示例
- 常见问题和解决方案
- 最佳实践和推荐设置

### 4. 清晰导航
- 明确的目录结构
- 快速导航章节
- 相关文档链接

## 📞 文档维护

### 更新文档时
1. 确认是否会影响其他文档
2. 更新相关的链接
3. 检查命令示例是否仍然有效
4. 更新 CHANGELOG_DOCS.md

### 添加新功能时
1. 在主文档中添加详细说明
2. 在 README.md 中添加简要介绍和链接
3. 提供命令示例
4. 更新快速命令参考

---

**维护者**: dahailu  
**最后更新**: 2026-03-09  
**版本**: v2.0
