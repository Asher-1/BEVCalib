# 🚀 BEVCalib 快速开始

5分钟快速上手BEVCalib训练和分析工具。

## ⚡ 最快的方式

### 1. 训练模型（3行命令）

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib
conda activate bevcalib
bash start_training.sh B26A v1
```

### 2. 批量训练（使用配置文件）

```bash
# 运行预配置的Z分辨率消融实验（3组，自动跳过已完成的实验）
bash batch_train.sh configs/batch_train_5deg.yaml

# 或先预览命令
bash batch_train.sh --dry-run configs/batch_train_5deg.yaml

# 强制重跑所有实验（忽略已存在的输出目录）
bash batch_train.sh --force configs/batch_train_5deg.yaml
```

### 3. 分析性能（1行命令）

```bash
# 分析训练结果（2-3秒）
bash utils/scripts/quick_analyze.sh 5deg --skip-test
```

## 📚 核心文档

| 场景 | 查看文档 | 耗时 |
| --- | --- | --- |
| 🚀 **训练模型** | [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | 5分钟 |
| 📊 **分析性能** | [utils/ANALYSIS_GUIDE.md](utils/ANALYSIS_GUIDE.md) | 3分钟 |
| 🔧 **批量实验配置** | [configs/README.md](configs/README.md) | 5分钟 |
| 📖 **项目总览** | [README.md](README.md) | 10分钟 |

## 🎯 常见任务

### 任务1: 快速验证配置

```bash
# 使用B26A小数据集（约1-2小时/100epochs）
bash start_training.sh B26A v1

# 监控训练
tail -f logs/B26A/model_small_10deg_v1/train.log
```

### 任务2: Z分辨率对比实验

```bash
# 方式1: 使用配置文件（推荐）
bash batch_train.sh configs/batch_train_5deg.yaml

# 方式2: 手动运行
BEV_ZBOUND_STEP=20.0 bash start_training.sh B26A v1-z1 --ddp
BEV_ZBOUND_STEP=4.0 bash start_training.sh B26A v1-z5 --ddp
BEV_ZBOUND_STEP=2.0 bash start_training.sh B26A v1-z10 --ddp
```

### 任务3: 学习率调优

```bash
# 使用配置文件运行4组学习率对比
bash batch_train.sh configs/batch_train_lr_ablation.yaml
```

### 任务4: 评估模型

```bash
python evaluate_checkpoint.py \
    --ckpt_path logs/B26A/model_small_5deg_v1/B26A_scratch/checkpoint/ckpt_400.pth \
    --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/bevcalib_training_data \
    --angle_range_deg 5.0 \
    --trans_range 0.3
```

### 任务5: 生成性能报告

new strategy
```bash
# 使用 B26A YAML 配置
python run_generalization_eval.py --config configs/eval_generalization_b26a.yaml
# 使用 ALL YAML 配置
python run_generalization_eval.py --config configs/eval_generalization_all.yaml
# 命令行覆盖
python run_generalization_eval.py --config configs/eval_generalization_all.yaml --angle_range 10.0 --output_dir logs/custom_eval
```


### 任务6: pytorch模型转drinfer模型

```bash
# 使用 B26A YAML 配置
python utils/torch2drinfer.py --config configs/drinfer_config_b26a.yaml
# 使用 ALL YAML 配置
python utils/torch2drinfer.py --config configs/drinfer_config_all.yaml
```


```bash
# 快速分析（使用已有测试结果）
bash utils/scripts/quick_analyze.sh 5deg --skip-test

# 查看报告
cat analysis_results/ANALYSIS_REPORT.md
```

## 🔧 自定义配置

### 创建批量训练配置

```bash
# 1. 复制模板
cp configs/batch_train_template.yaml configs/my_experiments.yaml

# 2. 编辑配置（示例）
cat > configs/my_experiments.yaml << 'EOF'
global:
  dry_run: false
  force_rerun: false             # true = 强制重跑已完成的实验
  wait_between_experiments: 10

experiments:
  - name: "基线实验"
    dataset: "B26A"
    version: "v1-baseline"
    params:
      angle_range_deg: 5
      use_ddp: true
      foreground: true
      no_tensorboard: true

  - name: "优化实验"
    dataset: "B26A"
    version: "v1-optimized"
    params:
      angle_range_deg: 5
      learning_rate: 0.00015
      use_ddp: true
      foreground: true
      no_tensorboard: true
EOF

# 3. 验证配置
bash batch_train.sh --dry-run configs/my_experiments.yaml

# 4. 运行
bash batch_train.sh configs/my_experiments.yaml
```

## 📊 监控训练

```bash
# GPU状态
nvidia-smi -l 1

# 训练日志
tail -f logs/B26A/model_small_5deg_v1/train.log

# TensorBoard
tensorboard --logdir logs/B26A/ --port 6006
# 访问: http://localhost:6006
```

## 🛑 停止训练

```bash
bash stop_training.sh
# 或
pkill -f train_kitti
```

## 💡 实用技巧

### 1. 快速查看命令帮助

```bash
bash start_training.sh         # 显示用法
bash batch_train.sh --help     # 显示帮助
```

### 2. 后台运行批量训练

```bash
nohup bash batch_train.sh configs/batch_train_5deg.yaml > batch.log 2>&1 &

# 监控进度
tail -f batch.log
```

### 3. 查看可用配置

```bash
ls configs/batch_train_*.yaml
# batch_train_5deg.yaml           - 5度Z消融
# batch_train_10deg_rotation.yaml - 10度rotation-only
# batch_train_lr_ablation.yaml    - 学习率消融
# batch_train_template.yaml       - 配置模板
```

### 4. 对比实验结果

```bash
# 训练完成后
bash utils/scripts/quick_analyze.sh 5deg --skip-test

# 查看对比报告
cat analysis_results/ANALYSIS_REPORT.md
```

## 📁 重要路径

```
训练脚本:     start_training.sh, train_universal.sh, batch_train.sh
训练配置:     configs/batch_train_*.yaml
分析脚本:     utils/scripts/analyze_experiments.py
分析配置:     utils/configs/experiment_config*.yaml
训练日志:     logs/<dataset>/model_*_<version>/
分析结果:     analysis_results/
```

## ❓ 遇到问题？

1. **查看详细文档**: 
   - 训练问题 → [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
   - 分析问题 → [utils/ANALYSIS_GUIDE.md](utils/ANALYSIS_GUIDE.md)

2. **查看更新日志**: [CHANGELOG_v2.1.md](CHANGELOG_v2.1.md)

3. **查看配置文档**: [configs/README.md](configs/README.md)

## 🎓 下一步

1. ✅ 完成快速开始 → 阅读 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
2. ✅ 理解训练流程 → 创建自定义配置
3. ✅ 训练完成 → 使用分析工具评估
4. ✅ 分析完成 → 根据报告优化参数

---

**文档版本**: v2.1  
**最后更新**: 2026-03-09  
**适合**: 首次使用BEVCalib的用户
