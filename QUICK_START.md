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

#### 评估脚本：测试集采样与均衡统计

评估与训练共用 `CustomDataset` 的采样语义；命令行参数带 `eval_` 前缀，避免与训练参数混淆。

| 参数 | 作用 | 备注 |
| --- | --- | --- |
| `--eval_sample_step N` | 测试集按步长下采样（每隔 N 帧取 1 帧） | 与 `--eval_max_frames_per_seq` **互斥**；不传则使用目录内全部帧 |
| `--eval_max_frames_per_seq K` | 每个序列最多保留 K 帧（均匀下采样） | 与训练 `--max_frames_per_seq` 语义一致；与 `--eval_sample_step` **互斥** |
| `--data_balance M` | 汇总指标以何种方式为主 | `0`（默认）：`PRIMARY_METRIC` 为 **Micro**（样本加权平均）；`1` 或 `2`：`PRIMARY_METRIC` 为 **Macro**（按序列等权平均，减轻长序列主导） |

说明：

- **训练侧**对应参数为 `--sample_step` 与 `--max_frames_per_seq`（同样在 `CustomDataset` 层互斥），批量训练 YAML 里常写作 `sample_step` / `max_frames_per_seq`。
- 泛化测试建议加 `--use_full_dataset`，否则默认按固定种子做 80/20 划分，仅评验证子集。
- `evaluate_drinfer.py` 同样支持 `--eval_sample_step`、`--eval_max_frames_per_seq`、`--data_balance`。

示例（全量测试数据 + 步长 2 + Macro 主指标）：

```bash
USE_DRCV_BACKEND=1 HF_HUB_OFFLINE=1 python evaluate_checkpoint.py \
  --ckpt_path <path/to/ckpt_best_val.pth> \
  --dataset_root /path/to/test_data \
  --use_full_dataset \
  --eval_sample_step 2 \
  --data_balance 1 \
  --angle_range_deg 5.0 \
  --trans_range 0.15 \
  --output_dir logs/my_eval_run
```

### 任务5: 行程标定（从 rosbag 自动标定 LiDAR-Camera 外参）

行程标定支持两种方式运行：**YAML 配置文件**（推荐）和**命令行参数**。

#### 方式1：YAML 配置文件（推荐）

```bash
# 一行命令即可运行
/opt/conda/envs/bevcalib310/bin/python run_bag_calibration.py \
  --config configs/calibration_default.yaml
```

配置文件示例 (`configs/calibration_default.yaml`)：

```yaml
# --- Input ---
input_file: /path/to/trips.txt        # 行程列表文件
input_format: trips                     # auto | trips | bag_list
trips_base: /path/to/trip_root          # 行程根目录（trips模式必填）

# --- Model ---
ckpt_path: /path/to/ckpt_400.pth       # 模型 checkpoint

# --- Output ---
output_dir: /path/to/calibration_outputs

# --- Calibration Parameters ---
max_frames: 400                         # 聚合帧数上限
min_agg_frames: 50                      # 最低聚合帧数（低于此值标记低置信度）
camera_name: traffic_2                  # 标定相机名称
batch_size: 24                          # 推理 batch size
img_h: 360                              # 推理图像高度
img_w: 640                              # 推理图像宽度
projection_ratio: 0.1                   # 投影图生成比例（0.1=10%帧数）

# --- Speed & Acceleration Filters ---
min_speed_kmh: 5.0                      # 低于此速度跳过
max_speed_kmh: 120.0                    # 高于此速度跳过
min_accel: -1.0                         # 加速度下限 (m/s^2)
max_accel: 2.0                          # 加速度上限 (m/s^2)
min_brightness: 25                      # 最低亮度

# --- Performance ---
parallel: 0                             # 0=顺序执行, -1=自动多GPU, N=指定GPU数
preparer_num_workers: 16                # 数据提取并行线程数
extract_buffer_multiplier: 1.5          # 提取缓冲倍率
```

CLI 参数可以覆盖 YAML 配置（CLI 优先级更高）：

```bash
# 用不同的 output_dir 覆盖 YAML 配置
/opt/conda/envs/bevcalib310/bin/python run_bag_calibration.py \
  --config configs/calibration_default.yaml \
  --output_dir /tmp/test_output
```

#### 方式2：纯命令行参数

```bash
/opt/conda/envs/bevcalib310/bin/python run_bag_calibration.py \
  --input_file /path/to/trips.txt \
  --input_format trips \
  --trips_base /path/to/trip_root \
  --ckpt_path /path/to/ckpt_400.pth \
  --output_dir /path/to/calibration_outputs \
  --max_frames 400 \
  --parallel 0
```

#### 输入格式

| 参数 | 说明 |
| --- | --- |
| `input_format: trips` | `input_file` 为行程名称列表，每行一个行程名。`trips_base` 指向行程根目录，程序自动拼接 `trips_base/<trip_name>/` 查找 bags/ 和 configs/ |
| `input_format: bag_list` | `input_file` 为 bag 文件路径列表（一行一个）。需额外指定 `config_dir` 指向 cameras.cfg 和 lidars.cfg 所在目录 |
| `input_format: auto` | 自动识别：若文件内含 `.bag` 路径则按 bag_list，否则按 trips |
| `input_format: remote` | **流式远程标定**。`input_file` 为行程名称列表，`trips_base` 为本地缓存目录。程序自动通过 `drfile` 下载行程数据并边下载边标定 |

#### `trips_base` 参数说明

`trips_base` 是行程数据根目录，用于 `trips` 模式定位各行程完整路径。例如：

```
trips_base/
├── YR-P177-95_20260412_061717/
│   ├── bags/
│   │   └── important/
│   │       ├── bag1.bag
│   │       └── bag2.bag
│   └── configs/
│       ├── cameras.cfg
│       └── lidars.cfg
├── YR-P01T-3_20260311_005058/
│   ├── bags/
│   └── configs/
```

#### 输出目录结构

```
output_dir/
├── SUMMARY_REPORT.md              # 所有行程汇总报告
├── <trip_name>/
│   ├── calibration_report.md      # 单行程详细标定报告
│   ├── calibration.log            # 完整标定日志
│   ├── calibrated_extrinsic.txt   # 标定后 LiDAR→Camera 外参 (4x4)
│   ├── original_extrinsic.txt     # 原始 LiDAR→Camera 外参 (4x4)
│   ├── configs/                   # 行程原始配置文件副本
│   │   ├── cameras.cfg
│   │   ├── lidars.cfg
│   │   └── lidars_calibrated.cfg  # 标定后外参写入的 lidars.cfg（sensing系）
│   └── projections/               # 点云投影对比图（原始 vs 标定后）
│       ├── frame_000000_comparison.png
│       └── ...
```

#### 流式远程标定（Remote Streaming Mode）

无需预先下载行程数据到本地，程序自动通过 `drfile` 工具边下载边标定：

```bash
# 方式1：YAML 配置文件
/opt/conda/envs/bevcalib310/bin/python run_bag_calibration.py \
  --config configs/calibration_remote.yaml

# 方式2：命令行参数
/opt/conda/envs/bevcalib310/bin/python run_bag_calibration.py \
  --input_file remote_trips.txt \
  --input_format remote \
  --trips_base /path/to/cache_dir \
  --ckpt_path /path/to/ckpt_400.pth \
  --output_dir /path/to/output
```

`remote_trips.txt` 内容为行程名列表（每行一个）：

```
YR-C01-81_20260427_112840
YR-P177-95_20260412_061717
```

**工作原理**（参考 C++ DPBag View 流式读取模式）：

1. 同步下载行程 `configs/` 和 `model/`（小文件，秒级完成）
2. 列出远程 Heavy/Medium/Light bag，按时间戳配对选择子集
3. 启动 8 个并发下载线程（后台）
4. 等待最少 4 组 bag 到位后立即开始数据提取（提取与下载并行）
5. 提取目标帧数达成后停止剩余下载（early termination）

**Bag 类型说明**：

| Bag 类型 | 内容 | 作用 |
| --- | --- | --- |
| Heavy_Topic_Group | 图像 + 点云 | 标定主数据（必需） |
| Medium_Topic_Group | 高频点云 | 补充点云数据 |
| Light_Topic_Group | `/localization/pose` | **点云去畸变所需的位姿数据**（缺失会跳过去畸变） |

下载完成的数据会缓存在 `trips_base/<trip_name>/` 中。再次运行时自动检测缓存完整性，若缺少 Light bags（pose 数据）会自动重新下载。

**自动功能**：

| 功能 | 说明 |
| --- | --- |
| **model 文件夹下载** | 自动从远程 `configs_onboard/sensors/model/` 下载工厂标定配置（`lidars.cfg` 等），用于 `install_angle_error` 计算。支持 `LINK` 类型目录的自动解析 |
| **install_angle_error 计算** | 标定完成后自动对比 BEVCalib 结果与工厂 model 外参的 RPY 差异，写入 `lidars_calibrated.cfg`。若 model 不可用，从 `configs/lidars.cfg` 的已有 IAE 反推工厂外参 |
| **图像分辨率自适应** | 自动检测 bag 中实际图像分辨率（如远程 bag 可能为 1920x1080）与 `cameras.cfg` 声明分辨率（如 3840x2160）不匹配的情况，自动按实际分辨率缩放内参并重建去畸变映射 |
| **点云去畸变** | 利用 Light bag 中的 `/localization/pose` 数据，对点云进行运动补偿（motion undistortion）。对齐 C++ `math_utils.cpp` 实现，将点云从 LiDAR 扫描时刻转换到图像曝光时刻，消除运动畸变。缺少 pose 数据时自动跳过 |
| **缓存完整性检测** | 再次运行已缓存的行程时，自动检测是否缺少 Light bags（pose 数据），缺失时自动重新下载补全 |

**远程模式输出目录结构**（相比标准模式多 `model/` 子目录）：

```
output_dir/<trip_name>/
├── configs/                     # 行程配置（含 lidars_calibrated.cfg）
├── model/                       # 工厂标定配置（若可用）
│   ├── lidars.cfg
│   └── cameras.cfg
├── projections/                 # 投影对比图
├── calibrated_extrinsic.txt
├── original_extrinsic.txt
├── calibration_report.md
└── calibration.log
```

#### GT 判定规则

程序会自动检查 `cameras.cfg` 中 traffic_2 和 `lidars.cfg` 中主 LiDAR 的 `install_angle_error` 字段。当所有 xyz 绝对值均 < 2.5 度时，判定初值为 GT，并计算多窗口标定精度误差（MEDW50, MEDW100, MEDW200, MEDW400）。

### 任务6: 生成泛化性能报告

```bash
# 使用 B26A YAML 配置
python run_generalization_eval.py --config configs/eval_generalization_b26a.yaml
# 使用 ALL YAML 配置
python run_generalization_eval.py --config configs/eval_generalization_all.yaml
# 命令行覆盖
python run_generalization_eval.py --config configs/eval_generalization_all.yaml --angle_range 10.0 --output_dir logs/custom_eval
# 测试集采样（与配置里 eval_params 一致，命令行优先）
python run_generalization_eval.py --config configs/eval_generalization_all.yaml \
  --eval_sample_step 2
python run_generalization_eval.py --config configs/eval_generalization_all.yaml \
  --parallel -1 --eval_max_frames_per_seq 400
```

YAML 中可在 `eval_params` 里设置采样与扰动范围（命令行 `--eval_sample_step` / `--eval_max_frames_per_seq` 会覆盖配置中的全局采样）：

```yaml
eval_params:
  angle_range: 5.0
  trans_range: 0.15
  eval_sample_step: 2          # 可选；与 eval_max_frames_per_seq 互斥
  # eval_max_frames_per_seq: 10000
```

多模型报告中 **`data_balance`** 写在每个模型条目下（`models`，不是 `eval_params`），例如：

```yaml
models:
  - label: "my-model"
    dir_name: "model_small_5deg_v1"
    data_balance: 1            # 该模型评估以 Macro 为主指标
    # ... 其余 bev_zbound_step、ckpt 等字段见 configs 内示例
```

### 任务7: pytorch模型转drinfer模型

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
# 彻底停止所有训练（含批量实验队列、重试、TensorBoard）
bash stop_training.sh --force

# 交互确认后停止
bash stop_training.sh

# 仅查看当前训练进程状态，不停止
bash stop_training.sh --status
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

**文档版本**: v2.5  
**最后更新**: 2026-05-19  
**适合**: 首次使用 BEVCalib 的用户
