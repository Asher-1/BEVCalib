# V30 实验计划：BEVCalib 终极泛化版本

## 1. 目标

V30 是面向量产落地的**最终优化版本**。目标在 V29-G3 (BEST=0.085°) 基础上，通过 DINOv2-Base backbone 升级进一步提升泛化精度至 **BEST < 0.07°**。

## 2. 历史最优结果回顾

| 版本 | 模型 | Backbone | 解冻 | Jitter | DANN | BEST Rot | 聚合方法 |
|------|------|----------|------|--------|------|---------|---------|
| V27 | E10 | Swin | N/A | strong (0.7/3.0) | 0.15 | 0.321° | SVDW400 |
| V28 | F5 | DINOv2-small (frozen) | frozen | light (0.15/1.5) | - | 0.296° | MEDW400 |
| V29 | G3 | DINOv2-small | last 2 | strong (0.7/3.0) | - | **0.085°** | MEDW400 |
| V29 | G2 | DINOv2-small (frozen) | frozen | strong (0.7/3.0) | 0.15 | 0.089° | MEDW400 |
| V29 | G1 | DINOv2-small (frozen) | frozen | strong (0.7/3.0) | - | 0.091° | MEDW400 |

### 关键发现
- **强 mount jitter** (prob=0.7, sigma=3.0) 是核心：将预测误差从偏差变成零均值随机噪声
- **MEDW (axis-angle median)** 在零均值噪声下极为有效，400帧改善率 > 96%
- **DINOv2 frozen backbone** 天然提供域不变特征
- **Partial unfreeze (last 2 layers)** 允许任务微调同时保留域不变性
- **DANN** 在全冻结时有明确贡献 (G2 vs G1: 0.089° vs 0.091°)

## 3. V30 实验矩阵

### 3.1 完整实验设计 (9组)

实验按训练优先级排序：

| 优先级 | ID | Backbone | 解冻 | Jitter | DANN | 验证目标 |
|:------:|:--:|----------|:----:|:------:|:----:|---------|
| 1 | **H8** | dinov2-base | last 2 | strong | 0.15 | **全优势组合** |
| 2 | H1 | dinov2-base | last 2 | strong | - | G3 直接升级 |
| 3 | H9 | dinov2-base | last 4 | strong | 0.15 | 深解冻+DANN |
| 4 | H3 | dinov2-base | frozen | strong | 0.15 | 冻结+DANN |
| 5 | H4 | dinov2-base | last 4 | strong | - | 更深解冻 |
| 6 | H2 | dinov2-base | frozen | strong | - | 纯冻结基线 |
| 7 | H6 | dinov2-base | last 2 | strong | - | lr 消融 (0.005) |
| 8 | H5 | dinov2-base | last 2 | moderate | - | jitter 消融 |
| 9 | H7 | dinov2-small | last 2 | strong | - | G3 复现基线 |

### 3.2 因子分析矩阵

```
                  DANN=0           DANN=0.15
               ┌──────────────┬──────────────┐
frozen         │   H2 (基线)   │   H3 (对照)  │
               ├──────────────┼──────────────┤
unfreeze-2     │   H1 (G3升级) │  H8 (最优候选)│
               ├──────────────┼──────────────┤
unfreeze-4     │   H4 (深解冻) │   H9 (深+DANN)│
               └──────────────┴──────────────┘

消融实验:
  H5: H1 + moderate jitter → jitter 强度敏感性
  H6: H1 + lower lr (0.005) → lr 敏感性
  H7: dinov2-small 基线 → backbone 升级收益
```

### 3.3 Backbone 参数对比

| 属性 | dinov2-small (V29) | dinov2-base (V30) |
|------|-------------------|-------------------|
| 总参数量 | 26.9M | 93.3M (3.5x) |
| Transformer Blocks | 12 | 12 |
| Embed Dim | 384 | 768 (2x) |
| FPN 输出 | 256x12x20 | 256x12x20 (一致) |
| Partial unfreeze (last 2) 可训练参数 | 4.2M | 20.9M (5x) |
| 推理显存 (batch=16) | ~1.5 GB | ~3.1 GB |
| 预训练权重 | 88 MB | 346 MB |

## 4. 训练配置

### 4.1 Quick 训练 (单机 8 GPU)
- 配置文件: `configs/batch8_train_all_v30_quick.yaml`
- 用途: 快速筛选最优配置
- 命令: `bash batch_train.sh configs/batch8_train_all_v30_quick.yaml`

### 4.2 Full 训练 (32 节点集群)
- 配置文件: `configs/batch8_train_all_v30.yaml`
- 用途: 量产级完整训练
- 命令: `bash batch_train.sh configs/batch8_train_all_v30.yaml`

### 4.3 共通参数
- Epochs: 400 (early stopping patience=30)
- Batch size: 16 per GPU
- LR: 1e-4 (StepLR, step_size=80)
- Loss: axis loss (weight_axis_rotation=0.5, balanced)
- 数据增强: truncated_normal perturbation, color jitter, PC jitter/dropout
- Evaluation: every 50 epochs

## 5. 泛化评估

### 5.1 评估配置
- 配置文件: `configs/eval_generalization_v30.yaml`
- 测试数据: test_data_v2 (4800 samples, 12 sequences, 每序列 400 帧)
- 扰动范围: ±5.0° rotation, ±0.15m translation
- 对照基线: V29-G3, V29-G1

### 5.2 评估指标
- Per-frame Rotation Error (Roll/Pitch/Yaw)
- 时序聚合: SVDW{50,200,400}, MEDW{50,200,400}, TRMW{50,200,400}
- **BEST**: 选择最低 Rot 的 (方法, 窗口) 组合
- Per-sequence 分析: 12 序列独立评估

### 5.3 评估命令
```bash
python run_generalization_eval.py \
  --config configs/eval_generalization_v30.yaml \
  --parallel -1
```

## 6. 部署方案

### 6.1 推理接口
```python
from utils.bevcalib_inference import BEVCalibInference, TemporalCalibrationAggregator

model = load_checkpoint("v30_H8_best/ckpt_best_val.pth")
wrapper = BEVCalibInference(model)
agg = TemporalCalibrationAggregator(min_frames=50, max_frames=400)

for frame in stream:
    pred_T = wrapper(frame.img, frame.pc, frame.init_T, eye4, frame.K)
    agg.add(pred_T)
    if agg.ready:
        calibration = agg.aggregate()
        confidence = agg.get_confidence()
```

### 6.2 聚合策略
- 默认方法: `axis_angle_median` (MEDW)
- 窗口大小: 400 帧 (最优)
- 最小帧数: 50 帧 (可提前输出初步结果)

### 6.3 部署指标目标
| 指标 | 目标 |
|------|------|
| BEST Rot (400帧) | < 0.07° |
| Roll / Pitch / Yaw 各轴 | < 0.03° |
| Per-sequence Std | < 0.04° |
| 50帧初步 Rot | < 0.30° |

## 7. 预期结果与决策流程

### 7.1 Quick 训练后的分析
1. 运行泛化评估 → 生成 GENERALIZATION_REPORT.md
2. 对比 H8 vs H1 → DANN 边际贡献
3. 对比 H1 vs H7 → base vs small 升级收益
4. 对比 H8 vs H9 → 解冻深度最优点
5. 对比 H1 vs H5 → jitter 敏感性
6. 对比 H1 vs H6 → lr 敏感性

### 7.2 决策树
```
If H8 BEST < 0.07° → H8 为量产配置, 启动 Full 训练
If H8 ≈ H1 → DANN 无贡献, H1 为量产配置 (更简单)
If H1 ≈ H7 → base 无优势, 保持 V29-G3
If all < 0.1° → 选最稳定 (最低 std) 的配置
```

### 7.3 Full 训练
仅对 Quick 阶段确认的最优 1-2 个配置进行 Full 训练，不需要全部 9 组。
