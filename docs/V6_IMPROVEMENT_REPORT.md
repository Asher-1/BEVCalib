BEVCalib V6 改进方案报告：突破训练精度瓶颈
================================================================================

报告日期: 2026-03-27
基于: v1-v5 全版本训练实验数据 + 原论文 BEVCALIB 对比分析
改进版本: V6 (MLP回归头 + Geodesic Loss + BEV高分辨率)

================================================================================
一、问题定义：训练精度瓶颈
================================================================================

1.1 现象描述

在 ALL 数据集 (78409帧, 8个序列) 的 v5 训练中 (model_small_5deg_v5_undist_step_z5_aug)，模型精度在 epoch 240 后完全停滞：

| 训练阶段 | Epoch 范围 | Train Rot | Best Val Rot | 特征 |
| --- | ---: | ---: | ---: | --- |
| 快速收敛 | 1→80 | 108°→3.1° | 112.9°→1.69° | 指数级下降 |
| 缓慢改善 | 80→240 | 3.1°→0.84° | 1.69°→0.35° | 线性缓降 |
| 完全停滞 | 240→400 | 0.62-0.77° | 0.28° (最终) | 振荡, 无进步 |

Train loss (total_loss) 也同步平坦, 说明不是过拟合, 而是模型能力达到上限。

1.2 多版本泛化性能汇总

B26A 训练模型 (含去畸变) 泛化评估:

数据来源: generalization_eval_b26a_undist_v5/GENERALIZATION_REPORT.md
测试数据: test_data (9852 samples, 5 sequences (00-04), ±5.0°/±0.15m 扰动)

| 排名 | 模型 | 训练Rot(deg) | 泛化Rot(deg) | 泛化衰退 | Median | P95 | Max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | undist-best-5deg | 0.61 | 1.047 | 1.7x | 0.857 | 2.790 | 4.294 |
| 2 | undist-v1cfg-5deg | 0.61 | 1.062 | 1.7x | 0.873 | 2.851 | 4.132 |
| 3 | v5-s42-original | 0.76 | 1.117 | 1.5x | 0.891 | 3.048 | 5.029 |
| 4 | v1-champion | 0.59 | 1.229 | 2.1x | 1.038 | 3.140 | 4.784 |
| 5 | undist-refine-v1champ | 0.31 | 1.391 | 4.5x | 1.222 | 3.227 | 4.061 |
| 6 | undist-angle20-eval5 | 1.28 | 1.443 | 1.1x | 1.258 | 3.235 | 3.688 |
| 7 | undist-angle10-eval5 | 0.94 | 1.542 | 1.6x | 1.358 | 3.262 | 3.834 |

注: "训练Rot"为最终 epoch 训练误差, 泛化衰退 = 泛化Rot / 训练Rot

ALL 训练模型泛化评估:

数据来源: generalization_eval_all_v5/GENERALIZATION_REPORT.md
测试数据: test_data (9852 samples, 5 sequences (00-04), ±5.0°/±0.15m 扰动)

| 排名 | 模型 | 训练Rot(deg) | 泛化Rot(deg) | 泛化衰退 | Median | P95 | Max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | v4-z5-no-tail | 0.53 | 0.633 | 1.2x | 0.528 | 1.502 | 2.502 |
| 2 | v5-cosinewr-noaug | 0.58 | 0.927 | 1.6x | 0.912 | 1.638 | 2.621 |
| 3 | v5-cosinewr-aug | 0.72 | 1.050 | 1.5x | 1.044 | 1.812 | 2.607 |
| 4 | v5-step-noaug | 0.57 | 1.145 | 2.0x | 1.103 | 2.047 | 3.026 |
| 5 | v5-step-aug | 0.58 | 1.231 | 2.1x | 1.196 | 2.253 | 3.519 |
| 6 | v1-champion | 0.59 | 1.233 | 2.1x | 1.043 | 3.146 | 4.550 |

================================================================================
二、根因分析：五大瓶颈因素
================================================================================

2.1 [核心瓶颈] BEV 空间分辨率不足

当前配置: X=[0, 200m], Y=[-100, 100m], 步长=2m → 100x100 BEV grid

LiDAR 点云距离分布实测 (B26A 数据):

| 距离段 | 点数占比 | 每帧平均点数 | BEV体素角分辨率(2m) | 有效信号 |
| --- | ---: | ---: | ---: | --- |
| 0-20m | 35.5% | 33490 | 5.7-11.3° | 高 (密集) |
| 20-40m | 36.1% | 34133 | 1.9-3.8° | 高 |
| 40-60m | 17.9% | 16876 | 1.1-2.3° | 高 |
| 60-80m | 4.8% | 4505 | 0.8-1.6° | 中 |
| 80-100m | 3.7% | 3480 | 0.6-1.3° | 中 |
| 100-150m | 1.4% | 1315 | 0.5-0.9° | 低 (稀疏) |
| 150-200m | 0.7% | 652 | 0.3-0.7° | 噪声级 |

关键发现:
- 97.9% 的有效点云集中在 0-100m 范围内
- 100-200m 仅 2.1% 的点云, 却消耗了 50% 的 BEV 体素
- 在远距离, 体素内仅 1-2 个点, 信号被 LiDAR 测距噪声淹没
- 2m 体素在 100m 处对应 arctan(2/100) = 1.15° 角度分辨率, 这是训练精度的理论下限

原论文对比:
- 论文原文: "for outdoor datasets, we extend the range to 90 meters" → 对应 XY = [-90m, 90m], 步长=2m
- 论文 Table 2 (KITTI) 旋转误差: ±1.5m/±20° → 0.08°; ±0.5m/±5° → 0.06°; ±0.25m/±10° → 0.06°; ±0.2m/±20° → 0.04°
- 论文 Table 4 (CALIBDB, 异构数据): Rot=2.5° (Roll=1.2°, Pitch=1.7°, Yaw=1.3°), Trans=38.0cm
- 我们在 ALL 数据集泛化 = 0.633° (v4-z5-no-tail 最佳), 在 B26A 泛化 = 1.047° (undist-best-5deg 最佳)
- 以上泛化数据均基于 test_data (9852 samples) 评估
- 我们的场景更接近 CALIBDB (多车型、多传感器), 精度已达到合理水平

2.2 [架构瓶颈] 回归头过于简单

原始代码:

    self.rotation_pred = nn.Linear(self.embed_dim, 4)  # 256维 → 4维四元数

整个模型的最终输出仅经过 1 个线性层。
对比论文: 论文原文 3.3节 "two separate multilayer perceptrons (MLPs) are used to predict translation and rotation, respectively", 说明原论文使用 MLP 而非单层 Linear。

我们的实现与论文不一致, 这是精度差距的重要来源。

2.3 [Loss 瓶颈] 四元数距离 Loss 的梯度问题

当前使用的 rotation_loss 基于四元数距离:
    loss = 2 * arctan2(||q_diff[1:3]||, |q_diff[0]|)

在 sub-degree 范围 (<1°):
- 四元数差异 < 0.01, 接近 float32 精度极限
- 需要经过 matrix→quaternion→distance 的转换链, 梯度链过长
- 点云重投影 loss 在近距离被 augment_pc_jitter=0.02m 的噪声淹没

对比论文: 论文 3.4节原文 "For rotation supervision, we adopt a geodesic loss [47] based on quaternion distance"，给出公式 L_ang = 2*arctan2(||q_delta[1:3]||_2, |q_delta[0]|)。论文的 rotation loss 实际与我们当前的 quaternion_distance 形式一致。但论文使用 lr=5e-5 (我们用 1e-4) 且使用 MLP 头，这两者组合使得 sub-degree 优化更有效。我们新增的 SO(3) geodesic loss 直接操作旋转矩阵, 梯度链更短。

2.4 [数据因素] 多场景混合 + 标定GT精度天花板

ALL 数据集包含 8 个序列、78409 帧:
- 序列 00-07, 帧数分别为: 1810, 12580, 11888, 13277, 3529, 11396, 6431, 17498
- 不同车辆型号、不同相机参数、不同安装位置
- 标定 GT 本身可能存在 0.1-0.3° 的系统误差
- 模型需要用单一网络适配所有场景

去畸变实验验证 (B26A → test_data 泛化评估):

数据来源: generalization_eval_b26a_undist_v5/GENERALIZATION_REPORT.md

| 条件 | 模型 | 泛化 Mean Rot |
| --- | --- | ---: |
| 去畸变 (D=0, pinhole) | undist-best-5deg | 1.047° |
| 原始 (有畸变参数) | v5-s42-original | 1.117° |
| 差异 | | -6.3% |

去畸变带来约 6% 的提升, 说明图像质量对精度有贡献。

2.5 [训练策略] DDP 大 batch + LR 调度

ALL 数据集训练使用 256 GPU (32节点 x 8卡), batch_size=8:
- 有效 batch size = 2048 (另一个实验使用 batch_size=16, 有效 batch size = 4096)
- 每 epoch 约 15 步 (78409/2048 ≈ 38 iters, 每 step 报告多个 iter), 梯度方差极低
- StepLR step_size=80 → 实际仅 ~600 步就开始衰减
- CosineWR 在 B26A 优于 StepLR, 但在 ALL 数据集上差异不大

LR 调度对比 (ALL v5 泛化):

数据来源: generalization_eval_all_v5/GENERALIZATION_REPORT.md

| LR调度 | 增强 | 训练Rot(deg) | 泛化Rot(deg) | 泛化衰退 |
| --- | --- | ---: | ---: | ---: |
| CosineWR | 无增强 | 0.58 | 0.927 | 1.6x |
| CosineWR | 强增强 | 0.72 | 1.050 | 1.5x |
| StepLR | 无增强 | 0.57 | 1.145 | 2.0x |
| StepLR | 强增强 | 0.58 | 1.231 | 2.1x |

发现: CosineWR 泛化更好; 强增强在大数据集上反而有害。

================================================================================
三、V6 改进方案
================================================================================

3.1 改进A: BEV 高分辨率 (聚焦有效范围)

方案: 缩小 BEV 范围到有效 LiDAR 覆盖区域, 保持相同 grid 大小

| 参数 | V5 (原始) | V6 (高分辨率) |
| --- | --- | --- |
| X 范围 | [0, 200m] | [0, 100m] |
| Y 范围 | [-100, 100m] | [-50, 50m] |
| XY 步长 | 2.0m | 1.0m |
| BEV Grid | 100x100 | 100x100 |
| 体素分辨率 | 2m/voxel | 1m/voxel |
| 100m 角分辨率 | 1.15° | 0.57° |
| 50m 角分辨率 | 2.29° | 1.15° |
| 点云覆盖率 | 100% | 97.9% |
| 稀疏体素 | 800x800x41 | 400x400x41 |
| Transformer计算量 | 不变 | 不变 |
| PC分支显存 | 减少 75% | (400x400 vs 800x800) |

实现方式: 通过环境变量配置, 无需修改模型代码:
    export BEV_XBOUND_MAX=100 BEV_YBOUND_MIN=-50 BEV_YBOUND_MAX=50 BEV_XY_STEP=1.0

3.2 改进B: 多层 MLP 回归头

原始 (单层): Linear(256, 4)

改进 (3层MLP):
    Linear(256, 128) → LayerNorm → GELU → Dropout(0.1)
    → Linear(128, 128) → LayerNorm → GELU → Dropout(0.1)
    → Linear(128, 4)

改进依据:
1. 论文原文使用 MLP ("two separate MLPs"), 我们之前用的是单层 Linear, 这是实现偏差
2. LayerNorm 稳定 sub-degree 范围的梯度幅度
3. GELU 相比 ReLU 在小值区间提供更平滑的非线性
4. Bottleneck 结构 (256→128→128→4) 避免过拟合

对参数量的影响:
    原始: 256*4 + 4 = 1028 参数 (旋转头)
    改进: 256*128 + 128 + 128*128 + 128 + 128*4 + 4 = 49796 参数
    增加 ~48K 参数 (占模型总参数量 <0.5%)

3.3 改进C: SO(3) Geodesic Loss

新增 GeodesicRotationLoss:
    loss = arccos((tr(R_pred^T @ R_gt) - 1) / 2)

优势对比:

| 特性 | 四元数距离 (原始, 论文同款) | SO(3) Geodesic Loss (V6新增) |
| --- | --- | --- |
| 数学含义 | 四元数空间测地距离 | SO(3)流形测地距离 |
| 公式 | 2*arctan2(norm(q_delta[1:3]), abs(q_delta[0])) | arccos((tr(R^T@R_gt) - 1) / 2) |
| 计算路径 | matrix→quat→distance | 直接 matrix 操作 |
| 梯度链长度 | 长 (经过 quat_from_matrix) | 短 (直接 bmm + acos) |
| sub-degree 梯度 | 接近数值精度极限 | 良好 (acos在0附近梯度稳定) |
| 与论文关系 | 论文原文使用此形式 | 等价但实现更高效 |

实现: 通过 --use_geodesic_loss 1 启用, 保持向后兼容。日志中同时输出两种 loss 用于对比。

================================================================================
四、与论文实验对比
================================================================================

4.1 论文关键实验结果

论文 Table 2 (KITTI, 与已发表结果对比, BEVCalib 行):

| 扰动范围 | 方法 | Trans(cm) | Rot(deg) | Roll | Pitch | Yaw |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ±1.5m, ±20° | BEVCalib | 2.4 | 0.08 | 0.0 | 0.1 | 0.0 |
| ±0.5m, ±5° | BEVCalib | 2.5 | 0.06 | 0.0 | 0.1 | 0.0 |
| ±0.25m, ±10° | BEVCalib | 1.8 | 0.06 | 0.0 | 0.1 | 0.0 |
| ±0.2m, ±20° | BEVCalib | 1.8 | 0.04 | 0.0 | 0.0 | 0.0 |

论文 Table 4 (CALIBDB, 开源可复现 baseline 对比):

| 方法 | Trans(cm) | Rot(deg) | Roll | Pitch | Yaw |
| --- | ---: | ---: | ---: | ---: | ---: |
| BEVCalib | 38.0 | 2.5 | 1.2 | 1.7 | 1.3 |
| CalibAnything | 86.2 | 3.3 | 2.3 | 2.1 | 1.0 |
| Koide3 | 96.8 | 16.5 | 5.3 | 10.2 | 11.9 |

注: 论文在 KITTI 上使用单GPU (RTX 6000 Ada), batch_size=16, 500 epochs, lr=5e-5, StepLR

4.2 我们的实验结果 vs 论文

| 场景 | 论文结果 | 我们的结果 | 差距分析 |
| --- | --- | --- | --- |
| KITTI 同分布 (±0.5m,±5°) | 0.06° (Table 2) | 0.28-0.29° (ALL best val) | 论文用[-90,90m]+500ep+单GPU+lr=5e-5 |
| KITTI 同分布 (±1.5m,±20°) | 0.08° (Table 2) | N/A (我们未测同分布) | 论文同分布测试, 不等同泛化 |
| CALIBDB (异构) | 2.5° (Table 4) | 1.047° (B26A undist-best→test_data 9852样本) | 我们更好 (B26A异构程度低于CALIBDB) |
| ALL→test_data 泛化 | N/A | 0.633° (v4-z5-no-tail) | 论文未测大规模混合数据泛化 |

重要说明: 论文 KITTI 结果是同分布验证 (训练和测试在同一 KITTI 数据集), 我们的泛化测试是跨数据集评估 (在 test_data 上), 两者并不直接可比。

4.3 架构差异分析

| 组件 | 论文配置 (原文3.4节) | 我们V5配置 | V6改进 |
| --- | --- | --- | --- |
| 回归头 | MLP (原文3.3节:"two separate MLPs") | Linear (单层) | MLP 3层 (已对齐论文) |
| Rotation Loss | Geodesic (quaternion-based, 原文3.4节) | Quaternion distance | Geodesic + 保留quat对比 |
| BEV范围 | [-90, 90m] (原文:"90 meters") | [0, 200m] | [0, 100m] (聚焦有效区) |
| BEV步长 | 2.0m | 2.0m | 1.0m (2倍分辨率) |
| Decoder | GGBD (feature selector + self-attention) | Deformable Attention | 保持不变 (不同设计) |
| FPN BEV Encoder | 有 | 可选 (bev_encoder=1) | 保持不变 |
| 优化器 | AdamW, lr=5e-5, weight_decay=1e-4 | AdamW, lr=1e-4 | 消融 1e-4 vs 5e-5 |
| Batch Size | 16 (单GPU: RTX 6000 Ada) | 8*256=2048 (DDP 32节点x8卡) | 保持不变 |
| Epochs | 500 | 400 | 保持不变 |
| LR Schedule | StepLR (gamma=0.5) | StepLR/CosineWR | 统一StepLR |
| Loss权重 | (LR, LT, LPC) = (1.0, 0.5, 0.5) | 同论文 | 保持不变 |

================================================================================
五、V6 消融实验设计
================================================================================

B26A 快速验证 (5组实验, 全部 StepLR, seed=42):

| 编号 | 实验名 | Loss | BEV范围 | 步长 | LR | 验证目标 |
| ---: | --- | --- | --- | ---: | --- | --- |
| 1 | v6_baseline | quat | [0,200] | 2m | 1e-4 | 对照组: 仅MLP头改进 |
| 2 | v6_geodesic | geodesic | [0,200] | 2m | 1e-4 | Geodesic loss效果 |
| 3 | v6_geo_lowlr | geodesic | [0,200] | 2m | 5e-5 | 低LR + geodesic |
| 4 | v6_hires | geodesic | [0,100] | 1m | 1e-4 | 核心实验: 高分辨率BEV |
| 5 | v6_hires_lowlr | geodesic | [0,100] | 1m | 5e-5 | 高分辨率 + 低LR |

预期效果:
- 实验1 vs V5: MLP头是否带来 val 精度提升 (预期 train rot 从 ~0.6° 降至 <0.5°)
- 实验2 vs 实验1: Geodesic loss 是否加速 sub-degree 收敛
- 实验4 vs 实验2: BEV 高分辨率是否打破精度地板 (预期 train rot 突破 0.4°)
- 实验5 vs 实验4: 论文使用 lr=5e-5 (原文:"initial learning rate of 5e-5"), 在高分辨率下是否更优

================================================================================
六、实验配置文件
================================================================================

配置路径: configs/batch8_train_b26a_v6_mlp_geodesic.yaml

已完成 dry-run 验证, 5组实验命令全部正确生成。

代码改动涉及文件:
- kitti-bev-calib/bev_calib.py: MLP回归头 + use_geodesic_loss参数传递
- kitti-bev-calib/losses/losses.py: 新增 GeodesicRotationLoss 类
- kitti-bev-calib/train_kitti.py: --use_geodesic_loss 参数 + 日志支持
- kitti-bev-calib/bev_settings.py: BEV_XY_STEP环境变量 + auto sparse_shape
- batch_train.sh / start_training.sh / train_universal.sh: 参数传递链

================================================================================
七、预期收益与风险
================================================================================

预期收益:

| 改进 | 影响范围 | 预期收益 | 信心度 |
| --- | --- | --- | --- |
| MLP回归头 | Train精度 | train rot 从 0.6-0.7° 降至 <0.5° | 高 (对齐论文实现) |
| Geodesic Loss | Sub-degree收敛 | 后期 loss 不再停滞 | 中-高 |
| BEV高分辨率 | 精度理论上限 | 角度分辨率从 1.15° 提升到 0.57° | 高 (物理原理) |
| 组合效果 | 整体 | 泛化误差从 1.05° (B26A) / 0.63° (ALL) 进一步降低 | 中 |

风险评估:

| 风险 | 可能性 | 影响 | 缓解措施 |
| --- | --- | --- | --- |
| 高分辨率BEV显存不足 | 低 | 训练失败 | sparse_shape缩小75%, 实际减少显存 |
| MLP头与旧ckpt不兼容 | 确定 | 无法refine旧模型 | 仅scratch训练, 不加载旧权重 |
| Geodesic loss梯度不稳定 | 低 | 训练发散 | eps=1e-7 clamp保护 |
| 100m范围丢失远距离信息 | 低 | 远距离精度降低 | 仅丢失2.1%稀疏点, 影响可忽略 |

================================================================================
八、总结
================================================================================

V6 改进的核心理念: 对齐论文实现 + 挖掘 BEV 物理极限

1. MLP 回归头: 修复了与论文的实现偏差 (论文用MLP, 我们用Linear)
2. Geodesic Loss: 提供更好的 sub-degree 优化梯度
3. BEV 高分辨率: 聚焦有效 LiDAR 范围 (0-100m), 体素分辨率翻倍

预期最终效果:
- B26A 数据集: best val 从 0.29° 提升至 <0.20°, 泛化从 1.05° (undist-best-5deg) 进一步降低
- ALL 数据集: train rot 突破 0.6° 瓶颈, best val 从 0.28° 降至 <0.20°
- 以上泛化数据均基于 test_data (9852 samples, 5 sequences) 评估

下一步: 启动 B26A 快速验证实验, 确认改进方向后再在 ALL 数据集上规模训练。

