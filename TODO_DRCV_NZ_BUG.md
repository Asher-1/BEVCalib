# BUG: drcv 后端 conv_out 缺少 Z stride 导致 image_branch 泛化有害

日期: 2026-04-24
状态: 待修复
优先级: 高（影响 drcv 模型的泛化性能）

## 问题描述

drcv 后端的 SparseEncoder.conv_out 使用 1x1x1 stride=1（仅做通道投影），
而 spconv 后端使用 3x3x3 stride=(1,1,2)（额外在 Z 维度降采样 2x）。

导致 drcv 的 PC 分支 BEV 特征 n_z=5，而 spconv 的 n_z=2。
在 to_bev_mode='concat' 下，drcv 的 ProjectionHead 输入维度为 640（5x128），
spconv 为 256（2x128）。

## 影响

drcv 的 PC 分支携带了 2.5x 的 Z 轴信息，使模型"自给自足"。
推理时 OOD camera 特征（不同车辆的 intrinsics/extrinsics）反而成为噪声，
导致 drcv 无增强模型的泛化精度恶化 20-24%。

消融实验数据:

| 模型 | 后端 | 增强 | Normal(°) | NO-IMG(°) | Camera 作用 |
|------|------|------|-----------|-----------|------------|
| v20-v8-z10-drcv | drcv | 无 | 1.169 | 0.889 | 有害 -24% |
| v20-v8-z10-pitch-wt3-drcv | drcv | 无 | 1.200 | 0.945 | 有害 -21% |
| v16-pitch3-signflip | drcv | 少量 | 2.062 | 1.866 | 有害 -10% |
| v16-ultimate | drcv | 全量 | 0.913 | 1.151 | 有益 +26% |
| v20-v8recipe-z10 | spconv | 无 | 0.736 | 1.248 | 有益 +70% |
| v20-v8recipe-pitch-wt3 | spconv | 无 | 0.646 | 0.818 | 有益 +27% |

## 涉及代码

- `kitti-bev-calib/pc_branch/pc_encoders.py`
  - SparseEncoder.__init__ (line ~259): drcv conv_out 的 kernel/stride 定义
  - SparseEncoder._compute_output_spatial: n_z 计算
  - SpconvToDenseBEV.__init__: concat 模式下 out_channels = n_z * in_channels

- `kitti-bev-calib/bev_calib.py`
  - ConvFuser: 融合 cam_bev_feats 和 pc_bev_feats

## 修复方案

### 方案 A（零代码，推荐当前）
继续使用 spconv 后端（n_z=2），当前最优模型 v20-v8recipe-pitch-wt3 (0.646°)

### 方案 B（推理时修复）
添加 DRCV_ALIGN_NZ 环境变量，在 SpconvToDenseBEV.forward 中通过
z_idx = z_idx * target_n_z // raw_n_z 将 5 层 Z 合并为 2 层。
仅影响推理，不改已有权重。

修改位置:
1. SpconvToDenseBEV.__init__: 添加 raw_n_z 参数
2. SpconvToDenseBEV.forward: 当 raw_n_z > n_z 时做 Z 索引重映射
3. SparseEncoder.__init__: 读取 DRCV_ALIGN_NZ, 计算 spconv 等效 n_z
4. run_generalization_eval.py: 透传 drcv_align_nz 配置

### 方案 C（训练时修复，需重训）
修改 drcv conv_out 为 kernel_size=3, stride=(1,1,2)（需确认 drcv Conv3d 是否支持 tuple stride）
或添加 Z-only pooling 层。从训练阶段就对齐 n_z=2。

### 方案 D（DST-Calib Difference Map）
替换原始图像特征为 Difference Map，从根本上解决 OOD camera 特征问题。

## 消融实验配置
configs/eval_generalization_ablation_no_image.yaml (已创建)
评估结果: logs/evaluations/ablation_no_image/
