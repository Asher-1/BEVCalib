# V70.2 论文方案与验收协议确认

## 结论

V70.1 没有达到跨未知车辆 `<0.1 deg`，主要不是 S4R/S4Z 权重不足，而是验收泄漏、门控指标方向错误、旋转语义不一致，以及论文架构被 CF-BEV-R 路径替代。S2、S4R、S4Z 的最佳 checkpoint 都靠近 epoch 1，说明后续课程只继续拟合同 rig 偏置，没有增加 OOD rig 的可观测校正能力。

## 已确认的根因

1. 原 train/val 在每个 sequence 内按帧切分。同一安装外参同时出现在两侧，supervised validation 测到的是同 rig 插值，不是未知 rig 泛化。
2. `PredIndep <= max` 会奖励输出不随输入注入变化的绝对外参捷径，和“能根据观测纠正安装误差”的目标相反。
3. fixed inject 使用 `R_init = R_gt @ dR`，而旧 signed error/Jacobian 使用 `R_out @ R_gt^T`。注入在 LiDAR 轴、测量在相机轴，不同 rig 会得到不同斜率。
4. 旧 coarse head 监督绝对旋转，却在组合时再次乘 `R_init`；同时 coarse-corrected `uv_feat_c` 没有送入 LocalCorr/RoCR。
5. 旧 V60 路径没有实现论文的显式 BEV Alignment，而是继续使用现有投影/相关性或 SECOND/Transformer 精配准。
6. mount randomization 被关闭，部署端也没有可观测 ZD/TTA，仅靠 DANN、GIN 和同 rig 标签无法识别绝对安装偏置。

## 唯一旋转协议

- 定义：`per_axis`。
- `2 deg` 表示 LiDAR roll、pitch、yaw 各 `2 deg`，不是总旋转 `2 deg`。
- 组合：`R_init = R_gt @ Rz(yaw) @ Ry(pitch) @ Rx(roll)`。
- `[2, 2, 2] deg` 的总 geodesic 约为 `3.4437 deg`。
- 训练随机扰动、mini-eval、Jacobian/gdiag 和 OOD gate 全部使用右乘 LiDAR 轴误差 `R_gt^T @ R_pred`。
- `rotation_target_definition=total` 只保留给历史复现实验，V70.2 禁止使用。

## 数据与 checkpoint 合同

- 默认 `val_split_mode=group`，完整 sequence 只能属于 train 或 val 一侧。
- V70.2 固定 holdout：`03,07,11,20,22`；启动时打印两侧 sequence，并对交集直接报错。
- `inject_recovery_eval_batches=0`、`jacobian_eval_batches=0` 表示跑完整 OOD validation，不是关闭。
- Recovery checkpoint 必须同时满足：
  - Genuine Recovery `>= 95%`，以 `+2/-2 deg` 去除 ZD 后的对称残差计算；
  - 三轴最差 signed correction slope `>= 0.80`；
  - OOD zero-drift 最大轴 `<= 0.10 deg`。
- dual checkpoint 还必须满足 OOD MEDW 和 Jacobian 阈值。PredIndep 不再进入任何 checkpoint 决策。

## 论文实现对应

- Implicit Alignment：ResNet-50 输出 H/16、W/16 图像特征，`d_model=128`；加入 LID 3D 与 sinusoidal position encoding；下采样点特征作为 query，图像特征和 registry token 作为 key/value；T1 的 3 层相似度矩阵与 FOV 均直接监督。
- 式 (5)：原始点特征与 T1 输出拼接，经两组双层 MLP 和 max pooling 融合。
- 式 (6)：512 维、8 heads、3 层 T2 单查询聚合，粗旋转头预测每轴 sin/cos，并按 `Rz Ry Rx` 解码绝对 `T_coarse`。
- 式 (7)、(12)：RGB lift 到相机中心 EDN 体素；LiDAR 坐标由绝对 `T_coarse` 变换到同一体素空间。
- 式 (13)：两路 volume 固定为 `200 x 8 x 200`、水平 `+/-25m`、垂直 `+/-5m`、`C=128`；高度维与通道维展开后的 BEV 直接拼接，经 `3x3 Conv + InstanceNorm + ReLU` 融合。
- BEV encoder：独立 ResNet-18，C5 经 `7x7 Conv -> 512`，再由双层 MLP sin/cos 头产生 `T_fine`。
- 式 (14)：只做一次 `T_final = T_fine @ T_coarse`。coarse 是绝对量，fine 是左乘残差，两者不再混用。

## 训练边界

V70.2 是单阶段端到端训练，不加载 V70.1/S4 checkpoint，不继续调 S4R/S4Z 权重。未知车辆目标通过 `augment_mount_jitter_prob=0.7` 和 OOD 可观测门控约束。mount jitter 同步应用 `T_aug^{-1} @ T_original` 到点云，严格保持 `T_aug @ P_aug = T_original @ P`，不会把随机外参当成错误标签。未通过三项门控的 checkpoint 不得声明 `<0.1 deg` 达标。

每个 sequence 均匀采样最多 500 帧。优化器按论文使用统一 `Adam, lr=1e-4, wd=0`。约 140 optimizer steps/epoch，因此正式配置使用 1860 epoch 对齐论文约 260k iterations，并每 365 epoch（约 51k iterations）将 LR 衰减 0.5。

## 2026-07-21 收敛故障确认

旧 run 在 epoch 61 后进入平台：OOD validation 总旋转约 1.2 deg，MEDW 最差轴约 0.5--0.8 deg；到 epoch 156 仍为 `R/P/Y=0.649/0.315/0.282 deg`，不是接近 0.1 deg 的正常收敛轨迹。

根因是旧 paper forward 虽生成 `T_init = T_gt @ delta`，却没有把 `delta` 施加到输入点云，且完全忽略 `T_init` 的旋转。同一帧在所有注入下具有完全相同的网络输入，模型只能预测 rig 的平均绝对外参。此时 Jacobian 接近 1 是 `correction = init_error - constant_output_error` 的代数结果，不是观测驱动恢复。

修复后的接口显式构造 `P_mis = delta @ P` 和 `T_target = T_gt @ inv(delta)`，满足 `T_target @ P_mis = T_gt @ P`；论文网络预测 `T_target`，最后以 `T_pred_abs = T_pred @ delta` 恢复仓库的绝对外参输出。由此 +inject、-inject 和 ZD 会产生不同观测，signed slope 才是可辨识指标。旧 checkpoint 同时缺少新增的 C5/FPN 参数并使用旧输入语义，禁止 resume，必须从头训练。

## 已知边界与后续判据

- C1 保留 960x540 输入以避免把 98 度鱼眼图像强制裁成 KITTI 的 160x512；这属于数据域适配，不是论文 KITTI 预处理的逐像素复现。
- 对 C1 每个 sequence 抽取 1 帧统计，论文固定 `+/-25m`、`+/-5m` volume 的点覆盖率平均 64.1%，最低 35.8%。当前仍有足够点数启动试验，但必须记录 `point_volume` 有效率；如果 OOD sequence 明显受此限制，应做“固定论文范围”和“扩大 C1 范围”的单变量对照。
- 500 帧配置约有 9000 train frames；8 GPU x batch 8 时约 140 optimizer iterations/epoch。必须按 iteration 数判断训练量，不能再用旧 200 epoch/28k iteration run 宣称论文方案已经收敛。
- 当前任务按需求是 rotation-only；论文原模型的 translation head 和完整 6-DoF benchmark 未在本配置中启用。

配置文件：`configs/c1_retrain/c1_v70.2_paper_explicit_bev_group_holdout.yaml`。

OOD 评估配置：`configs/c1_retrain/eval_generalization_c1_v70.2.yaml`。该配置要求 `generalization_diagnostics.json` 包含 v70.2 `acceptance_gate`，并以 Genuine Recovery、三轴最差 signed slope、ZD 联合判定 PASS/FAIL。

启动命令：

```bash
bash batch_train.sh configs/c1_retrain/c1_v70.2_paper_explicit_bev_group_holdout.yaml
```
