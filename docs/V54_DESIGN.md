# V54 设计方案：TLC 启发的光度-几何联合标定（Photo-Geometric Joint Alignment）

**日期**: 2026-06-12  
**状态**: 设计稿  
**前置**: V53e（Partial GIN + MGDA）训练中；V53c test MEDW 0.209°；GenuineRec 仍 ~62%  
**参考**: [TLC-Calib](https://github.com/SNU-VGILab/TLC-Calib) — Targetless LiDAR-Camera Calibration with Neural Gaussian Splatting (RA-L 2026)

---

## 1. 动机：TLC 证明了什么，V53 还缺什么

### 1.1 TLC-Calib 核心机制（可借鉴部分）

| TLC 机制 | 实现 | 标定信号来源 |
|----------|------|-------------|
| **联合优化** | 3D Gaussian 场景 + `cam_rot/trans_delta` 同步优化 | 渲染图 vs GT 图的 **稠密光度一致** |
| **无目标标定** | 自然场景纹理，无需标定板 | L1 + SSIM photometric loss |
| **Pose Release** | `min_viewpoint_cycle=5` 后释放 `opt_pose` | 先建场景表征，再标定外参 |
| **Rig 约束** | `use_rig`：同 cam_id 共享一个 rig 变换 | 多帧联合约束单一外参 |
| **From-LiDAR 初始化** | `from_lidar`：相机位姿从 LiDAR 轨迹 + blueprint c2l 初始化 | 降低优化自由度 |
| **两阶段 Refine** | 主训练 30k iter → 冻结 pose，精修场景 10k iter | 先粗标定，再精修几何 |

TLC 的标定本质是：**当外参正确时，LiDAR 锚定的 3D 结构投影到相机平面应与真实图像光度一致**。这是比稀疏 `PC_reproj_loss`（3D 点自洽）更强的 **2D-3D 稠密耦合**。

### 1.2 BEVCalib V53 现状与缺口

| 已有 | 缺口（TLC 可补） |
|------|-----------------|
| `PC_reproj_loss`：3D 点范数约束 | **无稠密光度对齐**；shortcut 可绕过几何理解 |
| `corr_alignment_loss`：RoCR 峰值监督 | 仅在 patch 级，非全图 SSIM |
| `GeoConsistencyLoss`（V40 P0a，未接入 CF-BEV-R 主线） | pred_uv vs gt_uv 采样，**依赖 GT T**，非 TLC 式渲染一致 |
| `seq_consistency_loss`：帧间 Δq 平滑 | 非 **rig 级**外参一致性（同 bag 零扰动应相同） |
| V53c Partial GIN → MEDW 优秀 | GenuineRec ~62%，**大扰动矫正增益不足** |
| V53e MGDA 平衡多任务 | 仍在 **同损失空间** 移动，未引入新标定信号 |

**结论**：V54 不引入完整 3DGS（部署不可行），而是将 TLC 的 **稠密光度-几何联合标定** 蒸馏为 CF-BEV-R 可训练的前馈 loss + 训练调度。

---

## 2. V54 核心思想

> **在 V53c（Partial GIN）+ V53e（MGDA）基座上，增加 TLC 式 LiDAR Splat Photo Loss（LSP）与 Rig/Pose 训练调度，用稠密 2D-3D 光度一致补强 Jacobian/GenuineRec，同时保持单帧前馈推理。**

```
Stage-1/2 (不变):  Swin+DLA → CrossAttn → LocalCorr → F_corr

Stage-3 (V53c GIN + 可选 DP-Head):  CorrTransformerHead → Δq

Stage-4 (V54 新增，训练期 only):
  LSP:  LiDAR 点云软光栅 → 合成稀疏 RGB-D → L1+SSIM vs 原图
  RigC: 同 bag 零扰动帧 Δq 方差约束
  PRS:  Pose Release Scheduler（先 corr，后 pose）

Loss 组合 (MGDA 四任务):
  pose | zero_drift | inject_recovery | photo_splat (LSP)
```

与 TLC 对照：

| | TLC-Calib | V54 BEVCalib |
|--|-----------|--------------|
| 场景表示 | 完整 3D Gaussian Splatting | **稀疏 LiDAR 点软光栅**（轻量代理） |
| 标定方式 | 每场景迭代优化 ~30k steps | **前馈网络** + LSP 作为训练监督 |
| 光度 loss | 全图 render vs GT | **LiDAR 覆盖区域** patch L1+SSIM |
| 推理 | 需场景重建 | **单帧 ms 级**（与 V53 相同） |
| Rig | 显式 `use_rig` 优化 | **Rig Consistency Loss** 训练约束 |

---

## 3. 模块设计

### 3.1 LiDAR Splat Photo Loss（LSP）— 主模块

**灵感**：TLC `photometric_loss + ssim`；BEVCalib 已有 `compute_projection_v42` / `sample_image_at_pixels`。

```python
class LiDARSplatPhotoLoss(nn.Module):
    """TLC-inspired dense photo-geometric alignment (train-only).

    1. 用 T_pred 将 LiDAR 点投影到图像平面 (u, v, depth)
    2. 按 depth 软光栅（可微 splat kernel，σ=1.5px）合成稀疏 RGB + depth map
    3. 在有效覆盖 mask 上计算:
       L_photo = (1-λ) * L1(splat_rgb, img) + λ * (1 - SSIM(splat_rgb, img))
    4. 辅助: L_depth = Huber(splat_depth, z_pred) 归一化深度一致
    """
```

**关键设计选择**：

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `lsp_lambda_ssim` | 0.2 | 对齐 TLC `lambda_dssim` |
| `lsp_weight` | 0.15 → 0.35 (ramp 15ep) | 从小权重起步，避免压制 pose loss |
| `lsp_start_epoch` | 10 | 对齐 TLC Pose Release（先 corr 后 photo） |
| `lsp_max_points` | 4096 | 与 `max_pcd_points` 协调 |
| `lsp_patch_size` | 32 | 在 splat 有效区域取局部 patch 算 SSIM |
| `lsp_depth_sigma` | 0.5m | 软 depth ordering，避免硬 z-buffer 不可微 |

**与 `GeoConsistencyLoss` 的区别**：

- GeoConsistency：在 **同一图像** 上比较 `sample(img, uv_pred)` vs `sample(img, uv_gt)` → 教网络「正确像素对应」
- LSP：用 **T_pred 合成渲染图** 与 **原图** 比较 → 更接近 TLC「错误外参 → 渲染错位 → 高 loss」的因果链

**训练时 T 来源**：

- 主路径：`T_pred = network(init_T, features)`（梯度回传至 pose head）
- 辅助路径（可选）：对 `T_init` 加小扰动做 consistency，类似 TLC 多视角

### 3.2 Rig Consistency Loss（RigC）— 来自 TLC `use_rig`

车载 LiDAR-Camera 外参是 **整车级 rig**，同 sequence/bag 内应为常数。

```python
class RigConsistencyLoss(nn.Module):
    """同 sequence 内零扰动帧的 Δq_pred 应一致（TLC rig 约束的前馈版）。"""
    def forward(self, delta_q_list, seq_ids, is_zero_pert_mask):
        # 对每个 seq_id 分组，计算 Δq 方差或 pairwise geodesic distance
        L_rig = mean(Var(Δq_i | seq, zero_pert))
```

| 参数 | 建议值 |
|------|--------|
| `rig_consistency_weight` | 0.1 |
| `rig_consistency_start_epoch` | 5 |
| 数据要求 | dataloader 返回 `seq_id`；`zero_perturbation_prob=0.15` batch 子集 |

**预期收益**：压低 seq03 类 MEDW outlier（V53 门控 MEDW 0.215° 的主因之一）。

### 3.3 Pose Release Scheduler（PRS）— 来自 TLC `min_viewpoint_cycle` + `pose_scheduler`

TLC 先训练 Gaussian 场景 5 个完整 viewpoint cycle，再释放 pose 优化。V54 映射为 epoch 级调度：

| 阶段 | Epoch | corr_head LR | pose_head LR | LSP | 说明 |
|------|-------|-------------|-------------|-----|------|
| **Warmup-Corr** | 0–9 | 1.0× | **0.0×** | off | 仅训 backbone + RoCR corr |
| **Release-Pose** | 10–49 | 1.0× | 1.0× | ramp 0→0.35 | 释放 pose + 开启 LSP |
| **Joint** | 50–79 | 0.5× | 1.0× | 0.35 | MGDA 四任务全开 |

实现：在 `train_kitti.py` 对 `head_params` / `corr_head` 乘 `pose_release_factor(epoch)`。

### 3.4 Refine Stage（可选 v54b）— 来自 TLC `--refine`

| | Stage-1 (v54a) | Stage-2 (v54b refine) |
|--|----------------|----------------------|
| Epoch | 80 | +20 |
| Backbone | 训练 | **冻结** |
| `lsp_weight` | 0.35 | **0.5** |
| 扰动范围 | 0–5° progressive | **仅 0–2°** |
| `zero_drift` / `inject` | MGDA 平衡 | 加大 ZD 权重 |
| 目标 | 泛化 + Rec | MEDW ≤ 0.18° 冲刺 |

---

## 4. 损失函数与 MGDA 扩展

### 4.1 任务损失（V54 = V53e + LSP + RigC）

| Loss | 权重 | 启动 epoch | MGDA 任务 |
|------|------|-----------|-----------|
| L_pose + axis + PC_reproj | 1.0 | 0 | `pose` |
| L_zero_drift | ramp 0.1→0.3 | 5 | `zd` |
| L_inject_recovery | 0.25 | 5 | `inject` |
| **L_lsp** | ramp 0→0.35 | 10 | **`photo`** |
| **L_rig** | 0.1 | 5 | 并入 `zd` |
| L_jacobian | 0.10 | 10 | 并入 `inject` |
| L_route (若启用 DP-Head) | 0.1 | 0 | — |

### 4.2 MGDA 四任务（扩展 V53e 三任务）

```python
mgda_tasks = {
    'pose': L_pose_total,
    'zd': L_zero_drift + L_rig,
    'inject': L_inject + L_jacobian,
    'photo': L_lsp,          # V54 新增
}
# mgda_weighted_loss 仍在 corr_head 瓶颈上算 α，单次 backward
```

**注意**：`photo` 任务需等 `lsp_start_epoch` 后才有非零 loss；之前 MGDA 退化为 V53e 三任务。

---

## 5. 数据与评估

### 5.1 训练数据（不变）

继续 `all_training_data`；V54 仅需 dataloader 补充 `seq_id`（若尚未暴露）。

### 5.2 TLC 格式交叉验证（推荐）

使用已下载的 `data/TLC-Calib`（KITTI-360 / FAST-LIVO2）做 **外部基准**：

| 协议 | 指标 | 目的 |
|------|------|------|
| BEVCalib gdiag | MEDW, ZD, GenuineRec | 主验收（与 V53 可比） |
| TLC `metrics_pose.py` | ARE / ATE per camera | 与论文对标 |
| TLC `metrics_nvs.py` | PSNR / SSIM / LPIPS | 光度一致代理（V54 LSP 有效性） |

实现路径：新增 `tools/tlc_eval_adapter.py`，将 BEVCalib 单帧预测 rig 写入 TLC `cams_to_lidar.txt` 格式。

### 5.3 验收目标

#### Phase A — v54a smoke（15ep）

| 指标 | 阈值 |
|------|------|
| 训练稳定 | 无 NaN；LSP 开启后 loss 可下降 |
| LSP backward | < 2s/batch（8×L20） |
| ZD max(R,P,Y) | ≤ 0.12° |
| GenuineRec @2° | ≥ 65% |

#### Phase B — v54a full（80ep）

| 指标 | 阈值 | vs V53 |
|------|------|--------|
| MEDW400 | ≤ **0.18°** | V53 门控 0.215° |
| ZD max(R,P,Y) | ≤ 0.10° | V52 0.146° |
| GenuineRec @2° | ≥ **70%** | V52 62% |
| Jacobian overall | ≥ 0.85 | 维持 |
| 单 ckpt | 是 | — |

#### Phase C — v54b refine（+20ep）

| 指标 | 阈值 |
|------|------|
| MEDW400 | ≤ **0.16°** |
| GenuineRec @2° | ≥ **75%** |

---

## 6. 实验矩阵

| 实验 | version | ep | 结构 | pretrain | 说明 |
|------|---------|-----|------|----------|------|
| **v54a_lsp_smoke** | v54a_lsp_smoke | 15 | V53c GIN + LSP + RigC + PRS | v53c dual | 验证 LSP 可训、MGDA 四任务 |
| **v54a_lsp_full** | v54a_lsp_full | 80 | 同上 + MGDA 4-task | v53c dual | 主攻线 |
| v54a_nolsp_ablation | v54a_nolsp | 80 | 仅 RigC + PRS，无 LSP | v53c dual | 消融：调度 vs 光度 |
| v54b_refine | v54b_refine | 20 | Stage-2 refine | v54a best | MEDW 冲刺 |
| v54c_dphead_lsp | v54c_dphead_lsp | 80 | v54a + DP-Head | v53a dual | 若 Rec 仍不足 |

**并行策略**：

| 机器 | 实验 |
|------|------|
| 本机 | v54a_lsp_smoke → full |
| 远端 | v54a_nolsp 消融 |
| smoke 通过后 | v54b refine |

---

## 7. 实现清单

| 项 | 文件 | 优先级 |
|----|------|--------|
| `LiDARSplatPhotoLoss` | `kitti-bev-calib/losses/lidar_splat_photo_loss.py` | P0 |
| `RigConsistencyLoss` | `kitti-bev-calib/losses/rig_consistency_loss.py` | P0 |
| CF-BEV-R forward 集成 LSP | `cf_bev_r_calib.py` | P0 |
| PRS epoch 调度 | `train_kitti.py` | P0 |
| MGDA 四任务 `photo` | `train_kitti.py`, `utils/mgda.py` | P0 |
| CLI: `--lsp_*`, `--rig_*`, `--pose_release_epoch` | `train_kitti.py`, `start_training.sh` | P0 |
| dataloader `seq_id` | dataset 相关 | P1 |
| v54a smoke/full yaml | `configs/v54a_lsp_cf_bev_r.yaml` | P1 |
| TLC eval adapter | `tools/tlc_eval_adapter.py` | P2 |
| Refine stage 脚本 | `configs/v54b_refine_cf_bev_r.yaml` | P2 |

### 7.1 LSP 实现要点（避免踩坑）

1. **可微 splat**：用 Gaussian kernel 累加颜色，避免硬 z-buffer `argmin`
2. **有效 mask**：仅 LiDAR 覆盖 > 50 像素的 patch 参与 SSIM
3. **与 rotation_only 兼容**：LSP 仍用 GT translation 投影（同 PC_reproj）
4. **AMP**：LSP 在 FP32 路径计算（同 PC_reproj）
5. **算力**：LSP 增加 ~15–25% step 时间；`lsp_max_points=4096` 可控

### 7.2 开关隔离

```python
# cf_bev_r_calib.py
if getattr(args, 'use_lsp_loss', 0):
    loss += lsp_weight * L_lsp(T_pred, img, pc, K)
# 默认 use_lsp_loss=0 → 等价 V53e
```

---

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| LSP 与 pose loss 冲突（光度局部最优） | PRS 延迟启动；权重 ramp；MGDA 平衡 |
| LSP 算力过高 | 限制点数；patch SSIM；仅主相机 |
| 夜间/低纹理场景 LSP 弱 | `lsp_valid_ratio` 门控；低纹理时降权 |
| RigC 需 seq_id | 无 seq_id 时 RigC=0，退化为 V53e |
| GenuineRec 仍 < 75% | 启用 v54c DP-Head；或加大 inject + jacobian |
| 与 TLC 数字不可直接比 | TLC 是 per-scene 优化；BEVCalib 报 feedforward 单帧指标 |

---

## 9. 决策树

```
v54a smoke 通过（LSP 可训、无 OOM）
  ├─ full 后 MEDW ≤ 0.18° & Rec ≥ 70%
  │    └─ v54b refine 冲 0.16° / 75% Rec → 上车候选
  ├─ MEDW 达标但 Rec < 70%
  │    └─ v54c DP-Head + LSP
  └─ LSP 无提升（vs v54a_nolsp 消融）
       └─ 回退 V53e；LSP 归档为负结果

TLC 交叉 eval PSNR 高但 gdiag Rec 低
  └─ LSP 过拟合光度；降 lsp_weight / 加 jacobian 约束
```

---

## 10. 参考

- TLC-Calib: `code/TLC-Calib/train.py`（photometric + SSIM + pose release）
- TLC poses: `scene/poses.py`（SE3 delta 累积）
- BEVCalib GeoConsistency: `losses/geo_consistency_loss.py`（V40 原型，非主线）
- BEVCalib corr projection: `losses/corr_alignment_loss.py`
- V53 设计: `docs/V53_DESIGN.md`
- V53e 配置: `configs/v53e_gin_mgda_cf_bev_r.yaml`

---

## 附录 A：TLC → V54 概念映射速查

| TLC 概念 | V54 对应 |
|----------|----------|
| `photometric_loss` + `ssim` | `LiDARSplatPhotoLoss` (L1 + SSIM) |
| `opt_pose` + `pose_scheduler` | Pose Release Scheduler (epoch 10) |
| `min_viewpoint_cycle` | `pose_release_epoch=10` |
| `use_rig` | `RigConsistencyLoss` |
| `from_lidar` | 已有 `T_init` from LiDAR odometry |
| `adaptive_voxel` | 已有 `cf_n_groups` / `max_pcd_points` |
| 3D Gaussian scene | **不引入**；用 LiDAR 稀疏 splat 代理 |
| `refine` 二阶段 | v54b refine（冻结 backbone） |
| `metrics_nvs` PSNR | TLC 交叉验证指标 |
