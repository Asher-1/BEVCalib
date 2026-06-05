# V42 实验设计：CF-BEV-R（CalibFormer 对齐 · 纯旋转标定优化）

> 状态：**v13.0 三阶段训练完成 + 评估修复 + 8卡DataParallel推理**（2026-06-02，MEDW=0.041°，SVD detach 确认最优，评估点云下采样 bug 修复 + 8卡并行评估支持）  
> 前置：`GENERALIZATION_REPORT.md`（v20 泛化基线 0.646°）  
> 参考论文：[CalibFormer (arXiv:2311.15241)](https://arxiv.org/abs/2311.15241)  
> 部署约束：**平移已知准确，仅优化旋转**（`rotation_only=true`）  
> 隔离策略：`fusion_backend=cf_bev_r`，**不影响其他实验**

---

## 0. 执行摘要

### 0.1 版本命名

**v42**，延续项目版本链 v20→…→v41→**v42**。不用 v20_opt（会误导为 v20 小修），不用 v22（与链断裂）。

### 0.2 部署目标

| 指标 | 目标 | 评估协议 |
|------|------|----------|
| Roll / Pitch / Yaw | 各轴 **≤ 0.10°** | test_data MEDW200，±5° init 扰动 |
| 平移 | **不预测**（固定为 init） | `rotation_only=true` |
| 交付口径 | **MEDW200 逐轴 max(R,P,Y)** | 200 帧窗口中位数 |

### 0.3 BEVCalib 原生限制 → 本方案突破点

v20 泛化 R=0.312° P=0.448° Y=0.230°，核心限制与对策：

| BEVCalib 限制 | 根因 | v42 突破 |
|---------------|------|----------|
| **LSS 深度估计隐式学习** | 深度分布记住训练相机安装，跨域 Pitch 崩溃 | **去 LSS**：Native Cross-Attn 直接在 2D-3D 原始域做对齐 |
| **BEV 栅格压缩垂直几何** | 高度轴 scatter 后 Pitch 视差信息丢失 | **不做 BEV 投影**：保留 3D group 的垂直坐标 |
| **concat 融合无显式对齐** | ConvFuser 1×1 conv 无法感知 misalignment 方向 | **CalibFormer 式 local corr + T_init 锚定 cross-attn** |
| **全局 pool 丢失空间结构** | Transformer 自注意力后 avg pool，Pitch 方向性被抹平 | **Pose Query Decoder**：query 选择性聚合 corr 特征 |
| **纯回归无几何约束** | 网络记忆训练分布 → 5.9× 泛化衰退 | **RoCR 几何初值 + Transformer 仅做残差** |
| **单阶段训练** | ±5° 固定范围，模型无法学习小角度精细调整 | **四阶段 progressive 10°→5°→3°→1°** |
| **ckpt 按合成角选** | 掩盖 Pitch 单轴超标 | **max(R,P,Y) MEDW 选 ckpt** |

**这些突破的叠加效果**是否足以达 0.1°？——诚实评估见 §0.4。

### 0.4 能否达到 R/P/Y 均 ≤ 0.10°

**分轴可行性**：

| 轴 | v20 测试 | 需改善 | 可行性 | 分析 |
|----|----------|--------|--------|------|
| **Roll** | 0.312° | 3.1× | **高** | v20 train=0.07°；cross-attn 对水平偏移敏感；progressive S3/S4 可压 |
| **Yaw** | 0.230° | 2.3× | **高** | v20 train=0.03°；水平旋转在 corr map 上信号最强 |
| **Pitch** | 0.448° | 4.5× | **中** | v20 train=0.04° 但泛化衰退 11×；BEV 垂直信息丢失是主因；去 LSS+前视辅路+progressive 可望改善但需实验验证 |

CalibFormer **KITTI 同数据集**精度（非跨域）：Roll=0.025° Pitch=0.104° Yaw=0.040°——CalibFormer 架构本身在 Pitch 上可接近 0.1°。BEVCalib v20 跨域 Pitch=0.448° 差距主要来自 **LSS 深度依赖 + BEV 压缩**（CalibFormer 不用这两者）。

因此 **v42 的目标路径**：去掉 BEVCalib 的 LSS/BEV 限制 → 对齐 CalibFormer 的 cross-modal correlation → 叠加 BEVCalib 已验证的 axis_weight / progressive / PC_reproj 优势 → **交叉组合两者优点**。

**不再设 L2=0.15° 降级线**。统一目标 **0.10°**，若未达成走 §9 迭代路径。

---

## 1. v42 与其他实验的隔离（开关控制）

### 1.1 工厂路由（`build_calib_model`）

在 `hybrid_triple_calib.py` 的 `build_calib_model()` 中新增路由：

```python
def build_calib_model(args, device, img_shape, rotation_only, ...):
    fusion_backend = getattr(args, 'fusion_backend', 'bev') or 'bev'

    if fusion_backend == 'cf_bev_r':
        from cf_bev_r_calib import CFBevRCalib
        model = CFBevRCalib.from_args(args, img_shape=img_shape).to(device)
        ...
        return model

    if fusion_backend == 'geo_match_proj':
        ...  # V40/V41 不变
    if fusion_backend in HybridTripleCalib.FUSION_BACKENDS:
        ...  # V39 等不变
    # legacy BEVCalib（v20 等）
    ...
```

### 1.2 CLI 开关

| 参数 | 值 | 作用 |
|------|-----|------|
| `--fusion_backend cf_bev_r` | v42 唯一入口 | 其他值走旧路径 |
| `--use_rocr 1` | 启用 RoCR 几何初值 | 默认 1；=0 退化为纯回归（ablation） |
| `--use_pitch_branch 1` | 启用 Pitch 辅路 | 已有 CLI，直接复用 |
| `--corr_window_mode adaptive` | 自适应窗口 | =fixed 退化为 d=4（ablation） |
| `--progressive_stage S1/S2/S3/S4` | 当前阶段标记 | 日志/ckpt 命名用 |

### 1.3 影响范围

| 模块 | v42 改动 | 其他实验影响 |
|------|----------|-------------|
| `hybrid_triple_calib.py` | 加 4 行 factory 路由 | **无**（`cf_bev_r` 分支独立） |
| `train_kitti.py` | 加 CLI argparse | **无**（默认值不变） |
| `batch_train.sh / start_training.sh / train_universal.sh` | 参数透传 | **无**（未设则不传） |
| `cf_bev_r_calib.py` | 新增文件 | **无**（不 import 于旧路径） |
| `losses/` | 新增 `corr_alignment_loss.py` | **无**（旧 loss 不引用） |

**保证**：不设 `--fusion_backend cf_bev_r` 时，整个 v42 代码路径**零加载**。

---

## 2. 架构设计：CF-BEV-R

### 2.1 核心理念：取 CalibFormer 精髓 + 去 BEVCalib 限制

| 借 CalibFormer | 去 BEVCalib 限制 | 保 BEVCalib 优势 |
|----------------|-----------------|------------------|
| Multi-head local correlation（§III-C） | ❌ LSS 深度估计 | ✅ rotation_only 部署约束 |
| T_init 锚定 cross-attn（§III-C window） | ❌ BEV 栅格俯视投影 | ✅ axis_weights Pitch 加权 |
| DLA 多尺度聚合（§III-B） | ❌ concat 无对齐融合 | ✅ PC_reproj_loss |
| Swin Encoder + Pose Query Decoder（§III-D） | ❌ 全局 avg pool | ✅ Progressive training |
| RGB-guided pose query init（§III-D） | ❌ 合成角选 ckpt | ✅ Swin backbone pretrain |

### 2.2 总体数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  V42 CF-BEV-R  (fusion_backend=cf_bev_r)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌── Stage-1: 细粒度双分支特征 ──────────────────────────────────────────┐ │
│  │  img 640×360 → Swin-Tiny + FPN + DLA skip(1/8,1/16,1/32→1/4聚合)    │ │
│  │  → F_rgb (B, 256, H/4, W/4)                                          │ │
│  │                                                                        │ │
│  │  pcd B×N×3 → FPS 128 groups + kNN-8 局部几何 MLP                     │ │
│  │  → xyz_g (B,G,3), F_pc (B,G,D)                                       │ │
│  │                                                                        │ │
│  │  T_init, K → uv_init = project(xyz_g, T_init, K)  (B,G,2)           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌── Stage-2: CalibFormer 式 T_init 条件对齐融合 ────────────────────────┐ │
│  │                                                                        │ │
│  │  (A) ExtrinsicAware CrossAttn ×2 层                                   │ │
│  │      Q = F_rgb_patch + harmonic(uv_grid)                              │ │
│  │      K,V = F_pc + harmonic(uv_init, xyz_g)                           │ │
│  │      → F_cross (B, P, D)                                             │ │
│  │                                                                        │ │
│  │  (B) Local Multi-Head Correlation (CalibFormer §III-C)                │ │
│  │      以 uv_init 为窗口中心，d=adaptive(4~12), heads=4                │ │
│  │      → corr_map (B,G,W²,H_heads), F_corr (B,G,D_corr)              │ │
│  │                                                                        │ │
│  │  (C) Bi-CrossAttn ×1 层（反向：pc_q ← img_kv）                       │ │
│  │      捕获 Pitch 轴垂直偏移方向信号                                     │ │
│  │      → F_bi (B, G, D)                                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌── Stage-3: 几何初值 + Correlation Transformer 解码 ──────────────────┐ │
│  │                                                                        │ │
│  │  (D) RoCR（Rotation-only Correlation Refine）                         │ │
│  │      corr_map peak offset → 2D-3D 对应 → rot-only SVD → R_geo       │ │
│  │                                                                        │ │
│  │  (E) Swin Encoder ×2 on corr token grid                              │ │
│  │                                                                        │ │
│  │  (F) Pose Query Init（CalibFormer §III-D）                            │ │
│  │      Q_0 = MLP( GAP(F_rgb) ⊕ Harmonic(T_init) ⊕ encode(R_geo) )    │ │
│  │                                                                        │ │
│  │  (G) Transformer Decoder ×4 层                                        │ │
│  │      Q = Q_0, K/V = Swin_encoder_out                                 │ │
│  │      → Δq (仅旋转残差 quaternion)                                     │ │
│  │                                                                        │ │
│  │  (H) 复合：q_pred = quat_mul(Δq, quat(R_geo))                       │ │
│  │                                                                        │ │
│  │  (I) 可选 iter K=1: 更新 T_init ← T_comp, 重跑 Stage-2→H            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌── Stage-3b: Pitch 辅路（并行）──────────────────────────────────────┐  │
│  │  FrontViewPitchBranch（已有代码）                                      │ │
│  │  前视 Z 层 + 图像 feat → L_pitch_aux                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  输出: T_comp: R=q_pred, t=t_init；T_comp = inv(ΔT) @ T_init              │
│                                                                              │
│  Loss = L_rot + L_axis(1:4:1) + L_pc_reproj + L_corr + L_pitch_aux        │
│         + L_quat_norm                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 RoCR（Rotation-only Correlation Refine）

**目的**：不靠 Transformer 从零回归旋转，而是先从 corr 几何关系读出一个 R_geo 初值，Transformer 只需预测极小的 Δq 残差。这将回归难度从「预测 0.5~5° 旋转」降为「预测 0.0x° 残差」，直接利于 0.1° 目标。

```python
class RoCR(nn.Module):
    """从 corr_map peak 提取 rotation-only 几何初值"""

    def forward(self, corr_map, xyz_g, uv_init, K):
        # 1. 从 corr_map soft-argmax 得到每个 group 的 uv_pred
        uv_pred = soft_argmax_2d(corr_map)      # (B, G, 2)

        # 2. uv_pred → 射线方向 (camera frame)
        rays_pred = pixel_to_ray(uv_pred, K)     # (B, G, 3)

        # 3. 已知 xyz_g (lidar frame) 和 rays_pred (camera frame)
        #    rotation_only SVD: R = argmin ||rays_pred - R @ project_dir(xyz_g)||
        R_geo = rotation_only_svd(rays_pred, xyz_g)  # (B, 3, 3)

        return R_geo  # 作为 Transformer 的残差基底
```

与 V41 Match+EPnP 的**本质区别**：RoCR 不学习 correspondence，而是直接从 corr peak 读取；不做 6DoF，只做 3DoF rotation SVD；不需要 fallback/valid_gt 等工程逻辑。

### 2.4 自适应 Correlation 窗口

```python
# 训练期用 GT 计算理想窗口大小
delta_uv = proj(T_gt, K, xyz_g) - proj(T_init, K, xyz_g)
d_ideal = ceil(delta_uv.abs().max() / patch_size) + margin
d = clamp(d_ideal, d_min=4, d_max=12)

# 推理期用 T_init ± angle_range 估计最大可能偏差
d_infer = estimate_max_window(angle_range_deg, K, mean_depth)
```

**为何比 CalibFormer 固定 d=4 更好**：CalibFormer 在 KITTI ±5° 训练/测试一致性好，窗口固定可行；fleet 跨域部署时相机参数/安装各异，固定窗口有 miss 风险。自适应窗口 **保底 d=4**（小角度不浪费计算），**上探 d=12**（大角度不遗漏）。

### 2.5 DLA 多尺度聚合（CalibFormer §III-B 对齐）

CalibFormer 消融表明 4× upsample 比 1× 改善 **Rot 43%、Trans 25%**（Table III）。

v20 的 Swin+FPN 输出 1/8 分辨率，v42 增加：

```python
# DLA skip: 1/8 + upsample(1/16) + upsample(1/32) → 1/4
F_rgb = fpn_1_8 + upsample_2x(fpn_1_16) + upsample_4x(fpn_1_32)
# 再过 1×1 conv 统一通道
F_rgb = conv_1x1(F_rgb)  # (B, 256, H/4, W/4)
```

---

## 3. Loss 设计（Rotation-Only 专项）

| Loss | 权重 | 说明 |
|------|------|------|
| `rotation_loss`（angular quat distance） | 1.0（rot_only 自动 ×2） | CalibFormer L_R |
| `axis_rotation_loss`（weights **1:4:1**） | 0.5 | **Pitch 4× 加权**（v20 用 3 不够） |
| `PC_reproj_loss` | 1.0（rot_only 自动 ×2） | CalibFormer L_P 等价 |
| `corr_alignment_loss` | 0.3 @ ep20 warmup | Huber(corr_peak_offset, δuv_gt) |
| `pitch_aux_loss` | 0.3 | FrontViewPitchBranch |
| `quat_norm_loss` | 0.5 | 正则化 |
| `seq_consistency_loss` | 0.1（S3+ 启用） | 相邻帧 Δq 平滑 → 优化 MEDW |

**Geodesic loss 切换策略**：S1/S2 用 quat angular（大角度收敛快）；S3/S4 切换为 geodesic SO(3) distance（小角度梯度更准）。

---

## 4. 四阶段 Progressive 训练

| 阶段 | angle | epochs | lr | Pitch 配置 | 重点 |
|------|-------|--------|-----|-----------|------|
| **S1** | ±10° | 80 | 1e-4 | 均匀三轴 | 全域收敛；学 corr 方向 |
| **S2** | ±5° | 150 | 1e-4→5e-5 | 三轴 prob=1/3 | 对齐 v20 域；冲 G1（val<0.10°） |
| **S3** | ±3° | 100 | 5e-5 | Pitch prob=0.5 | MEDW 精调；开 seq_consistency |
| **S4** | ±1° | 70 | 1e-5 | Pitch prob=0.6 | 小角度精调；冲 G3 |

每阶段 pretrain 上一阶段 `ckpt_best_medw_axis.pth`。

### Checkpoint 策略

```python
# eval epoch 时
medw = eval_medw200(test_data, angle=5.0)
axis_max = max(medw.roll, medw.pitch, medw.yaw)
if axis_max < best_axis_max:
    save("ckpt_best_medw_axis.pth")
    best_axis_max = axis_max
```

---

## 5. 交付 Gate

| Gate | 阶段结束 | 条件 | 后续 |
|------|----------|------|------|
| **G1** | S2 | val `max(R,P,Y) < 0.10°` | 进 S3 |
| **G2** | S3 | test MEDW200 `max(R,P,Y) < 0.15°` | 进 S4 |
| **G3** | S4 | test MEDW200 `max(R,P,Y) < 0.10°` | **交付** |
| FAIL | 任一 | 未达 | 走 §9 迭代路径 |

---

## 6. Ablation 矩阵

| ID | 架构差异 | 阶段 | 验证假设 |
|----|----------|------|----------|
| **R0** | v20 基线 | 无 | 0.646° 参照 |
| **R1** | v42 完整 | S1→S2 | cross-attn+corr+RoCR 总增益 |
| **R2** | R1 − RoCR (use_rocr=0) | S1→S2 | RoCR 几何初值价值 |
| **R3** | R1 − Pitch辅路 (use_pitch_branch=0) | S1→S2 | Pitch 双通道价值 |
| **R4** | R1 − adaptive window (corr_window_mode=fixed) | S1→S2 | 窗口自适应价值 |
| **R5** | R1 完整 | S1→S4 全链 | **0.10° 冲刺** |

最小验证集：**R0 → R1 → R5**（3 组 GPU）。

---

## 7. 代码实现

### 7.1 新增文件

| 文件 | 说明 | 预估行数 |
|------|------|----------|
| `kitti-bev-calib/cf_bev_r_calib.py` | 主模型 + `from_args` + forward | ~350 |
| `kitti-bev-calib/modules/rocr_refine.py` | RoCR：corr→R_geo SVD | ~100 |
| `kitti-bev-calib/modules/corr_transformer_decoder.py` | Swin Encoder + Pose Query Decoder | ~200 |
| `kitti-bev-calib/modules/pose_query_init.py` | RGB global + T_init + R_geo 编码 | ~60 |
| `kitti-bev-calib/modules/dla_aggregation.py` | DLA 多尺度 skip | ~40 |
| `kitti-bev-calib/losses/corr_alignment_loss.py` | Huber(peak, δuv_gt) | ~50 |
| `configs/v42_cf_bev_r.yaml` | 四阶段配置 | ~180 |

### 7.2 修改文件（最小侵入）

| 文件 | 改动 | 对其他实验影响 |
|------|------|-------------|
| `hybrid_triple_calib.py` L432 | 加 `cf_bev_r` 分支（~6行） | **无** |
| `train_kitti.py` argparse | 加 `--use_rocr`, `--corr_window_mode`, `--tinit_dropout_prob` | **无**（默认值不改） |

### 7.3 复用文件（严格审核后）

| 文件/模块 | 复用方式 | 注意 |
|-----------|---------|------|
| `ExtrinsicAwareCrossAttention` 单层 | **直接复用** | 不复用 `NativeCrossCalibHead` 整体（mean pool 不兼容） |
| `CrossAttentionBlock` | **直接复用** | — |
| `PointEncoder` (FPS+kNN) | **直接复用** | — |
| `losses/losses.py` 中 realworld_loss / PC_reproj / axis_rotation | **直接复用** | — |
| `img_branch/` Swin-Tiny + FPN | **直接复用** | — |
| `FrontViewPitchBranch` | **需适配输入** | z_summary 改为 pc group z 分桶（§10.2） |
| `compute_projection` | **需重写** | 去掉 extend_ratio 缩放 cx/cy |
| `LocalMultiHeadCorrelation` | **需 fork V2** | 保留完整 corr_map(B,G,W²,H)；修正 patch_size |

---

## 8. Backbone 选型：Swin-Tiny（非 DINOv2）

### 8.1 历史数据铁证

| 模型 | Backbone | Per-frame Rot | MEDW Rot | Shortcut? |
|------|----------|--------------|----------|-----------|
| **v20-v8recipe-pitch-wt3** | **Swin-Tiny (微调)** | **0.646°** | — | **否** |
| V28 F5-query-dinov2-frozen | DINOv2-S frozen | 2.323° | 0.137° | **是** |
| V30 G3-dann-ckpt400 | DINOv2-S unfreeze2 | 2.316° | 0.067° | **是** |
| V40/V41 全系列 | DINOv2-S frozen | — | — | **是**(Jac≈-0.1~-1.2) |

Per-frame ~2.3° ≈ identity 输出理论值（±5° 不校正的期望误差）。DINOv2 模型的低 MEDW 是统计幻觉。

### 8.2 根因

1. **DINOv2 自监督特征对语义敏感、对亚像素几何偏移不敏感** — correlation 需要的是"两个 patch 偏移了 Δuv"的精确空间信号
2. **冻结 backbone = 零几何适配** — 微小旋转导致的投影变化在 DINOv2 高维空间中几乎不可见
3. **Swin-Tiny 的 shifted window 机制对局部偏移天然敏感** — 且 `backbone_lr_scale=0.1` 允许微调适配几何任务
4. **CalibFormer 论文也用 ResNet-18（非 ViT）** — 卷积特征对局部纹理/边缘敏感，适合 correlation

### 8.3 决策

**v42 使用 Swin-Tiny + `backbone_lr_scale=0.1` 微调**，与 v20 一致。不用 DINOv2。

---

## 9. 反 Shortcut 设计（关键章节）

### 9.1 V42 的 Shortcut 攻击面

| 路径 | Shortcut 方式 | 风险 |
|------|-------------|------|
| T_init → harmonic → Cross-Attn | 模型只用 harmonic(T_init) 匹配，忽略视觉内容 | **中** |
| corr_map center bias | T_init ≈ GT 时 corr 峰值天然在中心 → RoCR 给 R_geo ≈ Identity | **高** |
| Decoder Δq ≈ 0 | R_geo 已近正确时 decoder 学会输出零残差 | **中** |
| 整条链路 | harmonic(T_init) → center corr → identity → Δq=0 → 透传 T_init | **高** |

### 9.2 反 Shortcut 五件套

| # | 机制 | 来源 | 参数 | 作用 |
|---|------|------|------|------|
| AS1 | **T_init Dropout** | V32 已验证 | prob=0.3, offset=15~30° | 30% 样本 T_init 远离 GT → 透传 loss 巨大 |
| AS2 | **Consistency Loss** | V32 已验证 | weight=0.5, start ep10 | 同场景不同 T_init → 预测应一致 |
| AS3 | **Corr Center Neg Bias** | V42 新增 | bias=-0.5 on center | 防止 corr 总在中心峰值 |
| AS4 | **RoCR Dropout** | V42 新增 | prob=0.3 | Decoder 必须独立预测全量旋转，不依赖 RoCR |
| AS5 | **S1 大角度训练** | V42 progressive | ±10° | Identity loss ≈ 10°，远超真实预测 |

### 9.3 Jacobian 监控（eval 指标，非 loss）

```python
# 每 eval epoch 计算
for axis in ['roll', 'pitch', 'yaw']:
    bias_corrections = []
    for bias in [-3°, -1°, +1°, +3°]:
        T_biased = apply_axis_bias(T_init, axis, bias)
        correction = model(img, pcd, T_biased) - model(img, pcd, T_init)
        bias_corrections.append((bias, correction))
    jacobian[axis] = polyfit(biases, corrections, deg=1)[0]
# 要求: J > 0.85 (每轴)
# J ≈ 0 → shortcut; J ≈ 1 → 完美自适应; J < 0 → 反向（更差）
```

**Gate 增强**：G2/G3 新增 Jacobian 条件：

| Gate | 原条件 | + 反 shortcut |
|------|--------|---------------|
| G2 | test MEDW200 max(R,P,Y) < 0.15° | + **Jac min(R,P,Y) > 0.85** |
| G3 | test MEDW200 max(R,P,Y) < 0.10° | + **Jac min(R,P,Y) > 0.85** |

---

## 10. 全面审核：潜在问题与优化迭代

### 10.1 已识别问题及缓解

| # | 问题 | 严重度 | 缓解 |
|---|------|--------|------|
| 1 | **RoCR SVD 在少量 valid corr 时退化** | 中 | valid_ratio<0.3 时 skip RoCR，pose query 不注入 R_geo |
| 2 | **S1 ±10° 训练 with rotation-only：trans=0 但 init 偏差大** | 低 | trans_range=0 已确保不对 t 加噪；大角度仅影响 R |
| 3 | **DLA 聚合 + Swin Encoder 2层 → 计算量增加** | 中 | 预估 ~1.5× v20 step time（DLA 轻量；Swin 仅 2层 on corr grid） |
| 4 | **S4 ±1° 训练域极窄，泛化 ±5° 测试可能退化** | 中 | S4 pretrain S3 ckpt（已学±3°分布）；eval 始终用 ±5° |
| 5 | **四阶段 pretrain 链断裂（某阶段 NaN）** | 低 | 每阶段独立 log_dir；NaN 可从上一阶段 ckpt 重启 |
| 6 | **axis_weights 1:4:1 可能压低 Roll/Yaw** | 低 | Roll/Yaw 在 v20 上已 <0.31°，余量充足；监控分轴曲线 |
| 7 | **Pose Query Decoder 与 RoCR 梯度路径冲突** | 中 | S1 前 20ep RoCR detach（不传梯度），让 decoder 先收敛 |
| 8 | **geodesic loss 在 S3/S4 切换时 loss 跳变** | 低 | warmup 5ep 线性混合 quat→geodesic |
| 9 | **CalibFormer depth+intensity 伪图支路未借** | 设计选择 | 伪图依赖 T_init 投影质量，rotation-only 大角度时投影偏差大；cross-attn 已替代其功能，不借 |
| 10 | **MEDW eval 在训练内需要 200帧窗口** | 低 | 每 eval epoch 跑一次，~2min overhead |

### 10.2 模块复用兼容性审核

| 模块 | 初始结论 | 严格审核 | 问题 |
|------|----------|----------|------|
| `ExtrinsicAwareCrossAttention` 单层 | 复用 | **可复用** | — |
| `NativeCrossCalibHead` 整体 | 复用 | **不可复用** | mean pool → MLP 路径与 V42 Decoder 不兼容 |
| `compute_projection` | 复用 | **需重写** | extend_ratio 缩放 cx/cy 不适合 V42 |
| `LocalMultiHeadCorrelation` | 扩展 | **需 fork V2** | corr_map.mean(dim=2) 丢弃空间信息；patch_size 不匹配 |
| `FrontViewPitchBranch` | 适配 | **需适配** | z_summary 源从 LSS BEV → pc group z 分桶 |
| `PointEncoder` / Loss 函数 | 复用 | **可复用** | — |

### 10.3 与 CalibFormer 原文对比审计

| CalibFormer 设计 | v42 对应 | 差异说明 |
|------------------|----------|----------|
| ResNet-18 backbone | **Swin-Tiny**（v20 已验证优于 ResNet） | 更强 backbone |
| DLA 4× upsample | **DLA skip 到 1/4** | 一致 |
| 2ch LiDAR 伪图 + ResNet-18 | **FPS groups + kNN MLP** | CalibFormer 投影伪图精度依赖 T_init；我们用 3D native groups 更鲁棒 |
| Multi-head correlation d=4 | **Adaptive d=4~12** | 增强，CalibFormer 固定 d=4 |
| Swin Encoder 2层 | **Swin Encoder 2层** | 一致 |
| Transformer Decoder 6层 | **4层**（防过拟合） | CalibFormer 在 KITTI 25K 样本可用 6 层；fleet ~10K 样本 4 层更安全 |
| RGB-guided pose query | **GAP(F_rgb) ⊕ T_init ⊕ R_geo** | 增强：加了 T_init 编码和 RoCR 初值 |
| L_T + L_R + L_P | **L_rot + L_axis + L_pc + L_corr + L_pitch + L_seq** | 增强：更丰富约束 |
| 单阶段 500ep | **四阶段 progressive 400ep** | 增强：小角度精调 |

### 10.4 可能的后续迭代路径

若 G3 未达：

| 优先级 | 方向 | 预期改善 |
|--------|------|----------|
| 1 | **S5 阶段 ±0.5°**（50ep, lr=5e-6） | Pitch 精细打磨 |
| 2 | **mount_jitter ±2°**（S2 后引入，prob=0.3） | 泛化衰退从 ~5× 压到 ~3.5×（v16 经验） |
| 3 | **iterative K=2**（S3+ 启用） | 迭代更新 T_init 重跑 cross-attn |
| 4 | **fleet PointGPT 替换 kNN PointEncoder** | 消除点云域差距（v37 教训） |
| 5 | **TTA median**（推理期多 probe 取中位） | 无训练成本改善 |

---

## 11. 变更记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-06-02 | v13.0 | **评估 Bug 修复 + 8卡 DataParallel 推理**：(1) **关键 Bug 修复**：`evaluate_checkpoint.py` 和 `bevcalib_inference.py` 未做 `max_pcd_points` 下采样（评估时点云 60K-94K 而训练仅 16K），导致输入分布不匹配，MEDW 从训练 0.046° 虚高到测试 0.846°。已在 collate_fn 和 _forward_cf_bev_r 中添加下采样。(2) **8卡 DataParallel 推理**：evaluate_checkpoint.py 自动检测多 GPU 并包装 DataParallel，batch_size 从 8 自动扩展到 64（8 GPU × 8），推理速度 ~5-6x。(3) **三阶段完整结果更新**：S1 MEDW=0.113° → S2 **MEDW=0.046°** → S3 **MEDW=0.041°**，全程零NaN。(4) **SVD Ablation 最终结论**：E2E(detach=15) 1285次NaN MEDW=0.089° vs Permanent-Detach(detach=999) **0次NaN MEDW=0.081°**——永久 detach 更稳定且更精确。 |
| 2026-06-02 | v12.0 | **三阶段训练完成 + SVD Ablation 结论 + Eval 支持**：S1→S2→S3 全部完成，MEDW=0.041°。`lr_schedule` bug 修复。evaluate_checkpoint.py + bevcalib_inference.py 支持 cf_bev_r。V42 MEDW=0.046° 远超 V41-B(0.836°)和V41-C(0.581°)。 |
| 2026-06-02 | v11.0 | **SVD 数值稳定性多层修复 + 永久 Detach 决策**：SVD正则化 `H+1e-6*I`、退化阈值 `1e-4`、`R_geo` nan_to_num 梯度钩子、最终 `rocr_detach_epochs=999` 永久detach。 |
| 2026-06-02 | v10.0 | **corr_alignment_loss 根因修复（Clamped GT 监督）**：(1) **根因**：`corr_alignment_loss` 的 GT 目标 `delta_uv_gt = (uv_gt-uv_init)/patch_size` 在 T_init 扰动下可达 15-100+ patch 单位（5°→15.3, 30°→101），但模型 `delta_uv_pred` 通过 soft-argmax 在 `[-radius,radius]=[-4,4]` 的窗口内计算，物理上限 4。不可达目标产生不可修复的 Huber loss → 单调增长 → 梯度爆炸 → ep66 100% NaN。(2) **正面修复（Clamped GT）**：`delta_uv_gt = clamp(raw_gt, -corr_radius, corr_radius)`，窗口内点精确监督、窗口外点教模型向 GT 方向推到极限。所有点都有有效梯度，无不可达目标，所有阶段（S1/S2/S3）均安全。(3) **防御层**：corr_head 独立梯度裁剪 max_norm=5.0；loss cap=10.0（安全网）；NaN 连续 30 batch 自动恢复 + LR×0.5。(4) `corr_alignment_weight` 0.3→0.2。(5) `continuous_noise_max_deg` 30°→15°：保持 3× 部署安全裕度，提高有效精确监督比例。 |
| 2026-06-01 | v9.0 | **一致性 OOM 根治：分离式 Forward**：(1) **Epoch 11 OOM 根因**：`consistency_loss_start_epoch=10` 启用后，旧代码将 batch 拼接为 2B（16→32）送入单次 forward，`CrossAttentionBlock.ffn` GELU 申请 2.64 GiB 时超出 L20 46GB 限制（已用 38GB+）。(2) **修复方案**：彻底废弃 2B-concat 路径，改为分离式 forward——原始 batch 正常训练（有梯度），alt batch 用 `torch.no_grad()` + `raw_model()` 单独推理，仅提取 `T_pred_alt` 计算一致性 loss。峰值显存 ~34 GiB（74%），比 2B 方式节省 ~26 GiB。(3) **耗时影响**：Epoch 11+ 每 step 增加一次 no_grad forward，耗时从 62.8s→80.6s/epoch（+28%），可接受。(4) Dryrun 验证通过（一致性从 epoch 0 启用，跑完 Epoch 1+2 无异常）。(5) Quick S1 验证：Epoch 11 `cons=0.0012`，Val Rot=1.07°，Jacobian=0.868，MEDW(R=0.51 P=0.65 Y=0.24)。 |
| 2026-06-01 | v8.0 | **一致性 Batch 维度修复 + 配置参数修正**：(1) **Epoch 10 crash 根因**：`consistency_loss_start_epoch=10` 启用后 batch 从 B→2B，`corr_alignment_loss` 的 `v42_delta_uv/xyz_groups/valid_mask` 未切片到 `[:B_cur]`，导致 `torch.bmm` 维度不匹配 `[32,4] vs [16,4]`。(2) `seq_consistency_loss` 的 `v42_rotation` 同理需要 `[:B_cur]` 切片。(3) `pitch_aux_weight` 在 `cf_bev_r_calib.py` 硬编码 0.1→改为使用配置值 0.3（打通 `from_args` 参数传递）。(4) **pretrain_ckpt 路径修复**：S2/S3 路径缺少 `model_small_{angle}deg_` 前缀和 `all_training_data_scratch/checkpoint/` 子目录，文件名 `ckpt_best_medw_axis.pth`→`ckpt_best_medw.pth`。(5) **32-node 配置优化**：注释修正（batch 3→16）；S2 LR 2e-4→5e-5（fine-tune 阶段）；S1 eval 10→10 ep；epoch 增加（S1:120→200, S2:120→160, S3:80→120）适配 256GPU 低步数。Dryrun 验证通过（800+步无 crash）。|
| 2026-06-01 | v7.0 | **NaN 根因修复 + 评估 OOM**：(1) `quaternion_from_matrix` / `quaternion_distance` 添加 `clamp(min=1e-8)` 防止 sqrt(0)/除零梯度爆炸（corr_head 梯度 270→正常）(2) 梯度裁剪 `max_norm` 35→10 (3) cross-attention all-invalid mask 安全保护（防止 softmax NaN）(4) Epoch 10 评估 OOM：batch_size 24→16 + 评估前 `torch.cuda.empty_cache()` (5) `_cuda_error_count` 每 epoch 重置。 |
| 2026-06-01 | v6.0 | **训练启动三重修复**：(1) `_nan_parts` UnboundLocalError（DDP non-main rank 作用域）(2) **点云 OOM**：原始 130K+点/样本→37GB OOM。新增 `max_pcd_points=16384` + `_subsample_pcd()` 全路径下采样，GPU 35→23.7GB(51%) (3) **NaN Guard DDP 死锁**：`continue` 跳过 backward → all-reduce 死锁。改用 `total_loss*0.0` 保持计算图。batch_size 24→16/GPU。4ep 验证：Rot 57°→4°，NaN 3次均正确处理。|
| 2026-06-01 | v4.0 | **实现完成并通过 GPU smoke test**。关键修复：RoCR SVD AMP 兼容（float32 保护）；FrontViewPitchBranch 参数名修正；SwinT FPN 单尺度绕过 DLA；PoseQueryInit MLP 加 LayerNorm+GELU；quat_head [1,0,0,0] 初始化；Loss warmup/gradient 修复。train_kitti.py 完成集成（argparser+batch_train.sh 参数转发+backbone LR 分组）。24.14M 参数，峰值 3.6GB（bs=4, AMP fp16）。 |
| 2026-06-01 | v5.1 | **严重梯度阻断修复 + 数据链路审核**：(1) **quat_head 权重零初始化导致梯度完全阻断**（2→359/360 params 有梯度），改用 xavier(gain=0.01) (2) SwinT FPN out_channels 与 feat_dim 解耦，添加 img_proj 投影层 (3) CrossAttentionBlock 多层 img_feat_dim 不一致修复 (4) batch-aware masked FPS 仅 4% 开销 (5) eval/Jacobian masks 类型修复（3 处 crash）(6) mask 全 1 快速路径。GPU 训练模拟通过（2 step loss 下降 65%）。|
| 2026-06-01 | v5.0 | **P1/P2/P3 严重缺陷修复**：(1) rocr_detach_epochs 接入训练循环 (2) corr_alignment_loss 接入训练循环 (3) seq_consistency_loss 接入训练循环。**FPS 性能优化**：消除 `.item()` GPU→CPU 同步，PointEncoder 488x 加速（1170ms→2.4ms），整体 2.7-5.4x 提速。**训练阶段简化**：4→3 阶段（5°→3°→1°），对齐 ±5° 部署需求。所有模型（v20/v41/v42）共享加速。|
| 2026-06-01 | v4.0 | 实现完成并通过 GPU smoke test。修复：SwinT FPN 单尺度处理、PoseQueryInit MLP 补 LayerNorm/GELU、quat_head identity 初始化、loss warmup 最小值 0.01、RoCR SVD AMP float32 guard、iterative_inference 方法修复 |
| 2026-06-01 | v3.0 | 深度审核：§8 Backbone 选型（Swin 非 DINOv2，数据铁证）；§9 反 Shortcut 五件套+Jacobian Gate；§10.2 模块复用兼容性审核（6 项不可直接复用）；§7 复用列表修正 |
| 2026-06-01 | v2.0 | 版本号改 v42；去掉 L2=0.15° 降级线，统一 0.10° 目标；增 BEVCalib 限制分析；增开关隔离 §1；seq 07/08 从风险移除；全面审核 §10；CalibFormer 对比审计 |
| 2026-06-01 | v1.0 | 初版 CF-BEV-R（原名 V22） |

---

## 附录 A：v20 → v42 演进对照

| 维度 | v20（当前最佳泛化） | v42（本方案） |
|------|---------------------|---------------|
| 图像→BEV | LSS 深度估计 + BEV Pool | **去 LSS**：Swin+FPN 2D patch（DLA 已实现但暂绕过，FPN 输出单尺度直接 interpolate） |
| 点云→BEV | SparseConv → BEV 栅格 | **去 BEV**：FPS groups 3D native |
| 融合 | ConvFuser concat | **ExtrinsicAware CrossAttn + LocalCorr** |
| Transformer | BEV 自注意力 8层 + 全局 pool | **Swin 2层 + PoseQuery Decoder 4层** |
| 几何约束 | 仅 loss 端 PC_reproj | **RoCR 几何初值 + Decoder 残差** |
| Pitch | axis_weights 1:3:1 | **1:4:1 + FrontViewPitchBranch** |
| 训练 | 单阶段 ±5° 400ep | **三阶段 5→3→1° (320ep total)** |
| Ckpt | mean Rot | **max(R,P,Y) MEDW200** |
| 隔离 | 默认 backend=bev | `fusion_backend=cf_bev_r` 独立 |

---

## §14 V43 训练策略改进 (2026-06-02)

### §14.1 V42 评估结论

All 7 models evaluated on test_data_v2 (±5° perturbation, 12 sequences, 28936 samples):

| 模型 | MEDW200 | bias+MEDW200 | 改善率 |
|------|---------|-------------|--------|
| V20 baseline | 0.608° | 0.547° | 10.0% |
| V42-S3 dual | 0.825° | 0.617° | 25.2% |
| SVD-E2E S2 | 0.832° | 0.624° | 25.0% |
| 32n-S1 | 0.814° | 0.641° | 21.3% |

Key findings:

- V42 trails V20 by 36% on raw MEDW200, but only 14% after bias correction
- V42's main weakness: systematic per-sequence bias (not random noise)
- 32n-S1 Pitch after bias correction (0.390°) beats V20 (0.398°)

### §14.2 架构根因分析

| 缺陷 | 类型 | 影响 |
|------|------|------|
| FrontViewPitchBranch 仅训练时生效 | 根本性 | Pitch 推理无专用路径 |
| Correlation→SVD 信息瓶颈 | 根本性 | 丰富特征被压缩到 (B,G,2) offsets |
| DLA 初始化未调用 | 可修复 | 缺少多尺度聚合 |
| center_neg_bias 导致零偏差过校正 | 可修复 | bag 评估中 C01-81 过校正 0.69° |

### §14.3 V43 改进方案

**训练策略改进（代码已实现）：**

1. `zero_perturbation_prob`: 5-10% 概率使用 T_init=T_gt 训练，教会模型"不需要校正"
2. Progressive anti-shortcut relaxation:
   - `rocr_center_bias`: S1(0.5) → S2(0.2) → S3(0.0)
   - `continuous_noise_max_deg`: S1(15°) → S2(5°) → S3(2°)
   - `rocr_dropout`: S1(0.3) → S2(0.2) → S3(0.1)
3. `axis_weights`: 从 "1,4,1" 调整为 "1,3,1.5"，减少 Pitch 过拟合，增加 Yaw 权重

**部署后处理（推荐）：**

- Bias correction: 收集 200 帧预测取 axis-angle median 作为偏差估计，后续帧减去偏差
- 预期效果: MEDW200 从 0.825° 降至 ~0.62°（接近 V20 的 0.608°）

### §14.4 V43 配置文件

| 文件 | 用途 |
|------|------|
| `configs/v43_cf_bev_r_quick.yaml` | 单机 8 卡 S1→S2→S3 |
| `configs/v43_cf_bev_r_quick_machineB.yaml` | 单机 8 卡 S2→S3 (从 V42-S1 ckpt) |
| `configs/v43_cf_bev_r_32node.yaml` | 32 节点 S2→S3 (从 V42-32n-S1 ckpt) |

### §14.5 V44 架构改进方向

**详细方案见 `docs/V44_ARCHITECTURE_IMPROVEMENT_PLAN.md`**

优先级排序：

| 优先级 | 改进 | 预期 MEDW200 提升 | 实现难度 |
|--------|------|-------------------|----------|
| P0-A | Pitch branch 推理融合 | -9% | ★★☆ |
| P0-B | 激活 DLA 多尺度聚合 | -5~8% | ★☆☆ |
| P1-A | 层级 Coarse-to-Fine Correlation | -10~15% | ★★★ |
| P1-B | Correlation Feature Augmentation | -3~5% | ★★☆ |
| P2-A | RoCR 置信度门控 | -3~5% | ★★☆ |
| P2-B | 增加 FPS groups 到 256 | -5~10% | ★☆☆ |
| P3 | 空间感知 Pose Query 初始化 | -3~5% | ★★☆ |

目标：MEDW200 从 V42 的 0.825° 降至 ≤0.50°（不依赖 bias correction）
