# V44 架构改进方案

> 状态：**设计阶段**（2026-06-02）
> 前置：`V42_CF_BEV_DESIGN.md`，`GENERALIZATION_REPORT.md`
> 基线：V42 CF-BEV-R 泛化 MEDW200=0.825°，V20 baseline MEDW200=0.608°
> 目标：**MEDW200 ≤ 0.50°**，Pitch 轴 MEDW200 ≤ 0.40°（无偏差矫正）

---

## 0. 执行摘要

V42 在训练集上达到了 MEDW=0.041° 的优秀精度，但泛化到 test_data_v2 后衰退至 0.825°（V20=0.608°）。V43 通过训练策略（zero-perturbation、progressive anti-shortcut）已经可以缓解零偏差过校正问题。V44 的任务是从**架构层面**解决 V42 的根本性缺陷，使模型在不依赖 bias correction 后处理的情况下达到或超越 V20。

### V42 根因诊断

| # | 缺陷 | 类型 | 影响量化 | V44 对策 |
|---|------|------|----------|----------|
| D1 | FrontViewPitchBranch 仅训练时生效 | **根本性** | Pitch MEDW200=0.493° vs V20=0.398° | P0: 推理融合 |
| D2 | Correlation→SVD 信息瓶颈 | **根本性** | 128 组 × 2D offset → 仅 256 个标量 | P1: 多尺度 + 密集 corr |
| D3 | DLA 多尺度聚合未激活 | **可修复** | 仅用 1/8 FPN，丢失细粒度特征 | P0: 激活 DLA |
| D4 | 单尺度 correlation 分辨率不足 | **结构性** | patch_size=4, radius=4 → 32px 窗口 | P1: 层级 coarse-to-fine |
| D5 | RoCR 无置信度门控 | **可改进** | 低置信度时错误初值误导 Transformer | P2: 置信度门控 |
| D6 | FPS groups 数量固定偏少 | **可调优** | 128 groups 对 Pitch 空间覆盖不足 | P2: 增加到 256 |
| D7 | Pose Query 初始化缺乏空间感知 | **可改进** | GAP 丢失空间分布信息 | P3: 空间感知 query |

---

## 1. P0-A：FrontViewPitchBranch 推理融合

### 1.1 问题分析

当前 `FrontViewPitchBranch` 在训练时通过 `pitch_aux_loss` 提供梯度信号，引导 backbone 学习 Pitch 相关特征。但**推理时 Pitch branch 的输出完全被丢弃**，最终旋转仅来自 `delta_q`（CorrTransformer）和 `R_geo`（RoCR）。

```
训练时:  loss = rotation_loss + 0.3 * pitch_aux_loss + corr_alignment_loss
推理时:  R_pred = delta_q ○ R_geo_q    ← pitch_branch 输出未参与
```

这导致训练和推理之间存在**信息泄漏**：训练时 backbone 为 Pitch branch 学到的特征在推理时没有被利用。

### 1.2 改进方案：Pitch-Aware Rotation Composition

将 Pitch branch 的预测在**推理时也参与最终旋转合成**，采用可学习的置信度加权：

```python
class PitchInferenceFusion(nn.Module):
    """在推理时融合 Pitch branch 预测到最终旋转中。

    策略：将 pitch_branch 预测的 pitch_delta 转为绕 X 轴的旋转四元数，
    然后与 CorrTransformer 的 delta_q 进行置信度加权融合。
    """

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.confidence_net = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, delta_q, pitch_q, rgb_gap, z_summary):
        """
        Args:
            delta_q:   (B, 4) CorrTransformer 预测的完整旋转残差
            pitch_q:   (B, 4) Pitch branch 预测的绕 X 轴旋转
            rgb_gap:   (B, D) 全局图像特征
            z_summary: (B, D) 点云高度分布特征
        Returns:
            fused_q: (B, 4) 融合后的旋转残差
        """
        confidence = self.confidence_net(
            torch.cat([rgb_gap, z_summary], dim=-1)
        )  # (B, 1) ∈ [0, 1]

        # Pitch 通道融合：从 delta_q 中分解 pitch 分量，与 pitch_q 加权混合
        fused_q = slerp(delta_q, compose(pitch_q, delta_q), confidence)
        return F.normalize(fused_q, dim=-1)
```

### 1.3 实现要点

1. **修改 `FrontViewPitchBranch`**：输出从标量 pitch_delta 改为四元数 `pitch_q = [cos(δp/2), sin(δp/2), 0, 0]`
2. **修改 `_core_forward`**：在 `corr_head` 输出 `delta_q` 后，调用 `PitchInferenceFusion` 进行融合
3. **新增 `PitchInferenceFusion` 模块**：置信度网络输入 `rgb_gap` + `z_summary`
4. **兼容性**：保留 `pitch_aux_loss` 作为辅助损失，融合模块独立于辅助损失

### 1.4 修改文件

| 文件 | 修改内容 |
|------|----------|
| `cf_bev_r_calib.py` | `_core_forward` 添加 pitch 融合路径 |
| `bev_calib.py` | `FrontViewPitchBranch` 新增 `predict_quaternion()` |
| `modules/pitch_fusion.py` | 新建：`PitchInferenceFusion` 模块 |

### 1.5 预期效果

- Pitch 轴 MEDW200：0.493° → ~0.40°（-19%）
- 总 MEDW200：0.825° → ~0.75°（-9%）
- 风险：**低**，pitch_q 置信度初始化为 0 可退化为 V42 行为

---

## 2. P0-B：激活 DLA 多尺度聚合

### 2.1 问题分析

V42 的 `DLAAggregation` 已经实例化（cf_bev_r_calib.py:110-113），但 `_core_forward` 中**完全没有调用**。实际代码仅使用 `img_encoder` 的第一层 FPN 输出（1/8 分辨率），然后双线性插值到 1/4：

```python
# 当前代码 (line 333-338)
fpn_out = self.img_encoder(imgs_norm.unsqueeze(1))
fpn_feat = fpn_out[:, 0]  # 仅取 1/8 层
fpn_feat = self.img_proj(fpn_feat)
F_rgb = F.interpolate(fpn_feat, size=(feat_h, feat_w), ...)  # 1/8 → 1/4
```

Swin-Tiny 的三层 FPN 输出（1/8, 1/16, 1/32）完全被浪费，**丢失了 1/16 的中间语义和 1/32 的全局上下文**。

### 2.2 改进方案：DLA 替代单层 FPN

```python
# V44 改进：使用 DLA 聚合三层 FPN
fpn_out = self.img_encoder(imgs_norm.unsqueeze(1))

# img_encoder 返回 (B, N, C, H_i, W_i)，N=1 for single cam
# 需要获取三层 FPN 特征
feat_list = self.img_encoder.get_multi_scale_features(imgs_norm.unsqueeze(1))
# feat_list = [feat_1_8, feat_1_16, feat_1_32]

F_rgb = self.dla(feat_list)  # (B, 256, H/4, W/4) — DLA 聚合
```

### 2.3 实现要点

1. **修改 `SwinT_tiny_Encoder`**：暴露 `get_multi_scale_features()` 方法，返回三层 FPN 的独立特征图
2. **修改 `_core_forward`**：用 `self.dla(feat_list)` 替代 `self.img_proj(fpn_out[:, 0])` + interpolate
3. **去掉 `self.img_proj`**：DLA 内部已包含 1×1 conv 对齐通道

### 2.4 修改文件

| 文件 | 修改内容 |
|------|----------|
| `cf_bev_r_calib.py` | `_core_forward` 调用 DLA 替代单层 FPN |
| `img_branch/swint_encoder.py` | 新增 `get_multi_scale_features()` |

### 2.5 预期效果

- 特征质量提升：1/16 和 1/32 层提供全局上下文，有助于跨场景泛化
- 总 MEDW200：预计 -5~8%（基于 CalibFormer 论文 DLA 消融实验）
- 风险：**极低**，DLA 模块已实现并测试，仅需调用

---

## 3. P1-A：层级 Coarse-to-Fine Correlation

### 3.1 问题分析

V42 的 `LocalMultiHeadCorrelationV2` 在**单一尺度**（1/4 分辨率，patch_size=4）上计算局部相关性。窗口半径 `r=4` 意味着每个 group 的搜索窗口为 `9×9=81` 个 tokens，对应原图 `36×36` 像素。

**问题**：
- 对大扰动（5° Pitch ≈ 32px 位移），81 tokens 窗口可能不够覆盖真实匹配位置
- 对小残差（<1°），1/4 分辨率的 correlation 精度仅 ~0.25°/pixel，难以达到 0.1° 精度
- 所有信息通过 `(B, G, W²)` 的扁平相关图传递给 RoCR，**几何丰度不足**

### 3.2 改进方案：两阶段层级 Correlation

```
Stage-1: Coarse Correlation (1/8 分辨率, r=6)
  → 大范围搜索（覆盖 ~96px），找到粗略匹配位置
  → 输出: coarse_offset (B, G, 2)

Stage-2: Fine Correlation (1/4 分辨率, r=3)
  → 以 coarse_offset 为中心开小窗口精细匹配
  → 输出: fine_corr_map (B, G, 49), fine_offset (B, G, 2)
```

```python
class HierarchicalCorrelation(nn.Module):
    """两阶段层级相关性：粗搜 → 细搜。"""

    def __init__(self, token_dim=256, num_heads=4):
        super().__init__()
        self.coarse_corr = LocalMultiHeadCorrelationV2(
            token_dim=token_dim, num_heads=num_heads,
            default_radius=6, out_dim=token_dim,
        )
        self.fine_corr = LocalMultiHeadCorrelationV2(
            token_dim=token_dim, num_heads=num_heads,
            default_radius=3, out_dim=token_dim,
        )
        self.offset_pred = nn.Linear(token_dim, 2)  # coarse offset predictor

    def forward(self, img_tokens_coarse, img_tokens_fine,
                pc_tokens, pc_uv_init, feat_h_c, feat_w_c, feat_h_f, feat_w_f):
        # Stage 1: coarse
        f_coarse, corr_coarse, info_c = self.coarse_corr(
            img_tokens_coarse, pc_tokens, pc_uv_init / 2,  # 1/8 坐标
            feat_h_c, feat_w_c)
        coarse_offset = self.offset_pred(f_coarse)  # (B, G, 2) in coarse grid

        # Stage 2: fine with shifted center
        uv_fine = pc_uv_init + coarse_offset * 2  # 转到 1/4 坐标
        f_fine, corr_fine, info_f = self.fine_corr(
            img_tokens_fine, pc_tokens, uv_fine, feat_h_f, feat_w_f)

        return f_fine, corr_fine, {
            'coarse_offset': coarse_offset,
            'fine_info': info_f,
            'coarse_corr': corr_coarse,
        }
```

### 3.3 实现要点

1. **需要 DLA 的多尺度特征**：coarse 用 1/8 FPN，fine 用 1/4 DLA 输出（与 P0-B 联动）
2. **coarse_offset 的监督**：在 RoCR 的 SVD 输出 delta_uv 和 coarse_offset 之间加 L2 loss
3. **CrossAttention 调整**：对 coarse 和 fine 两个尺度分别做 cross-attention，或共享 cross-attn 后分叉

### 3.4 修改文件

| 文件 | 修改内容 |
|------|----------|
| `modules/hierarchical_corr.py` | 新建：`HierarchicalCorrelation` |
| `cf_bev_r_calib.py` | 替换 `self.local_corr` 为 `self.hier_corr` |
| `losses/` | 新增 `coarse_offset_loss` |

### 3.5 预期效果

- 大扰动覆盖率提升：radius=6 @ 1/8 覆盖 104px（vs 当前 36px）
- 小残差精度提升：radius=3 @ 1/4 精度 ~0.13°/pixel
- 总 MEDW200：预计 -10~15%
- 风险：**中等**，需要仔细调参 coarse→fine 的坐标映射

---

## 4. P1-B：密集 Correlation 特征（替代 offset-only 瓶颈）

### 4.1 问题分析

当前信息流：

```
LocalCorrV2 → corr_map (B, G, 81) → RoCR soft_argmax → delta_uv (B, G, 2)
                                                    ↓
                           只保留了 peak 位置，丢弃了 correlation 形状信息
```

RoCR 通过 `soft_argmax` 将 81 维相关图压缩为 2 维 offset，然后通过 SVD 得到旋转。**correlation map 的形状、峰值锐度、多峰分布**等丰富信息全部丢失。

### 4.2 改进方案：Correlation Feature Augmentation

在送入 CorrTransformerHead 之前，将 correlation map 的统计特征编码为额外输入：

```python
class CorrFeatureAugmentation(nn.Module):
    """从 correlation map 提取几何统计特征。"""

    def __init__(self, window_size: int = 81, out_dim: int = 32):
        super().__init__()
        self.stats_mlp = nn.Sequential(
            nn.Linear(5, 32),  # peak_value, peak_sharpness, entropy, dx_var, dy_var
            nn.GELU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, corr_map, window_radius):
        """
        Args:
            corr_map: (B, G, W²) correlation scores
        Returns:
            corr_stats: (B, G, out_dim) correlation statistics features
        """
        B, G, W2 = corr_map.shape
        probs = F.softmax(corr_map, dim=-1)

        peak_val = corr_map.max(dim=-1).values          # (B, G)
        entropy = -(probs * (probs + 1e-8).log()).sum(-1) # (B, G)

        W = 2 * window_radius + 1
        offs = torch.arange(-window_radius, window_radius + 1,
                           device=corr_map.device, dtype=corr_map.dtype)
        dy, dx = torch.meshgrid(offs, offs, indexing='ij')
        dx, dy = dx.reshape(-1), dy.reshape(-1)

        mean_dx = (probs * dx).sum(-1)
        mean_dy = (probs * dy).sum(-1)
        var_dx = (probs * (dx - mean_dx.unsqueeze(-1))**2).sum(-1)
        var_dy = (probs * (dy - mean_dy.unsqueeze(-1))**2).sum(-1)
        sharpness = peak_val - corr_map.mean(dim=-1)

        stats = torch.stack([peak_val, sharpness, entropy, var_dx, var_dy], dim=-1)
        return self.stats_mlp(stats)
```

### 4.3 集成方式

```python
# 在 _core_forward 中，替换:
corr_tokens = f_corr + F_pc

# 改为:
corr_stats = self.corr_augment(corr_map, window_radius)
corr_tokens = f_corr + F_pc + corr_stats  # (B, G, D) 三路融合
```

### 4.4 预期效果

- CorrTransformer 获得更丰富的几何线索（峰值置信度、分布方差）
- 有助于 Transformer 判断哪些 groups 的 correlation 可靠
- MEDW200 预计 -3~5%
- 风险：**低**，仅增加 32 维特征到 corr_tokens

---

## 5. P2-A：RoCR 置信度门控

### 5.1 问题分析

当前 RoCR 的行为是二元的：

```
if rocr_dropout triggered:  R_geo = Identity
elif valid_ratio < 0.3:     R_geo = Identity
else:                       R_geo = SVD result (无论质量如何)
```

问题是 SVD 结果的质量**没有被量化并传递给下游**。当 correlation 峰值模糊（如遮挡、重复纹理区域）时，SVD 输出的旋转可能偏差较大，但 Transformer 无法区分好的和坏的 R_geo。

### 5.2 改进方案：Confidence-Gated RoCR

```python
class ConfidenceGatedRoCR(RoCR):
    """RoCR with learned confidence gating.

    SVD 结果通过置信度分数加权融入旋转估计：
    R_out = slerp(Identity, R_svd, confidence)
    confidence 由 correlation 统计特征学习得到。
    """

    def __init__(self, feat_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.conf_net = nn.Sequential(
            nn.Linear(3, 32),  # valid_ratio, peak_sharpness, residual_norm
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, corr_map, xyz_groups, uv_init, cam_intrinsic,
                valid_mask, window_radius, patch_size, feat_h, feat_w):
        base_out = super().forward(
            corr_map, xyz_groups, uv_init, cam_intrinsic,
            valid_mask, window_radius, patch_size, feat_h, feat_w)

        # 计算置信度特征
        valid_ratio = base_out['confidence']
        probs = F.softmax(corr_map, dim=-1)
        peak_sharp = (corr_map.max(-1).values - corr_map.mean(-1)).mean(1)
        residual = base_out['delta_uv'].norm(dim=-1).mean(1)
        conf_input = torch.stack([valid_ratio, peak_sharp, residual], dim=-1)
        confidence = self.conf_net(conf_input).squeeze(-1)  # (B,) ∈ [0,1]

        # 软门控：低置信度时 R_geo 趋向 Identity
        R_svd = base_out['R_geo']
        R_identity = torch.eye(3, device=R_svd.device).unsqueeze(0).expand_as(R_svd)
        alpha = confidence.view(-1, 1, 1)
        R_gated = R_identity + alpha * (R_svd - R_identity)
        # 重新正交化
        U, _, Vh = torch.linalg.svd(R_gated)
        R_gated = torch.bmm(U, Vh)

        base_out['R_geo'] = R_gated
        base_out['rocr_confidence'] = confidence
        return base_out
```

### 5.3 预期效果

- 避免低质量 SVD 结果误导 Transformer
- 在 Seq04/Seq07 等难序列上效果可能显著（这些序列的 32n-S1 MEDW 极差）
- MEDW200 预计 -3~5%
- 风险：**低-中**，置信度学习需要足够多样的训练数据

---

## 6. P2-B：增加 FPS Groups 到 256

### 6.1 问题分析

当前 `n_groups=128` 意味着每帧仅有 128 个 3D 点参与 correlation 和 SVD。对于 360×640 的图像，这相当于每 ~1800 像素才有一个锚点。特别是**Pitch 方向**的纵向分布密度不足，难以捕捉细微的垂直位移。

### 6.2 改进方案

```yaml
# 配置调整
cf_n_groups: 256  # 128 → 256
```

### 6.3 实现要点

1. 纯配置变更，无需修改代码
2. 计算开销：FPS O(N·G) + kNN O(G·N·k) + correlation O(G·W²·D)
   - 128→256：FPS ~2x，kNN ~2x，correlation ~2x
   - 实测预估：训练吞吐量降低 ~15-20%，推理时间增加 ~10ms
3. 需配合 batch_size 调整以适应 GPU 显存

### 6.4 预期效果

- 更密集的空间采样，特别是纵向（Pitch 相关）覆盖
- Pitch MEDW200 预计改善 5-10%
- 风险：**极低**，纯超参数调整

---

## 7. P3：空间感知 Pose Query 初始化

### 7.1 问题分析

当前 `PoseQueryInit` 使用 GAP（Global Average Pooling）的 `rgb_gap` 作为主要特征：

```python
encoded = cat([rgb_gap, harmonic(T_init), harmonic(R_geo)])
queries = MLP(encoded).reshape(B, num_queries, D)
```

GAP 丢失了所有空间信息。6 个 pose queries 的初始化**无法区分图像不同区域的对齐状态**（如上半部分对齐好、下半部分差），每个 query 接收完全相同的空间先验。

### 7.2 改进方案：Spatial-Aware Query Init

```python
class SpatialPoseQueryInit(nn.Module):
    """空间感知的 Pose Query 初始化。

    将 F_rgb 特征图分区域池化，每个 query 对应一个空间区域。
    6 queries = [上左, 上右, 中左, 中右, 下左, 下右]
    """

    def __init__(self, rgb_dim=256, num_queries=6, query_dim=256, n_harmonic=6):
        super().__init__()
        self.num_queries = num_queries
        # 每个 query 对应 feat_map 的一个 2×3 (H/2 × W/3) 区域
        region_dim = rgb_dim  # 区域池化后的维度
        harmonic_t = HarmonicEmbedding(n_harmonic)
        harmonic_r = HarmonicEmbedding(n_harmonic)
        t_dim = harmonic_t.get_output_dim(input_dims=16)
        r_dim = harmonic_r.get_output_dim(input_dims=9)
        self.harmonic_t = harmonic_t
        self.harmonic_r = harmonic_r

        mlp_in = region_dim + t_dim + r_dim
        self.per_query_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mlp_in, query_dim * 2),
                nn.LayerNorm(query_dim * 2),
                nn.GELU(),
                nn.Linear(query_dim * 2, query_dim),
            )
            for _ in range(num_queries)
        ])

    def forward(self, F_rgb_2d, T_init, R_geo):
        """
        Args:
            F_rgb_2d: (B, D, H, W) 2D 特征图（非 GAP）
            T_init: (B, 4, 4)
            R_geo: (B, 3, 3)
        """
        B, D, H, W = F_rgb_2d.shape
        # 2×3 grid → 6 regions
        h2, w3 = H // 2, W // 3
        regions = []
        for i in range(2):
            for j in range(3):
                region = F_rgb_2d[:, :, i*h2:(i+1)*h2, j*w3:(j+1)*w3]
                regions.append(region.mean(dim=[2, 3]))  # (B, D)

        t_emb = self.harmonic_t(T_init.reshape(B, -1))
        r_emb = self.harmonic_r(R_geo.reshape(B, -1))

        queries = []
        for q_idx, (region_feat, mlp) in enumerate(
            zip(regions, self.per_query_mlp)):
            q = mlp(torch.cat([region_feat, t_emb, r_emb], dim=-1))
            queries.append(q)

        return torch.stack(queries, dim=1)  # (B, 6, D)
```

### 7.3 预期效果

- Query 携带空间先验，decoder cross-attention 可以针对性地关注对齐差的区域
- 对 Pitch 特别有利：上下区域的 misalignment 差异是 Pitch 偏差的直接体现
- MEDW200 预计 -3~5%
- 风险：**中**，需要确保 spatial query 不会导致过拟合特定视角

---

## 8. 整体架构变更对比

### 8.1 V42 数据流

```
Swin-Tiny → FPN[0] → 1x1 proj → bilinear 1/8→1/4 → F_rgb
                                                        ↓
PointEncoder → F_pc ←── CrossAttnBlock ×2 ──→ F_cross
                                                        ↓
                        LocalCorrV2(F_cross, F_pc) → f_corr, corr_map
                                                        ↓
                        RoCR(corr_map) → R_geo → PoseQueryInit(GAP, T_init, R_geo)
                                                        ↓
                        CorrTransformerHead(f_corr+F_pc, queries) → delta_q
                                                        ↓
                        delta_q ○ R_geo_q → rotation
                        [Pitch branch: 仅训练时 aux loss]
```

### 8.2 V44 数据流

```
Swin-Tiny → FPN[0,1,2] → DLA聚合 → F_rgb (1/4)      [P0-B: 多尺度]
                        ↘ FPN[0] → F_rgb_coarse (1/8)  [P1-A: 粗搜]
                                                        ↓
PointEncoder(G=256) → F_pc ←── CrossAttnBlock ×2 ──→ F_cross
                                                        ↓
              HierarchicalCorr:                         [P1-A: 层级corr]
                Stage1: CoarseCorrV2(F_cross_1/8, F_pc) → coarse_offset
                Stage2: FineCorrV2(F_cross_1/4, F_pc, shifted) → f_corr, corr_map
                                                        ↓
              CorrFeatureAugment(corr_map) → corr_stats [P1-B: 密集特征]
                                                        ↓
              ConfidenceGatedRoCR(corr_map) → R_geo, conf  [P2-A: 置信度门控]
                                                        ↓
              SpatialPoseQueryInit(F_rgb_2d, T_init, R_geo)  [P3: 空间query]
                                                        ↓
              CorrTransformerHead(f_corr+F_pc+corr_stats, queries) → delta_q
                                                        ↓
              PitchInferenceFusion(delta_q, pitch_q, confidence) [P0-A: pitch融合]
                                                        ↓
              fused_q ○ R_geo_q → rotation
```

---

## 9. 实施计划与分期

### Phase 1（快速验证，1-2 天）

| 任务 | 优先级 | 预估改善 | 实现难度 |
|------|--------|----------|----------|
| 激活 DLA 多尺度聚合 | P0-B | 5-8% | ★☆☆ |
| Pitch branch 推理融合 | P0-A | 9% | ★★☆ |
| 增加 FPS groups 到 256 | P2-B | 5-10% | ★☆☆ |

**Phase 1 目标**：MEDW200 0.825° → ~0.65°

### Phase 2（架构升级，3-5 天）

| 任务 | 优先级 | 预估改善 | 实现难度 |
|------|--------|----------|----------|
| 层级 Coarse-to-Fine Correlation | P1-A | 10-15% | ★★★ |
| Correlation Feature Augmentation | P1-B | 3-5% | ★★☆ |
| RoCR 置信度门控 | P2-A | 3-5% | ★★☆ |

**Phase 2 目标**：MEDW200 ~0.65° → ~0.50°

### Phase 3（精细打磨，2-3 天）

| 任务 | 优先级 | 预估改善 | 实现难度 |
|------|--------|----------|----------|
| 空间感知 Pose Query Init | P3 | 3-5% | ★★☆ |
| 超参调优（axis_weights, lr schedule） | - | 3-5% | ★☆☆ |
| Iterative inference（2-iter refinement） | - | 2-3% | ★☆☆ |

**Phase 3 目标**：MEDW200 ~0.50° → ~0.45°

---

## 10. 消融实验设计

每个改进独立验证，使用 V42-S1 quick ckpt 作为起点，在 test_data_v2 上评估 MEDW200。

| 实验 | 配置变更 | 训练 epoch | 评估指标 |
|------|----------|------------|----------|
| V44-ablation-dla | 仅激活 DLA | S2 60 epoch | MEDW200 全轴 |
| V44-ablation-pitch-fusion | 仅加 pitch 融合 | S2 60 epoch | MEDW200 Pitch 轴 |
| V44-ablation-groups256 | 仅增 groups | S2 60 epoch | MEDW200 + 推理速度 |
| V44-ablation-hier-corr | 仅层级 corr | S2 80 epoch | MEDW200 全轴 |
| V44-ablation-corr-augment | 仅 corr 特征增强 | S2 60 epoch | MEDW200 |
| V44-ablation-conf-rocr | 仅置信度 RoCR | S2 60 epoch | MEDW200 + per-seq 方差 |
| V44-full | 所有改进 | S1→S2→S3 | MEDW200 全轴 + bag eval |

---

## 11. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| DLA 聚合引入过多参数导致过拟合 | 低 | 中 | DLA 参数量仅 ~130K，且带 conv bias=False |
| Pitch fusion 置信度网络学到 always-0 | 中 | 高 | 初始化 bias 使 sigmoid 输出 ~0.3，加 warmup |
| 层级 corr coarse→fine 坐标映射不准 | 中 | 高 | 先用 GT offset 验证 fine-stage 精度上界 |
| groups=256 显存不足 | 低 | 中 | 可降 batch_size 或用梯度累积 |
| 改进叠加后训练不稳定 | 中 | 高 | 按 Phase 分期，每阶段充分验证后再叠加 |

---

## 12. 与 V43 的关系

V44 与 V43 是**正交互补**的：

- **V43**：训练策略改进（zero-perturbation、progressive anti-shortcut、axis weights）→ 优化现有架构的训练效果
- **V44**：架构改进（DLA、pitch fusion、层级 corr）→ 从根本上提升模型的表达能力

推荐实施顺序：

```
V43 训练 (当前) → V43 验证 → V44 Phase 1 实现 → V44+V43 联合训练 → V44 Phase 2
```

V44 Phase 1 的代码修改保持**开关控制**，不影响 V43 实验：

```yaml
# V44 特性开关
use_dla_aggregation: true       # P0-B
use_pitch_inference_fusion: true # P0-A
cf_n_groups: 256                # P2-B
use_hierarchical_corr: false    # P1-A（Phase 2 再开启）
use_corr_augmentation: false    # P1-B
use_confidence_rocr: false      # P2-A
use_spatial_query_init: false   # P3
```

---

## 13. 关键指标追踪

| 指标 | V20 | V42 | V43 目标 | V44 目标 |
|------|-----|-----|----------|----------|
| MEDW200 总 | 0.603° | 0.821° | ~0.65° | **≤0.50°** |
| MEDW200 Roll | 0.294° | 0.327° | ~0.28° | ≤0.22° |
| MEDW200 Pitch | 0.434° | 0.630° | ~0.42° | **≤0.35°** |
| MEDW200 Yaw | 0.209° | 0.291° | ~0.20° | ≤0.16° |
| 泛化衰退比 | 5.9× | 20.0× | ~10× | **≤5×** |
| Per-seq MEDW200 Std | 0.232° | 0.510° | ~0.35° | ≤0.25° |
| 推理速度（ms/frame） | ~15 | ~25 | ~25 | ≤35 |

注：以上数据基于 2026-06-02 评估（7 模型, 12 seq × 400 帧, ±5° rotation-only）。

---

## 14. V44 域泛化优化方案（P-1 最高优先级）

### 14.1 训练/测试数据域差异分析

2026-06-02 域差异检查结果：

| 维度 | 训练数据 (21 seq) | 测试数据 (12 seq) | 差异评估 |
|------|-----------------|-----------------|----------|
| 焦距 fx | mean=7184.8 ±38.2 | mean=7182.2 ±52.8 | **微小**（<0.1%） |
| 安装 Roll | mean=-89.47° ±1.35° | mean=-89.35° ±1.02° | **微小** |
| 安装 Pitch | mean=0.20° ±0.50° | mean=0.15° ±0.38° | **微小** |
| 安装 Yaw | mean=-90.34° ±0.39° | mean=-90.22° ±0.29° | **微小** |
| 序列数量 | 21 | 12 | 覆盖不同行程 |
| 场景内容 | 城市/郊区/高速 | 城市/郊区/高速 | 需分析纹理差异 |

**关键发现：**
1. **相机参数域差异极小**（焦距、安装位姿分布高度重叠），不是泛化差的主因
2. **Test Seq09 = Train Seq09**（数据泄漏！fx 和 extrinsics 完全一致）
3. **Seq04/Seq07 在所有 V42 模型上崩溃**，但相机参数并无特殊——说明是**场景级差异**导致
4. V42 泛化问题的根源是**模型对场景纹理/光照的过敏性**，而非相机参数域偏移

### 14.2 域增强策略（已实现 & 在 V44 配置中启用）

| 增强方式 | 参数 | 理由 | 预期效果 |
|---------|------|------|----------|
| Mount Jitter | prob=0.3, rot_sigma=1.0° | 模拟安装位姿变化 | 降低对特定安装配置的过拟合 |
| Intrinsic Aug | fx/fy±2%, cx/cy±1% | 模拟焦距/光心变化 | 提高对内参变化的鲁棒性 |
| Color Jitter | strength=0.2 | 模拟光照/亮度变化 | 减少对纹理特征的依赖 |
| Zero Perturbation | prob=0.1 | V43 策略 | 消除零偏差过校正 |

### 14.3 进阶域泛化方案（Phase 2, 需要实验验证）

**方案 A: Instance Normalization (IN)**
- 在 backbone 中间层替换 BN 为 IN
- IN 去除每个样本的风格信息（亮度/对比度统计量），只保留结构信息
- 代价：可能降低训练集精度
- 实现：在 `img_proj` 层或 FPN 后加 IN

**方案 B: Domain-Adversarial Training (DANN)**
- 已有代码支持（`domain_adversarial=1`）
- 需要 domain label（可用 sequence ID 作为域标签）
- 通过 GRL 迫使特征表示忽略域信息

**方案 C: Style Normalization + Restitution (SNR)**
- 训练时随机替换 BN 统计量（模拟域偏移）
- 测试时重新估计 BN 统计量（TTA 变体）

**方案 D: 多域数据扩充**
- 当前 21 个训练序列来自同一批车辆/相机
- 最有效的泛化方案是增加来自**不同车辆/安装**的训练数据
- 估计需要 100+ 不同安装配置才能显著改善泛化

### 14.4 V44 训练配置（已创建）

配置文件: `configs/v44_cf_bev_r_quick.yaml`

```yaml
# V44 核心改动（相比 V42）
use_dla: 1                        # DLA 多尺度聚合
use_pitch_fusion: 1               # Pitch 推理融合
augment_mount_jitter_prob: 0.3    # 域增强：安装抖动
augment_intrinsic: 0.02           # 域增强：内参变化
augment_color_jitter: 0.2         # 域增强：颜色抖动
zero_perturbation_prob: 0.1       # V43 零扰动
```

Progressive training: S1(±5°, 150ep, warm from V42-S1) → S2(±3°, 120ep) → S3(±1°, 80ep)

### 14.5 预期效果分析

| 改进项 | 预期效果 | 信心度 | 来源 |
|--------|---------|--------|------|
| DLA 多尺度 | Pitch -0.03°~-0.05° | **中** | 补充细粒度特征 |
| Pitch 融合 | Pitch -0.02°~-0.05° | **中** | 训练时有效的信号推理时也利用 |
| Mount Jitter | 全轴 -0.05°~-0.10° | **高** | 直接增加域多样性 |
| Color Jitter | 全轴 -0.02°~-0.05° | **中** | 减少纹理过拟合 |
| Intrinsic Aug | Roll -0.01°~-0.03° | **低** | 域差异已经很小 |
| Zero Perturbation | 全轴 -0.02°~-0.03° | **高** | V43 已验证 |
| **合计（乐观）** | **~0.55°** (从 0.821°) | - | 仍距 0.1° 有差距 |

### 14.6 到 0.1° 目标的路线图

```
V42 MEDW200     = 0.821°  (baseline)
V44 Phase 1     ≈ 0.55°   (架构 + 域增强)
V44 + IN/DANN   ≈ 0.45°   (特征级域对齐)
V44 + 多域数据   ≈ 0.30°   (100+ 不同安装配置)
V44 + TTA       ≈ 0.20°   (测试时自适应)
V44 + 时序递归   ≈ 0.12°   (前帧结果反馈)
0.1° target     需要上述全部或接近全部
```

---

## 15. V43/V44 训练与泛化实验结果 (2026-06-03)

### 15.1 训练收敛总结

所有实验均已收敛（CONVERGED），训练总时间约 12 小时。

| 版本-阶段 | 范围 | Epochs | MEDW200 max(R,P,Y) | Val Rot | Jacobian | IN |
|-----------|------|--------|---------------------|---------|----------|----|
| V43-S1 | ±5° | 120 | 0.0743° | 0.22° | **0.999** | - |
| V43-S2 | ±3° | 120 | 0.0476° | 0.17° | 0.962 | - |
| V43-S3 | ±1° | 80 | **0.0413°** | 0.20° | 0.921 | - |
| V44-S1 | ±5° | 150 | 0.0888° | 0.26° | 0.971 | OFF |
| V44-S2 | ±3° | 120 | 0.0428° | 0.20° | 0.942 | **ON** |
| V44-S3 | ±1° | 80 | 0.0442° | 0.27° | 0.930 | **ON** |

### 15.2 泛化评估结果（test_data_v2, 12 sequences, 400 帧/seq）

**单帧推理排名：**

| 排名 | 模型 | Per-frame Rot | Roll | Pitch | Yaw |
|------|------|--------------|------|-------|-----|
| 1 | **V44-S1** (DLA+PF+Aug) | **0.649°** | 0.299 | 0.426 | 0.235 |
| 2 | V20 baseline | 0.646° | 0.312 | 0.447 | 0.230 |
| 3 | V43-S2 | 0.832° | 0.373 | 0.615 | 0.283 |
| 4 | V42-32n-S1 | 0.850° | 0.482 | 0.525 | 0.274 |
| 5 | V43-S3 | 0.860° | 0.420 | 0.600 | 0.291 |
| 6 | V44-S2 (IN) | **1.001°** | 0.376 | **0.765** | 0.309 |
| 7 | V44-S3 (IN) | **1.011°** | 0.383 | **0.770** | 0.309 |

**BEST 时序聚合排名：**

| 排名 | 模型 | BEST Rot | 方法 | Per-frame→BEST |
|------|------|---------|------|---------------|
| 1 | **V44-S1** | **0.582°** | MEDW100 | 10.3% |
| 2 | V20 baseline | 0.622° | - | 3.8% |
| 3 | V43-S3 | 0.756° | SVDW400 | 12.1% |
| 4 | V42-32n-S1 | 0.778° | - | 8.6% |

### 15.3 关键发现

**1. V44-S1 首次超越 V20 baseline**
- BEST 聚合: 0.582° vs V20 的 0.622°（**提升 6.4%**）
- Per-frame 与 V20 持平（0.649° vs 0.646°）
- 这是 V42 架构系列首次在泛化测试中追上 V20

**2. Instance Norm 严重破坏泛化**
- V44-S1（无IN）: 0.649° → V44-S2（有IN）: 1.001°（**恶化 54%**）
- Pitch 从 0.426° 暴涨到 0.765°（+79.6%）
- IN 消除了域内风格变化信息，但这些信息在跨域场景中是有用的

**3. V44 配置揭示：它融合了所有改进**
- DLA 多尺度聚合 + Pitch inference fusion（架构）
- mount_jitter=0.3, color_jitter=0.2, intrinsic=0.02（域增强）
- zero_perturbation_prob=0.1（zero_perturb，比 V43 的 0.05 更高）

**4. Shortcut 验证通过**
- Jacobian = 0.971（>0.85 阈值），矫正能力正常
- Seq09（数据泄漏序列）比值 0.68x，远高于 V43 的 0.23x-0.37x
- 泛化提升是全局性的，7/12 序列改善，不依赖数据泄漏

**5. V44-S1 Per-Sequence 分析**
- 改善最大: Seq04 (+70.7%), Seq06 (+61.7%), Seq05 (+55.5%)
- 退化: Seq03 (-25.6%), Seq10 (-77.0%), Seq09 (-86.1%)
- Seq09/Seq10 退化说明模型减少了对训练域 shortcut 的依赖

### 15.4 V44 vs V42 Per-Axis 改善

| 轴 | V42-32n-S1 | V44-S1 | 改善 |
|----|-----------|--------|------|
| Roll | 0.482° | 0.299° | **-38.0%** |
| Pitch | 0.525° | 0.426° | -18.9% |
| Yaw | 0.274° | 0.235° | -14.2% |
| Total | 0.850° | 0.649° | **-23.6%** |

### 15.5 距 0.1° 目标评估（更新）

| 轴 | V44-S1 BEST | 距 0.1° | 状态 |
|----|------------|---------|------|
| Roll | 0.275° | 2.8x | 需优化 |
| Pitch | 0.386° | 3.9x | 需优化 |
| Yaw | 0.199° | 2.0x | 需优化 |
| Total | 0.582° | 1.9x | 需优化 |

**更新路线图：**

```
V42 baseline       = 0.850° per-frame, 0.778° BEST
V44-S1 (achieved)  = 0.649° per-frame, 0.582° BEST  ← 当前最优
距 0.1° 目标仍需:    0.582° → 0.1° = 5.8x 改善
```

### 15.6 Bag 标定评估结果（2026-06-03 更新）

**重要：bag 评估结论与 test_data_v2 泛化评估存在显著差异！**

Bag 标定排名（Mean RotΔ，基于 2 个真实 bag：C01-81 + DE08-4）：

| 排名 | 模型 | Mean RotΔ | C01-81 | DE08-4 | IN |
|------|------|-----------|--------|--------|----|
| **1** | **V44-S2** | **0.356°** | 0.635 | 0.078 | **ON** |
| 2 | V44-S3 | 0.356° | 0.641 | 0.072 | ON |
| 3 | V43-S3 | 0.368° | 0.662 | 0.073 | - |
| 4 | V44-S1 | 0.369° | 0.634 | 0.104 | OFF |
| 5 | V43-S2 | 0.375° | 0.656 | 0.093 | - |

**Bag vs Test_data_v2 泛化评估对比：**

| 模型 | test_data_v2 Per-frame | test_data_v2 BEST | Bag RotΔ | 结论 |
|------|----------------------|-------------------|----------|------|
| V44-S1 (no IN) | **0.649°** | **0.582°** | 0.369° | test 最优 |
| V44-S2 (w/ IN) | 1.001° | 0.787° | **0.356°** | **bag 最优** |
| V44-S2-best-val | **0.763°** | **0.588°** | - | test 第 2 |

**关键发现：Instance Norm 在 bag 数据上没有退化，反而略优！**
- test_data_v2 上 V44-S1(0.649°) >>> V44-S2(1.001°)，IN 恶化 54%
- bag 评估上 V44-S1(0.369°) ≈ V44-S2(0.356°)，IN 略好 3.5%
- 说明 test_data_v2 可能高估了 IN 的负面影响
- V44-S2-best-val 在 test_data_v2 上的 BEST=0.588°（几乎追平 V44-S1 的 0.582°）

**Shortcut 抗性：所有模型全部通过**（RotΔ > 3.0° in shortcut test）

### 15.7 best_val vs best_dual Bag 标定对比（2026-06-03 完整版）

| 模型 | best_dual RotΔ | best_val RotΔ | 差异 | 胜者 |
|------|---------------|---------------|------|------|
| V43-S1 | 0.425° | 0.424° | -0.3% | ≈ 相同 |
| V43-S2 | 0.375° | 0.394° | +5.1% | dual |
| V43-S3 | 0.368° | 0.376° | +2.4% | dual |
| V43-from-v42S1-S2 | 0.434° | 0.384° | **-11.3%** | **val** |
| V43-from-v42S1-S3 | 0.383° | 0.411° | +7.3% | dual |
| V44-S1 (no IN) | 0.369° | 0.369° | 0.0% | 相同 |
| **V44-S2 (IN)** | **0.356°** | **0.424°** | **+18.8%** | **dual** |
| V44-S3 (IN) | 0.356° | 0.357° | +0.2% | ≈ 相同 |

**最重要发现：V44-S2 的 checkpoint 选择在不同数据集上有相反效果！**

| 数据集 | best_dual | best_val | 胜者 |
|--------|-----------|----------|------|
| test_data_v2 Per-frame | 1.001° | **0.763°** | **val (+24%)** |
| test_data_v2 BEST | 0.787° | **0.588°** | **val (+25%)** |
| Bag 标定 | **0.356°** | 0.424° | **dual (-16%)** |

解释：
- best_dual 针对 MEDW metric（多帧聚合稳定性）优化，在真实 bag 部署场景中更有效
- best_val 针对 val_rot（单帧旋转误差）优化，在 test_data_v2 上泛化更好
- V44-S1 和 V44-S3 两种 ckpt 结果一致，只有 V44-S2 差异大，可能与 S2 阶段 IN 初始化/优化路径有关

**这对 V44-opt 意味着什么：**
- "全程不用 IN" 的策略在 test_data_v2 上最优（V44-S1 独占前三）
- 但在 bag 部署场景中 IN 有轻微优势（V44-S2/S3 dual 是 bag 最优）
- **关键选择：优化 test_data_v2 泛化 还是 bag 部署？**
- V44-opt 策略：先优化 S1（无 IN），确保 test_data_v2 泛化最优作为 baseline
- 后续可做 V44-opt-S2-IN 对照组，评估 IN 在 bag 上的额外收益

---

## 16. V44-opt 优化方案设计 (2026-06-03)

### 16.1 设计依据

V44-S1 的泛化评估证明了以下关键发现：

| 发现 | 证据 | 优化方向 |
|------|------|---------|
| DLA + Pitch Fusion 有效 | V44-S1 首次超越 V20 | 保留 |
| Instance Norm 有害 | V44-S2(IN) 恶化 54% | **移除** |
| zero_perturb 有效 | V43/V44 Jacobian > 0.97 | 继续增加 |
| 域增强有效 | V44 7/12 seq 改善 | **加强** |
| Pitch 是瓶颈 | BEST: Pitch 0.386° vs Roll 0.275° | **增加 Pitch 权重** |
| S2/S3 fine-tune 无用（with IN） | V44-S3 比 V44-S1 差 | S2/S3 不用 IN |
| Shortcut 未被牺牲 | Jacobian 0.971, Seq09 比值 0.68x | 验证通过 |

### 16.2 V44-opt 与 V44 差异

| 参数 | V44-S1 | V44-opt |
|------|--------|---------|
| `axis_weights` | 1.0,4.0,1.0 | **1.5,4.0,1.5** (S1/S2), 2.0,4.0,2.0 (S3) |
| `pitch_aux_weight` | 0.3 | **0.4** |
| `zero_perturbation_prob` | 0.1 | **0.15** |
| `augment_mount_jitter_prob` | 0.3 | **0.4** |
| `augment_mount_jitter_rot_sigma` | 1.0 | **1.5** |
| `augment_color_jitter` | 0.2 | **0.3** |
| `augment_intrinsic` | 0.02 | **0.03** |
| `drop_path_rate` | 0.1 | **0.15** |
| `head_dropout` | 0.1 | **0.15** |
| `use_instance_norm` (S2/S3) | **1** | **0** |
| S1 `lr_schedule` | step | **cosine_warm_restarts** |
| S1 `num_epochs` | 150 | **200** |
| S1 warm-start from | V42-S1 best | **V44-S1 best_dual** |

### 16.3 优化策略解释

**1. 均衡提升 axis_weights: 1.0,4.0,1.0 → 1.5,4.0,1.5**
- 数据分析表明 Pitch weight=4x 已经不是瓶颈（P/Y=1.8 已最优均衡）
- 进一步增大 Pitch 权重有 Yaw 退化风险（跷跷板效应）
- 改为提升 Roll/Yaw 权重到 1.5，保持 Pitch 4x 不变
- 目标是三轴均衡下降而非单轴激进优化

**2. 更强的域增强**
- mount_jitter 从 0.3/σ=1.0 提高到 0.4/σ=1.5（增加外参变化多样性）
- color_jitter 从 0.2 提高到 0.3（更强的颜色鲁棒性）
- intrinsic 从 0.02 提高到 0.03（更大的焦距扰动范围）

**3. 从 V44-S1 best_dual warm-start**
- V44-S1 已经比 V42-S1 好得多，直接在其基础上继续优化
- 比从 V42 重新训练更高效

**4. Cosine Warm Restarts (T0=40, Tmult=2)**
- 替代 step scheduler，允许模型在多个重启周期中探索更多局部最优
- 200 epochs 配合 T0=40 会有约 3-4 个重启周期

**5. 更高的正则化**
- drop_path_rate: 0.1 → 0.15, head_dropout: 0.1 → 0.15
- 配合更强的增强，防止过拟合训练域

**6. S1 全程无 Instance Norm（S2 待定）**
- V44 在 test_data_v2 上 IN 伤害泛化，但 bag 评估中 IN 反而略优
- V44-opt-S1 先不用 IN 建立 baseline
- V44-opt-S2 是否用 IN 取决于 S1 泛化评估结果（可能做 IN/无IN 对照组）

### 16.4 预期效果

```
V44-S1 BEST       = 0.582° (Roll:0.275 Pitch:0.386 Yaw:0.199)
V44-opt 目标:
  Pitch 改善 15-25%: 0.386° → 0.29-0.33° (增加 Pitch 权重 + 更强增强)
  Roll  保持/微改:   0.275° → 0.26-0.28°
  Yaw   保持:        0.199° → 0.19-0.20°
  Total BEST ≈ 0.45-0.50°
```

### 16.5 训练配置

配置文件: `configs/v44opt_cf_bev_r_quick.yaml`

```bash
bash batch_train.sh configs/v44opt_cf_bev_r_quick.yaml
```

预计训练时间: ~7h (S1:200ep×70s + S2:120ep×70s + S3:80ep×70s)

---

## 17. V43/V44 全面评估总结 (2026-06-03 最终版)

### 17.1 综合排名总表

#### A. Test_data_v2 泛化排名（Per-frame / BEST 聚合）

| 排名 | 模型 | Per-frame | BEST | 改善率 | RPY分量(BEST) |
|------|------|-----------|------|--------|---------------|
| **1** | **V44-S1** (3个ckpt相同) | **0.649°** | **0.582°** | 10.3% | R:0.275 P:0.386 Y:0.199 |
| 2 | V44-S2-best-val | 0.763° | 0.588° | 22.9% | R:0.237 P:0.423 Y:0.200 |
| 3 | V43-S3-best-dual | 0.860° | 0.756° | 12.1% | R:0.323 P:0.579 Y:0.273 |
| 4 | V44-S3-best-dual/val (相同) | 1.011° | 0.781° | 22.7% | R:0.233 P:0.667 Y:0.204 |
| 5 | V43-from-v42S1-S3 | 0.903° | 0.786° | 13.0% | R:0.308 P:0.615 Y:0.276 |
| 6 | V44-S2-best-dual | 1.001° | 0.787° | 21.4% | R:0.231 P:0.674 Y:0.207 |
| 7 | V43-S2-best-dual | 0.832° | 0.798° | 4.1% | R:0.333 P:0.612 Y:0.274 |

#### B. Bag 标定排名（真实场景, 2个bags: C01-81 + DE08-4）

| 排名 | 模型 | Mean RotΔ | C01-81 | DE08-4 | Inject | Shortcut |
|------|------|-----------|--------|--------|--------|----------|
| **1** | **V44-S2 (w/IN)** | **0.356°** | 0.635 | 0.078 | 0.920 | 3.0 PASS |
| 2 | V44-S3 (w/IN) | 0.356° | 0.641 | 0.072 | 0.911 | 3.0 PASS |
| 3 | V43-S3 | 0.368° | 0.662 | 0.073 | 0.922 | 3.1 PASS |
| 4 | V44-S1 (no IN) | 0.369° | 0.634 | 0.104 | 0.907 | 3.5 PASS |
| 5 | V43-S2 | 0.375° | 0.656 | 0.093 | 0.935 | 3.2 PASS |

#### C. 训练精度排名（Val MEDW200 max(R,P,Y)）

| 排名 | 模型 | MEDW200 | Jacobian | Val Rot |
|------|------|---------|----------|---------|
| 1 | V43-S3 | 0.0413° | N/A | 0.20° |
| 2 | V44-S2 | 0.0428° | N/A | 0.20° |
| **3** | **V44-opt-S1** (训练中) | **0.0573°** | **0.949** | **0.24°** |
| 4 | V43-S1 | 0.0743° | 0.999 | 0.22° |
| 5 | V44-S1 | 0.0888° | 0.971 | 0.26° |

### 17.2 关键发现（颠覆性）

**发现 1：Instance Norm 效果是数据集依赖的（最重要发现）**

| 评估维度 | V44-S1 (no IN) | V44-S2 (w/IN) dual | V44-S2 (w/IN) val | 结论 |
|----------|----------------|---------------------|--------------------|----|
| test_data_v2 Per-frame | **0.649°** | 1.001° (+54%) | 0.763° (+18%) | IN 伤害 |
| test_data_v2 BEST | **0.582°** | 0.787° (+35%) | **0.588°** (+1%) | IN ≈ 中性(best_val) |
| Bag 标定 RotΔ | 0.369° | **0.356°** (-3.5%) | - | **IN 有益** |

- test_data_v2 高估了 IN 的负面影响
- 真实 bag 场景中 IN 反而略优
- best_val 选择策略可大幅弥补 IN 在 test 上的退化
- **说明：test_data_v2 和 bag 数据的域分布存在差异**

**发现 2：Checkpoint 选择策略极其关键**

| 模型 | best_dual | best_val | 差异 |
|------|-----------|----------|------|
| V44-S2 Per-frame | 1.001° | **0.763°** | **24% 改善** |
| V44-S2 BEST | 0.787° | **0.588°** | **25% 改善** |
| V43-S3 Per-frame | 0.860° | **0.835°** | 3% 改善 |

- IN 模型对 checkpoint 选择极度敏感
- best_dual 过度优化 MEDW metric，牺牲泛化
- **建议：IN 模型必须用 best_val 而非 best_dual**

**发现 3：V44 架构（DLA+PitchFusion）显著优于 V43**

| 对比维度 | V44-S1 | V43-S1 | V43-S3 (最好) | V44 优势 |
|----------|--------|--------|---------------|---------|
| Per-frame | **0.649°** | 0.887° | 0.860° | 24-27% |
| BEST | **0.582°** | 0.863° | 0.756° | 23-33% |
| Bag RotΔ | **0.369°** | 0.425° | 0.368° | 0-13% |

**发现 4：序列级难度分布**

| 难度 | Sequences | 特征 |
|------|-----------|------|
| 困难 (>1°) | Seq03, Seq07 | Pitch 系统性偏差 |
| 中等 (0.5-1°) | Seq02, Seq04, Seq08, Seq10 | 混合误差 |
| 简单 (<0.5°) | Seq00, Seq01, Seq05, Seq06, Seq09, Seq11 | - |

Seq07 和 Seq03 是所有模型的共同瓶颈（1.0-1.9° 误差），根本原因是 Pitch 系统性偏差。

### 17.3 V44-opt 训练中间状态 (Epoch 95/200)

| 指标 | V44-opt (best) | V44-S1 | 变化 |
|------|----------------|--------|------|
| MEDW200 max(R,P,Y) | **0.0573°** | 0.0888° | **-36%** |
| Jacobian | 0.949 | 0.971 | -2% |
| Val Rot | **0.24°** | 0.26° | -8% |
| RPY 均衡性 | R:0.036 P:0.057 Y:0.036 | R:0.049 P:0.077 Y:0.058 | 更均衡 |

**V44-opt 观察：**
1. 从 V44-S1 warm-start + 更强增强 → MEDW 大幅提升
2. 但 Epoch 41 后出现振荡（cosine restarts 导致）
3. Best checkpoint 已在 Epoch 41 保存，后续振荡不影响最终结果
4. 需等完成后做泛化评估才能判断是否真正改善

### 17.4 是否需要 V45？

**暂不需要，理由：**

1. **V44-opt 还在训练中**：MEDW 已改善 36%，但泛化效果未知
2. **V44 架构已被验证有效**：DLA+PF 是目前唯一超过 V20 baseline 的架构
3. **未探索完 V44 变体空间**：
   - V44-opt-S2（无 IN 版 S2 fine-tune）
   - V44-opt-S2-IN（有 IN 版 S2，基于 bag 数据表现）
   - V44-opt 加更多训练数据
4. **核心瓶颈是域偏移而非架构**：Seq03/Seq07 的 Pitch 系统偏差需要多域数据而非新架构

**V45 启动条件：**
- V44-opt 泛化评估 ≤ V44-S1 水平（即优化无效）
- 或发现 V44 架构存在无法通过超参/增强解决的根本缺陷
- 或需要引入全新模块（如 Test-Time Adaptation、多域对抗训练）

### 17.5 下一步行动

| 优先级 | 行动 | 预计时间 |
|--------|------|---------|
| P0 | 等 V44-opt-S1 训练完成（~3h） | 当天 |
| P0 | V44-opt 泛化评估 + bag 标定 | 1-2h |
| P1 | V44-opt-S2 设计（基于 S1 泛化结果决定是否用 IN） | 1h |
| P2 | V44-opt-S2 训练 + 评估 | 4-6h |
| P3 | 根据全部结果决定 V45 是否启动 | - |

---

## 18. Instance Norm 反向效果根因分析 (2026-06-03)

### 18.1 现象

V44-S2 (Instance Norm) 的 best_dual 和 best_val checkpoint 在不同评测数据集上表现完全相反：

| 数据集 | best_dual (Ep51) | best_val (Ep116) | 差异 |
|--------|-------------------|-------------------|------|
| test_data_v2 Per-frame | 1.001° | **0.763°** | val 好 24% |
| test_data_v2 BEST | 0.787° | **0.588°** | val 好 25% |
| Bag 标定 RotΔ | **0.356°** | 0.424° | **dual 好 16%** |

V44-S1 (无IN) 和 V44-S3 (IN继承) 不存在此现象。

### 18.2 根因

**Instance Norm 在 V44-S2 训练中创建了两个阶段：**

**早期 (Ep1-51)** — IN 参数正在适应:
- IN affine 参数（γ, β）还在学习中，保留了部分域特定信息
- 同一序列内的预测一致性高 → MEDW 稳定（0.043°）
- Jacobian 高（0.942），shortcut 抗性好
- 但跨域泛化还不够强（val_rot=0.25°）
- → `best_dual` 被保存在此阶段

**后期 (Ep51-120)** — IN 参数充分收敛:
- IN 完全学会去除域特定统计量（mean/var normalization）
- 跨域泛化增强（val_rot 持续下降到 0.20°）
- 但同一序列内一致性下降（MEDW 在 0.05-0.20° 间剧烈震荡）
- → `best_val` 被保存在此阶段

**本质矛盾：IN 创建了「域内一致性 vs 跨域泛化」tradeoff**

```
              IN收敛程度
    早期 ←─────────────→ 后期

  域内一致性: 高              低
  MEDW稳定性: 好              差(震荡)
  跨域泛化:   弱              强
  bag标定:    好              差
  test泛化:   差              好
```

### 18.3 V44-S3 不受影响的原因

V44-S3 从 V44-S2 加载 IN 参数后只做 ±1° 微调。S3 训练中 IN 变化极小，best_dual 和 best_val 在同一个 epoch（Epoch 1）。因此 bag 结果几乎一样（+0.2%）。

### 18.4 对 checkpoint 选择策略的影响

| 场景 | 推荐 checkpoint | 原因 |
|------|----------------|------|
| 真实部署（bag 标定） | best_dual | 需要域内多帧一致性 |
| 跨域泛化评估 | best_val | 需要跨域精度 |
| 同时需要两者 | 需要 V45 解决方案 | IN 无法同时满足 |

---

## 19. V45 设计方案：统一域内一致性与跨域泛化 (2026-06-03)

### 19.1 核心目标

**同时在 test_data_v2 泛化评估和 bag 标定部署中取得最优性能**，不再依赖 checkpoint 选择策略来切换优化目标。

量化目标：
- test_data_v2 BEST ≤ 0.55°（比 V44-S1 的 0.582° 改善 5%+）
- Bag 标定 RotΔ ≤ 0.35°（比 V44-S2-dual 的 0.356° 持平或更好）
- 两个指标在同一个 checkpoint 上同时达到

### 19.2 从第一性原理分析

为什么 IN 导致了 tradeoff？

1. **IN 的作用机制**：`IN(x) = (x - μ_instance) / σ_instance * γ + β`
   - 对每个 feature map 独立做零均值单位方差归一化
   - 移除了 instance 级别的统计量（=域/风格信息）
   - 保留了空间结构（=几何信息）

2. **问题所在**：IN 是全局开关——要么完全去除域信息，要么不去除
   - 对「颜色/光照」等域变量应该去除 ✓
   - 对「相机安装角度/焦距」等标定相关信息不应该去除 ✗
   - 但 IN 无法区分这两类信息

3. **理想解决方案**：选择性地去除"有害域信息"，保留"有用域信息"

### 19.3 V45 候选方案

#### 方案 A: Gated Instance Norm (GIN) — 推荐

```python
class GatedInstanceNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_norm = nn.InstanceNorm2d(channels, affine=True)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_normed = self.in_norm(x)
        gate = self.gate(x).unsqueeze(-1).unsqueeze(-1)
        return x * gate + x_normed * (1 - gate)
```

核心思想：让模型通过 learnable gate 自动决定每个 channel 的 IN 强度。
- gate ≈ 0: 完全应用 IN（去除域信息，跨域泛化）
- gate ≈ 1: 保留原始特征（保持域内一致性）
- 模型自动学习哪些 channel 需要域归一化，哪些需要保留

优点：
- 统一了跨域泛化和域内一致性
- 不需要额外训练技巧，end-to-end 学习
- 参数量增加极小（只增加一个线性层）

预期：同一个 checkpoint 可以同时在 bag 和 test_data_v2 上表现好

#### 方案 B: IBN-Net (Instance-Batch Normalization 混合)

```python
class IBNBlock(nn.Module):
    def __init__(self, channels, ratio=0.5):
        split = int(channels * ratio)
        self.IN = nn.InstanceNorm2d(split, affine=True)
        self.BN = nn.BatchNorm2d(channels - split)

    def forward(self, x):
        x_in, x_bn = x.split([split, channels-split], dim=1)
        return torch.cat([self.IN(x_in), self.BN(x_bn)], dim=1)
```

核心思想：前半 channels 用 IN（去域），后半 channels 用 BN（保留统计信息）。
- 仅在浅层（stem + stage1/2）使用 IBN，深层保持 BN/LN
- 理论：浅层特征包含更多域信息（纹理/颜色），深层特征更语义化

优点：
- 架构简洁，不增加推理开销
- 在 ImageNet 和 DG benchmark 上验证有效
- 不需要 gate 学习

缺点：
- ratio 是超参数，需要搜索
- 不如 GIN 灵活（fixed split vs learned gate）

#### 方案 C: Consistency-Regularized Training (CRT)

不修改架构，在训练中加入 MEDW 一致性正则化：

```python
loss_total = loss_rotation + λ_consist * loss_medw_consistency
```

其中 `loss_medw_consistency` 惩罚同一序列内多帧预测的方差：

```python
def medw_consistency_loss(predictions_per_seq):
    """对同一序列内的预测计算一致性损失"""
    losses = []
    for seq_preds in predictions_per_seq:
        mean_pred = seq_preds.mean(dim=0)
        var = ((seq_preds - mean_pred) ** 2).mean()
        losses.append(var)
    return torch.stack(losses).mean()
```

优点：
- 不改变模型架构，适用于任何 IN 配置
- 直接优化我们关心的 MEDW 指标
- 可以和方案 A/B 组合使用

缺点：
- 需要训练 batch 中来自同一序列的多帧（当前 dataloader 可能不支持）
- 增加 GPU 内存需求

#### 方案 D: Domain Augmentation + Hard Sequence Mining

不使用 IN，而是通过更强的数据增强来实现跨域泛化：

- **序列级增强**：对同一序列的所有帧施加一致的域变换（色彩/曝光/对比度）
- **Hard Sequence Mining**：识别困难序列（Seq03/Seq07）并在训练中增加其权重
- **Camera Intrinsic Augmentation**：更大范围的焦距/光心扰动
- **Multi-Camera Training**：混合不同相机型号的数据

优点：
- 不修改模型架构
- 增强数据多样性 → 提升泛化
- 避免 IN 带来的 tradeoff

缺点：
- 可能无法完全替代 IN 的域对齐效果
- 需要 Hard Sequence 标注或自动识别

### 19.4 推荐实施路线

| 阶段 | 方案 | 依赖 | 预计时间 | 优先级 |
|------|------|------|---------|--------|
| V45-Phase1 | 方案 A (GIN) 替换 V44 的 IN | V44-opt baseline | 0.5 天 | **P0** |
| V45-Phase2 | 方案 C (CRT) 加入一致性正则化 | Phase1 完成 | 0.5 天 | P1 |
| V45-Phase3 | 方案 D (Hard Mining) 加强数据增强 | 独立 | 0.5 天 | P1 |
| V45-Phase4 | 方案 B (IBN) 作为 GIN 的轻量级对照 | Phase1 完成 | 0.5 天 | P2 |

### 19.5 V45 训练配置草案

基于 V44-opt-S1 的最优配置，主要变更：

```yaml
# V45 Phase1: GIN 替换 IN
use_instance_norm: 0                # 不使用全局 IN
use_gated_instance_norm: 1          # 使用 GIN (新参数)
gin_layers: "fpn,dla"               # 在 FPN 和 DLA 输出后应用 GIN
gin_init_gate: 0.5                  # gate 初始值 0.5 (平衡 IN 和 bypass)

# V45 Phase2: 一致性正则化 (可选)
consistency_loss_weight: 0.1        # MEDW 一致性正则化权重
consistency_seq_batch: 4            # 每个 batch 中同一序列的帧数

# 保留 V44-opt 的所有其他优化
axis_weights: "1.5,4.0,1.5"
pitch_aux_weight: 0.4
zero_perturbation_prob: 0.15
augment_mount_jitter_prob: 0.4
augment_color_jitter: 0.3
drop_path_rate: 0.15
head_dropout: 0.15
lr_schedule: cosine_warm_restarts
```

### 19.6 成功标准

| 指标 | V44-S1 baseline | V44-S2 dual | V45 目标 |
|------|-----------------|-------------|---------|
| test_data_v2 Per-frame | **0.649°** | 1.001° | **≤ 0.62°** |
| test_data_v2 BEST | **0.582°** | 0.787° | **≤ 0.55°** |
| Bag RotΔ | 0.369° | **0.356°** | **≤ 0.35°** |
| MEDW200 max(R,P,Y) | 0.089° | 0.043° | ≤ 0.06° |
| Jacobian | 0.971 | 0.942 | ≥ 0.93 |

关键：**test_data_v2 和 Bag 指标在同一个 checkpoint 上同时达到**

### 19.7 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| GIN gate 塌缩到全 0 或全 1 | 中 | 退化为纯 IN 或无 IN | 加 gate regularization loss |
| 一致性正则化与旋转 loss 冲突 | 低 | 训练不收敛 | 用小权重 λ=0.1 |
| GIN 增加推理延迟 | 低 | 部署性能 | GIN 参数极少，negligible |
| Hard Mining 过拟合困难序列 | 中 | 简单序列退化 | 用 curriculum learning |
