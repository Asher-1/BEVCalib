# BEVCalib 泛化性能改进实施计划

**基于**: 诊断报告 + DESIGN_v23_generalization.md + DST-Calib 论文调研
**目标**: 突破当前 0.65° 架构地板，向 < 0.1° RPY 推进
**日期**: 2026-04-25 (初版) → 2026-04-30 (V3 更新: BEV Diff Fuser + Camera Dropout 详细实施) → 2026-05-03 (V4: DDP 兼容性修复) → 2026-05-04 (V5: 多帧融合实验 + V25 突破计划)

---

## 当前架构数据流

```
img (B,3,H,W) → Cam2BEV(SwinT+LSS) → cam_bev_feats (B,128,96,96)
                                                                    ↘
                                                      Fuser(可切换) → + pose_embed
                                                                    ↗           ↓
pc  (B,N,3)   → Lidar2BEV(spconv)    → pc_bev_feats  (B,128,96,96)   DeformableTransformer(×2)
                                                                                ↓
                                                                        masked_avg_pool → head_drop
                                                                                ↓
                                                                    rotation_pred(4) / translation_pred(3)
```

**Fuser 可切换**: ConvFuser (cat) / BEVDiffFuser (diff) / BEVDiffFuser-v2 (diff + cam-drop-aware dual path)

**诊断问题**: `cam_bev_feats` 跨场景完全一致（变化 <0.04%），实为固定 FOV 模板，不编码标定信号。

---

## Phase 1: BEV Diff Fuser + Camera Dropout（详细实施方案）

### 1.0 实现状态总览

| 组件 | 状态 | 位置 |
|------|------|------|
| BEVDiffFuser (v1, 单路径) | ✅ 已实现 | `bev_calib.py` L42-80 |
| Camera Dropout (batch级, zero模式) | ✅ 已实现 | `bev_calib.py` L384-395 |
| fuser_type 参数传递链路 | ✅ 完整 | train_kitti → batch_train → YAML |
| 训练/评估 YAML 配置 | ✅ 4组 | batch_train_phase1_phase2.yaml |
| BEVDiffFuser v2 (双路径, cam-drop-aware) | ✅ 已实现 | `bev_calib.py` L42-87 (cam_drop_aware + pc_only_conv) |
| cam_drop_mode (noise 模式) | ✅ 已实现 | `bev_calib.py` L390-392 |
| 统一 Fuser forward 签名 | ✅ 已实现 | ConvFuser.forward + BEVDiffFuser.forward 均支持 cam_dropped |
| 6→8 组 V24 实验配置 | ✅ 已创建 | batch8_train_all_v24.yaml / v24_quick / eval_v24 / eval_v24_quick |
| DDP 兼容性 (cam_drop + fuser) | ✅ 已修复 | `train_kitti.py` L1013 + `bev_calib.py` L386 (详见 §1.0.1) |

### 1.0.1 DDP 兼容性修复 (2026-05-03)

**问题**: Camera Dropout 触发时，部分模型参数脱离计算图，导致 DDP `RuntimeError: Expected to have finished reduction` 错误。

**根因分析 (两个独立问题)**:

| 根因 | 受影响实验 | 具体机制 |
|------|-----------|---------|
| noise 模式 `.detach()` 断图 | F (diff+noise) | `cam_bev_feats.std().detach()` + `torch.randn_like()` 创建全新张量，SwinT ~287个参数脱离计算图 |
| diff_v2 `pc_only_conv` 忽略 `img_bev_feat` | E, H (diff_v2+zero) | `cam_bev_feats*0` 保留图连接，但 fuser 的 `pc_only_conv` 路径完全不消费 `img_bev_feat`，camera encoder + `self.conv` 参数脱离 |

**不受影响的实验**: A, B (无 cam_drop), C, D (zero+concat/diff_v1: `cam_bev_feats*0` 被 fuser 正常消费), G (无 cam_drop)

**修复方案 (PyTorch 官方推荐: `find_unused_parameters=True`)**:

1. **`train_kitti.py` DDP 初始化** — 条件性启用:
   ```python
   need_find_unused = getattr(args, 'cam_drop_prob', 0) > 0
   model = DDP(model, device_ids=[local_rank], find_unused_parameters=need_find_unused)
   ```
   - `cam_drop_prob > 0` (实验 C, D, E, F, H) → `True`，允许 DDP 跳过未使用参数
   - `cam_drop_prob = 0` (实验 A, B, G) → `False`，零额外开销

2. **noise 模式** (`bev_calib.py` BEVCalib.forward) — 保持 camera encoder 在计算图中:
   ```python
   # 修复前: cam_bev_feats = torch.randn_like(cam_bev_feats) * noise_scale
   # 修复后: cam * 0 保持图连接，+ noise 提供替代特征
   cam_bev_feats = cam_bev_feats * 0 + torch.randn_like(cam_bev_feats) * noise_scale
   ```

**方案演进**: 初始使用 dummy parameter (`p.sum() * 0`) 手动注册计算图，后改为 `find_unused_parameters=True` — 更安全，PyTorch 官方推荐，且仅对 cam_drop 实验生效 (~10% DDP 通信开销)。

**精度影响**: 无。`find_unused_parameters=True` 只影响 DDP 的 allreduce 行为，不改变数值计算。已成功运行的实验 (A, B, C, D, G) 完全不受影响。

### 1.1 核心设计: BEVDiffFuser v2 (双路径, cam-drop-aware)

#### 1.1.1 问题分析: BEVDiffFuser + cam_drop 的交互退化

当 Camera Dropout 触发 (`cam_bev_feats = 0`) 时，BEVDiffFuser v1 的输入退化为:

| 通道 | cam 正常时 | cam 被 drop 时 (=0) | 问题 |
|------|----------|-------------------|------|
| pc_bev_feat | pc (128ch) | pc (128ch) | ✓ 正常 |
| \|cam - pc\| | 标定误差信号 | \|0 - pc\| = \|pc\| | ≈ pc 的绝对值（冗余） |
| cam * pc | 对齐增强 | 0 * pc = 0 | 完全浪费 |

结果: `[pc, |pc|, 0]` → 1/3 容量浪费，2/3 近乎冗余。

> **注**: `|pc|` 与 `pc` 并非完全相同 — pc_bev_feat 经过 conv 层可能有负值,
> 取绝对值会丢失符号信息。但信息冗余度仍然很高。

#### 1.1.2 解决方案: 双路径 Fuser

BEVDiffFuser v2 在 cam 被 drop 时走独立的 `pc_only_conv`，避免退化输入:

```
cam 正常:   [pc, |cam-pc|, cam*pc] → 384ch → conv     → 256ch (正常路径)
cam 被drop: pc                     → 128ch → pc_only_conv → 256ch (降级路径)
```

**设计决策**:

| 决策 | 选择 | 理由 |
|------|------|------|
| 双路径 vs 单路径+noise | 双路径 | 彻底避免退化输入；noise 方案作为对照实验 |
| pc_only_conv 结构 | 1x1 Conv + BN + ReLU | 与正常路径一致结构，仅输入维度不同 |
| pc_only_conv BN | 独立 BN | 训练时约 30% batch 走此路径，BN 统计独立 |
| 推理时路径选择 | 永远走正常路径 | cam 推理时始终可用，pc_only_conv 不参与推理 |
| cam_drop_aware 开关 | 构造器参数 | fuser_type="diff" 不启用，fuser_type="diff_v2" 启用 |

#### 1.1.3 统一 Fuser Forward 签名

**问题**: ConvFuser (nn.Sequential) 和 BEVDiffFuser (nn.Module) 的 forward 签名不一致，
需要 isinstance 检查来传递 `cam_dropped` 参数。

**解决方案**: 给 ConvFuser.forward() 加 `cam_dropped=False` 参数并忽略。
ConvFuser 已覆写了 forward()，加参数不影响 state_dict，不破坏现有调用:

```python
class ConvFuser(nn.Sequential):
    # __init__ 完全不变 (保留 nn.Sequential 继承，保证 state_dict 兼容)

    def forward(self, img_bev_feat, pc_bev_feat, cam_dropped=False):
        # cam_dropped accepted but ignored — ConvFuser has no dual-path
        return super().forward(torch.cat([img_bev_feat, pc_bev_feat], dim=1))
```

统一后的调用:

```python
# bev_calib.py forward() — 无需 isinstance 检查
x = self.conv_fuser(cam_bev_feats, pc_bev_feats, cam_dropped=cam_dropped)

# bevcalib_inference.py — 无需改动 (cam_dropped 默认 False)
x = m.conv_fuser(cam_bev_feats, pc_bev_feats)
```

**向后兼容**:
- 旧代码调用 `conv_fuser(cam, pc)` → cam_dropped 默认 False ✓
- 旧 checkpoint state_dict keys 不变 (nn.Sequential 内部结构未改) ✓
- 新代码调用 `conv_fuser(cam, pc, cam_dropped=True)` → 两种 Fuser 都能处理 ✓

### 1.2 核心设计: Camera Dropout 增强

#### 1.2.1 当前实现 (保留)

```python
# batch 级别 cam_drop，DDP broadcast 确保所有 rank 一致
if self.training and self.cam_drop_prob > 0:
    drop_flag = torch.rand(1, device=cam_bev_feats.device)
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(drop_flag, src=0)
    if drop_flag.item() < self.cam_drop_prob:
        cam_bev_feats = cam_bev_feats * 0
```

**保留 batch 级别的理由**:
- DDP 一致性更简单（所有 rank 同一决策，梯度同步无歧义）
- 模型明确学习两种模式（全有 cam vs 全无 cam），而非混合中间态
- 与 BEVDiffFuser v2 双路径设计更匹配（一个 batch 只走一条路径）

#### 1.2.2 新增 cam_drop_mode: "zero" / "noise"

| 模式 | 行为 | 适用场景 |
|------|------|---------|
| zero | `cam_bev_feats * 0` | 默认。简洁，模型学习 cam-absent 模式 |
| noise | `randn_like * (std * 0.1)` | 实验性。提供非零梯度流，更强正则化 |

**noise 模式设计**:

```python
if self.cam_drop_mode == "noise":
    noise_scale = cam_bev_feats.std().detach().clamp(min=1e-6) * 0.1
    cam_bev_feats = torch.randn_like(cam_bev_feats) * noise_scale
else:  # "zero"
    cam_bev_feats = cam_bev_feats * 0
```

**noise_scale = 0.1 * std 的理由**:
- 0.1 倍标准差 → 信噪比 10:1，noise 是弱信号，不会主导梯度
- 使用 `.detach()` 阻断 noise_scale 的梯度 → noise 只作为输入扰动，不影响 cam 分支参数更新
- `.clamp(min=1e-6)` 防止 std≈0 时 noise 消失

**noise 与 DiffFuser v1 的交互**:
当 cam_drop_mode="noise" 且 cam_dropped=True 时:
- `diff = |noise - pc|` → 随机差异信号（非退化）
- `interact = noise * pc` → 随机交互信号（非零）
- 模型必须学习在这种噪声输入下仍产出合理预测 → **更强正则化**

**noise 与 DiffFuser v2 的交互**:
cam_drop_mode="noise" 时 `cam_dropped=False`（cam 非完全缺失），
因此 diff_v2 走**正常路径**: `[pc, |noise-pc|, noise*pc]`。
这是一个有效组合，模型需要从噪声化的 diff/interact 信号中仍产出合理预测，
提供比 zero 模式更强的正则化（因为有非零梯度流经 cam 参数路径）。

### 1.3 代码改动详述

#### 1.3.1 `bev_calib.py` — 完整改动

**改动 1: ConvFuser.forward() 统一签名**

```python
# 原: def forward(self, img_bev_feat, pc_bev_feat):
# 新:
def forward(self, img_bev_feat, pc_bev_feat, cam_dropped=False):
    return super().forward(torch.cat([img_bev_feat, pc_bev_feat], dim=1))
```

**改动 2: BEVDiffFuser 新增双路径**

```python
class BEVDiffFuser(nn.Module):
    def __init__(self, img_in_channel, pc_in_channel, out_channel,
                 cam_drop_aware=False):
        super().__init__()
        assert img_in_channel == pc_in_channel, (
            f"BEVDiffFuser requires equal channel dims, "
            f"got img={img_in_channel} vs pc={pc_in_channel}")
        in_ch = pc_in_channel * 3
        self.cam_drop_aware = cam_drop_aware
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
        if cam_drop_aware:
            self.pc_only_conv = nn.Sequential(
                nn.Conv2d(pc_in_channel, out_channel, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True)
            )

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, img_bev_feat, pc_bev_feat, cam_dropped=False):
        if self.cam_drop_aware and cam_dropped:
            return self.pc_only_conv(pc_bev_feat)
        diff = (img_bev_feat - pc_bev_feat).abs()
        interact = img_bev_feat * pc_bev_feat
        return self.conv(torch.cat([pc_bev_feat, diff, interact], dim=1))
```

**改动 3: BEVCalib.__init__() 新增 fuser_type="diff_v2" 和 cam_drop_mode**

```python
def __init__(self, ..., fuser_type="concat", cam_drop_prob=0.0,
             cam_drop_mode="zero", ...):
    ...
    self.cam_drop_mode = cam_drop_mode

    if fuser_type == "diff_v2":
        self.conv_fuser = BEVDiffFuser(
            self.img_branch.out_channels,
            self.pc_branch.out_channels,
            self.embed_dim,
            cam_drop_aware=True,
        )
    elif fuser_type == "diff":
        self.conv_fuser = BEVDiffFuser(
            self.img_branch.out_channels,
            self.pc_branch.out_channels,
            self.embed_dim,
            cam_drop_aware=False,
        )
    else:
        self.conv_fuser = ConvFuser(
            self.img_branch.out_channels,
            self.pc_branch.out_channels,
            self.embed_dim,
        )
```

**改动 4: BEVCalib.forward() cam_drop 逻辑 + 统一 fuser 调用**

```python
# Camera Dropout (替换原 L354-359)
cam_dropped = False
if self.training and self.cam_drop_prob > 0:
    drop_flag = torch.rand(1, device=cam_bev_feats.device)
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(drop_flag, src=0)
    if drop_flag.item() < self.cam_drop_prob:
        if self.cam_drop_mode == "noise":
            noise_scale = cam_bev_feats.std().detach().clamp(min=1e-6) * 0.1
            cam_bev_feats = torch.randn_like(cam_bev_feats) * noise_scale
            # cam_dropped 保持 False: cam 被替换为噪声，非完全缺失
            # → fuser 走正常路径处理噪声 cam (diff/interact 均为随机信号)
        else:
            cam_bev_feats = cam_bev_feats * 0
            cam_dropped = True  # cam 完全缺失 → diff_v2 走 pc_only_conv

# Fuser 调用 — 统一签名，无需 isinstance 检查
x = self.conv_fuser(cam_bev_feats, pc_bev_feats, cam_dropped=cam_dropped)
```

**cam_dropped 语义**: `True` 仅表示 "cam 完全缺失"（zero 模式），`False` 包含 "cam 正常" 和 "cam 被替换为噪声" 两种情况。
这使得 diff_v2 + noise 成为有效组合（noise 通过正常 diff 路径处理）。

#### 1.3.2 `train_kitti.py` — 参数改动

```python
# fuser_type choices 新增 "diff_v2"
parser.add_argument("--fuser_type", type=str, default="concat",
                    choices=["concat", "diff", "diff_v2"],
                    help="BEV fuser: concat(ConvFuser) / diff(BEVDiffFuser) "
                         "/ diff_v2(BEVDiffFuser with cam-drop-aware dual path)")

# 新增 cam_drop_mode
parser.add_argument("--cam_drop_mode", type=str, default="zero",
                    choices=["zero", "noise"],
                    help="Camera dropout mode: zero(hard zeros) / noise(Gaussian noise)")

# BEVCalib 构造增加 cam_drop_mode
model = BEVCalib(
    ...
    fuser_type=args.fuser_type,
    cam_drop_prob=args.cam_drop_prob,
    cam_drop_mode=args.cam_drop_mode,
    ...
)
```

#### 1.3.3 `evaluate_checkpoint.py` — choices 更新

```python
# L2113-2114: fuser_type choices 新增 "diff_v2"
parser.add_argument("--fuser_type", type=str, default=None,
                   choices=["concat", "diff", "diff_v2"],
                   help="BEV fuser type (auto-detected from checkpoint if omitted)")
```

`_resolve_model_params_from_ckpt` 中的 fuser_type 自动检测 (`'fuser_type': ('fuser_type', 'concat')`)
**无需改动** — 从 ckpt args 读取 "diff_v2" 并传递给 BEVCalib。

`cam_drop_mode` **不需要添加到 evaluate_checkpoint.py** — 仅训练时有效，评估/推理时 cam 不会被 drop。

#### 1.3.4 `bevcalib_inference.py` — 无需改动

```python
# L67: 推理调用不变 (cam_dropped 默认 False)
x = m.conv_fuser(cam_bev_feats, pc_bev_feats)
# ConvFuser: 忽略 cam_dropped=False ✓
# BEVDiffFuser v1/v2: cam_dropped=False → 走正常路径 ✓
```

`load_bevcalib_inference()` 从 ckpt args 读取 `fuser_type` 并传给 BEVCalib 构造器，
**前提**: BEVCalib 已支持 "diff_v2"（在 1.3.1 中已完成）。

#### 1.3.5 Shell 脚本改动

**`train_universal.sh`**: 新增 `--cam_drop_mode` 解析

```bash
# 参数解析区新增:
--cam_drop_mode)
    CAM_DROP_MODE="$2"; shift 2 ;;

# OPTIM_FLAGS 拼接区新增:
[ -n "$CAM_DROP_MODE" ] && OPTIM_FLAGS="$OPTIM_FLAGS --cam_drop_mode $CAM_DROP_MODE"
```

**`start_training.sh`**: 同上模式

```bash
--cam_drop_mode)
    CAM_DROP_MODE="$2"; shift 2 ;;

# 两处 OPTIM_ARGS 拼接区 (非A30和A30) 各新增一行:
[ -n "$CAM_DROP_MODE" ] && OPTIM_ARGS="$OPTIM_ARGS --cam_drop_mode $CAM_DROP_MODE"
```

**`batch_train.sh`**: OPTIM_PARAMS 新增映射

```bash
('cam_drop_mode', '--cam_drop_mode'),
```

### 1.4 实验矩阵 (6 组)

#### 1.4.1 实验设计

| ID | 名称 | fuser_type | cam_drop_prob | cam_drop_mode | 验证目标 |
|----|------|-----------|---------------|---------------|---------|
| A | baseline | concat | 0.0 | - | 对照基线 |
| B | diff_only | diff | 0.0 | - | DiffFuser v1 单独效果 |
| C | drop_only | concat | 0.3 | zero | CamDrop 单独效果 |
| D | diff_drop_v1 | diff | 0.3 | zero | DiffFuser v1 + CamDrop (退化问题) |
| E | diff_drop_v2 | diff_v2 | 0.3 | zero | **双路径优化** (cam→pc_only_conv) |
| F | diff_drop_noise | diff | 0.3 | noise | **噪声替代** (cam→noise+正常diff) |

#### 1.4.2 关键对比链

```
DiffFuser 效果:           A vs B    (fuser_type: concat → diff)
CamDrop 效果:             A vs C    (cam_drop: 0 → 0.3)
组合效果:                 A vs D    (diff + cam_drop)
双路径优化 vs 退化输入:   D vs E    (diff 的 cam_dropped 处理方式)
噪声替代 vs 硬零:        D vs F    (cam_drop_mode: zero → noise)
最优组合:                E vs F    (双路径 vs 噪声, 哪个更好)
```

#### 1.4.3 共享训练超参 (基于 V22-B1 冠军配置)

```yaml
defaults:
  params:
    # 基础
    rotation_only: false         # joint 优化 (B1 冠军策略)
    lr: 0.0001
    backbone_lr_multiplier: 0.5  # A2 LR 配置
    no_amp: 1                    # FP32 训练

    # 正则化
    drop_path_rate: 0.1
    head_dropout: 0.1

    # 数据
    pose_aware_sampling: true
    max_frames_per_seq: 10000
    angle_range: 5

    # 架构
    z_voxels: 5
    voxel_mode: "scatter"
    scatter_reduce: "mean"
    to_bev_mode: "concat"

    # 训练
    epochs: 400
    eval_epoches: 50
    batch_size: 16

    # Fuser + CamDrop 默认 (被各实验 params 覆盖)
    fuser_type: "concat"
    cam_drop_prob: 0.0
    cam_drop_mode: "zero"
```

#### 1.4.4 YAML 配置 (单文件 batch_train_diffcamdrop.yaml)

```yaml
experiments:
  # A: baseline — concat fuser, no cam dropout
  - name: "dc_A_baseline"
    description: "A: baseline (concat, no drop)"
    version: "dc_A_baseline"

  # B: diff fuser only
  - name: "dc_B_diff_only"
    description: "B: BEVDiffFuser v1 only"
    version: "dc_B_diff_only"
    params:
      fuser_type: "diff"

  # C: cam dropout only
  - name: "dc_C_drop_only"
    description: "C: concat + cam_drop=0.3"
    version: "dc_C_drop_only"
    params:
      cam_drop_prob: 0.3

  # D: diff + dropout v1 (退化问题: [pc, |pc|, 0])
  - name: "dc_D_diff_drop_v1"
    description: "D: diff v1 + cam_drop=0.3 (退化 [pc,|pc|,0])"
    version: "dc_D_diff_drop_v1"
    params:
      fuser_type: "diff"
      cam_drop_prob: 0.3

  # E: diff + dropout v2 (双路径优化: cam→pc_only_conv)
  - name: "dc_E_diff_drop_v2"
    description: "E: diff v2 (dual-path) + cam_drop=0.3"
    version: "dc_E_diff_drop_v2"
    params:
      fuser_type: "diff_v2"
      cam_drop_prob: 0.3

  # F: diff + dropout noise (噪声替代: cam→noise+diff路径)
  - name: "dc_F_diff_drop_noise"
    description: "F: diff v1 + cam_drop=0.3 noise mode"
    version: "dc_F_diff_drop_noise"
    params:
      fuser_type: "diff"
      cam_drop_prob: 0.3
      cam_drop_mode: "noise"
```

### 1.5 兼容性矩阵

#### 1.5.1 Checkpoint 兼容性

| 场景 | 结果 | 说明 |
|------|------|------|
| 旧 ckpt (fuser_type=concat) + 新代码 | ✅ | 走 ConvFuser，cam_drop_mode 默认 "zero" |
| 旧 ckpt (fuser_type=diff) + 新代码 | ✅ | BEVDiffFuser v1，cam_drop_aware=False |
| 新 ckpt (fuser_type=diff_v2) + 新代码 | ✅ | BEVDiffFuser v2，cam_drop_aware=True |
| 新 ckpt (fuser_type=diff_v2) + 旧代码 | ❌ | 旧 BEVCalib 不识别 "diff_v2"（预期行为） |

#### 1.5.2 State Dict 兼容性

| fuser_type | State Dict Keys | 特殊参数 |
|------------|----------------|---------|
| concat | `conv_fuser.0.weight`, `conv_fuser.1.weight`, ... | 无 (nn.Sequential 直接子模块) |
| diff | `conv_fuser.conv.0.weight`, `conv_fuser.conv.1.weight`, ... | 无 |
| diff_v2 | 同 diff + `conv_fuser.pc_only_conv.0.weight`, ... | `pc_only_conv` 参数 (~130K 额外) |

#### 1.5.3 推理链路验证

```
训练:   bev_calib.py forward()  → conv_fuser(cam, pc, cam_dropped=T/F) ✓
推理:   bevcalib_inference.py   → m.conv_fuser(cam, pc)  (cam_dropped=False) ✓
评估:   evaluate_checkpoint.py  → 同推理路径 ✓
导出:   bevcalib_inference.py prepare_for_drinfer_export → m.conv_fuser(cam, pc) ✓
```

### 1.6 验证标准

#### 1.6.1 代码验证 (实施后、训练前)

| 检查项 | 方法 |
|--------|------|
| 旧 checkpoint 加载 | 加载 v22-B1 ckpt，确认 `strict=False` 无 missing key |
| diff_v2 模型构造 | 确认 `pc_only_conv` 在 `model.state_dict()` 中 |
| forward 正确性 | batch_size=2 的 dummy 输入，确认输出 shape 正确 |
| cam_drop 触发 | `model.training=True, cam_drop_prob=1.0`，确认 cam_bev_feats=0 |
| DDP 一致性 | 2 rank DDP，确认 drop_flag broadcast 正确 |
| 推理路径 | `model.eval()` 后 forward，确认不 drop cam |

#### 1.6.2 训练验证 (6组实验的评估指标)

| 指标 | baseline 预期 | 目标 |
|------|-------------|------|
| Val Rot Error | ~0.20° | 收敛到类似水平 |
| Gen Rot Error | ~0.665° | < 0.60° (改善 >10%) |
| Camera Bypass 退化 | < 5% (camera 几乎没用) | > 15% (证明 camera 贡献增加) |
| 训练收敛速度 | ~200 epoch 收敛 | ≤ 250 epoch |

#### 1.6.3 实验成功标准

```
B > A (diff fuser 有效):        Gen Rot 改善 > 3%
C > A (cam drop 有效):          Gen Rot 改善 > 2%
E > D (双路径优于退化):          Gen Rot 改善 > 1%
max(E, F) > D (优化方案有效):    Gen Rot 改善 > 2%
best(B,C,D,E,F) < 0.60°:       突破 0.65° 地板
```

### 1.7 已知风险和缓解

| 风险 | 影响 | 概率 | 缓解 |
|------|------|------|------|
| DiffFuser 训练不收敛 | 高 | 低 | 仅作调试用 warm start；正式实验均从随机初始化开始 |
| pc_only_conv BN 统计不准 (仅 30% batch 训练) | 中 | 中 | pc_only_conv 仅训练时使用，不影响推理 |
| 双路径导致 loss 震荡 | 中 | 低 | 监控 loss 曲线；必要时降低 cam_drop_prob 至 0.15 |
| noise_scale 需要调参 | 低 | 中 | 默认 0.1*std 保守设置；结果不理想时尝试 0.3 |
| 6 组实验训练时间长 (6 × ~2天) | 低 | 高 | Quick 模式先跑 (1N, ~6h/实验)，多 GPU 并行 3 组 × 2 轮 ≈12h；筛选后 Full 模式跑 top 3 |
| BEV 空间 Diff Map 效果弱于图像空间 | 中 | 中 | 如 Phase 1 效果不佳，Phase 4 考虑图像空间方法 |

---

## Phase 2: Camera Branch 特征多样性强化 (Phase 1 结果后)

(保留原设计方向，待 Phase 1 实验结果后细化)

### 2.1 核心思路

如果 Phase 1 证明 DiffFuser 有效（Camera 贡献增加），但泛化仍未达标，
则进一步修复 Camera 分支的特征退化问题:

- 傅里叶域增强 (FCVL 风格): 在频域交换 Camera BEV 特征的风格
- 特征多样性约束: 添加 loss 惩罚 cam_bev_feats 的跨样本一致性

### 2.2 依赖 Phase 1 的决策点

```
Phase 1 结果 < 0.55° → Phase 2 优先级降低 (DiffFuser 已足够有效)
Phase 1 结果 0.55-0.62° → Phase 2 正常推进 (Camera 修复是关键杠杆)
Phase 1 结果 > 0.62° → Phase 2 紧急推进 + 考虑 Phase 4 DST-Calib 式架构
```

---

## Phase 3: 双边 BEV 增强 (中等改造, 依赖 Phase 1 结果)

(保留原设计，依赖 Phase 1/2 结果)

### 3.1 核心思路

在训练时，不仅扰动 LiDAR→Camera 变换（当前做法），还扰动 Camera 的虚拟视角:
1. 用 Depth Anything V2 估计每帧的稠密深度
2. 将深度图转换为 3D 点云
3. 从随机虚拟视角重新渲染 Camera 深度投影
4. 与 LiDAR 的随机投影配对训练

### 3.2 决策点

Phase 2 结果将决定是否需要 Phase 3:
- 如果 Phase 2 达到 < 0.40° → Phase 3 可选
- 如果 Phase 2 仅到 ~0.50° → 考虑 Phase 3

---

## Phase 4: 完整 DST-Calib 式架构 (大改造，可选)

(保留原设计作为长期参考)

---

## 实施文件改动清单

### 必需改动 (Phase 1)

| 优先级 | 文件 | 改动内容 | 工作量 |
|--------|------|---------|--------|
| P0 | `bev_calib.py` | ConvFuser 签名 + BEVDiffFuser v2 + cam_drop_mode | 中 |
| P0 | `train_kitti.py` | fuser_type choices + cam_drop_mode 参数 | 小 |
| P0 | `evaluate_checkpoint.py` | fuser_type choices 更新 | 极小 |
| P0 | `batch_train.sh` | cam_drop_mode 参数映射 | 极小 |
| P0 | `train_universal.sh` | cam_drop_mode 参数解析和传递 | 小 |
| P0 | `start_training.sh` | cam_drop_mode 参数解析和传递 | 小 |
| P0 | `configs/batch_train_diffcamdrop.yaml` | 6 组实验配置 (新建) | 中 |
| P1 | `configs/eval_diffcamdrop.yaml` | 6 组评估配置 (新建, 见下方模板) | 中 |

### Eval YAML 配置模板 (eval_diffcamdrop.yaml)

```yaml
# 评估配置 — 自动从 checkpoint args 检测 fuser_type
# cam_drop_mode 仅训练时有效，评估无需指定
models:
  - label: "dc_A_baseline"
    dir_name: "model_small_5deg_dc_A_baseline"
    ckpt: "ckpt_best_val.pth"
    voxel_mode: "scatter"
    scatter_reduce: "mean"
    to_bev_mode: "concat"
    fuser_type: "concat"           # 显式指定，覆盖 auto-detect
    angle_deg: 5
    z_voxels: 5
    mode_desc: "A: baseline (concat, no drop)"

  - label: "dc_E_diff_drop_v2"
    dir_name: "model_small_5deg_dc_E_diff_drop_v2"
    ckpt: "ckpt_best_val.pth"
    voxel_mode: "scatter"
    scatter_reduce: "mean"
    to_bev_mode: "concat"
    fuser_type: "diff_v2"          # diff_v2 需要显式指定
    angle_deg: 5
    z_voxels: 5
    mode_desc: "E: diff v2 (dual-path) + cam_drop=0.3"
  # ... 其余 4 组类似结构
```

**注意**: `fuser_type` 也可省略，evaluate_checkpoint.py 从 ckpt args 自动检测。
但显式指定可避免旧代码加载新 ckpt 时的歧义。

### 实验初始化规则

**所有 6 组实验均从随机初始化开始**，不使用 warm start，保证公平对比。
仅当特定配置不收敛时，可使用 warm start 作为**调试手段**（不计入正式结果）。

### 无需改动

| 文件 | 原因 |
|------|------|
| `bevcalib_inference.py` | cam_dropped 默认 False，推理路径不变 |
| `run_generalization_eval.py` | fuser_type 是 passthrough |
| 现有训练 YAML | 旧配置不受影响 |
| 现有评估 YAML | 旧配置不受影响 |

---

## 实施时间线

```
Day 0: 代码实现
  1. bev_calib.py 改动 (ConvFuser 签名 + BEVDiffFuser v2 + cam_drop_mode)
  2. train_kitti.py 参数
  3. Shell 脚本 (3 个)
  4. evaluate_checkpoint.py choices
  5. 代码验证 (dummy 输入测试)

Day 0.5: 配置和启动
  6. 创建 batch_train_diffcamdrop.yaml (6 组)
  7. Quick 模式启动 (1N, ~6h/实验)

Day 1-2: Quick 训练 + 评估
  8. 6 组 Quick 训练完成
  9. 泛化评估 (test_data_v2)
  10. 筛选 top 3 实验

Day 3-5: Full 训练 + 最终评估
  11. Top 3 实验 Full 模式 (32N)
  12. 全量泛化评估
  13. 结果分析 + 决定 Phase 2 方向
```

---

## 2. Loss 优化方案 (V25 预研)

### 2.0 现有 Loss 架构分析

**当前 loss 公式** (`realworld_loss`, rotation_only 模式):

```
total = 1.0 * rotation_loss(quat_dist)
      + 1.0 * PC_reproj_loss
      + 0.5 * quat_norm_loss
      + 0.5 * axis_rotation_loss(weighted Euler)
```

**已有组件**:

| Loss | 实现 | 状态 | 备注 |
|------|------|------|------|
| `rotation_loss` | quaternion distance | ✅ 启用 | 先 mat→quat 再求距离，有 per-sample 循环 |
| `PC_reproj_loss` | 点云重投影误差 | ✅ 启用 | per-sample 循环 (慢) |
| `quat_norm_loss` | 四元数归一化惩罚 | ✅ 启用 | 向量化，性能好 |
| `axis_rotation_loss` | 加权 Euler 角 | ✅ 启用 | `AdaptiveAxisRotationLoss`, 固定权重 |
| `GeodesicRotationLoss` | SO(3) 测地线距离 | ⚪ 已实现未启用 | `use_geodesic_loss=0` |
| `translation_loss` | SmoothL1 | ⚪ joint 模式 | rotation_only 下不参与 |

**问题诊断**:

1. **quaternion distance 在小角度时梯度不稳定**: 当预测接近 GT (< 0.1°) 时，quat 距离的梯度变得不精确
2. **PC_reproj_loss 有 for 循环**: `for i in range(B)` 逐样本做矩阵运算，是训练瓶颈
3. **缺乏模态感知正则化**: Loss 不考虑 cam/pc 各自的贡献，与 CamDrop 无协同
4. **固定权重**: 轴权重 (1.0, 3.0, 1.0) 是手工调参，不一定是最优的

### 2.1 Phase 1: 快速改进 (无架构变更)

#### 2.1.1 启用 Geodesic Loss

**改动**: `use_geodesic_loss: 1`

Geodesic loss `arccos((tr(R_pred^T @ R_gt) - 1) / 2)` 直接在旋转矩阵上操作:
- 不需要 mat→quat 转换 (省去 per-sample `quaternion_from_matrix`)
- 在小角度 (<0.5°) 梯度更稳定
- 几何上更有意义 (SO(3) 流形上的真实距离)

**实施**: 仅需在 V24 YAML 中加一个实验：

```yaml
- name: "v24_I_geodesic"
  params:
    use_geodesic_loss: 1
    fuser_type: "diff_v2"
    cam_drop_prob: 0.3
```

**风险**: 低。已有实现，只需开关。

#### 2.1.2 PC_reproj_loss 向量化

**当前** (per-sample 循环):
```python
for i in range(B):
    T_pred = ...; R_total = inv(GT) @ T_pred
    points_transformed = points_h @ R_total.t()
    loss += (points_transformed - pc).norm()
```

**优化** (batch 向量化):
```python
R_total = torch.bmm(torch.linalg.inv(gt_T), T_pred)  # (B, 4, 4)
points_h = torch.cat([pc, ones], dim=-1)               # (B, N, 4)
transformed = torch.bmm(points_h, R_total.transpose(1, 2))[:, :, :3]
error = (transformed - pc).norm(dim=-1)                 # (B, N)
if mask is not None:
    error = error * mask
    loss = error.sum() / mask.sum().clamp(min=1)
else:
    loss = error.mean()
```

**收益**: ~3-5x 训练速度提升 (消除 Python 循环 + 利用 CUDA 并行)

### 2.2 Phase 2: CamDrop 协同 Loss

#### 2.2.1 辅助单模态预测头 (Auxiliary Modality Heads)

**动机**: CamDrop 从训练策略角度强迫模型不依赖 camera，但 loss 没有显式约束各模态独立预测能力。辅助头从 loss 角度互补。

**架构**:
```python
class BEVCalib:
    def __init__(self, ..., aux_head_weight=0.0):
        ...
        if aux_head_weight > 0:
            aux_dim = self.pc_branch.out_channels
            self.pc_aux_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(aux_dim, 4),  # quaternion
            )
```

**Loss**:
```python
if self.aux_head_weight > 0 and self.training:
    pc_quat = self.pc_aux_head(pc_bev_feats)
    L_aux_pc = quaternion_distance(pc_quat, gt_quat)
    total_loss += aux_head_weight * L_aux_pc
```

**设计要点**:
- 只加 PC 辅助头 (不加 Cam 辅助头): 因为 CamDrop 的目标是增强 PC 分支独立性
- `aux_head_weight` 建议 0.1-0.3: 辅助信号不应主导训练
- 辅助头很小 (1 AvgPool + 1 Linear): 几乎不增加计算量
- 推理时可移除辅助头

**实验**: 需要在 `bev_calib.py` 添加代码，作为 V25 实验。

#### 2.2.2 CamDrop 一致性 Loss

**动机**: 希望模型在有/无 camera 时产出相似预测，显式约束模态鲁棒性。

**设计**:
```python
if self.training and self.consistency_weight > 0 and not cam_dropped:
    with torch.no_grad():
        # 模拟 cam drop，用正常路径的预测作为 teacher
        cam_zero = cam_bev_feats * 0
        x_dropped = self.conv_fuser(cam_zero, pc_bev_feats, cam_dropped=True)
        # ... 通过 transformer + head 得到 pred_dropped
    pred_dropped = self._forward_head(x_dropped)  # detached
    L_consist = F.mse_loss(pred_normal, pred_dropped.detach())
    total_loss += consistency_weight * L_consist
```

**设计要点**:
- 仅在 cam 未被 drop 时计算 (cam 已 drop 时没有 "normal" 参考)
- Teacher 信号用 `torch.no_grad()` + `.detach()`: 不回传梯度到 teacher
- 计算开销: ~30% (需要额外一次 fuser+transformer 前向)
- `consistency_weight` 建议 0.01-0.1: 不宜过强，避免阻碍学习

**风险**: 中等。需要重构 `forward()` 以支持部分前向，实现复杂度较高。

### 2.3 Phase 3: 自适应权重 (Learnable Weighting)

#### 2.3.1 Homoscedastic Uncertainty 多任务权重

**替代固定的 `axis_weights` 和 loss component weights**:

```python
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))
    
    def forward(self, *losses):
        total = 0
        for i, L in enumerate(losses):
            precision = torch.exp(-2 * self.log_sigma[i])
            total += precision * L + self.log_sigma[i]
        return total
```

**效果**: 训练初期损失大的任务自动获得更高权重，后期自动平衡。

**风险**: 低。但需要仔细观察 `log_sigma` 的变化确保不发散。

### 2.4 优先级排序与实验计划

| 优先级 | 改进 | 工作量 | 收益 | 依赖 |
|--------|------|--------|------|------|
| **P0** | Geodesic loss (开关) | 极小 | 中 | 无 |
| **P0** | PC_reproj 向量化 | 小 | 训练加速 3-5x | 无 |
| **P1** | PC 辅助预测头 | 中 | 中-高 | V24 结果 |
| **P2** | CamDrop 一致性 loss | 大 | 高 | V24 结果 |
| **P3** | 自适应权重 | 中 | 中 | P0 |

**建议路径**:
1. V24 实验中加一组 `use_geodesic_loss=1` (零成本验证)
2. 先向量化 PC_reproj_loss (训练提速，对所有后续实验都有益)
3. V24 结果出来后，根据 CamDrop 效果决定是否实施辅助头和一致性 loss

---

## Phase 3: V25 架构突破 — 攻克 0.1° 目标

### 3.0 V24 实验总结 & 多帧融合实验结论 (2026-05-04)

#### 3.0.1 V24 最终性能

| 模型 | Mean Rot | Pitch | Roll | Yaw |
|------|----------|-------|------|-----|
| **v24-B-diff-only** (Best) | 0.664° | 0.458° | 0.338° | 0.195° |
| v24-A-baseline | 0.673° | 0.464° | 0.340° | 0.201° |
| v24-D-diff-drop-v1 | 0.678° | 0.444° | 0.372° | 0.219° |
| v23-A2-joint | 0.665° | 0.445° | 0.355° | 0.214° |

**结论**: V22→V24 从 0.75°→0.66°（改善 12%），但遇到 **~0.66° 架构地板**。

#### 3.0.2 多帧融合实验（关键发现）

使用 `evaluate_multiframe.py` 分析了 N=1~400 帧的滑动窗口平均效果:

| 窗口大小 | v24-B Mean Rot | Pitch | 实际改善 | 理论√N |
|----------|---------------|-------|---------|--------|
| N=1 (单帧) | 0.664° | 0.458° | 1.00x | 1.0x |
| N=10 | 0.634° | 0.441° | 1.05x | 3.2x |
| N=50 | 0.625° | 0.438° | 1.06x | 7.1x |
| N=400 | 0.612° | 0.435° | 1.08x | 20.0x |

**核心结论**:
1. **400帧平均仅改善 8%**（理论应 20x），说明 **92% 的误差是系统性偏差**
2. 多帧融合/静止窗口检测**对当前架构几乎无效**
3. Pitch 轴占总误差 ~69%，且几乎不受多帧影响
4. **必须从模型/训练层突破**

#### 3.0.3 系统性偏差根因

| 根因 | 影响 | 证据 |
|------|------|------|
| Z 体素分辨率不足 (4m/层, 仅 5 层) | Pitch 无法精细感知 | Pitch 始终 2x 于其他轴 |
| BEV XY 分辨率 (2m/格) | 0.1° 对应亚像素位移 | 所有轴都有系统性残差 |
| 单次预测 (无迭代) | 大扰动→粗修正 | 泛化衰退 <1x (模型偏保守) |
| Camera 信息未被有效利用 | 标定信号丢失 | cam_bev_feats 变化 <0.04% |

#### 3.0.4 深层根因: 序列级标定差异的欠拟合 (2026-05-05 signed error 分析)

**核心发现**: 泛化误差不是"全局偏差"也不是"随机噪声"，而是 **序列级系统性偏差**。

| 分析维度 | Pitch Bias² | Pitch Variance | 含义 |
|----------|-------------|---------------|------|
| 全局 | 7.5% | 92.5% | 全局看像"随机" |
| **序列内** | **~95%** | **~5%** | 序列内几乎全是bias |

**每个序列有固定的偏差方向** (signed axis-angle error):

| Seq | Pitch bias | 方向一致性 | Seq | Pitch bias | 方向一致性 |
|-----|-----------|-----------|-----|-----------|-----------|
| 00 | -0.650° | 98% neg | 03 | +0.438° | 99% pos |
| 08 | -0.759° | 99% neg | 06 | +0.343° | 98% pos |
| 05 | -0.628° | 98% neg | 09 | +0.356° | 95% pos |
| 01 | -0.512° | 99% neg | 10 | +0.403° | 95% pos |

**解读**:
- 12 个测试序列有 12 个不同的 GT 标定
- 模型学到了"平均修正策略"，对特定序列的 GT 系统性过修/欠修
- **多帧平均无效的直接原因**: 同一序列内所有帧偏差方向一致，平均无法消除
- **本质是序列级标定多样性的欠拟合**

**对 V25 的指导**:
- V25-A (Z精细化): 更高 Z 分辨率 → BEV 能编码不同 pitch GT 产生的细微差异
- V25-C (Coarse-to-Fine): Fine 模型在 ±1° 范围内 → 序列间偏差从 0.76° 缩小到 coarse 残差 ~0.1°
- **潜在 V26 方向**: 序列级条件化 (让模型"感知"当前序列的 GT 特征)

---

### 3.1 V25-A: Z 体素精细化 (Pitch 专项优化)

**优先级**: P0 (最快突破 Pitch 瓶颈)
**预期收益**: Pitch 降 30-50% → Total 降至 ~0.4°

#### 3.1.1 原理

当前 Z-bound: `[-10, 10]` step=4.0m → 5 个 Z 层
- 1° pitch 在 100m 距离处产生 1.74m Z 偏移
- 当前 4m 分辨率: 1° pitch ≈ 0.44 格 (严重欠采样!)
- 若 step=1.0m → 20 层: 1° pitch ≈ 1.74 格 (充分采样)

#### 3.1.2 实现方案

```python
# 环境变量控制 (backward-compatible)
BEV_ZBOUND_MIN = -10
BEV_ZBOUND_MAX = 10
BEV_ZBOUND_STEP = 1.0  # 4.0 → 1.0, 生成 20 个 Z 层

# bev_calib.py: LSS depth bins 自动适配
z_bound = [Z_MIN, Z_MAX, Z_STEP]  # → 20 bins
# Cam2BEV: depth_channels = (Z_MAX - Z_MIN) / Z_STEP = 20 (原 5)
# Lidar2BEV: spconv z_dim = 20 (原 5)
```

**关键修改点**:
1. `Cam2BEV`: depth_channels 从 5→20，LSS outer product 输出 `(B, C, 20, H_bev, W_bev)`
2. `Lidar2BEV`: spconv spatial_shape z 维从 5→20
3. BEV pool: `(B, C, 20, 96, 96)` → `(B, C, 96, 96)` (collapse Z 轴)
4. 内存: 4x 增长 (5→20 层)，需验证 L20 GPU 是否可容纳 batch_size=8

**实验配置**:
```yaml
v25_A_z20_rotonly:
  bev_zbound_step: 1.0  # 20 Z layers
  rotation_only: true
  epochs: 400
```

#### 3.1.3 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 显存不足 | 降 batch 为 4, 或用 gradient checkpointing |
| 稀疏性增大 | Z=20 层 voxel 很稀疏, spconv 自然处理 |
| 训练变慢 | 4x spconv 计算, 预计每 epoch 慢 2-3x |

---

### 3.2 V25-B: Coarse-to-Fine 两阶段预测

**优先级**: P1 (系统性降低所有轴误差)
**预期收益**: Total 降 30-40% → ~0.4°

#### 3.2.1 原理

当前 ±5° 扰动范围，模型一步预测存在"保守修正"偏向。两阶段:
1. Stage 1: 粗预测 (±5° → ±1°)
2. Stage 2: 精预测 (±1° → ±0.1°)

更小的搜索空间 → 更高的相对精度。

#### 3.2.2 实现方案

```python
class BEVCalib_CoarseToFine(nn.Module):
    def __init__(self, ...):
        self.coarse_model = BEVCalib(...)  # 正常 ±5° 训练
        self.fine_model = BEVCalib(...)    # ±1° 精细范围训练
    
    def forward(self, imgs, pcs, gt_T, init_T, ...):
        # Stage 1: 粗修正
        T_coarse, _, _ = self.coarse_model(imgs, pcs, gt_T, init_T, ...)
        
        # Stage 2: 精修正 (以 coarse 结果为新的 init)
        T_fine, _, _ = self.fine_model(imgs, pcs, gt_T, T_coarse.detach(), ...)
        
        return T_fine, T_coarse, None
```

**训练策略**:
- Phase A: 正常训练 coarse_model (±5°, 400 epochs)
- Phase B: 冻结 coarse_model, 用其输出作为 fine_model 的输入
- fine_model 训练时扰动范围缩小为 ±1° (从 coarse 预测出发)

#### 3.2.3 风险

| 风险 | 缓解 |
|------|------|
| coarse 预测差 → fine 失效 | 训练 fine 时加入 coarse 预测噪声 |
| 推理延迟翻倍 | 两个轻量模型 vs 一个重型模型 |
| 需要两阶段训练 | 可共享 backbone 权重 |

---

### 3.3 V25-C: 高分辨率 BEV + 前视图 Pitch 分支

**优先级**: P2 (更根本的架构改进)
**预期收益**: Pitch 降 40-50%, 但实现复杂度高

#### 3.3.1 原理

Pitch 信号主要编码在图像的垂直位移中 (物体在图像中上下移动)。
BEV 投影后丢失了大部分垂直位置信息。直接在前视图(FV)特征上回归 Pitch 更合理。

#### 3.3.2 实现方案

```python
class PitchAwareBEVCalib(nn.Module):
    def __init__(self, ...):
        self.bev_model = BEVCalib(...)  # 标准 BEV 流程 (Roll + Yaw)
        self.fv_pitch_head = FrontViewPitchHead(...)  # 前视图 Pitch 专用
    
    def forward(self, imgs, pcs, gt_T, init_T, ...):
        # BEV 流程: 预测 Roll + Yaw (BEV 擅长的)
        T_bev, _, _ = self.bev_model(imgs, pcs, gt_T, init_T, ...)
        
        # 前视图流程: 从图像特征直接预测 Pitch
        fv_feats = self.bev_model.cam_encoder.backbone(imgs)  # 共享 backbone
        pitch_pred = self.fv_pitch_head(fv_feats, pcs)
        
        # 合并
        T_final = compose_with_pitch(T_bev, pitch_pred)
        return T_final, T_bev, pitch_pred

class FrontViewPitchHead(nn.Module):
    """从前视图特征提取 Pitch 信号"""
    def __init__(self, in_channels=512, feat_h=12, feat_w=20):
        super().__init__()
        self.vertical_pool = nn.AdaptiveAvgPool2d((feat_h, 1))  # 保留垂直信息
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, (feat_h, 1)),  # collapse 高度维
        )
        self.fc = nn.Linear(64, 1)  # 输出 pitch 角度
```

#### 3.3.3 风险

| 风险 | 缓解 |
|------|------|
| 前视图 pitch 信号弱 | 加 depth-aware attention |
| 多分支训练不稳定 | 分阶段: 先 BEV, 再加 FV |
| 实现复杂度高 | 作为 V25 后期探索 |

---

### 3.4 V25-D: 迭代预测 + Transformer Refinement

**优先级**: P1 (配合 Coarse-to-Fine)
**预期收益**: 单阶段内降 15-25%

#### 3.4.1 方案

在 DeformableTransformer 内部增加 iteration:
```python
# 当前: 2 层 transformer, 直出
# V25-D: 2 层 transformer × K 次迭代, 每次用上一轮输出的 T_pred 重新做 pose_embed

for k in range(K):  # K=2-3
    pose_embed = self.pose_embed_fn(T_current)
    fused = fused_feats + pose_embed
    output = self.transformer(fused)
    delta_T = self.head(output)
    T_current = compose(T_current, delta_T)
```

共享 transformer 权重, 额外开销仅为 K-1 次前向。

---

### 3.5 V25 实验优先级总表

| 优先级 | 方案 | 工作量 | 预期收益 | V25 实验标签 |
|--------|------|--------|---------|-------------|
| **P0** | Z 体素精细化 (5→20层) | 小 (改配置) | Pitch 降 30-50% | v25_A |
| **P0** | Geodesic loss | 极小 | 中 (5-10%) | v25_B |
| **P1** | Coarse-to-Fine 两阶段 | 中 | Total 降 30-40% | v25_C |
| **P1** | 迭代 Refinement (K=3) | 中 | Total 降 15-25% | v25_D |
| **P2** | 前视图 Pitch 分支 | 大 | Pitch 降 40-50% | v25_E |
| **P2** | 高分辨率 BEV (1m/格) | 大 (显存) | 全轴 20-30% | v25_F |

**V25 目标**: 从 0.66° 降至 **< 0.3°** (单帧), 配合部署层 multi-frame → **< 0.1°**

**建议执行路径**:
1. **v25_A (Z精细化)**: 最快验证，仅改环境变量 + 验证显存
2. **v25_B (Geodesic loss)**: 零成本加入 v25_A 实验
3. **v25_C (Coarse-to-Fine)**: 在 v25_A 基础上叠加
4. **v25_D (迭代)**: 可与 C 并行探索
5. **v25_E/F**: 根据前面结果决定

---

### 3.6 V25 训练方案详细规划 (2026-05-05)

#### 3.6.1 实验矩阵 (8 组)

| ID | Z 层数 | Z step | Geodesic | 扰动范围 | batch | 验证目标 |
|----|--------|--------|----------|---------|-------|----------|
| **A1** | 10 | 2.0m | No | ±5° | 16 | Z=10 对 Pitch 的改善 |
| **A2** | 15 | 1.33m | No | ±5° | 16 | Z=15 进一步改善? |
| **A3** | 20 | 1.0m | No | ±5° | 8 | Z=20 极限 (显存紧张) |
| **B1** | 5 | 4.0m | Yes | ±5° | 16 | Geodesic loss 单独效果 |
| **B2** | 10 | 2.0m | Yes | ±5° | 16 | Z + Geodesic 叠加效果 |
| **C1** | 5 | 4.0m | No | **±1°** | 16 | 精细模型 (链式 Fine) |
| **C2** | 5 | 4.0m | No | **±2°** | 16 | 中等精细模型 |
| **C3** | 10 | 2.0m | No | **±1°** | 16 | Z=10 + Fine (终极组合) |

#### 3.6.2 训练配置文件

| 文件 | 节点 | 数据量 | 用途 |
|------|------|--------|------|
| `configs/batch8_train_all_v25.yaml` | 32N | 全量 (10000 帧/seq) | 正式训练 |
| `configs/batch8_train_all_v25_quick.yaml` | 1N | 500 帧/seq | 快速验证趋势 |
| `configs/eval_v25.yaml` | - | test_data_v2 | 泛化评估 |

#### 3.6.3 Quick 版本训练计划 (验证阶段)

**目的**: 用 ~2 小时快速验证各方案的趋势性，筛选出值得投入 full 训练的实验

**配置差异**:
- `nnodes: 1` (单节点 8 卡)
- `max_frames_per_seq: 500` (数据量 1/20)
- 其余参数与 full 版完全一致

**执行顺序** (按依赖关系):
1. **第一批 (并行)**: A1, A2, B1, C1, C2 — 无依赖，可同时启动
2. **第二批 (视显存)**: A3 (需降 batch=8，确认 OOM 情况)
3. **第三批 (组合)**: B2, C3 — 等 A1 确认 Z=10 没问题后启动

**Quick 版判定标准** (相对 V24-B baseline 0.664°):
- A 系列 Pass: Pitch < 0.40° (改善 > 12%)
- B 系列 Pass: Total Rot < 0.63° (改善 > 5%)
- C 系列 Pass: 在 ±1°/±2° 范围内 Total Rot < 0.15° (精度提升 > 4x)

#### 3.6.4 Full 版本训练计划 (正式阶段)

**目的**: 对 Quick 版验证通过的实验进行全量训练

**配置**:
- `nnodes: 32` (32 节点 256 卡)
- `max_frames_per_seq: 10000` (全量数据)
- `num_epochs: 400`

**执行策略**:
1. Quick 通过则启动对应 Full 训练
2. Quick 中 A1 vs A2 差异 < 2%: Full 仅跑 A1 (节省资源)
3. Quick 中 A3 OOM: 改用 `gradient_checkpointing` 或放弃 A3
4. Quick 中 C1 vs C2 差异 < 3%: Full 仅跑 C2 (更鲁棒的 ±2°)

**预估训练时间** (32N):
- A/B 系列: ~8h (与 V24 相同)
- A3 (batch=8): ~12h (batch 减半)
- C 系列: ~8h (数据相同，扰动范围不影响速度)

#### 3.6.5 Coarse-to-Fine (C 系列) 评估方案

C 系列的核心价值在于**链式推理**:

```
输入 (±5° 扰动) → V24-B (coarse, ±5°→±1° 残差) → V25-C1 (fine, ±1°→0°)
```

**三种评估模式**:

| 模式 | 方法 | 意义 |
|------|------|------|
| standalone | C1 单独评估 (±1° 扰动) | 验证 fine 模型在小范围内的精度 |
| chain-oracle | GT 标定 + ±1° 扰动 → C1 | Fine 模型理论上限 |
| **chain-real** | ±5° 扰动 → V24-B → C1 | 实际部署效果 |

**链式评估命令**:
```bash
python evaluate_coarse_fine.py \
  --coarse_ckpt logs/.../v24_B_diff_only/ckpt_best_val.pth \
  --fine_ckpt logs/.../v25_C1_fine_1deg/ckpt_best_val.pth \
  --coarse_angle 5 --fine_angle 1 \
  --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2
```

**C 系列成功标准**:
- chain-real Total Rot < 0.30°: 显著突破 (V24-B 的 0.664° → 55% 改善)
- chain-real Total Rot < 0.40°: 有效突破 (40% 改善)
- chain-real Total Rot > 0.50°: C 方案失败, coarse 残差太大

#### 3.6.6 运行命令速查

```bash
# === Quick 验证 (单节点) ===
bash batch_train.sh configs/batch8_train_all_v25_quick.yaml

# === Full 训练 (32 节点) ===
bash batch_train.sh configs/batch8_train_all_v25.yaml

# === 泛化评估 ===
HF_HUB_OFFLINE=1 python run_generalization_eval.py --config configs/eval_v25.yaml --parallel -1

# === Coarse-to-Fine 链式评估 ===
python evaluate_coarse_fine.py \
  --coarse_ckpt logs/all_training_data/model_small_5deg_v24_B_diff_only/ckpt_best_val.pth \
  --fine_ckpt logs/all_training_data/model_small_5deg_v25_C1_fine_1deg/ckpt_best_val.pth \
  --coarse_angle 5 --fine_angle 1 \
  --dataset_root /mnt/drtraining/user/dahailu/data/bevcalib/test_data_v2

# === Dry run (检查配置) ===
bash batch_train.sh --dry-run configs/batch8_train_all_v25.yaml
bash batch_train.sh --dry-run configs/batch8_train_all_v25_quick.yaml
```

#### 3.6.7 V25 里程碑

| 里程碑 | 标准 | 预计时间 |
|--------|------|---------|
| M1: Quick 完成 | 8 组 quick 训练完成 + 趋势分析 | +2 天 |
| M2: 筛选确认 | 确定 Full 训练候选 (3-5 组) | +2.5 天 |
| M3: Full 完成 | Full 训练 + 泛化评估 | +5 天 |
| M4: 链式验证 | C 系列 Coarse-to-Fine 链式评估 | +6 天 |
| M5: 最终报告 | 确认最优方案, 更新 PLAN | +7 天 |

**最终目标**: 单帧 < 0.3°, 多帧(N=50) < 0.1°
