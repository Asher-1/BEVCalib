# V39.1 Camera-BEV 架构重构总结

## 执行日期：2026-05-28

## 问题诊断

### 训练状态（两个训练均进行到中期）

| 训练 | 当前Epoch | Gate状态 | Train Rot | MEDW200 (Epoch 161/201) |
|------|-----------|----------|-----------|------------------------|
| **M1原始(f1000)** | 168-172 | **bev=0.000, entropy=0.000** ❌ | 1.89-1.95° ✅ | **0.416°** ✅ |
| **M1修复(f500)** | 209-214 | **bev=0.000, entropy=0.000** ❌ | 1.95-1.99° ✅ | **0.314°** ✅ |

### 修复尝试历史

1. **Attempt 1**：`gate_entropy_weight=0.05` + `bev_lr=1.0` + `deep_sup=0.5`
   - **结果**：Epoch 2稍好(bev=0.112)，Epoch 3完全坍塌(bev<0.01)
   - **结论**：正则化太弱，无效

2. **Attempt 2**：`gate_entropy_weight=0.2` + Gate初始化偏BEV `bias=[1.5, -1.5]`
   - **结果**：Epoch 3完美平衡(bev=0.489, entropy=0.900) → Epoch 7再次坍塌(bev=0.02)
   - **结论**：短暂有效，但Proj预训练优势最终压倒一切

### 🔴 震惊发现

**即使Gate完全坍塌为纯Proj（BEV权重=0%），MEDW仍达到v36 baseline水平（0.3-0.4°）**

这说明：
1. ✅ Proj分支单独已经足够强大
2. ❌ 当前BEV架构无法提供互补价值
3. ❌ 超参数调整无法解决根本问题
4. 🎯 **必须执行架构重构（方案3）**

---

## 根因分析（5个维度）

### 1. 预训练不对称（主因）

| 模块 | BEV分支 | Proj分支 |
|------|---------|----------|
| 图像编码器 | DINOv2 (frozen) ✅ | DINOv2 (frozen) ✅ |
| 点云编码器 | PointGPT (frozen) ✅ | PointGPT (frozen) ✅ |
| **融合网络** | **PointGPT2BEV (random init) ❌<br>BEVDiffFuser (random init) ❌<br>BEV-Transformer (random init) ❌** | **AttenDualFusion (Fleet经验) ⚠️** |

**结果**：Proj初期质量高，Gate理性选择强者。

### 2. 信息密度差异

```
BEV：100×100 grid → mask保留30-40% → pool → 128-d (信息瓶颈)
Proj：16×28 patches (100%利用) → cross-attn → 384-d (信息容量大)
```

### 3. 几何归纳偏置错位

```
标定任务本质：找旋转R使得 pc_camera = R @ pc_lidar

Proj优势：保留camera-pc对应，loss gradient直接指向ΔR
BEV劣势：破坏camera几何 → BEV俯视投影 → 需额外推理3D反演
```

### 4. 梯度路径长度

```
Proj：loss → GatedHead (1层) → AttenDualFusion cross-attn → 短链路
BEV：loss → GatedHead → Transformer (3层) → BEV fusion → PointGPT2BEV scatter → 长链路+梯度弥散
```

### 5. 信息损失不对称

```
BEV：cam_bev_mask丢失60-70%空间 + scatter稀疏聚合损失
Proj：所有patches参与，无信息损失
```

---

## 🚀 方案3：Camera-BEV Cross-Attention架构

### 核心改进

| 维度 | 原BEV | Camera-BEV |
|------|-------|-----------|
| **坐标系** | BEV俯视投影（破坏camera几何） | **Camera坐标系（保留前向对应）** |
| **链路长度** | 5层（DINOv2→Query→PointGPT2BEV→Fuser→Transformer→Pool） | **2层（DINOv2→CrossAttn→Pool）** |
| **信息利用** | mask 30-40%（cam FOV内） | **100%（所有patches）** |
| **与Proj互补** | 同任务不同视角（但更弱） | **不同分辨率(640×360 vs 224×448)+query方式** |

### 架构对比

```
【原BEV】
img(640×360) → DINOv2 patches → Query投影到BEV(100×100)
                                    ↓
pc → PointGPT groups → PointGPT2BEV scatter → BEV grid
                                    ↓
                              BEVDiffFuser (融合)
                                    ↓
                            BEV-Transformer (3层)
                                    ↓
                              AdaptivePool → 128-d
                                (信息瓶颈)

【原Proj】
img(640×360) → resize(224×448) → DINOv2 ViT patches (16×28)
                                           ↓
                            Cross-Attention Query
                                           ↓
pc → PointGPT groups (256) ← Key/Value
                                           ↓
                            AttenDualFusion → 384-d
                              (强预训练经验)

【新Camera-BEV】
img(640×360) → DINOv2 patches (45×80=3600) ← Camera坐标系
                               ↓
                    MultiheadAttention Query
                               ↓
pc → PointGPT groups (256) ← Key/Value (投影到Camera系)
                               ↓
                    2层Cross-Attention
                               ↓
                    GlobalPooling → 256-d
                      (梯度路径短)
```

### 实施文件

1. **核心架构**：`kitti-bev-calib/camera_bev_fusion.py` ✅
   - `CameraBEVCrossAttention`：核心cross-attention模块
   - `CameraBEVBranch`：高层wrapper
   - `build_camera_bev_branch`：工厂函数

2. **集成**：`kitti-bev-calib/hybrid_triple_calib.py` ✅
   - 添加`FUSION_BACKENDS = (..., 'camera_bev_triple')`
   - 添加`self.use_camera_bev`标志位
   - 构建Camera-BEV分支逻辑
   - forward方法集成

3. **配置文件**：`configs/v39_camera_bev.yaml` ✅
   - M2快速验证实验（100 epoch, f500）
   - Proj-only baseline对比（可选）

4. **文档**：
   - `docs/V39_DESIGN.md` ✅ (10.5节：Gate坍塌根因分析与架构反思)
   - `docs/V39_1_CAMERA_BEV_QUICKSTART.md` ✅ (快速启动指南)

### Smoke Test结果

```bash
✅ Camera-BEV model 实例化成功
Fusion backend: camera_bev_triple
use_camera_bev: True
use_proj: True
BEV feat_dim: 256
Proj feat_dim: 384
```

---

## 🎯 实验计划

### M2: Camera-BEV快速验证（100 epoch）

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib
bash batch_train.sh configs/v39_camera_bev.yaml
```

### 成功标志

| 维度 | 目标 | 对比基准 |
|------|------|---------|
| **Gate平衡** | bev=0.3-0.6, entropy>0.6 持续到Epoch 20+ | vs 原架构Epoch 3坍塌 |
| **MEDW性能** | ≤ 0.35° | vs 纯Proj 0.31-0.42° |
| **训练收敛** | Train Rot < 2.5° | vs 原架构1.9-2.0° |

### 决策树

```
Camera-BEV训练到Epoch 20
  |
  ├─ Gate保持平衡(bev>0.2, entropy>0.5)
  |    └─> ✅ 继续到Epoch 100，对比MEDW
  |          ├─ MEDW优于纯Proj → **成功！Camera-BEV有效**
  |          └─ MEDW持平/更差 → 放弃，简化为proj_only
  |
  └─ Gate坍塌(bev<0.05)
       └─> ❌ Epoch 40停止，放弃双分支架构
            → 正式结论：Proj预训练优势无法克服
            → 简化为fusion_backend='proj_only'
```

---

## 理论贡献

### 设计合理性审查结论

**设计理念正确，但实现细节不匹配预期**：

| 维度 | 设计目标 | 实际表现 |
|------|----------|----------|
| BEV分支 | 时序平滑，降帧间variance | 从头训练，初期质量差 |
| Proj分支 | 单帧精度，空间对应 | 强预训练，立即有效 |
| 融合策略 | 互补（BEV稳定+Proj精度） | 坍塌（Proj主导，BEV边缘化） |

### 未来方向

1. **接受现状**：继续使用Proj-dominant训练作为baseline（MEDW已达标）
2. **两阶段预训练**：先单独预训练BEV分支到Proj同等质量，再融合
3. **Camera-BEV重构**（当前方案）：跳过BEV投影，直接camera系cross-attention

---

## 监控命令

```bash
# 实时监控Gate状态
tail -f logs/*/v39_M2_camera_bev_quick/train.log | grep "Gate mean"

# 预期健康状态：
# Epoch [1-5]: bev=0.4-0.6, entropy>0.8
# Epoch [10-20]: bev=0.3-0.5, entropy>0.6
# Epoch [50+]: bev=0.2-0.4, entropy>0.5（允许轻微倾斜但不完全坍塌）

# 检查MEDW趋势
grep "MEDW200" logs/*/v39_M2_camera_bev_quick/train.log

# 对比原架构坍塌点
# Epoch 2-3是关键观察窗口，原架构此时bev已<0.01
```

---

## 风险与缓解

| 风险 | 缓解策略 |
|------|----------|
| Camera-BEV仍坍塌 | entropy_weight已设0.1（温和），Gate初始化已偏BEV |
| 训练速度慢 | f500快速验证，总epoch仅100 |
| MEDW无改善 | 保留Proj-only作为fallback，决策树明确停止条件 |

---

## 交付物清单

- [x] 根因分析报告（5个维度）
- [x] Camera-BEV架构实现（`camera_bev_fusion.py`）
- [x] HTCN集成（`hybrid_triple_calib.py`）
- [x] 配置文件（`v39_camera_bev.yaml`）
- [x] 文档更新（`V39_DESIGN.md`, `V39_1_CAMERA_BEV_QUICKSTART.md`）
- [x] Smoke test通过
- [ ] **待启动**：M2 Camera-BEV训练
