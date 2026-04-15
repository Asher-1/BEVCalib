# BEVCalib 训练速度优化效果对比报告

## 测试环境

| 项目 | 配置 |
|------|------|
| GPU | 8× NVIDIA L20 (48GB) |
| Batch size | 16/GPU |
| 精度 | AMP float16 + GradScaler |
| 数据集 | ALL (130,434 帧, 11 sequences) |
| 采样数据 | 11,000 帧 (1000帧/序列) |
| DDP | NCCL AllReduce |

## 1. 各优化项逐步贡献

基于 v8_quick 实际训练日志 (8 GPU L20) + SwinT DDP Benchmark (8 GPU A30) 联合分析:

| 步骤 | 配置 | per-step | epoch | 400ep 总时间 | 节省 |
|------|------|----------|-------|-------------|------|
| **Baseline** | 无优化+全量数据+eval/1ep | ~1343ms | ~1094s | **7496min (125h)** | - |
| **①+采样** | max_frames_per_seq=1000 | ~1343ms | ~91s | **809min** | 6687min (96.9%) |
| **②+eval/50** | eval_epoches=50 | ~1343ms | ~91s | **613min** | 196min (2.8%) |
| **③+find_unused** | find_unused_parameters=False | ~1322ms | ~90s | **603min** | 9min |
| **④+skip_stage3** | SwinT Stage3 前向跳过 | ~1317ms | ~90s | **601min** | 2min |
| **⑤+grad_accum** | grad_accum_steps=2 | ~1310ms | ~89s | **598min (10h)** | 3min |

**总加速: 125h → 10h = 12.5x**

## 2. 关键发现

### 采样策略是压倒性的加速因子

采样 `max_frames_per_seq=1000` 贡献了 **96.9%** 的总节省时间:
- Batches/epoch: 815 → 68 (12x 减少)
- 这是因为 per-step 耗时不变, 纯粹是 epoch 内 batch 数量的减少

### 计算优化 per-step 效果有限

SwinT+FPN DDP benchmark 实测各优化对 SwinT 子模型加速 14.5%:

| 优化项 | SwinT+FPN 加速 | 完整模型 per-step 效果 |
|--------|---------------|---------------------|
| find_unused=False | +10.4% | ~1.5% (~21ms) |
| skip Stage3 | +1.9% | ~0.4% (~5ms) |
| grad_accum=2 | +2.2% | ~0.5% (~7ms) |
| **合计** | **14.5%** | **~2.5% (~33ms)** |

**原因**: SwinT+FPN 只占完整模型 compute 的 ~15-20%, 其余 80%+ 被 Lidar2BEV (sparse conv)、ConvFuser、DeformableTransformer、Loss 占据。

### 计算优化的**非速度价值**

虽然计算优化对速度贡献有限, 但它们有重要的**非速度价值**:

| 优化项 | 非速度价值 |
|--------|----------|
| `find_unused_parameters=False` | **DDP 正确性** - 避免反向图遍历 bug, 减少 GPU 显存占用 |
| freeze + skip Stage3 | **显存节省** - 减少 14.2M 参数的梯度缓存 |
| `grad_accum=2` | **等效大 batch** - 256 effective batch size, 改善训练稳定性 |

## 3. eval_epoches 优化

| 设置 | 验证次数 (400 epochs) | 验证总开销 | 节省 |
|------|----------------------|-----------|------|
| eval_epoches=1 | 400 次 | ~200 min | - |
| eval_epoches=50 | 8 次 | ~4 min | **196 min (98%)** |

## 4. 端到端总效果

```
  Baseline 全量:     125.0 h (7496 min)
  全优化+采样:         10.0 h  (598 min)
  ────────────────────────────────────
  总加速:             12.5x
  
  贡献分解:
    采样策略           96.9%  (6687 min saved)
    eval_epoches=50     2.8%  ( 196 min saved)
    计算优化合计         0.2%  (  14 min saved)
```

## 5. 精度影响

| 优化项 | 精度影响 | 说明 |
|--------|---------|------|
| find_unused=False | ✅ 无 | 冻结的参数本就不参与梯度计算 |
| skip Stage3 | ✅ 无 | Stage3 输出从未被 FPN 使用, 数值完全一致 |
| grad_accum=2 | ⚠️ 微弱 | 等效 batch size 翻倍 (128→256), 可能需微调学习率 |
| max_frames_per_seq=1000 | ⚠️ 待验证 | 每序列仅采样 8.4% 帧, 需通过 quick 训练评估泛化能力 |
| eval_epoches=50 | ✅ 无 | 仅减少验证频率 |

## 6. 模块级计算耗时分布

基于训练日志 + 架构分析推算 (v8_quick, 8 GPU L20, ~1310ms/step):

| 模块 | 估算耗时 | 占 compute | 占 step | 特点 |
|------|---------|-----------|--------|------|
| **pc_branch** (SparseEncoder 3D) | ~450ms | **~40%** | 34% | FP32, 无 AMP 加速, 3D sparse conv |
| **img_branch** (SwinT+FPN+LSS) | ~394ms | ~35% | 30% | 已优化 (Stage3 skip), AMP FP16 |
| **deformable_tx** (BEV attention) | ~225ms | ~20% | 17% | 100×100 BEV 特征图 |
| head + loss | ~56ms | ~5% | 4% | 轻量 |
| data_load + prep | ~134ms | - | 10% | I/O |

### 关键发现

- **pc_branch 是最大计算瓶颈** (~40%), 因为:
  - DRCV sparse conv 只支持 FP32, 无法受益于 AMP 半精度加速
  - 体素化采用 Python 循环逐样本处理, 无批量 GPU kernel
  - 3D sparse conv 在高分辨率网格 (800×800×41) 上运算密集
- img_branch 虽然参数最多 (31M), 但 AMP FP16 + Stage3 skip 已显著降低耗时
- deformable_transformer 在 100×100 BEV 上的 attention 计算不可忽视

## 7. 进一步优化建议

| 优化项 | 目标模块 | 预估收益 | 难度 | 说明 |
|--------|---------|---------|------|------|
| **conv+BN 融合** | pc_branch | +3-5% | 低 | DRCV 已有 `fused()` 接口但未使用 |
| **spconv v2 替代** | pc_branch | +5-15% | 中 | 更优的 CUDA kernel, 需验证兼容性 |
| **PointPillar 2D** | pc_branch | +30-50% | 高 | 2D conv 替代 3D sparse, 需重新训练 |
| torch.compile | 全模型 | +5-15% | 中 | 已有 CLI 支持, 需验证 spconv 兼容性 |
| **更粗体素** | pc_branch | +10-20% | 低 | 减少初始活跃体素, 可能影响精度 |
| BF16 替代 FP16 | img_branch | +2-3% | 低 | 省去 GradScaler 开销 |
| 数据预加载 .pt | data_load | +1-2% | 低 | I/O 占比小, 空间有限 |

### 投入产出优先级

1. **conv+BN 融合** — 改动最小, 已有接口, 立即可试
2. **max_num_voxels 调优** — 当前 cap=120000, 可能过高; 减少可降低 indice_pairs 开销
3. **spconv v2** — 如果 DRCV 持续有兼容性问题, 切换到社区版 spconv v2 是长期解
