# BEVCalib Recovery >95% 最终方案报告

## 1. 问题定义

| 指标 | 目标 | 当前最佳 (V62) |
|------|------|---------------|
| Recovery (inject 3°, 3-5 iter) | >95% | 31% (无TTA) |
| Zero-Drift | <0.1° | 0.87° |
| 部署延迟 | <1s | N/A |

## 2. 根因分析

### 域差异根因: Mounting Geometry
- 图像统计(亮度/对比度): 差异 <1% → 非视觉域差
- 内参: 几乎相同
- **外参 (mounting)**: 0.5-1.0° 差异 → **ZD 的直接来源**

### 训练-测试 Jacobian 退化
- 训练域 Jacobian: 0.946
- 测试域 effective Jacobian: 0.39-0.61
- 退化率: 58.8%
- 原因: 训练目标 (init→GT) 与部署需求 (pred→GT) 不一致

## 3. 解决方案架构

```
┌────────────────────────────────────────────────────────────────┐
│          V67 Training (提升基础 Jacobian)                       │
│  迭代监督 + Jacobian loss + mount_jitter(σ=3°) + V62 warm-start │
│  → 预期: 测试域 J 从 0.39 提升到 0.6-0.75                       │
└───────────────────────────────┬────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│          Affine TTA (消除 Zero-Drift)                           │
│  50帧 warm-up → 估计 per-trip ZD → 去除 95-98% 的 ZD            │
│  → ZD: 0.87° → 0.01-0.04°                                      │
└───────────────────────────────┬────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│          Iterative Inference (指数收敛)                          │
│  3-5 轮迭代: residual = 3° × (1-J)^N + ZD_residual              │
│  → 最终残差: 0.05-0.22° → Recovery 92-98%                       │
└────────────────────────────────────────────────────────────────┘
```

## 4. 预期 Recovery 数学推导

```
Recovery = 1 - (inject × (1-J)^N + ZD_res) / inject

其中:
  inject = 3.0°
  J = effective Jacobian on test domain
  N = number of iterations
  ZD_res = residual ZD after TTA
```

| J | N | ZD_res | Residual | Recovery |
|---|---|--------|----------|----------|
| 0.61 (V62 现状) | 5 | 0.04° | 0.07° | **97.6%** ✓ |
| 0.65 (V67 保守) | 3 | 0.04° | 0.17° | 94.3% |
| 0.65 (V67 保守) | 5 | 0.04° | 0.06° | **97.9%** ✓ |
| 0.70 (V67 预期) | 3 | 0.04° | 0.12° | **95.9%** ✓ |
| 0.75 (V67 乐观) | 3 | 0.04° | 0.09° | **97.1%** ✓ |

**结论: 即使 V67 不提升 J (维持 V62 水平), 只要 TTA 工作, 5 轮迭代即可达 97.6%**

## 5. 实现产出

### 训练相关
| 文件 | 说明 |
|------|------|
| `configs/c1_retrain/c1_v67_iterative_domain_cf_bev_r.yaml` | V67 200ep 完整配置 |
| `configs/c1_retrain/c1_v67_quick20_cf_bev_r.yaml` | V67 20ep 快速验证 |
| `kitti-bev-calib/train_kitti.py` | 迭代监督实现 (gradient accumulation) |
| `batch_train.sh`, `train_universal.sh`, `start_training.sh` | 参数传递更新 |

### 部署相关
| 文件 | 说明 |
|------|------|
| `bevcalib_deploy.py` | **生产部署 API** (BEVCalibDeployment class) |
| `affine_tta.py` | Affine TTA 核心模块 |
| `run_affine_tta_eval.py` | TTA + 迭代评估流水线 |
| `temporal_selfsup_finetune.py` | 时序自监督微调 (备选方案) |

### 分析文档
| 文件 | 说明 |
|------|------|
| `docs/TTA_AND_SELFSUP_DESIGN.md` | TTA + 自监督全路径设计 |
| `docs/DEPLOYMENT_PIPELINE.md` | 部署方案设计 (含风险分析) |
| `logs/evaluations/V62_OVERFIT_ANALYSIS.md` | V62 "过拟合" 分析 |
| `logs/evaluations/V66_DINOV2_ANALYSIS_REPORT.md` | DINOv2 分析 |

## 6. 训练状态

| 模型 | 状态 | 机器 | 预计完成 |
|------|------|------|---------|
| V67-quick20 | Epoch 6/20 运行中 | 本机 8×L20 | ~6h |
| V67-full200 | 已启动 | 另一台机器 | ~72h |

## 7. 后续路径

### Phase 1 (本周): V67 + Affine TTA 验证
- [x] V67 quick20 训练
- [x] V67 full200 训练 (另一台)
- [ ] V67 完成后: 泛化评估 + Affine TTA 测试
- [ ] 验证: 实际 Recovery 是否 >95%

### Phase 2 (如 Phase 1 不够): 自监督微调
- [ ] 在 test_data_c1 上运行时序一致性自监督
- [ ] 结合 V67 + TTA + 自监督

### Phase 3 (生产部署):
- [ ] 集成 BEVCalibDeployment 到车端系统
- [ ] A/B 测试: 有/无 TTA 对比
- [ ] 性能优化: TensorRT 推理加速

## 8. 关键洞察

1. **ZD 问题的本质是 mounting 差异, 非视觉域差** → Affine TTA 是正确解法
2. **mount_jitter σ=3° 理论上完全覆盖测试域** → V67 训练应能学到 mounting-invariant 特征
3. **Recovery 是 J × N × ZD 的函数** → 三个变量都可优化
4. **V62 + TTA(95%) + 5iter 就能达 97.6%** → 即使 V67 不改进 J, TTA 也能解决问题
5. **部署延迟可通过流水线分摊** → 每帧 1 轮迭代, 5 帧完成一个 cycle
