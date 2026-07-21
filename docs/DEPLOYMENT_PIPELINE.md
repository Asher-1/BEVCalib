# BEVCalib 部署方案: TTA + 多轮迭代 Pipeline

## 目标
- 任意新车辆/安装配置，inject 3° 后 Recovery > 95%
- 部署延迟要求: 首帧标定 < 5s, 稳态标定 < 200ms/frame

## 部署 Pipeline 架构

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: TTA Warm-up (车辆首次上电/标定初始化)              │
│  Duration: 3-5 seconds (50 frames @ 10Hz)                    │
│                                                              │
│  1. Model forward on 50 frames (no perturbation)             │
│  2. Collect predictions: euler_pred[50, 3]                   │
│  3. Compute Affine correction:                               │
│     bias = mean(euler_pred)  ← 这就是 ZD 估计               │
│     scale = 1.0 (or learned from temporal consistency)        │
│  4. Cache: affine_params = {bias, scale}                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Iterative Inference (稳态标定)                     │
│  Duration: ~1s per calibration cycle (5 iterations × 200ms)  │
│                                                              │
│  for iteration in 1..5:                                      │
│    1. Model forward: pred = model(image, pcd, T_current)     │
│    2. Apply Affine: corrected = pred * scale + bias          │
│    3. Update: T_current = apply_correction(T_current, corr)  │
│    4. Check convergence: if |corrected| < threshold → break  │
│  end                                                         │
│                                                              │
│  Output: T_calibrated = T_current                            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: Online ZD Tracking (可选, 持续运行)                │
│                                                              │
│  每 100 帧更新 ZD 估计 (滑动窗口):                           │
│    new_bias = 0.9 * old_bias + 0.1 * recent_mean_pred       │
│                                                              │
│  用途: 处理 ZD 随时间缓慢漂移的情况                          │
│  (如: 温度变化导致相机安装微变)                              │
└─────────────────────────────────────────────────────────────┘
```

## 部署代码接口设计

```python
class BEVCalibDeployment:
    def __init__(self, model_path, device='cuda'):
        self.model = load_model(model_path, device)
        self.affine_bias = torch.zeros(3)   # RPY bias (radians)
        self.affine_scale = torch.ones(3)   # RPY scale
        self.is_adapted = False
        self.zd_history = []

    def warmup(self, frames: List[Frame], n_frames=50) -> dict:
        """Phase 1: TTA Warm-up"""
        predictions = []
        for frame in frames[:n_frames]:
            pred = self.model.forward(frame)
            predictions.append(pred.euler_rad)
        
        predictions = torch.stack(predictions)
        self.affine_bias = predictions.mean(dim=0)
        self.is_adapted = True
        
        return {
            'zd_estimate_deg': self.affine_bias.numpy() * 180/np.pi,
            'zd_std_deg': predictions.std(dim=0).numpy() * 180/np.pi,
            'confidence': 1 - predictions.std(dim=0).mean().item() / 0.1
        }

    def calibrate(self, frame: Frame, max_iter=5, threshold_deg=0.05) -> CalibResult:
        """Phase 2: Iterative Inference"""
        T_current = frame.init_T
        
        for it in range(max_iter):
            pred = self.model.forward(frame, T_current)
            corrected = pred.euler_rad * self.affine_scale - self.affine_bias
            
            if corrected.norm() * 180/np.pi < threshold_deg:
                break
            
            T_current = apply_correction(T_current, corrected)
        
        self.zd_history.append(pred.euler_rad.detach())
        
        return CalibResult(
            T_calibrated=T_current,
            confidence=1.0 - corrected.norm().item(),
            n_iterations=it + 1
        )

    def update_zd(self, window=100):
        """Phase 3: Online ZD Tracking"""
        if len(self.zd_history) >= window:
            recent = torch.stack(self.zd_history[-window:])
            new_bias = recent.mean(dim=0)
            self.affine_bias = 0.9 * self.affine_bias + 0.1 * new_bias
```

## 风险分析 (Task E)

### 风险 1: ZD 非恒定 (帧间变化)
- **证据**: V66 seq00 ZD std=0.11°, seq01 std=0.08°
- **影响**: Affine TTA 只能消除 ZD 的稳态分量, 无法跟踪帧间变化
- **缓解**: Phase 3 滑窗跟踪 + 选择 ZD 稳定的帧段做估计

### 风险 2: 迭代推理在 OOD 场景 J < 0.61
- **证据**: V62 测试域 effective J=0.61 是在 inject 3° 的首轮
- **影响**: 后续迭代 J 可能更低 (小残差时模型不敏感)
- **缓解**: V67 的迭代监督专门训练小残差场景的 J

### 风险 3: Affine TTA warm-up 期间标定不可用
- **证据**: 需要 50 帧 (~5s) 才能估计 ZD
- **影响**: 系统上电后前 5s 无可靠标定
- **缓解**: 使用上次关机前保存的 ZD 参数作为初值, 渐进更新

### 风险 4: Affine 假设过于简单 (线性)
- **证据**: 实际 ZD 可能随 Pitch 角度非线性变化
- **影响**: 部分帧的 ZD 去除不完全
- **缓解**: 升级到 C4 方案 (LoRA TTA) 或使用分段线性

### 风险 5: 5 轮迭代延迟过大
- **证据**: 每次推理 ~150ms (960×540, L20), 5轮 = 750ms
- **影响**: 高速场景下标定延迟可能导致使用过时标定
- **缓解**: 
  - 边沿触发: 只在检测到大偏移时触发完整 5 轮
  - 流水线: 迭代过程可跨帧分摊 (每帧只做 1 轮, 5帧完成一个cycle)

### 综合风险评级

| 风险 | 严重性 | 概率 | 对策成熟度 |
|------|--------|------|-----------|
| ZD 非恒定 | 中 | 40% | 高 (滑窗跟踪) |
| 小残差J下降 | 高 | 60% | 高 (V67迭代监督) |
| Warm-up期 | 低 | 100% | 高 (缓存上次值) |
| 线性假设不足 | 低 | 20% | 中 (可升级LoRA) |
| 迭代延迟 | 中 | 30% | 高 (流水线) |

## 预期性能

| 场景 | 方案 | 预测 Recovery | 延迟 |
|------|------|-------------|------|
| 最优: V67 + TTA + 5iter | V67(J≈0.7)+TTA(95%) | **98%+** | ~750ms |
| 保守: V62 + TTA + 5iter | V62(J≈0.6)+TTA(95%) | **98.4%** | ~750ms |
| 快速: V67 + TTA + 3iter | V67(J≈0.7)+TTA(95%) | **95-97%** | ~450ms |
| 最小: V62 + TTA + 3iter | V62(J≈0.6)+TTA(95%) | **92.6%** | ~450ms |

## 建议

1. **Phase 1 立即可部署**: 使用 V62 + Affine TTA + 5 迭代
2. **Phase 2 提升效率**: V67 完成后, 3 迭代即可达标
3. **Phase 3 终极优化**: 自监督微调使 J→0.8, 2 迭代达标
