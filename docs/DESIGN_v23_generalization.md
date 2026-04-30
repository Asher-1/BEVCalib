# V23 泛化优化方案设计文档

## 1. 问题诊断

### 1.1 核心发现 (V22 Diagnosis 泛化评估)

| 模型 | Val Rot | Gen Rot | Gap |
|------|---------|---------|-----|
| A2 full (16N, bb=0.5) | 0.07° | 1.027° | 14.7x |
| A1 full (16N, bb=0.1) | 0.09° | 0.774° | 8.6x |
| B1 quick (1N, joint) | 0.20° | 0.665° | 3.3x |
| A1 quick (1N, bb=0.1) | 0.18° | 0.670° | 3.7x |

**关键结论**: 训练精度最优 ≠ 泛化最优。过拟合是核心瓶颈。

### 1.2 根因分析

参考 "What Really Matters for Learning-based LiDAR-Camera Calibration" (Huang et al., 2025):

1. **回归式标定方法本质是检索网络**: 学习深度图空间分布→参数映射，而非跨模态对应关系
2. **随机扰动数据生成管线有固有缺陷**: 仅生成 de-calibration 数据，不代表真实标定多样性
3. **多节点大 batch 加剧过拟合**: SGD 噪声减小→隐性正则化消失→sharp minimum
4. **Pitch 误差占主导** (0.422°/1.018° ≈ 41% of axis sum, 且为最大单轴误差): 不同序列 pitch 特征分布差异最大

### 1.3 泛化目标

| 指标 | 当前最佳 | 目标 | 需改善 |
|------|---------|------|--------|
| Mean Rot Total | 0.665° | < 0.30° | 2.2x |
| Roll | 0.324° | < 0.10° | 3.2x |
| Pitch | 0.422° | < 0.10° | 4.2x |
| Yaw | 0.272° | < 0.10° | 2.7x |

## 2. 解决方案概览

采用**三层堆叠**策略，每层独立且正交叠加:

```
Layer 1: V23 训练优化 (正则化 + joint + intrinsic_input)
          预期: 0.665° → 0.40°

Layer 2: SWAD 权重平均 (寻找平坦极小值)
          预期: 0.40° → 0.35°

Layer 3: 迭代几何精炼 (ICP / Chamfer Distance 后处理)
          预期: 0.35° → 0.22-0.35° (保守)
```

## 3. Layer 1: V23 训练优化

### 3.1 配置文件

- `configs/batch8_train_all_v23_optimized.yaml` (32N full)
- `configs/batch8_train_all_v23_optimized_quick.yaml` (1N quick)

### 3.2 核心改动 (vs V22)

| 维度 | V22 | V23 |
|------|-----|-----|
| 优化模式 | rotation_only | **joint** (B1 冠军策略) |
| LR | lr=1e-4, bb=0.1 | lr=1e-4, **bb=0.5** (A2 最优) |
| intrinsic_input | 部分实验 | **默认开启** |
| AMP | 默认开启 | **no_amp=1** (FP32) |
| 正则化 | 标准 | **cam_drop/强dropout/强增强** |

### 3.3 实验矩阵 (8组)

| 实验 | Joint | cam_drop | dp/hd | intr_aug | scene_aug | LR sched |
|------|-------|----------|-------|----------|-----------|----------|
| A1 | No | 0 | 0.1/0.1 | 0% | standard | Step |
| A2 | Yes | 0 | 0.1/0.1 | 0% | standard | Step |
| B1 | Yes | 0.15 | 0.1/0.1 | 0% | standard | Step |
| B2 | Yes | 0 | 0.2/0.2 | 0% | standard | Step |
| B3 | Yes | 0 | 0.1/0.1 | ±5% | standard | Step |
| B4 | Yes | 0 | 0.1/0.1 | 0% | strong | Step |
| C1 | Yes | 0.15 | 0.2/0.2 | ±5% | strong | Step |
| C2 | Yes | 0.15 | 0.2/0.2 | ±5% | strong | CosineWR |

## 4. Layer 2: SWAD 权重平均

### 4.1 原理

SWAD (Stochastic Weight Averaging Densely, NeurIPS'21) 通过在训练过程中密集收集权重快照并平均，找到 loss landscape 中更平坦的极小值。

**为什么适合我们**:
- 我们的核心问题是 sharp minimum (训练好但泛化差)
- SWAD 在 DomainBed 上是 DG SOTA 方法
- 实现简单，不增加推理开销
- 与其他正则化手段正交叠加

### 4.2 算法

```
输入: 训练过程中的 eval 检查点序列 {θ_1, θ_2, ..., θ_T}
      对应验证 loss {L_1, L_2, ..., L_T}

1. 寻找 T_start:
   - 监控验证 loss
   - 当 loss 首次达到局部最优后 patience_start 个 eval 周期: T_start

2. 收集快照:
   - 从 T_start 开始，每次 eval 保存权重快照到 CPU

3. 寻找 T_end (过拟合检测):
   - 当验证 loss 持续上升 patience_end 个 eval 周期: T_end
   - 或训练结束

4. 计算 SWAD 模型:
   θ_SWAD = (1/N) * Σ θ_i, for i in [T_start, T_end]

5. 保存 ckpt_swad.pth
```

### 4.3 代码修改

**文件**: `kitti-bev-calib/train_kitti.py`

新增 `SWADCollector` 类:

```python
class SWADCollector:
    """SWAD: Stochastic Weight Averaging Densely (NeurIPS'21)
    
    在训练过程中密集收集权重快照, 自动检测最优收集窗口:
    - T_start: 验证 loss 首次达到局部最优后 patience_start 个 eval 周期
    - T_end: 验证 loss 持续上升 patience_end 个 eval 周期 (过拟合检测)
    """
    
    def __init__(self, patience_start=3, patience_end=3, tolerance=0.02):
        self.patience_start = patience_start
        self.patience_end = patience_end
        self.tolerance = tolerance
        
        self.snapshots = []
        self.snapshot_epochs = []
        self.val_losses = []
        
        self.started = False
        self.ended = False
        self.best_val_loss = float('inf')
        self.best_loss_idx = -1
        self.no_improve_count = 0
        
        # Phase 2 独立跟踪
        self._phase2_best = float('inf')
        self._phase2_losses = []  # 仅 Phase 2 期间的 loss (避免被 Phase 1 污染)
        
    def update(self, model, val_loss, epoch):
        """每次 eval 后调用。注意: 仅在 is_main rank 上调用以避免冗余内存占用。"""
        import math
        if math.isnan(val_loss) or math.isinf(val_loss):
            return  # 跳过异常 eval, 避免损坏 SWAD 状态
        eval_idx = len(self.val_losses)
        self.val_losses.append(val_loss)
        
        # Phase 1: 等待 loss 达到局部最优
        if not self.started:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_loss_idx = eval_idx
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
                if self.no_improve_count >= self.patience_start:
                    self.started = True
                    self._phase2_best = self.best_val_loss
        
        # Phase 2: 收集快照 + 过拟合检测
        if self.started and not self.ended:
            raw_model = model.module if hasattr(model, 'module') else model
            snapshot = {k: v.clone().cpu() 
                       for k, v in raw_model.state_dict().items()}
            self.snapshots.append(snapshot)
            self.snapshot_epochs.append(epoch)
            self._phase2_losses.append(val_loss)
            
            # 收集期间持续跟踪最佳 loss
            if val_loss < self._phase2_best:
                self._phase2_best = val_loss
            
            # 过拟合检测: 仅基于 Phase 2 的 loss (避免 Phase 1 旧值误触发)
            threshold = self._phase2_best * (1 + self.tolerance)
            if (len(self._phase2_losses) >= self.patience_end and 
                all(v > threshold 
                    for v in self._phase2_losses[-self.patience_end:])):
                self.ended = True
                n_remove = min(self.patience_end, len(self.snapshots))
                if n_remove > 0:
                    self.snapshots = self.snapshots[:-n_remove]
                    self.snapshot_epochs = self.snapshot_epochs[:-n_remove]
                
    def get_averaged_state_dict(self):
        """返回平均后的权重 (跳过 BN 统计量, 使用最后快照的)
        
        注: SWA/SWAD 论文建议平均后重跑训练数据更新 BN stats。
        此处简化为使用最后快照的 BN stats, 因 SWAD 收集的是邻近快照,
        权重差异小, BN 偏差有限。如需更精确, 可在保存后额外执行:
            model.load_state_dict(swad_state); model.train()
            for batch in train_loader: model(batch)  # 仅更新 BN stats
        """
        if not self.snapshots:
            return None
        n = len(self.snapshots)
        avg = {}
        for key in self.snapshots[0]:
            if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                avg[key] = self.snapshots[-1][key]
            else:
                # 逐步累加避免 O(N) 内存 (不用 torch.stack)
                total = self.snapshots[0][key].float().clone()
                for s in self.snapshots[1:]:
                    total += s[key].float()
                avg[key] = total / n
        return avg
        
    def get_info(self):
        return {
            'n_snapshots': len(self.snapshots),
            'snapshot_epochs': self.snapshot_epochs,
            'started': self.started,
            'ended': self.ended,
            'best_val_loss': self.best_val_loss,
            'phase2_best': self._phase2_best,
            'total_evals': len(self.val_losses),
        }
```

集成到训练循环 (train_kitti.py 的 eval 阶段):

```python
# 初始化 (训练开始时, 仅 rank 0)
if args.swad and is_main:
    swad = SWADCollector(
        patience_start=args.swad_patience_start,
        patience_end=args.swad_patience_end,
        tolerance=args.swad_tolerance
    )

# 每次 eval 后 (在 best_val 判断之后, 仅 rank 0 收集快照以节省内存)
# 注意: SWAD 固定使用 rot_error 作为过拟合检测指标 (而非 best_val 的复合分数),
# 因为泛化优化的核心目标是旋转精度, translation 的泛化差距较小
if args.swad and is_main:
    swad.update(model, val_pose_errors['rot_error'], epoch)
    info = swad.get_info()
    tprint(f"[SWAD] started={info['started']}, "
           f"snapshots={info['n_snapshots']}, "
           f"ended={info['ended']}, "
           f"phase2_best={info['phase2_best']:.4f}")

# 训练结束后
if args.swad and is_main:
    swad_state = swad.get_averaged_state_dict()
    if swad_state:
        swad_ckpt = {
            'model_state_dict': swad_state,
            'swad_info': swad.get_info(),
            'args': vars(args),
            'epoch': -1,
            'rotation_only': rotation_only,
        }
        swad_ckpt.update(_build_ckpt_metadata(model, args))
        swad_path = os.path.join(ckpt_save_dir, 'ckpt_swad.pth')
        torch.save(swad_ckpt, swad_path)
        tprint(f"[SWAD] Saved: {swad_path} "
               f"({swad.get_info()['n_snapshots']} snapshots averaged)")
    else:
        tprint("[SWAD] WARNING: No snapshots collected, skipping SWAD save")
```

新增 argparse 参数:

```python
parser.add_argument("--swad", action="store_true",
                    help="Enable SWAD (Stochastic Weight Averaging Densely)")
parser.add_argument("--swad_patience_start", type=int, default=3,
                    help="SWAD: eval cycles after best loss before start collecting")
parser.add_argument("--swad_patience_end", type=int, default=3,
                    help="SWAD: eval cycles of rising loss to stop collecting")
parser.add_argument("--swad_tolerance", type=float, default=0.02,
                    help="SWAD: loss rise tolerance ratio for overfit detection")
```

**文件**: `batch_train.sh`

在 OPTIM_PARAMS 列表中新增:
```bash
('swad', '--swad'),
('swad_patience_start', '--swad_patience_start'),
('swad_patience_end', '--swad_patience_end'),
('swad_tolerance', '--swad_tolerance'),
```

注: `swad` 是 boolean flag, `batch_train.sh` 中需要处理为:
```bash
# 对于 boolean flag, 当 YAML 值为 true 时传递 --swad, 为 false 时不传
if [ "$SWAD" = "true" ] || [ "$SWAD" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --swad"
fi
```

### 4.4 SWAD 参数调优指南

| 参数 | 默认值 | 调优建议 |
|------|--------|---------|
| `patience_start` | 3 | 对于400 epoch/50 eval = 8次eval, 3次≈150个epoch |
| `patience_end` | 3 | 过拟合检测, 3次≈150个epoch的上升 |
| `tolerance` | 0.02 | 2% loss上升视为过拟合, 可调至0.05更保守 |

**快速模式**: 如果总 eval 次数少 (如 quick 版), 减小 patience:
```yaml
swad_patience_start: 2
swad_patience_end: 2
```

### 4.5 显存/存储影响

- 快照存 CPU RAM，不占 GPU
- 假设 400 epochs / 50 eval_epoches = 8 次 eval
- 收集 ~4 个快照，每个 ~100MB → 总 ~400MB CPU RAM
- 完全可接受
- DDP 模式: 只有 rank 0 收集快照

### 4.5.1 边界情况

**无过拟合 (loss 持续改善)**: Phase 1 永不结束，SWAD 不产生输出。这是期望行为：如果模型不过拟合，则不需要 SWAD。对应实验仅使用 `ckpt_best_val.pth`。

**极少 eval 次数 (quick 模式 4-6 次 eval)**: 即使 `patience_start=2`，Phase 2 可能只收集 1-2 个快照，平均效果有限。建议 quick 模式使用 `patience_start=1`。

### 4.5.2 训练中断恢复

SWAD 状态不保存到 checkpoint 中 (因快照数据量大)。如果训练中断并恢复:
- SWAD 从头开始收集 (Phase 1 重新检测)
- 这通常可接受: 恢复后模型已在合理 loss 范围内，Phase 1 会快速跳过
- 如需完整保留 SWAD 状态，可将 `_phase2_losses` 和 `snapshot_epochs` 保存到 checkpoint (不保存快照本身)，恢复时重新进入 Phase 2 收集

### 4.5.3 配置文件更新

SWAD 实现后，需在 V23 训练 YAML 的 `defaults.params` 中添加:

```yaml
# configs/batch8_train_all_v23_optimized.yaml
# configs/batch8_train_all_v23_optimized_quick.yaml
defaults:
  params:
    swad: true
    swad_patience_start: 3   # quick 版可改为 2
    swad_patience_end: 3     # quick 版可改为 2
    swad_tolerance: 0.02
```

### 4.6 评估方法

- 对每个 V23 实验，同时保存 `ckpt_best_val.pth` 和 `ckpt_swad.pth`
- 评估时对比两者的泛化性能
- 在 eval YAML 中为每个模型添加 SWAD 变体:

```yaml
models:
  - label: "v23_A2_joint"
    dir_name: "model_small_5deg_v23_A2_joint"
    ckpt: "ckpt_best_val.pth"
    mode_desc: "A2 joint (best_val)"
    
  - label: "v23_A2_joint_swad"
    dir_name: "model_small_5deg_v23_A2_joint"
    ckpt: "ckpt_swad.pth"
    mode_desc: "A2 joint (SWAD)"
```

## 5. Layer 3: 迭代几何精炼

### 5.1 方法选择

| 方法 | 原理 | 精度 | 速度 | 实现复杂度 |
|------|------|------|------|-----------|
| **ICP** | 最近邻点匹配+变换优化 | 中等 | 快 | 低 |
| **Chamfer Distance TTA** | 梯度下降优化投影一致性 | 高 | 中 | 中 |
| **Edge-based** | 投影点云与图像边缘对齐 | 中高 | 中 | 中 |

### 5.2 方法 A: ICP 精炼

**输入**: 点云, 图像, 内参K, 初始标定, BEVCalib预测的delta

**算法**:
```
1. 将 BEVCalib 预测的 delta_R, delta_T 应用到初始标定
2. 提取图像 Canny 边缘, 构建 KD-Tree
3. 投影3D点云到2D, 查询最近边缘点
4. 使用 solvePnP 最小化重投影误差:
   min_{R,t} Σ ||π(K · [R|t] · P_i) - q_i||²
   其中 P_i 是3D世界点, q_i 是匹配的边缘点, π 是投影函数
5. 阻尼迭代直到收敛 (通常 5-20 步, 单步 ≤0.02rad)
```

**依赖**: OpenCV (`cv2.solvePnP`), scipy (`cKDTree`)

**预期**:
- 当初始误差 < 1° 时可靠收敛
- 典型改善: 10-30% 误差降低 (学术论文典型范围)

### 5.3 方法 B: Chamfer Distance TTA

**输入**: 训练好的模型, 测试样本

**算法**:
```
1. 前向推理得到 delta_R, delta_T
2. 将 delta 参数设为可优化
3. 定义 Chamfer Distance loss:
   - 使用预测的 delta 投影点云到图像
   - 计算投影点与图像结构特征的距离
4. 梯度下降 5-10 步优化 delta
5. 输出精炼后的 delta
```

**关键**: loss function 设计
- **投影深度一致性**: 投影点云深度 vs 图像估计深度
- **边缘对齐**: 投影点云轮廓 vs 图像边缘 (Canny/Sobel)
- **Chamfer Distance**: 投影2D点集 vs 图像特征点集

**预期**:
- 每样本精炼时间: ~0.1-0.5s
- 典型改善: 15-40% 误差降低 (优于 ICP，因可微分优化更精确)
- 对 Pitch 特别有效 (几何信号最强)
- 注意: 在 <0.5° 误差范围内效果递减，因信号接近边缘检测噪声

### 5.4 实现文件

**新建**: `tools/refine_calibration.py`

```python
import torch
import numpy as np
from scipy.spatial import cKDTree

class CalibrationRefiner:
    """标定预测后处理精炼器
    
    支持两种方法:
    - ICP: 基于最近邻点匹配的经典优化 (快速, 无需GPU)
    - Chamfer: 基于梯度下降的可微分优化 (更精确, 需要GPU)
    """
    
    def __init__(self, method='icp', max_iter=20, lr=0.001, rotation_only=True):
        self.method = method
        self.max_iter = max_iter
        self.lr = lr
        self.rotation_only = rotation_only
        
    def refine(self, point_cloud, image, K, T_initial, delta_pred):
        """
        Args:
            point_cloud: (N, 3) numpy array, LiDAR点云
            image: (H, W, 3) numpy array, 相机图像
            K: (3, 3) numpy array, 相机内参
            T_initial: (4, 4) numpy array, 初始外参标定
            delta_pred: dict, BEVCalib预测的 {'rotation': (3,), 'translation': (3,)}
                        rotation 为 euler angles (rad), translation 为 m
                        
        Returns:
            delta_refined: dict, 精炼后的 {'rotation': (3,), 'translation': (3,)}
        """
        if self.method == 'icp':
            result = self._refine_icp(point_cloud, image, K, 
                                       T_initial, delta_pred)
        elif self.method == 'chamfer':
            result = self._refine_chamfer(point_cloud, image, K,
                                          T_initial, delta_pred)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # rotation_only 安全保障: ICP/Chamfer 内部可能引入微小 translation 分量
        # (旋转-平移耦合导致), 此处强制保留原始 translation
        if self.rotation_only:
            result['translation'] = delta_pred['translation'].copy()
        return result
            
    def _refine_icp(self, point_cloud, image, K, T_initial, delta_pred):
        """
        ICP 精炼算法:
        1. 将 BEVCalib 预测应用到初始标定
        2. 提取图像深度边缘特征
        3. 投影点云到图像, 与边缘对齐
        4. 使用 solvePnP 迭代优化直到收敛
        """
        import cv2
        from scipy.spatial.transform import Rotation as R_scipy
        
        T_current = apply_delta(T_initial, delta_pred)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_points = np.array(np.where(edges > 0)).T  # (M, 2) y,x
        edge_points = edge_points[:, ::-1].astype(np.float64)  # (M, 2) x,y
        
        if len(edge_points) < 100:
            return delta_pred
        
        edge_tree = cKDTree(edge_points)
        H, W = image.shape[:2]
        K_cv = K.astype(np.float64)
        prev_residual = float('inf')
        
        for iteration in range(self.max_iter):
            pc_cam = (T_current[:3, :3] @ point_cloud.T + 
                      T_current[:3, 3:4])  # (3, N)
            
            valid = pc_cam[2, :] > 0.1
            pc_cam_valid = pc_cam[:, valid]
            pc_world_valid = point_cloud[valid]
            
            proj_2d = K @ pc_cam_valid
            proj_2d = proj_2d[:2, :] / proj_2d[2:3, :]
            proj_2d = proj_2d.T  # (N', 2)
            
            in_image = ((proj_2d[:, 0] >= 0) & (proj_2d[:, 0] < W) &
                       (proj_2d[:, 1] >= 0) & (proj_2d[:, 1] < H))
            proj_2d = proj_2d[in_image]
            pc_3d_world = pc_world_valid[in_image]
            
            if len(proj_2d) < 50:
                break
                
            dists, indices = edge_tree.query(proj_2d, k=1)
            
            good = dists < 50
            if good.sum() < 30:
                break
                
            tgt_2d = edge_points[indices[good]]
            src_3d = pc_3d_world[good]
            
            mean_residual = np.linalg.norm(
                tgt_2d - proj_2d[good], axis=1).mean()
            
            if mean_residual < 0.5:
                break
            if abs(prev_residual - mean_residual) < 0.1:
                break
            prev_residual = mean_residual
            
            src_3d_cv = src_3d.astype(np.float64)
            tgt_2d_cv = tgt_2d.astype(np.float64)
            
            rvec_init, _ = cv2.Rodrigues(T_current[:3, :3].astype(np.float64))
            tvec_init = T_current[:3, 3].astype(np.float64).reshape(3, 1)
            
            success, rvec, tvec = cv2.solvePnP(
                src_3d_cv, tgt_2d_cv, K_cv, None,
                rvec=rvec_init, tvec=tvec_init,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                break
                
            R_new, _ = cv2.Rodrigues(rvec)
            T_new = np.eye(4)
            T_new[:3, :3] = R_new
            T_new[:3, 3] = tvec.flatten() if not self.rotation_only else T_current[:3, 3].copy()
            
            # 限制单步更新幅度 (防止跳跃)
            delta_step = T_new @ np.linalg.inv(T_current)
            step_angle = np.linalg.norm(
                R_scipy.from_matrix(delta_step[:3, :3]).as_rotvec())
            if step_angle > 0.02:  # > 1.15° 单步太大, 做阻尼
                damping = 0.02 / step_angle
                rotvec_step = R_scipy.from_matrix(delta_step[:3, :3]).as_rotvec()
                delta_step[:3, :3] = R_scipy.from_rotvec(
                    rotvec_step * damping).as_matrix()
                delta_step[:3, 3] *= damping
            
            T_current = delta_step @ T_current
        
        return matrix_to_delta(T_current, T_initial)
    
    def _refine_chamfer(self, point_cloud, image, K, 
                         T_initial, delta_pred):
        """
        Chamfer Distance 精炼 (可微分, 梯度下降)
        使用 rotation vector (轴角) 作为优化变量, Rodrigues 公式构建旋转矩阵
        """
        import cv2
        from scipy.spatial.transform import Rotation as R_scipy
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        H, W = image.shape[:2]
        
        pc = torch.tensor(point_cloud, dtype=torch.float32, device=device)
        K_t = torch.tensor(K, dtype=torch.float32, device=device)
        T_init = torch.tensor(T_initial, dtype=torch.float32, device=device)
        
        # 将 Euler 角转换为 rotation vector (轴角表示)
        rotvec_np = R_scipy.from_euler('xyz', delta_pred['rotation']).as_rotvec()
        rotvec_init = torch.tensor(rotvec_np, dtype=torch.float32, device=device)
        trans_init = torch.tensor(delta_pred['translation'], dtype=torch.float32,
                                  device=device)
        
        rotvec = rotvec_init.clone().requires_grad_(True)
        if self.rotation_only:
            trans = trans_init.clone()  # 不优化 translation
        else:
            trans = trans_init.clone().requires_grad_(True)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_pts = np.array(np.where(edges > 0)).T[:, ::-1].astype(np.float32)
        if len(edge_pts) < 100:
            return delta_pred
        edge_t = torch.tensor(edge_pts, device=device)
        
        opt_params = [rotvec] if self.rotation_only else [rotvec, trans]
        optimizer = torch.optim.Adam(opt_params, lr=self.lr)
        
        max_refine_rot = 0.02    # ±1.15° rotation vector norm 限制
        max_refine_trans = 0.05  # ±5cm
        
        for step in range(self.max_iter):
            optimizer.zero_grad()
            
            T_delta = rodrigues_to_matrix(rotvec, trans)
            T_curr = T_delta @ T_init
            
            pc_cam = (T_curr[:3, :3] @ pc.T + T_curr[:3, 3:4])
            valid = pc_cam[2, :] > 0.1
            pc_cam = pc_cam[:, valid]
            
            proj = K_t @ pc_cam
            proj_2d = proj[:2, :] / proj[2:3, :]
            proj_2d = proj_2d.T  # (N, 2)
            
            # 过滤图像范围外的投影点
            in_image = ((proj_2d[:, 0] >= 0) & (proj_2d[:, 0] < W) &
                       (proj_2d[:, 1] >= 0) & (proj_2d[:, 1] < H))
            proj_2d = proj_2d[in_image]
            
            if len(proj_2d) < 50:
                break
            
            # 随机子采样加速 cdist (每步不同样本, 类似 SGD 效果)
            # 注: 不固定 seed, 因随机性有助于跳出局部极值; 如需复现可设 torch.manual_seed
            n_sample = min(2000, len(proj_2d))
            idx = torch.randperm(len(proj_2d), device=device)[:n_sample]
            src = proj_2d[idx]
            
            dists = torch.cdist(src.unsqueeze(0), 
                               edge_t.unsqueeze(0))  # (1, n, M)
            min_dists = dists.min(dim=2).values  # (1, n)
            
            loss = min_dists.mean()
            loss.backward()
            optimizer.step()
            
            # 相对 clamp: 限制与初始预测的偏差
            with torch.no_grad():
                # rotation: norm-based clamp (限制实际旋转角度)
                delta_rv = rotvec - rotvec_init
                delta_norm = delta_rv.norm()
                if delta_norm > max_refine_rot:
                    rotvec.copy_(rotvec_init + delta_rv * (max_refine_rot / delta_norm))
                # translation: per-component clamp (各轴独立限制, rotation_only 时跳过)
                if not self.rotation_only:
                    trans.clamp_(trans_init - max_refine_trans,
                                trans_init + max_refine_trans)
        
        # 将 rotation vector 转回 Euler 角 (与系统其他部分一致)
        rotvec_result = rotvec.detach().cpu().numpy()
        euler_result = R_scipy.from_rotvec(rotvec_result).as_euler('xyz')
        return {
            'rotation': euler_result,
            'translation': trans.detach().cpu().numpy()
        }
```

辅助函数:

```python
def apply_delta(T_initial, delta):
    """将 delta (euler + translation) 应用到 T_initial"""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler('xyz', delta['rotation']).as_matrix()
    T_delta = np.eye(4)
    T_delta[:3, :3] = R
    T_delta[:3, 3] = delta['translation']
    return T_delta @ T_initial

def matrix_to_delta(T_current, T_initial):
    """从 T_current 和 T_initial 计算 delta"""
    from scipy.spatial.transform import Rotation
    T_delta = T_current @ np.linalg.inv(T_initial)
    rot = Rotation.from_matrix(T_delta[:3, :3]).as_euler('xyz')
    trans = T_delta[:3, 3]
    return {'rotation': rot, 'translation': trans}

def rodrigues_to_matrix(rotvec, trans):
    """可微分的 rotation vector → 4x4 变换矩阵 (Rodrigues 公式, 精确)
    
    全程可微, 无 if/else 分支, 对 θ≈0 梯度安全.
    
    Args:
        rotvec: (3,) rotation vector (axis * angle)
        trans:  (3,) translation vector
    Returns:
        T: (4, 4) transformation matrix, R 保证正交
    """
    # clamp 避免 norm(0) 的 NaN 梯度, 保持可微
    theta_sq = (rotvec * rotvec).sum()
    theta = torch.sqrt(theta_sq.clamp(min=1e-12))
    
    k = rotvec / theta
    K = torch.zeros(3, 3, device=rotvec.device, dtype=rotvec.dtype)
    K[0, 1] = -k[2]; K[0, 2] = k[1]
    K[1, 0] = k[2];  K[1, 2] = -k[0]
    K[2, 0] = -k[1]; K[2, 1] = k[0]
    
    # Rodrigues: R = I + sin(θ)*K + (1-cos(θ))*K²
    # 当 θ→0: sin(θ)/θ→1, (1-cos(θ))/θ²→0.5, 自然退化为 R≈I
    R = (torch.eye(3, device=rotvec.device, dtype=rotvec.dtype) + 
         torch.sin(theta) * K + 
         (1 - torch.cos(theta)) * (K @ K))
    
    T = torch.eye(4, device=rotvec.device, dtype=rotvec.dtype)
    T[:3, :3] = R
    T[:3, 3] = trans
    return T
```

**集成到评估**: `evaluate_checkpoint.py` 新增 `--refine` 参数:

```python
parser.add_argument("--refine", type=str, default="none",
                    choices=["none", "icp", "chamfer"],
                    help="Post-processing refinement method")
parser.add_argument("--refine_steps", type=int, default=20,
                    help="Refinement iteration steps")
parser.add_argument("--refine_lr", type=float, default=0.001,
                    help="Chamfer refinement learning rate")
# rotation_only 自动从 checkpoint 读取, 传递给 CalibrationRefiner
```

注: `rotation_only` 已从 checkpoint 自动检测，无需额外参数。refinement 自动适配:
- `rotation_only=True`: 仅优化旋转，translation 保持固定
- `rotation_only=False`: 同时优化旋转和平移

评估循环中的集成位置 (`evaluate_checkpoint.py` 的 eval loop):

```python
# 在 model forward 之后, compute_pose_errors 之前:
if args.refine != "none":
    from tools.refine_calibration import CalibrationRefiner, matrix_to_delta, apply_delta
    refiner = CalibrationRefiner(
        method=args.refine, max_iter=args.refine_steps,
        lr=args.refine_lr, rotation_only=rotation_only)

# 在 per-sample 循环中 (T_pred_np 已有, init_T_to_camera_np 已有):
if args.refine != "none":
    for i in range(len(T_pred_np)):
        delta_pred = matrix_to_delta(T_pred_np[i], init_T_to_camera_np[i])
        delta_refined = refiner.refine(
            point_cloud=pcs_np[i, :, :3],
            image=imgs_np[i],      # 已 resize 的 BGR 图像
            K=np.array(intrinsics)[i],  # 已 resize 调整的内参
            T_initial=init_T_to_camera_np[i],
            delta_pred=delta_pred)
        T_pred_np[i] = apply_delta(init_T_to_camera_np[i], delta_refined)
    # 后续 compute_pose_errors 自动使用精炼后的 T_pred_np
```

**注**: `apply_delta` 使用左乘约定 `T_delta @ T_initial`，与 BEVCalib 模型的 `T_pred = loss_fn` 输出的 `T_gt_expected` 一致 (已通过 `bev_calib.py` 确认)。

使用方式:
```bash
# 无精炼 (默认)
python evaluate_checkpoint.py --ckpt path/to/ckpt.pth

# ICP 精炼
python evaluate_checkpoint.py --ckpt path/to/ckpt.pth --refine icp --refine_steps 20

# Chamfer 精炼
python evaluate_checkpoint.py --ckpt path/to/ckpt.pth --refine chamfer --refine_steps 10 --refine_lr 0.001
```

### 5.5 精炼参数调优指南

| 参数 | ICP 推荐 | Chamfer 推荐 | 说明 |
|------|----------|-------------|------|
| max_iter | 20 | 10 | ICP 收敛快, Chamfer 步数少但每步更精确 |
| lr | N/A | 0.001 | Chamfer 学习率, 过大会不稳定 |
| max_refine_rot | N/A | 0.02 rad (±1.15°) | **相对于初始预测**的最大偏差 |
| max_refine_trans | N/A | 0.05 m (±5cm) | **相对于初始预测**的最大偏差 |
| convergence | 0.5 pixel | N/A | ICP 收敛阈值 + 进展检测 (Δ<0.1px) |
| damping | 0.02 rad/step | N/A | ICP 单步最大旋转角度 |
| edge_threshold | Canny(50,150) | Canny(50,150) | 可调整边缘检测灵敏度 |
| min_points | 50 | 50 | 最少有效投影点 (含图像范围过滤) |
| rotation_only | 自动 | 自动 | 从 checkpoint 读取, True 时仅精炼旋转 |

### 5.6 注意事项

1. **ICP 适合大多数场景**: 当 BEVCalib 预测误差 < 1° 时, ICP 可靠收敛
2. **Chamfer 对 Pitch 更有效**: 因为 pitch 变化导致的投影位移最明显
3. **精炼增加推理延迟**: ICP ~50ms/sample, Chamfer ~200ms/sample
4. **不适用于极端误差**: 如果 BEVCalib 预测误差 > 5°, 精炼可能失败
5. **需要足够的图像纹理**: 无纹理区域 (天空/白墙) 边缘少, 精炼效果差
6. **Chamfer 显存注意**: `torch.cdist(2000, M)` 产生 2000×M 矩阵。典型 M=20000 (边缘点) 时占 ~160MB，可接受。如 M>50000 导致 OOM，需子采样 edge_t

## 6. 实施时间线

```
Day 1 上午: 实现 SWAD (train_kitti.py + batch_train.sh)
Day 1 下午: 启动 V23 Quick + SWAD 训练 (8个实验, 后台)
Day 2 上午: 实现 ICP 精炼 (tools/refine_calibration.py)
Day 2 下午: 实现 Chamfer Distance 精炼
Day 3:     V23 Quick 完成 → 评估 SWAD vs best_val
           → 测试 ICP/Chamfer 精炼效果
Day 4:     参数调优 → 联合评估 → 确认最终精度
           → 如达标, 启动 32N full 训练
```

## 7. 预期效果

| 阶段 | 乐观 | 保守 | Per-Axis (乐观) | 方法 |
|------|------|------|----------------|------|
| V22 最佳 (当前) | 0.665° | 0.665° | ~0.22° | B1 joint quick |
| V23 训练优化 | ~0.40° | ~0.50° | ~0.13° | joint + 正则化 |
| + SWAD | ~0.35° | ~0.45° | ~0.12° | 平坦极小值 |
| + ICP 精炼 | ~0.28° | ~0.40° | ~0.09° | 几何后处理 |
| + Chamfer TTA | ~0.22° | ~0.35° | ~0.07° | 梯度精炼 |

**重要说明**:
- **保守估计更接近现实**: 亚度级精炼依赖图像纹理/边缘质量，在 <0.5° 时信号接近噪声
- Layer 3 (几何精炼) 学术论文典型改善为 10-30%，非 40-70%
- **达到 0.1° 需要额外架构改进** (Phase 3/4 或匹配式方法)，仅靠 V23+SWAD+精炼大概率不够
- 多层叠加策略的核心价值：即使单层改善有限，组合效果也显著缩小与目标的差距

## 8. 风险与备选

| 风险 | 应对 |
|------|------|
| SWAD 对回归任务效果不如分类 | 可用 EMA (指数滑动平均) 替代 |
| ICP 收敛到错误极值 | 限制精炼幅度 (max delta per step) |
| Chamfer loss 梯度不稳定 | 加入 smooth loss / gradient clipping |
| 32N 训练仍然过拟合 | 减少到 8N 或增加 epoch early stopping |
| 所有方案组合仍达不到 0.1° | 考虑匹配式方法 (长期架构重写) |

## 9. 参考文献

1. Cha et al., "SWAD: Domain Generalization by Seeking Flat Minima", NeurIPS 2021
2. Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization", UAI 2018
3. Huang et al., "What Really Matters for Learning-based LiDAR-Camera Calibration", 2025
4. Wang et al., "CalibRefine: Deep Learning-Based Online Automatic Targetless LiDAR-Camera Calibration with Iterative and Attention-Driven Post-Refinement", 2025
5. Li et al., "PEGO: Parameter-Efficient Group with Orthogonal regularization for DG", 2024
6. Naseer et al., "TFS-ViT: Token-level Feature Stylization for Domain Generalization", 2023
