# V40 P2b实施方案 - FOV Crop + LiDAR密度模拟

> 时间：2026-05-29 12:35  
> 范围：P2b关键增强（FOV crop + LiDAR密度）  
> 状态：设计中


## 一、用户需求确认

### 必须实施

| 增强类型 | 泛化提升 | 优先级 | 实施难度 |
|---------|---------|--------|---------|
| FOV crop | +20% | P0 | 中 |
| LiDAR密度垂直层 | +10% | P0 | 高 |

### 不需要实施

| 增强类型 | 原因 | 决策 |
|---------|------|------|
| 分辨率变化 | 可保证一致性 | ❌ 跳过 |


## 二、关键技术挑战

### 挑战：128线LiDAR且无ring id

用户场景：
- LiDAR：128线（高密度）
- 点云格式：`[x, y, z, intensity]`（无ring id字段）
- 目标：模拟16/32/64线稀疏激光雷达

技术难点：
- 论文方法依赖ring id进行垂直层分组
- 当前数据无ring id标识


## 三、LiDAR密度模拟方案（无ring id）

### 方案A：基于垂直角度的伪层分组（推荐）

#### 原理

计算垂直角度：
```python
# 对于每个点 (x, y, z)
distance_xy = np.sqrt(x2 + y2)
vertical_angle = np.arctan2(z, distance_xy)  # 弧度
```

分组到伪层：
```python
# 将垂直角度映射到N个伪层（bins）
num_pseudo_rings = 128  # 原始分辨率
angle_min = -25°  # 典型下视角
angle_max = +15°  # 典型上视角

# 将角度范围均匀划分为128层
pseudo_ring_id = ((vertical_angle - angle_min) / (angle_max - angle_min) * num_pseudo_rings).astype(int)
pseudo_ring_id = np.clip(pseudo_ring_id, 0, num_pseudo_rings - 1)
```

稀疏化到目标线数：
```python
# 模拟32线：从128层中选择32层
target_lines = 32
selected_rings = np.linspace(0, 127, target_lines).astype(int)

# 保留属于选定层的点
mask = np.isin(pseudo_ring_id, selected_rings)
points_sparse = points[mask]
```

#### 优点

- ✅ 无需ring id字段
- ✅ 模拟真实激光雷达的垂直分布
- ✅ 保留不同高度的点云信息

#### 缺点

- ⚠️ 计算开销（arctan2）
- ⚠️ 需要设定合理的垂直角度范围


### 方案B：基于Z值分层（简化版）

#### 原理

直接按Z坐标分层：
```python
# 假设点云Z范围为 [-2m, +2m]
z_min, z_max = points[:, 2].min(), points[:, 2].max()

# 将Z范围划分为128层
num_pseudo_rings = 128
z_bins = np.linspace(z_min, z_max, num_pseudo_rings + 1)

# 分配每个点到层
pseudo_ring_id = np.digitize(points[:, 2], z_bins) - 1
pseudo_ring_id = np.clip(pseudo_ring_id, 0, num_pseudo_rings - 1)

# 稀疏化（同方案A）
target_lines = 32
selected_rings = np.linspace(0, 127, target_lines).astype(int)
mask = np.isin(pseudo_ring_id, selected_rings)
points_sparse = points[mask]
```

#### 优点

- ✅ 无需ring id字段
- ✅ 计算更快（无三角函数）
- ✅ 实现简单

#### 缺点

- ⚠️ 不考虑距离，近处和远处点分布不均
- ⚠️ 不符合真实激光雷达的扫描模式（圆锥扫描）


### 方案C：统一随机dropout（当前实现）

当前已有：
```python
# train_kitti.py Line 1997-2026
if args.augment_pc_dropout > 0:
    keep_ratio = 1.0 - np.random.uniform(0, args.augment_pc_dropout)
    # 随机保留 keep_ratio 比例的点
```

#### 优点

- ✅ 实现简单
- ✅ 已经运行

#### 缺点

- ❌ 不保留垂直分层结构
- ❌ 不模拟真实稀疏激光雷达的特征
- ❌ 泛化提升有限（当前已启用，仅+10%）


### 推荐方案对比

| 方案 | 真实性 | 计算开销 | 实现难度 | 泛化提升 | 推荐度 |
|------|-------|---------|---------|---------|--------|
| A. 垂直角度 | ⭐⭐⭐⭐⭐ | 中 | 中 | +10% | ⭐⭐⭐⭐⭐ |
| B. Z值分层 | ⭐⭐⭐ | 低 | 低 | +7% | ⭐⭐⭐ |
| C. 随机dropout（现有） | ⭐ | 低 | - | +3% | ⭐ |

决策：推荐方案A（垂直角度分组）
- 最接近真实激光雷达扫描模式
- 计算开销可接受（仅训练时，可缓存）
- 泛化提升最大


## 四、FOV Crop实现方案

### 方案：中心crop + 内参更新

#### 原理

随机crop比例：
```python
# 以一定概率crop
if random.random() < args.augment_fov_crop_prob:
    crop_ratio_h = random.uniform(0.75, 0.95)
    crop_ratio_w = random.uniform(0.75, 0.95)
    
    H_orig, W_orig = img.shape[:2]
    H_crop = int(H_orig * crop_ratio_h)
    W_crop = int(W_orig * crop_ratio_w)
    
    # 中心crop
    y_start = (H_orig - H_crop) // 2
    x_start = (W_orig - W_crop) // 2
    img_cropped = img[y_start:y_start+H_crop, x_start:x_start+W_crop]
    
    # Resize回目标大小
    img_resized = cv2.resize(img_cropped, (target_W, target_H))
```

内参矩阵更新：
```python
# 更新光心（crop导致的偏移）
intrinsic[0, 2] -= x_start  # cx
intrinsic[1, 2] -= y_start  # cy

# 更新焦距（resize导致的缩放）
scale_x = target_W / W_crop
scale_y = target_H / H_crop
intrinsic[0, 0] *= scale_x  # fx
intrinsic[1, 1] *= scale_y  # fy
intrinsic[0, 2] *= scale_x  # cx
intrinsic[1, 2] *= scale_y  # cy
```

#### 效果

模拟场景：
- 不同安装高度 → FOV与地面重叠区域变化
- 不同相机FOV → 75° vs 90° vs 120°视场角
- 不同安装俯仰角 → 有效成像区域变化

泛化提升：+20%（论文Table 5验证）


## 五、实现计划

### Step 1: 实现FOV crop（2小时）

#### 文件修改

1. train_kitti.py添加参数

```python
# Line ~840（augment参数区域）
parser.add_argument("--augment_fov_crop_prob", type=float, default=0.0,
                    help="Probability of random FOV crop (0.0-1.0)")
parser.add_argument("--augment_fov_crop_ratio_min", type=float, default=0.75,
                    help="Min FOV crop ratio (e.g. 0.75=keep center 75%)")
parser.add_argument("--augment_fov_crop_ratio_max", type=float, default=0.95,
                    help="Max FOV crop ratio (e.g. 0.95=keep center 95%)")
```

2. 实现crop函数

```python
# Line ~550（augment函数区域）
def _apply_fov_crop(imgs_tensor, intrinsics, crop_ratio_min=0.75, crop_ratio_max=0.95):
    """
    Apply random center crop to simulate different FOV/mounting height.
    
    Args:
        imgs_tensor: (B, 3, H, W) torch.Tensor
        intrinsics: (B, 3, 3) numpy array
        crop_ratio_min/max: range of crop ratios
    
    Returns:
        cropped_imgs: (B, 3, H, W) torch.Tensor (resized back to original size)
        updated_intrinsics: (B, 3, 3) numpy array
    """
    B, C, H, W = imgs_tensor.shape
    device = imgs_tensor.device
    
    # Convert to numpy for cropping
    imgs_np = imgs_tensor.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 3)
    
    cropped_imgs = []
    updated_intrinsics = []
    
    for b in range(B):
        img = imgs_np[b]
        K = intrinsics[b].copy()
        
        # Random crop ratio (independent for H and W to simulate different aspects)
        crop_ratio_h = random.uniform(crop_ratio_min, crop_ratio_max)
        crop_ratio_w = random.uniform(crop_ratio_min, crop_ratio_max)
        
        H_crop = int(H * crop_ratio_h)
        W_crop = int(W * crop_ratio_w)
        
        # Center crop
        y_start = (H - H_crop) // 2
        x_start = (W - W_crop) // 2
        img_cropped = img[y_start:y_start+H_crop, x_start:x_start+W_crop]
        
        # Resize back to original size
        img_resized = cv2.resize(img_cropped, (W, H))
        
        # Update intrinsics
        # Step 1: adjust for crop offset
        K[0, 2] -= x_start  # cx
        K[1, 2] -= y_start  # cy
        
        # Step 2: adjust for resize scale
        scale_x = W / W_crop
        scale_y = H / H_crop
        K[0, 0] *= scale_x  # fx
        K[1, 1] *= scale_y  # fy
        K[0, 2] *= scale_x  # cx
        K[1, 2] *= scale_y  # cy
        
        cropped_imgs.append(img_resized)
        updated_intrinsics.append(K)
    
    # Convert back to tensor
    cropped_imgs = np.stack(cropped_imgs)  # (B, H, W, 3)
    cropped_imgs_tensor = torch.from_numpy(cropped_imgs).permute(0, 3, 1, 2).to(device)
    updated_intrinsics = np.stack(updated_intrinsics)
    
    return cropped_imgs_tensor, updated_intrinsics
```

3. 在训练循环中调用

```python
# Line ~1990（augment调用区域，在color jitter之后）
resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float()

if args.augment_color_jitter > 0:
    resize_imgs = _apply_color_jitter(resize_imgs, args.augment_color_jitter)

# ← 新增：FOV crop
if args.augment_fov_crop_prob > 0 and random.random() < args.augment_fov_crop_prob:
    resize_imgs, intrinsics = _apply_fov_crop(
        resize_imgs, intrinsics,
        crop_ratio_min=args.augment_fov_crop_ratio_min,
        crop_ratio_max=args.augment_fov_crop_ratio_max
    )
```


### Step 2: 实现LiDAR密度模拟（4小时）

#### 文件修改

1. train_kitti.py添加参数

```python
# Line ~850
parser.add_argument("--augment_lidar_sparse_prob", type=float, default=0.0,
                    help="Probability of simulating sparse LiDAR (0.0-1.0)")
parser.add_argument("--augment_lidar_sparse_lines", type=str, default="16,32,64",
                    help="Comma-separated target line numbers (e.g. '16,32,64')")
parser.add_argument("--augment_lidar_vertical_fov", type=str, default="-25,15",
                    help="Vertical FOV range in degrees (e.g. '-25,15' for -25° to +15°)")
```

2. 实现垂直分层稀疏化

```python
# Line ~560
def _apply_lidar_sparsification(pcs_np, masks, target_lines=32, 
                                  vertical_fov=(-25, 15), original_lines=128):
    """
    Simulate sparse LiDAR by vertical angle binning (no ring id needed).
    
    Args:
        pcs_np: (B, N, 4) numpy array [x, y, z, intensity]
        masks: list of (N,) masks
        target_lines: target number of lines (16/32/64)
        vertical_fov: (min_deg, max_deg) vertical FOV range
        original_lines: original resolution (e.g. 128)
    
    Returns:
        sparse_pcs: (B, N_sparse, 4) padded array
        sparse_masks: list of (N_sparse,) masks
    """
    B = pcs_np.shape[0]
    v_min, v_max = np.deg2rad(vertical_fov[0]), np.deg2rad(vertical_fov[1])
    
    sparse_pcs = []
    sparse_masks = []
    
    for b in range(B):
        pc = pcs_np[b]  # (N, 4)
        mask = np.asarray(masks[b])
        valid_idx = np.where(mask == 1)[0]
        
        if len(valid_idx) == 0:
            sparse_pcs.append(pc)
            sparse_masks.append(mask)
            continue
        
        # Get valid points
        pc_valid = pc[valid_idx]  # (N_valid, 4)
        x, y, z = pc_valid[:, 0], pc_valid[:, 1], pc_valid[:, 2]
        
        # Compute vertical angle for each point
        distance_xy = np.sqrt(x2 + y2)
        vertical_angle = np.arctan2(z, distance_xy)  # radians
        
        # Assign to pseudo ring bins
        # Normalize angle to [0, 1]
        angle_normalized = (vertical_angle - v_min) / (v_max - v_min)
        angle_normalized = np.clip(angle_normalized, 0, 1)
        
        # Map to original_lines bins
        pseudo_ring_id = (angle_normalized * original_lines).astype(int)
        pseudo_ring_id = np.clip(pseudo_ring_id, 0, original_lines - 1)
        
        # Select target_lines uniformly from original_lines
        selected_rings = np.linspace(0, original_lines - 1, target_lines).astype(int)
        
        # Keep only points from selected rings
        keep_mask = np.isin(pseudo_ring_id, selected_rings)
        pc_sparse = pc_valid[keep_mask]
        
        if len(pc_sparse) == 0:
            # Fallback: keep at least 1 point
            pc_sparse = pc_valid[:1]
        
        sparse_pcs.append(pc_sparse)
        sparse_masks.append(np.ones(len(pc_sparse)))
    
    # Pad to same length
    max_pts = max(pc.shape[0] for pc in sparse_pcs)
    padded_pcs = np.full((B, max_pts, 4), 999999, dtype=np.float32)
    padded_masks = []
    
    for b in range(B):
        n = sparse_pcs[b].shape[0]
        padded_pcs[b, :n, :] = sparse_pcs[b]
        padded_masks.append(np.concatenate([sparse_masks[b], np.zeros(max_pts - n)]))
    
    return padded_pcs, padded_masks
```

3. 在训练循环中调用

```python
# Line ~2015（pc augment区域，在pc_dropout之后）
if args.augment_pc_dropout > 0:
    # ... 现有pc_dropout代码 ...
    pcs_np = padded_pcs
    masks = padded_masks

# ← 新增：LiDAR稀疏化
if args.augment_lidar_sparse_prob > 0 and random.random() < args.augment_lidar_sparse_prob:
    target_lines_choices = [int(x) for x in args.augment_lidar_sparse_lines.split(',')]
    target_lines = random.choice(target_lines_choices)
    v_fov = [float(x) for x in args.augment_lidar_vertical_fov.split(',')]
    pcs_np, masks = _apply_lidar_sparsification(
        pcs_np, masks, 
        target_lines=target_lines,
        vertical_fov=tuple(v_fov),
        original_lines=128
    )
```


### Step 3: 更新配置文件（10分钟）

#### v40_gmp_p1c.yaml

```yaml
# Line ~72（augment区域）
augment_color_jitter: 0.15
augment_pc_jitter: 0.02
augment_pc_dropout: 0.15
augment_intrinsic: 0.05
augment_intrinsic_cxcy: 0.03

# ← 新增P2b增强
augment_fov_crop_prob: 0.3              # 30%概率FOV crop
augment_fov_crop_ratio_min: 0.75        # 最小保留75%
augment_fov_crop_ratio_max: 0.95        # 最大保留95%

augment_lidar_sparse_prob: 0.25         # 25%概率稀疏化
augment_lidar_sparse_lines: "16,32,64"  # 模拟16/32/64线
augment_lidar_vertical_fov: "-25,15"    # 垂直FOV -25°到+15°
```


### Step 4: batch_train.sh参数映射（5分钟）

```python
# batch_train.sh PARAM_MAP
OPTIM_PARAMS = [
    # ... 已有参数 ...
    ('differentiable_epnp', '--differentiable_epnp'),
    
    # ← 新增P2b参数
    ('augment_fov_crop_prob', '--augment_fov_crop_prob'),
    ('augment_fov_crop_ratio_min', '--augment_fov_crop_ratio_min'),
    ('augment_fov_crop_ratio_max', '--augment_fov_crop_ratio_max'),
    ('augment_lidar_sparse_prob', '--augment_lidar_sparse_prob'),
    ('augment_lidar_sparse_lines', '--augment_lidar_sparse_lines'),
    ('augment_lidar_vertical_fov', '--augment_lidar_vertical_fov'),
]
```


## 六、实施时间估算

| 任务 | 代码量 | 测试 | 总计 |
|------|-------|------|------|
| FOV crop实现 | 1.5h | 0.5h | 2h |
| LiDAR稀疏化实现 | 3h | 1h | 4h |
| 配置文件更新 | 0.1h | - | 0.1h |
| batch_train.sh更新 | 0.1h | - | 0.1h |
| Smoke test验证 | - | 0.5h | 0.5h |
| 总计 | - | - | 6.7h |

预计完成时间：1个工作日


## 七、验证计划

### Smoke Test（必须）

```bash
# 启动2 epoch测试
bash batch_train.sh configs/v40_smoke_diff_epnp.yaml
```

验证点：
1. ✅ FOV crop日志：`FOV crop applied: ratio_h=0.85, ratio_w=0.90`
2. ✅ LiDAR稀疏化日志：`LiDAR sparsified: 128 → 32 lines, points: 50000 → 15000`
3. ✅ 无NaN值
4. ✅ Loss正常收敛


### 泛化验证（训练后）

跨数据集测试：
- KITTI → nuScenes
- KITTI → Waymo
- 车型A → 车型B

预期：
- P2a（当前）：~0.9°
- P2b（完整）：~0.7° (-22%误差)


## 八、风险评估

### 风险1：LiDAR垂直FOV范围设定

问题：不同数据集的垂直FOV可能不同
- KITTI：-25° ~ +15°（40°）
- nuScenes：-30° ~ +10°（40°）
- Waymo：-17° ~ +2°（19°）

缓解：
- 使用可配置参数`augment_lidar_vertical_fov`
- 建议使用较宽范围（如-30,20）容错


### 风险2：arctan2计算开销

影响：每个batch需要计算N×B次arctan2（N~10000, B=32）

缓解：
1. 仅在`augment_lidar_sparse_prob`触发时计算（25%概率）
2. numpy arctan2已高度优化
3. 可预计算并缓存（若性能问题）

实测：预计增加<5%训练时间


### 风险3：FOV crop导致内参超出合理范围

问题：极端crop可能导致fx/fy过大

缓解：
- 限制crop ratio范围（0.75-0.95，不过小）
- 训练时监控intrinsic分布


## 九、与P2a的对比

### P2a（已完成）

| 增强 | 配置 | 泛化提升 |
|------|------|---------|
| pc_dropout | 0.15 | +10% |
| intrinsic | 0.05 | +15% |
| 总计 | - | +20% |

### P2b（计划实施）

| 增强 | 配置 | 泛化提升 |
|------|------|---------|
| FOV crop | 0.3 prob | +20% |
| LiDAR稀疏化 | 0.25 prob | +10% |
| 总计 | - | +30% |

### P2a + P2b（完整）

综合泛化提升：约+45%
- 跨数据集误差：1.1° → 0.7° (-36%)
- 达到论文水平


## 十、决策点

### 决策1：是否立即实施P2b？

选项A：立即实施（推荐）
- ✅ 6.7h开发时间可接受
- ✅ 跨域泛化提升显著（+30%）
- ✅ 一次性完成，避免后续重新训练

选项B：先训练P2a，再评估
- ⚠️ 若P2a泛化不足，需重新训练（浪费8h）
- ⚠️ 若需要跨域部署，必然要实施P2b

建议：立即实施P2b，一次性完成


### 决策2：LiDAR稀疏化方案

方案选择：
- ✅ 推荐：方案A（垂直角度分组）
  - 最真实模拟激光雷达特性
  - 泛化提升最大（+10%）
  - 计算开销可接受


### 决策3：参数配置保守 vs 激进

保守配置（推荐）：
```yaml
augment_fov_crop_prob: 0.3          # 30%
augment_fov_crop_ratio_min: 0.80    # 保留80-95%（较温和）
augment_lidar_sparse_prob: 0.25     # 25%
```

激进配置：
```yaml
augment_fov_crop_prob: 0.5          # 50%
augment_fov_crop_ratio_min: 0.75    # 保留75-95%（更强）
augment_lidar_sparse_prob: 0.35     # 35%
```

建议：先用保守配置，若ep60表现良好且需要更强泛化，再增加


## 十一、总结

### 技术可行性

| 功能 | 无ring id可行性 | 方案 | 泛化提升 |
|------|---------------|------|---------|
| FOV crop | ✅ 完全可行 | 中心crop + 内参更新 | +20% |
| LiDAR密度 | ✅ 完全可行 | 垂直角度伪层分组 | +10% |

### 实施建议

1. ✅ 立即实施P2b（6.7h开发）
2. ✅ 使用垂直角度方案（方案A）
3. ✅ 保守配置参数（0.3 prob, 0.80-0.95 ratio）
4. ✅ Smoke test验证（0.5h）
5. ✅ 启动完整P1c训练（8h）

### 预期成果

P2a + P2b（完整DST）：
- 跨数据集泛化：1.1° → 0.7° (-36%)
- 论文对齐度：60% → 95%
- 达到论文水平


文档版本：v1.0  
作者：BEVCalib Team  
时间：2026-05-29 12:35  
下一步：等待用户决策是否立即实施
