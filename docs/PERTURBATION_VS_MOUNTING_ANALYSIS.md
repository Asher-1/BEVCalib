# 扰动 vs 跨车型安装差异：为什么 ±5° 训练扰动无法覆盖传感器配置偏移

本文回答 BEVCalib 泛化能力的核心概念问题：训练时的随机外参扰动与测试时面对不同物理安装的相机-LiDAR 配置并不等价，即使标量角度或平移差异看起来"小于扰动范围"。

---

## 1. 训练流水线的实际行为

### 1.1 扰动的应用位置

在 `train_kitti.py` 中，每个 batch 使用：

- `gt_T_to_camera_np`：数据集中的真值 LiDAR → 相机变换矩阵。
- `init_T_to_camera_np`：由 `generate_single_perturbation_from_T(gt_T_to_camera_np, ...)` 生成的扰动外参。

模型同时接收 `gt_T_to_camera_t` 和 `init_T_to_camera_t`。

### 1.2 扰动机制 (`tools.py`)

`generate_single_perturbation_from_T`：

- 构建随机旋转 `delta_rots`（轴角表示，角度范围 `[-angle_range_deg, angle_range_deg]`，默认 20°，可配置为 5°）。
- 组合方式：`new_rots = delta_rots * orig_rots`（SciPy `Rotation` 乘积：小旋转与 GT 旋转的组合）。
- 可选地沿随机方向扰动平移，幅度不超过 `trans_range`（除非设置 `rotation_only`）。

因此，对于每个样本，`init_T` 是该样本自身 `gt_T` 的随机邻域，而非来自"所有可能车辆标定"的全局分布采样。

### 1.3 图像分支：BEV 使用扰动后的外参

在 `bev_calib.py` `BEVCalib.forward` 中：

```python
cam2ego_T = torch.linalg.inv(init_T_to_camera)
cam_bev_feats, cam_bev_mask = self.img_branch(
    cam2ego_T=cam2ego_T, cam_intrins=cam_intrinsic, ...
)
```

约定：

- `init_T_to_camera`：LiDAR → 相机。
- `cam2ego_T = inv(init_T_to_camera)`：相机 → LiDAR（ego）—— 代码将 LiDAR 坐标系作为 LSS 的 ego 坐标系。

因此 LSS 几何由扰动后的外参驱动，而非 GT。RGB 图像像素不变；只有提升几何（frustum 点在 3D / BEV 体素中的落点位置）使用 `init_T`。

### 1.4 监督信号：目标仍然是 GT

`losses/losses.py` `realworld_loss` 从回归头构建预测的 `T`，计算：

- `T_gt_expected = inv(T_pred) @ init_T_to_camera`（参见代码内注释和实现）。

训练损失将此与 `gt_T_to_camera` 比较。因此网络被训练为：给定 `init_T`，预测一个校正量，使组合后的位姿匹配 `gt_T`。

对显式问题的回答：

| 问题 | 回答 |
|------|------|
| 扰动是否改变了 BEV 投影使用的 T？ | 是。BEV 使用 `inv(init_T)`，即扰动后的 LiDAR→cam 取逆得到 camera→ego。 |
| 扰动是否只改变了"标签"？ | 否。前向图像几何和损失目标都依赖于 `init_T` 与 `gt_T`。 |
| 模型看到的是 `(图像, 点云, 扰动T) → …` 吗？ | 是，其中 `扰动T` = `init_T` 用于 LSS；GT 用于损失和点云重投影项。 |

---

## 2. LSS 如何使用 T 和 K（`img_branch.py`）

### 2.1 `get_geometry`

Frustum 在图像空间构建，然后：

1. 通过 `post_cam2ego_*` 撤销 resize/crop 增强（训练时通常传入单位阵 `post_cam2ego_T`）。
2. 转换为相机射线并应用 `cam2ego_rot @ inv(cam_intrins)` 得到 ego（LiDAR）坐标系下的方向。
3. 加上 `cam2ego_trans`：相机原点在 ego 坐标系中的位置。

因此每个体素分配都依赖于：

- 完整的旋转 `cam2ego_rot`（来自 `inv(init_T)[:3,:3]`）。
- 完整的平移 `cam2ego_trans`（来自 `inv(init_T)[:3,3]`）。
- 完整的内参 `cam_intrins`（通过 `inv(K)`）。

不存在仅使用"相对于某个标准安装的 ΔR"而独立于基础安装的路径；流水线始终使用 `init_T` 隐含的绝对 `cam2ego`。

### 2.2 深度 + 散射

`get_cam_feature` 预测每像素的深度分布（depth bins 上的 softmax）并与图像特征相乘。错误的几何意味着即使深度完美，特征也会被散射到错误的 3D 单元格中，而且深度网络是在训练序列的统计特性上训练的。

### 2.3 `bev_pool`

几何计算之后，点被 `bev_settings.py` 中的固定 `xbound, ybound, zbound` 离散化。改变基础外参会改变：哪些单元格接收质量、网格的遮挡关系、以及相机视锥体与固定 BEV ROI 的重叠——这是对特征图的全局改变，而非网络其余部分可以忽略的简单等变平移。

---

## 3. 点云分支（与相机外参正交）

`Lidar2BEV` 使用 `bev_settings.py` 中的固定 `coors_range` 在 ego 坐标系中对 LiDAR 点进行体素化。它不使用 `init_T`。

因此对于给定帧：

- LiDAR BEV（近似）与安装无关，因为它仅依赖于 LiDAR 坐标系中的点云。
- 相机 BEV 依赖于 `init_T` 和 K。

融合模块（`ConvFuser` + transformer）因此看到的错位模式取决于错误图像几何与正确 LiDAR 几何之间的差异。该模式是在训练安装分布上学习的。

---

## 4. 为什么"~3° 倾斜在 ±5° 范围内"是错误的比较

### 4.1 扰动是围绕训练 GT 的局部采样，而非对其他车辆 GT 的覆盖

训练过程反复采样：

  T_init = noise(T_gt_train)

测试序列 03 有不同的真值：

  T_gt_test03 ≠ T_gt_train

所谓"倾斜仅与训练差~3°"是比较两个确定性标定值（如训练平均倾斜 vs Seq03）。5° 增强并不意味着模型见过了从每个绝对位姿出发的所有 3° 偏移；它意味着每个训练样本都以其自身 T_gt_train 为中心，在最大 5° 的误差范围内被观察到。

集合的不同：

- 训练覆盖（几何）：{ inv(noise(T_gt_train)) } 遍历帧和噪声采样——始终锚定在训练序列的外参上。
- 测试（Seq03）：inv(noise(T_gt_test03)) 或 inv(T_gt_test03)（取决于评估方式）——锚定在 Seq03 的外参上。

即使标量 pitch 差距为 3°，完整的 (R, t, K) 联合决定了 LSS 几何。不存在这样的定理：在噪声-训练-GT 外参上训练的网络能泛化到噪声-测试-GT 外参，除非表示是解耦的或等变的，而标准 LSS + CNN 融合不具备这些性质。

### 4.2 相同的标称角度 ≠ 相同的 BEV 特征图

`get_geometry` 对 R 和 t 是非线性的，且耦合了内参。3° 的 pitch 变化：

- 移动整个视锥体在 ego 坐标系中的位置。
- 改变哪些深度 bins 和哪些 BEV 单元格被激活。
- 与真实图像内容（地平线、道路在图像中的布局）交互，而这些内容与该相机的物理安装直接相关——像素不是从标准相机合成的。

因此 Seq03 上的失败模式（如系统性 pitch 误差 ~2.1°）与 5° 增强并不矛盾：模型可能在训练安装流形内插值良好，但在另一辆车的联合 (R, t, K, 外观) 上外推失败。

### 4.3 平移示例（`t_y`：−0.010 m vs −0.069 m）

~59 mm 的相机在 ego 坐标系中的横向位置差异不会被仅旋转扰动覆盖（当 `rotation_only=1` 时）：`init_T` 中的平移保持等于每个样本的 GT，因此跨序列的 t_y 偏移在训练中从未被模拟。

即使使用完整位姿扰动，平移也是沿随机方向从当前 GT 随机化的，而非系统性地扫过以匹配另一辆车的固定偏差。因此结构化的安装偏移（始终存在的 t_y 差异）相对于围绕训练 GT 的各向同性噪声仍然是分布外的。

### 4.4 内参

`get_geometry` 使用 `inv(K)`。如果 Seq03/04 在 resize 后的 fx, fy, cx, cy 与训练不同，这是一个独立的领域偏移。训练可能使用 `--augment_intrinsic` 来随机化 K，但这与"与外参 5° 相同的增强"不同；没有它，内参不匹配仍然是一个独立的失败模式。

---

## 5. 概念总结

1. BEV 投影使用完整的扰动外参（`inv(init_T)`），而非 GT，也不是独立于基础安装的"仅 delta"。
2. 扰动相对于每个训练帧的 GT，因此无法均匀覆盖来自其他车辆的绝对外参。
3. LSS + 池化依赖于绝对的 R、t 和 K；数据集之间的小标量角度差异不等同于训练过程中围绕不同基础位姿看到的随机小误差。
4. LiDAR BEV 基本与安装无关；相机 BEV 不是。融合网络学习修复训练分布下的错位；测试安装产生的 OOD（分布外）联合统计量涉及（图像外观、相机 BEV、LiDAR BEV）。
5. 因此 ±5°（或任何固定球）外参噪声在分布覆盖或等变性的意义上无法"覆盖"跨车型安装差异——它只能围绕训练标定进行正则化。

---

## 6. 启示（简要）

- 多车型/多序列训练（使用多样的 T_{\text{gt}} 和 K）是直接拓宽 LSS 所见绝对位姿分布的方法。
- 领域自适应或显式标定先验（如标称安装 + 残差）可以减少外推差距。
- 内参增强（`train_kitti.py` `_augment_intrinsics`）解决 K 偏移但需要调参；它本身无法消除基础外参的 OOD 问题。

---

## 7. 代码参考（导航用）

| 主题 | 文件 / 位置 |
|------|------------|
| 扰动组合 | `kitti-bev-calib/tools.py` `generate_single_perturbation_from_T` |
| `cam2ego_T = inv(init_T)` | `kitti-bev-calib/bev_calib.py` `BEVCalib.forward` |
| LSS 几何 | `kitti-bev-calib/img_branch/img_branch.py` `LSS.get_geometry`, `Cam2BEV.forward` |
| BEV 范围 / 深度范围 | `kitti-bev-calib/bev_settings.py` |
| 损失组合 `T_gt_expected` | `kitti-bev-calib/losses/losses.py` `realworld_loss.forward` |
| 训练 batch 构建 | `kitti-bev-calib/train_kitti.py`（扰动 + `model(...)`） |
| 可选 K 增强 | `train_kitti.py` `_augment_intrinsics` |

---

本文档为 BEVCalib 项目生成：扰动与物理安装偏移的分析。
