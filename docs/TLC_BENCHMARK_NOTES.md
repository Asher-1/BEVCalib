# TLC × BEVCalib Benchmark Notes

## large_zigzag ~88° 失败根因（非坐标系 bug）

对 `KITTI-360/large_zigzag` 做四相机 sweep（v54a_lsp, 5° perturb, 30 frames）：

| cam_id | rig_rot | 说明 |
|--------|---------|------|
| 0 | **2.18°** | 推荐默认 |
| 1 | 2.30° | 可用 |
| 2 | **88.83°** | 旧默认，失败 |
| 3 | 4.51° | 可用 |

**结论：**
- TLC 每帧 `pcds/` 与 `params/lidars.txt` 同步，点云在**当帧 LiDAR 系**，静态 `cams_to_lidar_gt` 用法正确
- cam2 的 GT 外参含 ~90° 安装角（与 KITTI-Odometry `image_2` 训练分布不一致）
- BEVCalib 在 cam2 + 大 zigzag 场景下会退化为近恒等旋转预测（Roll 误差 ~89°）
- **已将 KITTI-360 默认相机从 cam2 改为 cam0**（`tools/prepare_tlc_for_bevcalib.py`）

## 复现 cam sweep

```bash
for cam in 0 1 2 3; do
  python3 tools/tlc_bevcalib_benchmark.py \
    --scenes large_zigzag --cam_id $cam --max_frames 30 \
    --ckpt_path logs/all_training_data/model_small_5deg_v54a_lsp_full/.../ckpt_best_dual.pth \
    --output_dir logs/evaluations/tlc_cam_sweep/cam_$cam
done
```

## NaN GUARD 与 jacobian

`v54a_lsp_stable_smoke` 中 NaN 从 **ep6** 开始（377 params），早于 `jacobian_loss_start_epoch=10`。
需 ablation 验证：关闭 jacobian / 降低权重 / 延后 MGDA。
