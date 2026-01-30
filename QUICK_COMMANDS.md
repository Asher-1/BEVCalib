# BEVCalib 快速命令参考

本文档提供常用命令的快速参考。

---

## 🚀 数据准备

### 从 ROS Bag 准备数据集
```bash
python tools/prepare_custom_dataset.py \
    --bag_dir /path/to/bags \
    --config_dir /path/to/config \
    --output_dir /path/to/output \
    --batch_size 500 \
    --num_workers 4
```

### 查看点云
```bash
# 查看单个 PLY 点云
python tools/view_pointcloud.py /path/to/temp/pointclouds/000000.ply

# 查看 BIN 点云
python tools/view_pointcloud.py /path/to/sequences/00/velodyne/000000.bin

# 只显示统计信息
python tools/view_pointcloud.py /path/to/file.ply --info
```

### 验证数据集
```bash
python tools/validate_kitti_odometry.py --dataset_root /path/to/dataset
```

### 可视化投影
```bash
python tools/visualize_projection.py --dataset_root /path/to/dataset --sequence 00
```

---

## 🎯 训练

### KITTI 数据集
```bash
python kitti-bev-calib/train_kitti.py \
    --log_dir ./logs/kitti \
    --dataset_root /path/to/kitti-odometry \
    --batch_size 16 \
    --num_epochs 500
```

### B26A 自定义数据集（从头训练）
```bash
python kitti-bev-calib/train_kitti.py \
    --log_dir ./logs/B26A_model \
    --dataset_root /home/ludahai/develop/data/eol/B26A_online/YR-B26A1-1_20251117_031232_lidar/bevcalib_training_data \
    --label B26A_20_1.5 \
    --batch_size 8 \
    --num_epochs 150 \
    --save_ckpt_per_epoches 15 \
    --angle_range_deg 20 \
    --trans_range 1.5 \
    --lr 1e-4 \
    --scheduler 1 \
    --step_size 50
```

### B26A 自定义数据集（微调）
```bash
python kitti-bev-calib/train_kitti.py \
    --log_dir ./logs/B26A_finetuned \
    --dataset_root /home/ludahai/develop/data/eol/B26A_online/YR-B26A1-1_20251117_031232_lidar/bevcalib_training_data \
    --pretrain_ckpt ./ckpt/kitti.pth \
    --label B26A_finetuned \
    --batch_size 4 \
    --num_epochs 50 \
    --lr 5e-5 \
    --scheduler 1 \
    --step_size 20
```

---

## 📊 评估

### KITTI 数据集
```bash
python kitti-bev-calib/inference_kitti.py \
    --log_dir ./logs/kitti \
    --dataset_root /path/to/kitti-odometry \
    --ckpt_path ./ckpt/kitti.pth \
    --batch_size 16
```

### 自定义数据集
```bash
python kitti-bev-calib/inference_kitti.py \
    --log_dir ./logs/B26A_eval \
    --dataset_root /home/ludahai/develop/data/eol/B26A_online/YR-B26A1-1_20251117_031232_lidar/bevcalib_training_data \
    --ckpt_path ./logs/B26A_model/checkpoints/best_model.pth \
    --batch_size 4
```

---

## 🔍 调试工具

### 检查数据集结构
```bash
python tools/visualize_kitti_structure.py /path/to/dataset
```

### 检查特定序列
```bash
python tools/visualize_kitti_structure.py /path/to/dataset --sequence 00
```

### 测试数据集加载
```bash
python -c "
from kitti_dataset import KittiDataset
dataset = KittiDataset('/path/to/dataset', auto_detect=True)
print(f'Total frames: {len(dataset)}')
img, pcd, gt, K = dataset[0]
print(f'Image size: {img.size}')
print(f'Point cloud shape: {pcd.shape}')
"
```

---

## 📈 监控

### TensorBoard
```bash
tensorboard --logdir ./logs --port 6006
```

### 查看训练日志
```bash
tail -f ./logs/B26A_model/train.log
```

---

## 🔧 常用参数组合

### 快速实验（调试用）
```bash
--batch_size 2 --num_epochs 10 --save_ckpt_per_epoches 5
```

### 标准训练
```bash
--batch_size 8 --num_epochs 150 --save_ckpt_per_epoches 15 --lr 1e-4
```

### 高精度训练
```bash
--batch_size 4 --num_epochs 200 --save_ckpt_per_epoches 20 --lr 5e-5 --scheduler 1 --step_size 80
```

### 微调模式
```bash
--pretrain_ckpt ./ckpt/kitti.pth --batch_size 4 --num_epochs 50 --lr 5e-5
```

---

## 📝 环境设置

### 激活环境
```bash
conda activate bevcalib
```

### 安装依赖
```bash
pip install -r requirements.txt
cd kitti-bev-calib/img_branch/bev_pool && python setup.py build_ext --inplace
```

### 下载预训练模型
```bash
# Google Drive
gdown https://drive.google.com/uc\?id\=1gWO-Z4NXG2uWwsZPecjWByaZVtgJ0XNb

# 或 Hugging Face
huggingface-cli download cisl-hf/BEVCalib --revision kitti-bev-calib --local-dir ./ckpt
```

---

## 🎯 工作流示例

### 完整流程：从 ROS Bag 到训练

```bash
# 1. 准备数据集
python tools/prepare_custom_dataset.py \
    --bag_dir /path/to/bags \
    --config_dir /path/to/config \
    --output_dir /path/to/dataset \
    --batch_size 500 \
    --num_workers 4

# 2. 验证数据集
python tools/validate_kitti_odometry.py --dataset_root /path/to/dataset

# 3. 查看点云样例
python tools/view_pointcloud.py /path/to/dataset/temp/pointclouds/000000.ply

# 4. 训练模型
python kitti-bev-calib/train_kitti.py \
    --dataset_root /path/to/dataset \
    --log_dir ./logs/my_model \
    --batch_size 8 \
    --num_epochs 150

# 5. 监控训练
tensorboard --logdir ./logs

# 6. 评估模型
python kitti-bev-calib/inference_kitti.py \
    --dataset_root /path/to/dataset \
    --ckpt_path ./logs/my_model/checkpoints/best_model.pth
```

---

## 🔗 快速链接

- **数据准备**: [README.md#custom-dataset](README.md#custom-dataset)
- **训练指南**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **参数详解**: [TRAINING_GUIDE.md#参数调优指南](TRAINING_GUIDE.md)
- **故障排除**: [TRAINING_GUIDE.md#常见问题](TRAINING_GUIDE.md)

---

**使用建议**：
- 复制需要的命令，修改路径后直接使用
- 保存为自己的脚本以便重复使用
- 根据实际情况调整参数

**更新时间**: 2026-01-28  
**版本**: v1.0
