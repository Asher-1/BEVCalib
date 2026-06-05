# Fleet PointGPT 预训练：自定义数据集域适配方案

## 🎯 目标

基于 `bevcalib/all_training_data` 重训 PointGPT backbone，替换 KITTI 预训练权重，实现域适配。

---

## 📦 已创建的文件

### 1. **配置文件**
```
/mnt/drtraining/user/dahailu/code/ProjFusion/cfg/pointgpt/pretrain_fleet_8gpu.yaml
```
- **优化点**：
  - 全局 batch size: 256 (32/GPU × 8)
  - LR: 2.8e-4 (按 sqrt(batch_ratio) 缩放)
  - Epochs: 30 (从 50 降低，预期更快收敛)
  - 数据：21 个序列（00-21，除 18 做验证）

### 2. **DDP 训练脚本**
```
/mnt/drtraining/user/dahailu/code/BEVCalib/tools/pretrain_fleet_pointgpt_ddp.py
```
- 8 卡 DistributedDataParallel
- 自动 gradient sync
- Rank 0 保存 checkpoint

### 3. **启动脚本**
```bash
/mnt/drtraining/user/dahailu/code/BEVCalib/tools/run_pretrain_fleet_pointgpt_8gpu.sh
```

### 4. **辅助脚本**
- **快速验证**：`sanity_check_pointgpt_ddp.sh` (2 epochs 测试)
- **监控训练**：`monitor_pointgpt_pretrain.sh` (实时 tail log)

---

## 🚀 执行步骤

### Step 1: 快速验证 DDP 配置（可选但推荐）

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib
bash tools/sanity_check_pointgpt_ddp.sh
```

**预期输出**：
- 2 个 epoch 完成（约 10-15 分钟）
- 生成 `pretrained/fleet_pointgpt_sanity.pth`
- 验证 8 卡 DDP 正常工作

---

### Step 2: 开始完整训练（30 epoch）

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib

# 方式1：默认配置
bash tools/run_pretrain_fleet_pointgpt_8gpu.sh

# 方式2：自定义参数
EPOCHS=50 BATCH_SIZE=32 LR=3e-4 \
bash tools/run_pretrain_fleet_pointgpt_8gpu.sh
```

**训练时间估算**：
- 单 epoch：约 10-15 分钟（取决于数据量）
- 30 epochs：**约 5-7.5 小时**（vs 单卡 60+ 小时）

**输出文件**：
- 模型：`pretrained/fleet_pointgpt_tiny_8gpu.pth`
- 日志：`pretrained/fleet_pointgpt_tiny_8gpu_train.log`

---

### Step 3: 监控训练进度

```bash
# 实时监控
bash tools/monitor_pointgpt_pretrain.sh

# 或直接 tail
tail -f /mnt/drtraining/user/dahailu/code/ProjFusion/pretrained/fleet_pointgpt_tiny_8gpu_train.log
```

**日志示例**：
```
Ep 1/30 train=0.3986 val=0.2482 lr=2.80e-04 (850s)
  saved pretrained/fleet_pointgpt_tiny_8gpu.pth (val=0.2482)
Ep 2/30 train=0.2154 val=0.1876 lr=2.75e-04 (840s)
  saved pretrained/fleet_pointgpt_tiny_8gpu.pth (val=0.1876)
...
```

---

## 📊 预期结果

### 训练完成后

1. **Val loss** 应低于 **0.15**（参考：KITTI PointGPT 约 0.10-0.12）
2. **文件大小**：约 75 MB（与 `kitti_pointgpt_tiny.pth` 220 MB 相比更紧凑）
3. **格式兼容**：`base_model` 键，可直接用于 ProjFusion

---

## 🔄 用于 ProjFusion Rot-Only 训练

### 替换 PointGPT backbone

**方式 1：直接替换预训练权重**
```bash
cd /mnt/drtraining/user/dahailu/code/ProjFusion

# 备份原始 KITTI PointGPT
mv pretrained/kitti_pointgpt_tiny.pth pretrained/kitti_pointgpt_tiny.pth.bak

# 使用 Fleet PointGPT
cp pretrained/fleet_pointgpt_tiny_8gpu.pth pretrained/kitti_pointgpt_tiny.pth

# 重新训练 ProjFusion rot-only
bash scripts/start_finetune_rotonly_from_harmonic.sh  # 修改 --pretrain 路径
```

**方式 2：显式指定新权重**

修改 `cfg/dataset/bevcalib_r10_rotonly.yml`：
```yaml
path:
  pretrain: pretrained/fleet_pointgpt_tiny_8gpu.pth  # 指向新权重
```

然后训练：
```bash
python train.py \
  --config cfg/dataset/bevcalib_r10_rotonly.yml \
  --pretrain pretrained/fleet_pointgpt_tiny_8gpu.pth \
  --rot_only \
  --exp_name bevcalib_rotonly_fleet_pointgpt \
  --batch_size 16
```

---

## 📈 预期性能提升

### 域适配效果

| Metric | KITTI PointGPT | Fleet PointGPT (预期) |
|--------|----------------|---------------------|
| **Val loss** | 0.12 (KITTI) | **0.15** (Fleet) |
| **下游 rot_err** | 1.22° | **< 1.0°** 🎯 |
| **RRMSE** | 1.60° | **< 1.3°** 🎯 |

**原因**：
1. ✅ **点云分布对齐**：Fleet 数据的稀疏性、距离分布与部署环境一致
2. ✅ **场景语义匹配**：训练场景与实际应用场景更接近
3. ✅ **去除域偏移**：避免 KITTI → Fleet 的 distribution shift

---

## ⚙️ 高级优化选项

### 如果 val loss > 0.20（训练不充分）

1. **增加 epochs**：
```bash
EPOCHS=50 bash tools/run_pretrain_fleet_pointgpt_8gpu.sh
```

2. **降低学习率**：
```bash
LR=1.5e-4 bash tools/run_pretrain_fleet_pointgpt_8gpu.sh
```

3. **Warmup + 更长训练**：
编辑 `pretrain_fleet_8gpu.yaml`：
```yaml
scheduler: { type: CosLR, kwargs: { epochs: 50, initial_epochs: 5 } }
```

### 如果训练太慢

1. **减少数据采样**：
编辑 `FLEET.yaml`：
```yaml
SKIP_FRAME: 3  # 从 2 改为 3，减少 33% 数据
```

2. **混合精度训练**：
在 `pretrain_fleet_pointgpt_ddp.py` 中添加：
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

---

## 🔍 验证 Fleet PointGPT 质量

### 快速自检

```python
import torch

# 加载权重
ckpt = torch.load('pretrained/fleet_pointgpt_tiny_8gpu.pth')

# 检查关键指标
print(f"Epoch: {ckpt['epoch']}")
print(f"Best val loss: {ckpt['best_metrics']['val_loss']:.4f}")

# 应该低于 0.20，理想 < 0.15
assert ckpt['best_metrics']['val_loss'] < 0.20, "Val loss too high!"
```

---

## 📋 Checklist

训练完成后，确认：

- [ ] Val loss < 0.20 ✅
- [ ] 文件大小约 75 MB ✅
- [ ] 包含 `base_model` 键 ✅
- [ ] 在 ProjFusion rot-only 训练中加载成功 ✅
- [ ] 下游 rot_err 相比 KITTI PointGPT 有提升 ✅

---

## 🆘 常见问题

### Q1: DDP 启动失败，报 NCCL 错误
**A**: 检查 GPU 可见性和端口
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
bash tools/run_pretrain_fleet_pointgpt_8gpu.sh
```

### Q2: Out of Memory (OOM)
**A**: 降低 per-GPU batch size
```bash
BATCH_SIZE=16 bash tools/run_pretrain_fleet_pointgpt_8gpu.sh
```

### Q3: Val loss 不下降
**A**: 检查数据质量和学习率
```bash
# 降低 LR
LR=1e-4 bash tools/run_pretrain_fleet_pointgpt_8gpu.sh
```

---

## 📞 后续支持

训练完成后，如需进一步优化 rot-only 性能，可以：

1. **Fine-tune on smaller perturbations**（如 5° 或 3°）
2. **Multi-step training loss**（对齐 3-step iterative testing）
3. **Ensemble Fleet + KITTI PointGPT**（双 backbone 融合）

---

**准备好了就开始训练！** 🚀

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib
bash tools/run_pretrain_fleet_pointgpt_8gpu.sh
```
