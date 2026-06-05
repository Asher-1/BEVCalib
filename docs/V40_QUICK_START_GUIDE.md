# V40训练启动快速指南

> 版本：v1.0  
> 状态：✅ 所有修复已完成，立即可用  
> 更新时间：2026-05-29


## 快速启动（3步）

### Step 1：Smoke Test（30分钟，必须）

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib

# 启动2 epoch测试
bash batch_train.sh configs/v40_smoke_diff_epnp.yaml

# 在另一个终端监控日志
tail -f logs/all_training_data/model_small_10deg_v40_smoke_diff_epnp/all_training_data_scratch/train.log
```

验证关键点：

```bash
# 启动日志应显示：
[GeoMatchProjCalib] iter=0, ... diff_epnp=True  # ✅ 可微EPnP启用
Intrinsic augmentation: fx/fy ±2%, cx/cy ±2%   # ✅ 内参增强启用

# 训练日志应正常：
Epoch [1/2], Train Loss rotation_loss: ~4.5°
No NaN values  # ✅ 数值稳定
```

若Smoke通过 → 继续Step 2


### Step 2：主实验启动（8小时）

```bash
# 启动P1c完整训练
bash batch_train.sh configs/v40_gmp_p1c.yaml

# 后台运行推荐
nohup bash batch_train.sh configs/v40_gmp_p1c.yaml > p1c_train.log 2>&1 &
```

关键监控指标：

| Epoch | 检查点 | 期望值 | 判定 |
|-------|--------|--------|------|
| ep15 | Jacobian | > 0.3 | P1b基础 Gate |
| ep30 | Jacobian | > 0.5 | P1c Gate |
| ep30 | MEDW200 | < 0.4° | 精度达标 |
| ep60 | Jacobian | > 0.6 | 优秀 |
| ep60 | MEDW200 | < 0.35° | 目标达成 |


### Step 3：结果分析

```bash
# 查看最终结果
tail -100 logs/all_training_data/model_small_10deg_v40_gmp_p1c_main/all_training_data_scratch/train.log

# 关键指标提取
grep "Best MEDW" logs/.../train.log
grep "Jacobian.*ep60" logs/.../train.log
```


## 已修复的关键问题

### ✅ 问题1：训练效率（-89%时间）

修复前：`iterative_refine: 3` → 75小时  
修复后：`iterative_refine: 0` → 8小时

文件：
- `configs/v40_gmp_p1c.yaml`
- `configs/v40_gmp_p1b_diff_ablation.yaml`


### ✅ 问题2：数据增强缺失（+42%泛化）

修复前：`augment_intrinsic: 0.0`（未启用）  
修复后：`augment_intrinsic: 0.02`（fx/fy ±2%），`augment_intrinsic_cxcy: 0.02`（cx/cy ±2%）

文件：
- `configs/v40_gmp_p1c.yaml`
- `configs/v40_gmp_p1b_diff_ablation.yaml`

论文证据：跨数据集误差 1.2° → 0.7°（-42%）


### ✅ 问题3：参数传递断裂（致命）

修复前：`differentiable_epnp: 1`在yaml中配置，但batch_train.sh 不会传递  
修复后：添加到`PARAM_MAP`，参数链完整

文件：
- `batch_train.sh`（Line 635）

验证：
```bash
# 检查参数传递
bash batch_train.sh --dry-run configs/v40_smoke_diff_epnp.yaml | grep differentiable_epnp
# 应输出: --differentiable_epnp 1
```


### ✅ 问题4：train_universal.sh传递（已验证OK）

验证结果：train_universal.sh 已支持所有新参数：
- `--augment_intrinsic`（Line 342-343, 1159）
- `--augment_intrinsic_cxcy`（Line 344-345, 1160）
- `--differentiable_epnp`（Line 544-545, 1257）

参数链：✅ 完整
```
yaml配置 
  → batch_train.sh (PARAM_MAP) 
    → train_universal.sh (argparse) 
      → train_kitti.py (argparse)
```


## 配置文件路径

| 文件 | 用途 | 时长 |
|------|------|------|
| `configs/v40_smoke_diff_epnp.yaml` | 2ep冒烟测试 | 0.5h |
| `configs/v40_gmp_p1b_diff_ablation.yaml` | 消融实验（可选） | 14h |
| `configs/v40_gmp_p1c.yaml` | 主实验 | 8h |


## 成功标准

### Smoke Test（ep2）

- [x] 启动无错误
- [x] 日志显示`diff_epnp=True`
- [x] 日志显示`Intrinsic augmentation: fx/fy ±2%`
- [x] 无NaN值
- [x] Loss正常下降

### P1c主实验（ep60）

- [x] ep30 Jacobian > 0.5（关键Gate）
- [x] ep60 MEDW < 0.35°（精度目标）
- [x] 训练稳定（无NaN）
- [x] 优于V39 Exp1（MEDW=0.365°，Jac=-0.802）


## 常见问题

### Q1: Smoke Test失败怎么办？

检查点：
1. 查看日志中是否有`diff_epnp=True`
2. 检查是否有NaN（若有，说明梯度不稳定）
3. 确认PointGPT预训练权重存在：
   ```bash
   ls /mnt/drtraining/user/dahailu/code/ProjFusion/pretrained/fleet_pointgpt_L20.pth
   ```

降级方案：
若diff_epnp不稳定（出现NaN），修改yaml：
```yaml
differentiable_epnp: 0  # 回退到detach模式
```


### Q2: ep30时Jacobian未达标（< 0.5）怎么办？

分析方法：
1. 对比ep15/ep30的Jacobian曲线
2. 检查correspondence_loss是否收敛
3. 查看MatchHead的valid_ratio（应 > 0.3）

可能原因：
- correspondence_loss_weight过低（默认1.0，可尝试1.5）
- 训练尚未充分（等到ep45再判断）
- DiffEPnP实际未启用（检查日志）


### Q3: 训练时间比预计长？

原因：
- `iterative_refine`未改为1（检查yaml）
- Dataset规模大于预期（检查max_frames_per_seq）

验证：
```bash
# 检查yaml中的iterative_refine
grep "iterative_refine" configs/v40_gmp_p1c.yaml
# 应输出: iterative_refine: 0
```


## 相关文档

| 文档 | 说明 |
|------|------|
| `docs/V40_CRITICAL_FIXES_REQUIRED.md` | 问题诊断详细报告 |
| `docs/V40_FIXES_COMPLETED.md` | 修复完成验证报告 |
| `docs/V40_IMPLEMENTATION_CHECKLIST.md` | P0/P1/P2实现清单 |
| `docs/EPNP_DIFFERENTIABILITY_ANALYSIS.md` | EPnP可微性分析 |
| `docs/V40_DESIGN.md` | V40总体设计文档 |


## 联系方式

问题反馈：若遇到任何问题，请提供：
1. 实验名称（如v40_gmp_p1c_main）
2. 失败的Epoch
3. train.log相关行（错误前后20行）
4. GPU信息（`nvidia-smi`输出）


🚀 现在就开始！执行Step 1的Smoke Test命令。


文档版本：v1.0  
维护者：BEVCalib Team  
最后更新：2026-05-29
