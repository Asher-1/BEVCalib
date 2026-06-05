# V40 P1d 训练无法启动 - 根因与修复

> **时间**: 2026-05-29  
> **现象**: `train.log` 只有配置表，无 `Epoch [1/60]`，训练进程未持续运行

---

## 已确认的三个根因

### 1. argparse 无法解析负值参数（根本原因）

**问题**: YAML 中 `augment_lidar_vertical_fov: "-25,15"` 被展开为：

```bash
--augment_lidar_vertical_fov -25,15
```

Python `argparse` 会把 `-25,15` 当成**新的 option**（因为以 `-` 开头），报错：

```
train_kitti.py: error: argument --augment_lidar_vertical_fov: expected one argument
```

仅加 shell 引号 `'-25,15'` **仍不够**，必须用 **`--flag=value`** 形式：

```bash
--augment_lidar_vertical_fov=-25,15   ✅
```

**修复**:
- `batch_train.sh`: `_cli_arg()` 对负值生成 `--flag=value`
- `start_training.sh` / `train_universal.sh`: `_opt_flag_append()` + 解析 `--flag=*` 形式
- `start_training.sh`: nohup  stderr 写入 `train.stderr.log`（不再丢弃到 `/dev/null`）

---

### 2. PointGPT 配置文件路径错误

**问题**: `v40_gmp_p1d.yaml` 中写成了不存在的路径：

```
finetune_fleet_pointgpt_L20.yaml  ❌
```

**正确路径**:
```
finetune_fleet_L20.yaml  ✅
```

GeoMatch 使用 `pointgpt2bev` 编码器，启动时会加载该配置；路径错误会导致模型初始化失败（错误可能只出现在 stderr，未写入 `train.log`）。

**修复**: 已更正 `configs/v40_gmp_p1d.yaml`

---

### 3. 参数传递链断裂（此前已修）

`batch_train.sh` → `start_training.sh` → `train_universal.sh` 三层中，P2b 参数（`augment_fov_crop_*`, `augment_lidar_sparse_*`）曾缺失解析，导致 `Unknown option: --augment_fov_crop_prob`。

**修复**: 已在 `start_training.sh` 和 `train_universal.sh` 补全解析与传递。

---

## 为何 train.log 只有配置表？

日志写入顺序：

1. `train_universal.sh` 打印配置表 → 写入 `train.log` ✅（你看到的部分）
2. `torchrun` 启动 `train_kitti.py`
3. `train_kitti.py` 的 `tprint()` 才开始写 Epoch 日志

若在步骤 2–3 之间崩溃（参数错误 / 配置路径错误 / GPU 被占满），`train.log` 就会**只有配置、没有 Epoch**。

---

## 次要风险：GPU 被其他任务占满

当前机器上可能同时运行 `v40_gmp_p0ab` 等任务（`nvidia-smi` 显示 8 卡均有进程）。P1d 再申请 8 卡 DDP 可能：

- NCCL 初始化挂起
- OOM 后 worker 静默退出

**建议**: 启动 P1d 前确认 8 卡空闲，或先停掉冲突实验。

---

## 重新启动步骤

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib

# 1. 验证配置（可选）
python3 -c "
import yaml, os, shlex
cfg = yaml.safe_load(open('configs/v40_gmp_p1d.yaml'))
exp = [e for e in cfg['experiments'] if not e.get('skip', False)][0]
p = {**cfg['defaults']['params'], **exp['params']}
assert os.path.isfile(p['native_cross_pointgpt_config'])
print('OK:', shlex.quote(str(p['augment_lidar_vertical_fov'])))
"

# 2. 确认 GPU 空闲
nvidia-smi

# 3. 启动训练
bash batch_train.sh configs/v40_gmp_p1d.yaml
```

---

## 启动成功判定（约 10–15 分钟内）

```bash
tail -f logs/all_training_data/model_small_10deg_v40_gmp_p1d_strong_match/train.log
```

应看到：

```
[GeoMatchProjCalib] ... diff_epnp=True ...
Epoch [1/60], Step [1/32] ...
correspondence_loss_weight 生效（total_loss 中 corr 占比高）
```

**ep1 关键检查**:
```bash
grep -E "GeoMatchProjCalib|correspondence_loss_weight|augment_fov|Epoch \[1" \
  logs/all_training_data/model_small_10deg_v40_gmp_p1d_strong_match/train.log | head -20
```

---

## 修改文件清单

| 文件 | 修改 |
|------|------|
| `configs/v40_gmp_p1d.yaml` | 修正 pointgpt config 路径 |
| `batch_train.sh` | `shlex.quote` / `_cli_quote` 防止 `-25,15` 被误解析 |
| `start_training.sh` | P2b 参数解析 + `_quote_cli_val` 传递 |
| `train_universal.sh` | P2b 参数解析 + `_quote_cli_val` 传递 |

---

**状态**: 代码已修复，请重新执行 `bash batch_train.sh configs/v40_gmp_p1d.yaml`
