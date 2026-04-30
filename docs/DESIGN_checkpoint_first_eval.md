# 设计文档：Checkpoint-First 评估架构

## 1. 背景与动机

当前评估配置存在大量参数冗余，YAML 中手动指定的模型架构参数（如 `rotation_only`, `voxel_mode`, `intrinsic_input` 等）与 checkpoint 中保存的 `args` 重复，导致：

1. **配错风险高**：手动配置可能与实际训练参数不一致，导致模型加载失败或评估结果无意义
2. **维护成本高**：每新增一个训练参数，都需同步更新 eval YAML、`run_generalization_eval.py` 的命令构建、`evaluate_checkpoint.py` 的 argparse
3. **代码重复**：3 条路径（evaluate_checkpoint / evaluate_drinfer / bevcalib_inference）各自独立处理参数检测

## 2. 现状分析

### 2.1 参数传递全链路

```
训练 YAML → batch_train.sh → train_kitti.py → checkpoint['args'] = vars(args)
                                                        ↓
eval YAML → run_generalization_eval.py → evaluate_checkpoint.py → ckpt_args.get(...)
                                                                        ↓
                                                                   BEVCalib(**params)
```

### 2.2 evaluate_checkpoint.py 当前参数分类（41 个命令行参数）

| 类别 | 参数 | 当前来源 | checkpoint 中是否有 |
|------|------|----------|-------------------|
| **模型架构** | rotation_only | CLI 传入，default=-1 自动检测 | ✅ args.rotation_only |
| | use_mlp_head | CLI 传入，default=-1 自动检测 | ✅ args.use_mlp_head |
| | deformable | CLI 传入，default=0 | ✅ args.deformable |
| | bev_encoder | CLI 传入，default=1 | ✅ args.bev_encoder |
| | bev_pool_factor | CLI 传入，default=0 | ✅ args.bev_pool_factor |
| | voxel_mode | CLI 传入或 ckpt_args fallback | ✅ args.voxel_mode |
| | scatter_reduce | CLI 传入或 ckpt_args fallback | ✅ args.scatter_reduce |
| | to_bev_mode | CLI 传入或 ckpt_args fallback | ✅ args.to_bev_mode |
| | fuser_type | CLI 传入或 ckpt_args fallback | ✅ args.fuser_type |
| | intrinsic_input | 仅从 ckpt_args 读取 | ✅ args.intrinsic_input |
| | use_foundation_depth | CLI 传入 | ✅ args.use_foundation_depth |
| | depth_model_type | CLI 传入 | ✅ args.depth_model_type |
| | fd_mode | CLI 传入 | ✅ args.fd_mode |
| **训练噪声** | angle_range_deg | CLI 传入 | ✅ args.angle_range_deg |
| | trans_range | CLI 传入 | ✅ args.trans_range |
| | perturb_distribution | ckpt_args fallback | ✅ args.perturb_distribution |
| | per_axis_prob | ckpt_args fallback | ✅ args.per_axis_prob |
| | per_axis_weights | ckpt_args fallback | ✅ args.per_axis_weights |
| **评估特有** | dataset_root | CLI 必填 | ❌ |
| | output_dir | CLI 传入 | ❌ |
| | batch_size | CLI 传入 | ❌ 训练 batch_size 无意义 |
| | max_batches | CLI 传入 | ❌ |
| | vis_interval | CLI 传入 | ❌ |
| | eval_sample_step | CLI 传入 | ❌ |
| | eval_max_frames_per_seq | CLI 传入 | ❌ |
| | use_full_dataset | CLI flag | ❌ |
| | target_width/height | CLI 传入 | ✅ 可从 ckpt 读取 |
| | eval_seed | CLI 传入 | ❌ |
| | data_balance | CLI 传入 | ❌ |
| | zero_image | CLI flag | ❌ |

### 2.3 eval YAML 冗余度（以 eval_v22_intrinsic_input.yaml 为例）

每个模型配置 17 个字段，其中 **11 个可从 checkpoint 自动读取**：

```yaml
# 当前配置 - 17 个字段
- label: "v22_intr_input"
  dir_name: "model_small_5deg_v22_intr_input"
  ckpt: "ckpt_best_val.pth"
  bev_zbound_step: "4.0"        # 环境变量，需保留
  rotation_only: 1              # ← 可自动检测
  intrinsic_input: 1            # ← 可自动检测（且当前根本没传！）
  use_drcv: 0                   # eval 选项，需保留
  use_mlp_head: 0               # ← 可自动检测
  voxel_mode: "hard"            # ← 可自动检测
  scatter_reduce: "sum"         # ← 可自动检测
  to_bev_mode: "concat"         # ← 可自动检测
  angle_deg: 5                  # ← 可自动检测
  z_voxels: 5                   # BEV 配置，需保留
  version: "v22"                # 描述字段
  mode_desc: "..."              # 描述字段

# 精简后 - 6 个字段
- label: "v22_intr_input"
  dir_name: "model_small_5deg_v22_intr_input"
  ckpt: "ckpt_best_val.pth"
  bev_zbound_step: "4.0"
  mode_desc: "intrinsic_input + scene aug"
  # 所有模型架构参数从 checkpoint 自动读取
```

## 3. 设计方案

### 3.1 核心原则

1. **Checkpoint 是唯一事实来源（Single Source of Truth）**
2. **命令行/YAML 可覆盖但不必填**：`CLI 值 > checkpoint 值 > 默认值`
3. **精简评估 YAML**：只配置 eval-specific 参数 + 模型路径
4. **不破坏现有行为**：已有的显式配置继续生效（向后兼容）

### 3.2 参数优先级链

```
命令行显式传入  >  eval YAML 配置  >  checkpoint['args']  >  硬编码默认值
     (覆盖用)       (eval-specific)     (训练时保存)          (最后兜底)
```

### 3.3 改动范围

#### 3.3.1 evaluate_checkpoint.py — 统一 checkpoint-first 自动检测

当前状态：部分参数已实现（rotation_only, use_mlp_head, voxel_mode 等），但不一致。

**改动**：新增 `_resolve_from_checkpoint()` 统一函数：

```python
def _resolve_from_checkpoint(args, ckpt_args):
    """Resolve model params: CLI > checkpoint > default."""
    resolved = {}
    
    AUTO_DETECT_PARAMS = {
        # param_name: (cli_attr, ckpt_key, default, sentinel)
        'rotation_only':  ('rotation_only',  'rotation_only',  True,    -1),
        'use_mlp_head':   ('use_mlp_head',   'use_mlp_head',   0,       -1),
        'deformable':     ('deformable',     'deformable',     0,       -1),
        'bev_encoder':    ('bev_encoder',    'bev_encoder',    1,       -1),
        'bev_pool_factor':('bev_pool_factor','bev_pool_factor',0,       -1),
        'voxel_mode':     ('voxel_mode',     'voxel_mode',     'hard',  None),
        'scatter_reduce': ('scatter_reduce', 'scatter_reduce', 'sum',   None),
        'to_bev_mode':    ('to_bev_mode',    'to_bev_mode',    'concat',None),
        'fuser_type':     ('fuser_type',     'fuser_type',     'concat',None),
        'intrinsic_input':('intrinsic_input','intrinsic_input', False,  None),
        'use_foundation_depth': ('use_foundation_depth', 'use_foundation_depth', 0, -1),
        'depth_model_type': ('depth_model_type', 'depth_model_type', 'midas_small', None),
        'fd_mode':        ('fd_mode',        'fd_mode',        'replace', None),
    }
    
    for param, (cli_attr, ckpt_key, default, sentinel) in AUTO_DETECT_PARAMS.items():
        cli_val = getattr(args, cli_attr, sentinel)
        if cli_val != sentinel and cli_val is not None:
            resolved[param] = cli_val
            source = "cli"
        elif ckpt_key in ckpt_args:
            resolved[param] = ckpt_args[ckpt_key]
            source = "checkpoint"
        else:
            resolved[param] = default
            source = "default"
        # Optional: log source for debugging
    
    return resolved
```

**需要改动的 argparse 参数**：将目前 `default=0/1` 的硬编码默认值改为 `default=-1`（sentinel），表示"未指定，从 checkpoint 检测"：

| 参数 | 当前 default | 改为 |
|------|-------------|------|
| `--deformable` | 0 | -1 |
| `--bev_encoder` | 1 | -1 |
| `--bev_pool_factor` | 0 | -1 |
| `--use_foundation_depth` | 0 | -1 |

已经是 -1 或 None 的参数无需改动：`--rotation_only`(-1), `--use_mlp_head`(-1), `--voxel_mode`(None), etc.

#### 3.3.2 run_generalization_eval.py — 精简命令构建

当前 `_build_eval_cmd_and_env()` 有 ~50 行参数传递代码。

**改动**：移除冗余参数传递，只保留 eval-specific 参数：

```python
def _build_eval_cmd_and_env(mcfg, per_model_dir):
    # 核心命令 - 只传 eval-specific 参数
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--ckpt_path", ckpt_path,
        "--dataset_root", TEST_DATA,
        "--output_dir", per_model_dir,
        "--use_full_dataset",
        "--max_batches", "0",
        "--vis_interval", str(VIS_INTERVAL),
        "--batch_size", str(mcfg.get("batch_size", BATCH_SIZE)),
    ]
    
    # 可选的 eval-scope 覆盖（仅当 YAML 显式指定时传递）
    if "angle_range_deg" in mcfg or ANGLE_RANGE:
        cmd.extend(["--angle_range_deg", str(mcfg.get("angle_range_deg", ANGLE_RANGE))])
    if "trans_range" in mcfg or TRANS_RANGE:
        cmd.extend(["--trans_range", str(mcfg.get("trans_range", TRANS_RANGE))])
    
    # 架构覆盖（仅当 YAML 显式指定且需要覆盖 checkpoint 值时）
    OVERRIDE_PARAMS = [
        "rotation_only", "deformable", "use_mlp_head", "bev_pool_factor",
        "voxel_mode", "scatter_reduce", "to_bev_mode", "fuser_type",
        "use_foundation_depth", "depth_model_type", "fd_mode",
    ]
    for p in OVERRIDE_PARAMS:
        if p in mcfg:
            cmd.extend([f"--{p}", str(mcfg[p])])
    
    # eval-specific flags
    if mcfg.get("zero_image"):
        cmd.append("--zero_image")
    if mcfg.get("use_drcv"):
        cmd.append("--use_drcv")
    
    return cmd, env, ckpt_path
```

#### 3.3.3 train_kitti.py — 增强 checkpoint 元数据

当前保存 `vars(args)` 已包含所有训练参数。建议额外保存：

```python
torch.save({
    # 现有字段
    'epoch': epoch + 1,
    'model_state_dict': model_to_save.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'args': vars(args),
    
    # 新增: 模型结构快照（不依赖 argparse 命名）
    'model_config': {
        'rotation_only': model.rotation_only,
        'intrinsic_input': model.intrinsic_input,
        'embed_dim': model.embed_dim,
        'deformable': model.deformable,
        'bev_encoder_use': model.bev_encoder_use,
        'use_mlp_head': hasattr(model, 'rotation_pred') and 
                        isinstance(model.rotation_pred, nn.Sequential) and
                        len(model.rotation_pred) > 1,
    },
    
    # 新增: 环境信息（调试用）
    'env': {
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'hostname': os.uname().nodename,
        'timestamp': datetime.now().isoformat(),
    },
})
```

`model_config` 的价值：即使 argparse 命名变更，从模型实例直接读取属性更可靠。
评估脚本的读取优先级变为：`CLI > model_config > args > default`。

#### 3.3.4 eval YAML 精简

精简前（17 字段/模型）：

```yaml
- label: "v22_intr_input"
  dir_name: "model_small_5deg_v22_intr_input"
  ckpt: "ckpt_best_val.pth"
  bev_zbound_step: "4.0"
  rotation_only: 1
  intrinsic_input: 1
  use_drcv: 0
  use_mlp_head: 0
  voxel_mode: "hard"
  scatter_reduce: "sum"
  to_bev_mode: "concat"
  angle_deg: 5
  z_voxels: 5
  version: "v22"
  mode_desc: "intrinsic_input head + scene aug"
```

精简后（5 字段/模型）：

```yaml
- label: "v22_intr_input"
  dir_name: "model_small_5deg_v22_intr_input"
  ckpt: "ckpt_best_val.pth"
  bev_zbound_step: "4.0"
  mode_desc: "intrinsic_input head + scene aug"
  # rotation_only, intrinsic_input, voxel_mode, etc. 全部从 checkpoint 自动读取
  # 仅当需要覆盖 checkpoint 值时才显式指定
```

## 4. 向后兼容策略

1. **已有 eval YAML 继续可用**：显式指定的参数仍然被传递和使用
2. **新建 eval YAML 更简洁**：省略的参数从 checkpoint 自动检测
3. **渐进式迁移**：可以逐步删除 YAML 中的冗余字段
4. **sentinel 值不影响现有调用**：`-1` 表示"未指定"，`0/1` 继续作为显式值

## 5. 实施步骤（建议顺序）

| 步骤 | 改动 | 影响范围 | 风险 |
|------|------|----------|------|
| **Phase 1** | `evaluate_checkpoint.py`: 统一 `_resolve_from_checkpoint()` | 仅评估脚本内部重构 | 低 |
| **Phase 2** | `evaluate_checkpoint.py`: argparse default 改为 sentinel | 可能影响直接调用 | 中（需回归测试） |
| **Phase 3** | `run_generalization_eval.py`: 精简 `_build_eval_cmd_and_env()` | YAML→CLI 传递 | 低（可逐步减少传递） |
| **Phase 4** | `train_kitti.py`: 增加 `model_config` + `env` 字段 | 新 checkpoint 格式 | 低（向后兼容） |
| **Phase 5** | eval YAML 精简 | 配置文件 | 低（可选） |
| **Phase 6** | `bevcalib_inference.py` / `evaluate_drinfer.py`: 统一检测逻辑 | drinfer 路径 | 低 |

## 6. 评审清单

- [ ] `evaluate_checkpoint.py` 的 `_resolve_from_checkpoint()` 覆盖所有模型架构参数
- [ ] argparse sentinel 值不破坏现有脚本调用
- [ ] `run_generalization_eval.py` 在 YAML 省略参数时仍能正确评估
- [ ] 新 checkpoint 格式对旧评估脚本向后兼容
- [ ] 旧 checkpoint（无 model_config）对新评估脚本向后兼容
- [ ] 文档中列出所有可从 checkpoint 自动检测的参数
