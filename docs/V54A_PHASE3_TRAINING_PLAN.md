# V54a Phase3 训练方案

## 背景

| 阶段 | 结论 |
|------|------|
| ep6 ablation | ZD/inj/rig@ep5 致 NaN；延后至 ep15 修复 |
| v2 15ep smoke | ep1–14 全 0 NaN（但 **未激活** 全栈，epoch 0-indexed） |
| v2_full | 曾提前启动于 ep1，**应先验证 ep16+** |

## Phase3 流水线

```
ep20 stack smoke (3×20ep)
    ├─ stack      默认 v2 全栈 + MGDA photo
    ├─ nophoto    mgda_include_photo=0
    └─ safe       MGDA@15 + photo=0
         ↓ 选 ep16+ NaN=0 胜者
80ep full (v3_stack / v3_nophoto / v3_safe)
         ↓
gate eval + TLC benchmark vs v54a_lsp_full
```

## 配置

| 文件 | 用途 |
|------|------|
| `configs/v54a_lsp_stable_v3_ep20_cf_bev_r.yaml` | 3×20ep smoke |
| `configs/v54a_lsp_stable_v3_fallback_cf_bev_r.yaml` | 3×80ep full 备选 |
| `run_v54a_phase3_pipeline.sh` | 一键 smoke → 选胜者 → full |

## 监控

```bash
tail -f logs/v54a_phase3/ep20_smoke_batch.log
grep -E 'NaN GUARD|mgda_n_tasks|lsp_loss' logs/all_training_data/model_small_5deg_v54a_v3_ep20_*_smoke/train.log
```

## 部署

当前仍用 `v54a_lsp_full/ckpt_best_dual.pth`；Phase3 full 完成后对比 gate + TLC。
