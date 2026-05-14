# V30 实验设计文档

## 1. 背景与动机

V29-G3 (dinov2-small + partial unfreeze + strong jitter) 在泛化评估中达到 MEDW400=0.092°, MEDW800=0.070°。
V30 的目标是通过消融实验验证最优配置: DANN / auto_scale / 扰动范围。

**重大决策 (2026-05-14)**: dinov2-base 方案已淘汰。800帧统一评估显示 G3(small) 在 MEDW800 以 13.3%
优势领先 H8(base), 且模型仅为 H8 的 43%。所有 V30 实验统一使用 dinov2-small。

### 关键发现 (800帧统一评估, 2026-05-14, geodesic metric)

| 模型 | Backbone | DANN | Params | MEDW400 | MEDW800 | Per-frame |
|---|---|---|---:|---:|---:|---:|
| **V29-G3** | dinov2-small (unfreeze2) | No | 49.7M | 0.092° | **0.070°** | 2.316° |
| V30-H8 | dinov2-base (unfreeze2) | Yes | 116.2M | 0.087° | 0.080° | 2.316° |

| 指标 | V29-G3 (small) | V30-H8 (base) | 对比 |
|---|---:|---:|---:|
| MEDW400 | 0.092° | 0.087° | H8 +5.4% |
| MEDW800 | **0.070°** | 0.080° | **G3 +13.3%** |
| 400→800帧提升 | 24.2% | 7.9% | G3 受益更大 |
| √N 缩放效率 | 93% | 77% | G3 噪声更独立 |
| 模型大小 | 49.7M | 115M | 2.3x 更小 |

### 核心洞察

1. **MEDW400 时 H8 略优** (0.087° vs 0.092°), **但 MEDW800 时 G3 反超** (0.070° vs 0.080°)
2. **G3 √N 效率 93%** — 帧间误差近似 i.i.d., H8 仅 77% (噪声不够独立)
3. **DANN = 过拟合修补器** — 对 dinov2-base 是必需的, 对 dinov2-small 待验证
4. **800帧聚合突破 0.07°** — MEDW800 = 0.070°, 远超 0.1° 目标
5. **聚合帧数越多, G3 优势越大** — 部署时选择更长序列可获得更高精度

## 2. 消融矩阵

### 2.1 V30 Full 训练 (4 experiments, dinov2-small only)

训练规模: 32 nodes, batch_size=16, max_frames=10000, epochs=400, ddp_auto_scale=1

```
                    auto_scale=1              auto_scale=0
                  ─────────────────────     ─────────────────────
dinov2-small:
  无 DANN          G3 (部署首选)  ★           G3-noAS (AS 消融)
  + DANN          G3-DANN (DANN 效果)
  10° 扰动         G3-10deg (大扰动消融)
```

### 2.2 V30 Opt Quick 训练 (3 experiments)

训练规模: 1 node, batch_size=16, max_frames=500, epochs=400

| 实验 | Backbone | DANN | Angle Range | 目标 |
|---|---|---|---:|---|
| G3 | dinov2-small | No | 5° | 部署首选候选 |
| G3-DANN | dinov2-small | Yes | 5° | 填补 small+DANN+unfreeze2 缺失数据 |
| G3-10deg | dinov2-small | No | 10° | 大扰动训练消融 |

注: auto_scale 消融 (G3-noAS) 对 1-node 训练无意义, 仅在 full 训练中。

## 3. 回答的关键问题

| 对比 | 回答的问题 |
|---|---|
| G3 vs G3-noAS | ddp_auto_scale 对 dinov2-small 32-node 训练的提升幅度 |
| G3 vs G3-DANN | DANN 对 dinov2-small + unfreeze2 是否有效 |
| G3 vs G3-10deg | 10° 训练→5° 评估能否突破单帧精度上限 |
| Quick vs Full (G3) | max_frames=10000 大数据是否有额外收益 |

## 4. 配置文件清单

### 训练配置
| 文件 | 节点 | Batch | Frames | 实验数 |
|---|---:|---:|---:|---:|
| `batch8_train_all_v30.yaml` | 32 | 16 | 10000 | 4 |
| `batch8_train_all_v30_a30.yaml` | 32 | 8 | 10000 | 4 |
| `batch8_train_all_v30_opt_quick.yaml` | 1 | 16 | 500 | 3 |

### 评估配置
| 文件 | 模型数 | 帧数/seq |
|---|---:|---:|
| `eval_generalization_v30.yaml` | 8 (4×2ckpt) | 400 |
| `eval_generalization_v30_opt_quick.yaml` | 6 (3×2ckpt) | 400 |
| `eval_generalization_v29_v30_unified_800frames.yaml` | 2 | 800 |

## 5. 预期结果与部署决策树

```
IF G3-full (small+AS=1+10000帧) MEDW800 < 0.065°:
    → 部署 G3-full, V30 全规模训练有明确收益
    → 进一步验证 1600 帧聚合

ELIF G3-DANN < G3:
    → DANN 对 small + unfreeze2 有正面效果
    → 考虑部署 G3-DANN

ELIF G3-10deg < G3:
    → 10° 训练在 5° 评估中有优势
    → 更新训练配置

ELSE:
    → 坚持 V29-G3 配方直接部署 (MEDW800=0.070°)
    → 聚焦增加聚合帧数 (800→1600)
```

## 6. 时序聚合优化路线 (已验证, geodesic metric)

| 窗口大小 | G3 实测 | 400→X 缩放 | √N 效率 |
|---:|---:|---:|---:|
| W=1 (单帧) | 2.316° | — | — |
| W=100 | 0.206° | — | — |
| W=200 | 0.135° | — | — |
| W=400 | 0.092° | 基准 | 基准 |
| W=800 | **0.070°** | 0.761 | 93% |
| W=1600 (预测) | ~0.050° | ~0.543 | — |

**最大杠杆: 增加聚合帧数** (√N 缩放, 已验证效率 93%)

## 7. 模型参数量参考

| 模型 | Backbone | Total Params | Checkpoint | 部署评估 |
|---|---|---:|---:|---|
| G3 (dinov2-small) | 22.1M | **49.7M** | **400 MB** | **推荐部署** |
| ~~H8 (dinov2-base)~~ | ~~86.6M~~ | ~~116.2M~~ | ~~749 MB~~ | ~~已淘汰~~ |
