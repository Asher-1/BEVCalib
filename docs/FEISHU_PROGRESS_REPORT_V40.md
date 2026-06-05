# BEVCalib 标定方向组内进展汇报（V40 阶段）

汇报人：BEVCalib 标定组  
汇报日期：2026-05-29  
适用场景：组内技术同步 / 飞书文档直接粘贴


## 一、一句话结论

我们在一条业内公认的难题上走了很久：v32–v35 阶段主要卡在 shortcut（模型几乎不跟 T_init 变化）；v37 在 Jacobian 上突破，但部署指标 MEDW 没跟上；v39 回归路线 shortcut 复发。近一周按 2025 顶会论文 What Really Matters 重构了 V40 架构，并完成了工程侧「致命 bug」修复与首轮 P1c 训练诊断——精度起点已经很好（MEDW≈0.34°），但新组件 MatchHead 尚未学起来。问题已被定位，P1d 针对性实验已就绪，下一步用约 8h GPU 做一次有明确 Go/No-Go 的验证，而不是盲目加长训。


## 二、研究背景：为什么这件事本身就难

LiDAR–Camera 在线外参标定是自动驾驶感知的基础能力，也是业界长期未彻底解决的难题：

| 难点 | 具体表现 |
|------|----------|
| Shortcut（T_init 泄漏） | 模型输出 ≈ T_init + 固定偏移，改 T_init 误差几乎不变 |
| 双指标分裂 | 单帧精度好 ≠ 序列部署指标 MEDW 好（v37 已实证） |
| 泛化 | 训练集 KITTI 固定 FOV/分辨率/点云密度，跨车型/跨传感器掉点明显 |
| 范式混乱 | 回归、Cost Volume、匹配+PnP 混用，很多「看起来新」仍是 retrieval |

2025 年论文 What Really Matters 通过系统实验指出：匹配 + 几何监督 + 真实传感器模拟增强，才是泛化和抗 shortcut 的主路径——这直接驱动了 V40 GeoMatch-ProjCalib（GMP）设计。


## 三、历史实验回顾：v32–v35 及后续为何「看起来慢」

### 3.1 v32–v35：核心问题是 Shortcut，不是「没训够」

logs/all_training_data/v32-v35 目录下的实验链（v32→v33 corr→v34 xcorr→v35 ExplicitTInit）结论一致：

| 版本 | Val 最优 | Jacobian ±10° | 本质问题 |
|------|----------|---------------|----------|
| v30–v34 | ~2.3°–2.8° | ≈0.0005（FIXED） | 学固定矫正量，不感知 T_init 偏差 |
| v35 | 2.05° | 0.283（ADAPTIVE） | 首次打破 shortcut，但距理想 1.0 仍差很远 |
| v35b 等变体 | ~2.07° | 0.120 | 加 sensitivity loss 反而削弱自适应 |

根因（TECH_REPORT_v35）：T_init 仅经 BEV 投影间接进入网络，多层卷积/Transformer 后信号衰减，模型只能学训练集统计平均偏移。

结论：v32–v35 不是「调参不够」，而是范式/信息通路有问题；v35 是重要里程碑，但未同时解决泛化与部署指标。


### 3.2 v37：Jacobian 赢了，MEDW 输了——双 KPI 分裂被证实

| 指标 | v36 基线 | v37 NativeCross（Ep400） | 判定 |
|------|----------|------------------------|------|
| Jacobian ±10° | 0.29（WEAK） | 0.88+（ADAPTIVE） | v37 成功克服 shortcut |
| MEDW200（±5° test） | 0.413° | 0.457°（最优 ckpt） | 部署指标未超越 v36 |
| 单帧 Rot | — | 0.84°（Ep350） | 单帧有改善 |

关键洞察：Val 最优 ≠ MEDW 最优；继续用 val loss 早停会损害部署指标。这解释了「训练 loss 在降、上线指标不稳」的长期困惑。


### 3.3 v39：当前对照基线——MEDW 好，Shortcut 仍在

v39 fast Exp1（±10°，60ep，已跑完）观测：

| 指标 | 数值 | 说明 |
|------|------|------|
| Best MEDW200 | 0.362°（Ep1） | 部署精度起点优秀，已达 P1c Gate（<0.4°） |
| Jacobian ep31 | -0.802（WEAK） | shortcut 未解决，甚至比 v37 退化 |
| Train Rot ep60 | 3.74° | 训练误差仍高，依赖 pretrain 路径 |

结论：v39 的 proj_only 回归头把 MEDW 做漂亮了，但没有继承 v37 的 Jacobian 能力——这正是 V40 要修的「分裂」问题。


## 四、为什么整体进展「感觉慢」——四个层次的原因

### 4.1 科学层：在错误范式上迭代了很久

v30–v39 大量实验本质是「更强的回归头 / 更强的融合」，而 2025 论文指出：Cost Volume 仍是 retrieval；真正有效的是显式 2D–3D 匹配 + 可微 PnP + 几何监督。V40 直到本周才按论文完成架构对齐（GMP = MatchHead + DiffEPnP + GeoConsistency + P2 增强）。


### 4.2 工程层：近期才发现并修复的「隐形失败」

5/29 全面审计发现，若不修复以下问题，P1c 训练等于白跑：

| 问题 | 严重性 | 修复状态 | 影响 |
|------|--------|----------|------|
| differentiable_epnp 未传入训练脚本 | 致命 | 已修复 | yaml 配了也不生效，P1c 退化为 proj_only |
| iterative_refine=3 | 严重 | 已改为 0/1 | 60ep 从约 75h 降至约 8h（-89%） |
| P2 内参增强未启用 | 严重 | P2a 已加强 | 跨数据集泛化受限 |
| P2b FOV/LiDAR 稀疏 | 中等 | 5/29 代码已完成 | 论文对齐度 60%→约 95% |

这些不是「模型不行」，而是「实验根本没按设计跑」——修复后迭代速度会实质性加快。


### 4.3 训练层：P1c 首轮已跑，诊断清晰

P1c（GeoMatch + DiffEPnP + P2a）Epoch 1–15 监控结论：

| 维度 | 状态 | 数据 |
|------|------|------|
| 配置正确性 | 通过 | fusion_backend、MatchHead、DiffEPnP 均已加载 |
| 起点质量（MEDW） | 优秀 | ep1 MEDW=0.341°，优于 Gate 0.4° |
| 训练收敛 | 极差 | Train Rot 4.52°→4.16°，15ep 仅改善 8%（预期约 50%） |
| MatchHead | 失效 | 77% fallback 到 refine-only；match_valid_ratio 22%→18% |
| Jacobian | 恶化 | -1.195（目标 >0.5），继承 v39 shortcut |

根因（已写入监控报告）：V39 pretrain 过强 → 惰性 fine-tune，模型走老 RefineHead 捷径，拒绝学 Match 路径。这不是失败，而是「第一次把真问题量化了」。


### 4.4 资源层：GPU 排队是硬约束

| 项目 | 数值 |
|------|------|
| P1c 主实验（8×L20，60ep） | 约 8h / 次 |
| 完整 V40 矩阵（P0→P1→P2→P3） | 数十到上百 GPU·小时 |
| 队列等待 | 实验设计必须「每次有结论」，不能堆无效长跑 |

策略：smoke（约 30min）→ 主实验 → ep15 Go/No-Go，避免 70% 概率失败仍跑满 60ep。


## 五、V40 当前进展：本周完成了什么

### 5.1 架构与设计（按论文对齐）

```
论文结论                    V40 GMP 实现
────────────────────────────────────────────
匹配 > 回归              → MatchHead + DiffEPnP（P1b/P1c）
几何监督必要              → GeoConsistency（P0a）
Cost Volume = retrieval  → LocalCorr 作特征，不作 pose 回归
数据增强 = 泛化核心        → P2a 内参/dropout + P2b FOV/LiDAR 稀疏
```

设计文档、实现清单、验证报告、P2 差距分析均已完成；Dry-run 验证参数链完整。


### 5.2 工程修复清单（5/29 全部落地）

| 修复项 | 修复前 | 修复后 |
|--------|--------|--------|
| 训练耗时（60ep） | ~75h | ~8h |
| DiffEPnP 参数传递 | 断裂 | 完整 |
| 内参增强 | 0% 启用 | fx/fy ±5%，cx/cy ±3% |
| 点云 dropout | 5% | 15% |
| P2b FOV crop | 未实现 | 已实现并验证 |
| P2b LiDAR 垂直层稀疏 | 未实现 | 已实现并验证 |


### 5.3 实验状态一览

| 实验 | 状态 | 关键结果 |
|------|------|----------|
| v39 fast Exp1 | 已完成 | MEDW=0.362°；Jacobian WEAK（-0.802） |
| V40 P0/P0ab（5°） | 已有 log | 基础设施验证 |
| V40 P1c main | ep15 监控完成 | MEDW 达标；Match/Jacobian 未达标 |
| V40 P1d strong_match | 配置就绪 | 针对 P1c 根因的三项改动 |
| V40 smoke | 脚本就绪 | 启动前 30min 必跑 |


## 六、有希望的地方——为什么值得继续投入

### 6.1 最难的「起点精度」已经有了

P1c ep1 MEDW=0.341°（Roll 0.18° / Pitch 0.21° / Yaw 0.13°），说明 DINOv2 + PointGPT + AttenDualFusion + V39 pretrain 这条特征通路是有效的。我们不是在从零爬精度，而是在已有 0.34° 部署精度基础上，补 Jacobian 和 Match 路径——问题范围被大幅缩小。


### 6.2 v37 证明 Shortcut 可以被架构解决

v37 NativeCross 做到 Jacobian 0.88+（ADAPTIVE），说明团队有能力做抗 shortcut 结构。V40 是把 v37 的 Jacobian 能力与 v39 的 MEDW 能力合并到一条路径上——方向正确，只是 v39 pretrain 抑制了新 head 学习。


### 6.3 问题已从「不知道为什么」变成「知道改什么」

P1c 失败模式非常具体：

| 症状 | 根因 | P1d 对策 |
|------|------|----------|
| 77% match fallback | fallback 阈值过高 + 监督太弱 | corr_loss_weight 1→5；min_valid 0.3→0.15 |
| MatchHead 不学习 | warmup 5ep 无梯度 | warmup 5→2 |
| Train Rot 不降 | 走 RefineHead 捷径 | 强制更多 batch 走 EPnP 路径 |

P1d 不是「再试一次」，而是有假设、有指标、有 ep15 决策阈值的实验。


### 6.4 工程债务已基本还清

参数传递、训练效率、P2 增强代码——这些曾导致「跑了等于没跑」的坑已填平。后续实验结果的可信度会显著提高，每次 8h 训练都能真实反映架构改动效果。


### 6.5 论文对齐度已达可发表工程水准

| 维度 | 对齐度 |
|------|--------|
| P0/P1（匹配+几何） | ~100% |
| P2 数据增强 | ~70%（P2b 完成后约 95%） |
| 综合 | ~70%，跨数据集外推仍差论文约 30%（可接受为 Phase 2 目标） |


## 七、接下来怎么办（资源优先、结论驱动）

### 7.1 近期实验计划（按优先级）

| 顺序 | 实验 | GPU 成本 | 决策点 | 成功标准（ep15） |
|------|------|----------|--------|------------------|
| 1 | P1d smoke（2ep） | ~30min | 配置生效 | corr_loss_weight=5.0 出现在日志 |
| 2 | P1d strong_match（60ep） | ~8h | ep15 Go/No-Go | fallback<50% 且 Train Rot<3.5° |
| 3a | 若 ep15 通过 | 继续至 ep60 | 交付候选 |
| 3b | 若 fallback 仍>65% | match_only 消融 | 强制禁用 fallback |
| 3c | 若仍失败 | from_scratch 消融 | 去掉 pretrain head 抑制 |

### 7.2 ep15 决策表（避免浪费队列）

| match_fallback | Train Rot | 决策 |
|----------------|-----------|------|
| <50% | <3.5° | Go：继续 ep60 |
| 50–65% | 3.5–4.0° | 观察：跑到 ep30 再评估 |
| >65% | >4.0° | Stop：切换 match_only 或 from_scratch |

### 7.3 中期路线（P1d 通过后）

| Phase | 目标 | 关键指标 |
|-------|------|----------|
| P1d 成功 | Match 路径 + Jacobian 恢复 | Jacobian>0.5 @ ep30；MEDW 不劣于 0.35° |
| P2 完整 | 跨数据集泛化 | 外推误差从 ~0.9° 向 ~0.7° 靠拢 |
| P3 三阶段 | 大角度交付 | 10°→5°→3° progressive；Jacobian>0.85 |


## 八、资源需求说明（给管理者）

### 8.1 我们需要的不是「更多盲目训练」

| 请求 | 理由 |
|------|------|
| 1 次 P1d 主实验（8h × 8卡） | 有明确 ep15 结论，成功则直接冲交付 |
| smoke 插队（30min） | 避免参数链再次静默失败 |
| 失败时 1 次消融（8h） | match_only 或 from_scratch，二选一 |

不建议：在 P1c 已知 70%+ 失败概率的情况下继续跑满剩余 45ep（约 4h GPU，无新信息）。


### 8.2 与业务目标的对应关系

| 业务目标 | 当前最近距离 | V40 终局目标 |
|----------|--------------|--------------|
| 部署 MEDW200 | 0.341°（已达标 Gate 0.4°） | <0.30° |
| 抗 shortcut（Jacobian） | -1.195（未达标） | >0.85 |
| 跨车型/跨传感器 | P2 部分对齐 | 论文级 ~0.7° 外推 |


## 九、给组内同事的信心总结

1. 慢，主要是因为在错误范式上走了很久（v32–v39），且近期才发现工程级致命 bug——不是团队不努力，是问题本身难 + 诊断成本高。

2. 快，体现在本周：论文驱动重构、工程修复、P1c 首轮量化诊断、P1d 方案就绪——迭代周期从「数周无结论」压缩到「8h 一个决策点」。

3. 有希望，因为最难的 MEDW 起点（0.34°）已达成，v37 已证明 Jacobian 可修复，P1c 失败原因明确且 P1d 有针对性——这是可验证的下一跳，不是赌博。

4. 需要耐心的是：LiDAR–Camera 标定是 2025 年仍在发顶会的工作，GPU 排队是硬约束；我们用 Go/No-Go 机制确保每次排队都有结论。


## 十、附录：关键数据速查

| 项目 | 数值 |
|------|------|
| v35 Jacobian 突破 | 0.0005 → 0.283 |
| v37 Jacobian | 0.88+（ADAPTIVE） |
| v36 MEDW200 基线 | 0.413° |
| v39 MEDW200 | 0.362°（Ep1 best） |
| v39 Jacobian ep31 | -0.802（WEAK） |
| P1c ep1 MEDW | 0.341° |
| P1c ep15 Train Rot | 4.16°（仅改善 8%） |
| P1c match fallback | 77% |
| P1c Jacobian | -1.195 |
| 训练时间修复 | 75h → 8h（-89%） |
| P1d 成功概率（文档估计） | 60–70%（vs P1c 20–30%） |


## 十一、相关文档索引

| 文档 | 内容 |
|------|------|
| PAPER_What_Really_Matters_for_LiDAR_Camera_Calibration.md | 2025 论文研读与 V40 映射 |
| V40_DESIGN.md | GMP 架构与 Phase 规划 |
| V40_COMPLETE_SUMMARY_WITH_P2.md | 修复与 P2 增强总结 |
| V40_P1C_TRAINING_MONITOR_EP15_FINAL.md | P1c 训练诊断（核心证据） |
| V40_P1D_CONFIG_CHANGES.md | P1d 改动与 Go/No-Go |
| V40_QUICK_START_GUIDE.md | 启动与监控命令 |
| TECH_REPORT_v35.md | v32–v35 shortcut 历史 |
| V37_GENERALIZATION_REPORT.md | v37 双 KPI 分裂实证 |
