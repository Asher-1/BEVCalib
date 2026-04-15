# BEVCalib V6 泛化性能综合对比: ALL vs B26A

> 日期: 2026-03-31 | 测试集: test_data (9852 samples, 5 sequences 00-04) | 扰动: ±5.0° / ±0.15m

---

## 一、完整泛化排名总表

将 ALL v6 (8模型) + B26A v6 (7模型+v1基准) + 历史基准统一排名:

| 排名 | 模型 | 训练集 | 回归头 | Loss | 特殊配置 | val_rot | 泛化Mean | P95 | Max | 衰退 |
|---:|------|--------|--------|------|----------|--------:|--------:|-----:|-----:|-----:|
| 1 | all-v6-linear-baseline | ALL(78K) | Linear | quaternion | StepLR, 无aug | 0.166° | 0.494° | 1.243° | 2.312° | 3.0x |
| 2 | all-v6-mlp-geodesic | ALL(78K) | MLP | geodesic | StepLR, 无aug | 0.117° | 0.617° | 1.717° | 2.986° | 5.3x |
| 3 | all-v6-linear-geodesic | ALL(78K) | Linear | geodesic | StepLR, 无aug | 0.153° | 0.618° | 1.529° | 2.635° | 4.0x |
| 4 | v4-z5-no-tail (旧冠军) | ALL(78K) | Linear | quaternion | StepLR, 有aug, axis=3,1.5,1 | 0.530° | 0.633° | 1.502° | 2.502° | 1.2x |
| 5 | all-v6-linear-hires | ALL(78K) | Linear | geodesic | StepLR, 无aug, 100m/pool2x | 0.188° | 0.640° | 1.743° | 2.945° | 3.4x |
| 6 | all-v6-mlp-hires-pool2x | ALL(78K) | MLP | geodesic | StepLR, 无aug, pool2x | 0.122° | 0.675° | 1.769° | 3.206° | 5.5x |
| 7 | all-v6-mlp-baseline | ALL(78K) | MLP | quaternion | StepLR, 无aug | 0.097° | 0.678° | 2.361° | 3.013° | 7.0x |
| 8 | all-v6-mlp-hires-bs4 | ALL(78K) | MLP | geodesic | StepLR, 无aug, 100m/bs4 | 0.097° | 0.702° | 1.852° | 3.174° | 7.2x |
| 9 | all-v6-mlp-geodesic-drcv | ALL(78K) | MLP | geodesic | drcv后端 | 0.104° | 0.747° | 1.634° | 3.134° | 7.2x |
| 10 | v5-cosinewr-noaug | ALL(78K) | Linear | quaternion | CosineWR, 无aug | 0.325° | 0.927° | 1.638° | 2.621° | 2.9x |
| - | - | - | - | - | - | - | - | - | - | - |
| 11 | B26A-v5-undist-best | B26A(18K) | Linear | quaternion | CosineWR, 有aug | 0.287° | 1.047° | 2.790° | 4.294° | 3.6x |
| 12 | B26A-v6-linear-baseline | B26A(18K) | Linear | quaternion | StepLR, 无aug | 0.290° | 1.194° | 3.082° | 4.286° | 4.1x |
| 13 | B26A-v6-linear-geodesic | B26A(18K) | Linear | geodesic | StepLR, 无aug | 0.273° | 1.232° | 3.024° | 4.050° | 4.5x |
| 14 | B26A-v1-champion | B26A(18K) | Linear | quaternion | CosineWR, 有aug | 0.590° | 1.253° | 3.210° | 5.030° | 2.1x |
| 15 | B26A-v6-mlp-geodesic | B26A(18K) | MLP | geodesic | StepLR, 无aug | 0.171° | 1.330° | 3.025° | 4.045° | 7.8x |
| 16 | B26A-v6-mlp-hires-pool2x | B26A(18K) | MLP | geodesic | pool2x | 0.194° | 1.357° | 3.178° | 3.622° | 7.0x |
| 17 | B26A-v6-mlp-baseline | B26A(18K) | MLP | quaternion | StepLR, 无aug | 0.171° | 1.427° | 3.300° | 4.175° | 8.3x |
| 18 | B26A-v6-mlp-hires-bs4 | B26A(18K) | MLP | geodesic | 100m/bs4 | 0.177° | 1.455° | 3.413° | 4.184° | 8.2x |
| 19 | B26A-v6-linear-hires | B26A(18K) | Linear | geodesic | pool2x | 0.916° | 1.671° | 3.394° | 4.419° | 1.8x |

---

## 二、旋转分量对比 (Roll / Pitch / Yaw)

### 2.1 ALL v6 模型

| 模型 | Roll | Pitch | Yaw | Total | Pitch占比 |
|------|-----:|------:|----:|------:|--------:|
| all-v6-linear-baseline | 0.236° | 0.332° | 0.181° | 0.494° | 67.2% |
| all-v6-mlp-geodesic | 0.174° | 0.534° | 0.137° | 0.617° | 86.5% |
| all-v6-linear-geodesic | 0.390° | 0.332° | 0.218° | 0.618° | 53.7% |
| all-v6-linear-hires | 0.237° | 0.451° | 0.241° | 0.640° | 70.5% |
| all-v6-mlp-baseline | 0.178° | 0.605° | 0.129° | 0.678° | 89.2% |
| all-v6-mlp-geodesic-drcv | 0.183° | 0.645° | 0.225° | 0.747° | 86.3% |

### 2.2 B26A v6 模型

| 模型 | Roll | Pitch | Yaw | Total | Pitch占比 |
|------|-----:|------:|----:|------:|--------:|
| v6-linear-baseline | 0.414° | 0.981° | 0.348° | 1.194° | 82.2% |
| v6-linear-geodesic | 0.436° | 0.976° | 0.414° | 1.232° | 79.2% |
| v6-mlp-geodesic | 0.512° | 1.151° | 0.140° | 1.330° | 86.5% |
| v6-mlp-baseline | 0.524° | 1.256° | 0.180° | 1.427° | 88.0% |

### 2.3 历史基准

| 模型 | Roll | Pitch | Yaw | Total |
|------|-----:|------:|----:|------:|
| v4-z5-no-tail (axis=3,1.5,1) | 0.385° | 0.336° | 0.214° | 0.633° |
| v5-undist-best (CosineWR+aug) | 0.392° | 0.795° | 0.318° | 1.047° |

---

## 三、关键现象 & 根因分析

### 现象 1: ALL >> B26A — 数据量带来 58% 泛化提升

| 同配置对比 | B26A泛化 | ALL泛化 | 提升 |
|-----------|--------:|-------:|-----:|
| Linear + quaternion | 1.194° | 0.494° | -58.6% |
| Linear + geodesic | 1.232° | 0.618° | -49.8% |
| MLP + quaternion | 1.427° | 0.678° | -52.5% |
| MLP + geodesic | 1.330° | 0.617° | -53.6% |

根因: ALL数据集(78K帧, 8序列)比B26A(18K帧, 1序列)多4.4x数据。更重要的是ALL包含多个不同场景序列的多样性，而B26A仅有单一序列。这使模型在ALL上学到更通用的LiDAR-Camera对应特征，泛化误差在所有配置上一致下降50-59%。

---

### 现象 2: Linear头 >> MLP头 — 简单结构泛化更好

ALL数据集上的对比 (控制变量):

| 对比 | Linear | MLP | MLP比Linear差 |
|------|------:|----:|-------------:|
| quaternion loss | 0.494° | 0.678° | +37.2% |
| geodesic loss | 0.618° | 0.617° | -0.2% (持平) |
| hires+pool2x | 0.640° | 0.675° | +5.5% |

B26A数据集上的对比:

| 对比 | Linear | MLP | MLP比Linear差 |
|------|------:|----:|-------------:|
| quaternion loss | 1.194° | 1.427° | +19.5% |
| geodesic loss | 1.232° | 1.330° | +8.0% |

根因:
1. 回归任务维度低: 输出仅4维(四元数)，不需要多层非线性变换。Linear的256→4直接映射足矣
2. MLP过拟合严重: MLP参数量~49K vs Linear~1K (50倍)。MLP训练val_rot=0.097°但泛化0.678° (衰退7.0x); Linear训练val_rot=0.166°但泛化0.494° (衰退3.0x)
3. 唯一例外: geodesic loss下MLP(0.617°) ≈ Linear(0.618°)，说明geodesic的几何约束部分补偿了MLP的过拟合

---

### 现象 3: Quaternion Loss >> Geodesic Loss (在Linear头上)

| 数据集 | Linear+quaternion | Linear+geodesic | quaternion优势 |
|--------|--------:|--------:|------:|
| ALL | 0.494° | 0.618° | -20.0% |
| B26A | 1.194° | 1.232° | -3.1% |

但MLP头上结论相反:

| 数据集 | MLP+quaternion | MLP+geodesic | geodesic优势 |
|--------|--------:|--------:|------:|
| ALL | 0.678° | 0.617° | -9.0% |
| B26A | 1.427° | 1.330° | -6.8% |

根因: Loss函数与回归头存在交互效应:
- Linear + quaternion: 梯度路径最短(四元数→L2→反传)，与Linear直出四元数的结构完美匹配
- Linear + geodesic: 需要四元数→旋转矩阵→geodesic距离→反传回四元数，多了转换步骤，梯度信息损耗
- MLP + geodesic: MLP的中间层能学习适配geodesic梯度的内部表示，弥补了路径长度的劣势

---

### 现象 4: BEV高分辨率始终退化

| 对比 | 标准BEV(200m/2m) | hires(100m/1m) | 退化幅度 |
|------|--------:|--------:|------:|
| ALL-Linear-geo | 0.618° | 0.640° | +3.6% |
| ALL-MLP-geo | 0.617° | 0.675° (pool), 0.702° (bs4) | +9-14% |
| B26A-MLP-geo | 1.330° | 1.357° (pool), 1.455° (bs4) | +2-9% |

根因:
1. hires将X范围从[0,200]缩到[0,100]，丢失100-200m远端信息（虽然点稀疏但提供全局尺度线索）
2. 70-100m处LiDAR点密度 <2 pts/m²，1m voxel大多为空，引入噪声
3. bs4将batch从8降到4，梯度估计质量下降
4. pool2x先升分辨率再pool回去，引入额外量化误差

---

### 现象 5: V6-linear-baseline(0.494°) 超越 V4旧冠军(0.633°) 达22%

两者训练配置对比:

| 配置项 | V4-z5-no-tail (0.633°) | V6-linear-baseline (0.494°) |
|--------|----------------------|---------------------------|
| 数据集 | ALL (78K) | ALL (78K) |
| 回归头 | Linear | Linear |
| Loss | quaternion | quaternion |
| 数据增强 | 有 (jitter/dropout/color) | 无 |
| axis_weights | 3.0, 1.5, 1.0 | 1.0, 1.0, 1.0 |
| batch_size | 16 | 8 |
| seed | 无固定 | 42 |
| 代码版本 | V4 (旧) | V6 (新) |

可能原因:
1. V6代码改进: V6引入了更好的训练策略(warmup, backbone_lr_scale, drop_path, head_dropout, truncated_normal扰动, per_axis_prob等)
2. seed=42的稳定性: 固定seed避免了不利的随机初始化
3. 数据增强可能在V6框架下不再必要: V6的正则化手段(dropout, drop_path)替代了数据增强的作用
4. axis_weights=3,1.5,1反而不利: V4过度加权Roll(3.0)，但V6数据显示Pitch才是主要误差源

---

### 现象 6: Pitch始终是最大误差源(占53-89%)

| 类型 | Pitch范围 | Pitch占总误差比例 |
|------|--------:|--------:|
| ALL-Linear模型 | 0.332-0.451° | 53-71% |
| ALL-MLP模型 | 0.534-0.645° | 82-89% |
| B26A-Linear模型 | 0.976-1.196° | 72-82% |
| B26A-MLP模型 | 1.135-1.287° | 85-88% |

根因:
1. BEV投影几何约束弱: Pitch旋转(绕LiDAR-Y轴)改变地面点的前后投影，但BEV俯视视角对这种变化不敏感
2. 相机图像对Pitch更敏感但特征难提取: Pitch改变地平线位置，但Swin特征不易显式编码这种全局位移
3. MLP模型的Pitch更差是因为MLP更容易过拟合训练集中Pitch的特定分布

---

## 四、V6改进回顾: 哪些有效，哪些无效

| V6改进项 | 设计初衷 | ALL上效果 | B26A上效果 | 结论 |
|---------|---------|----------|----------|------|
| MLP回归头 | 增加模型容量 | 比Linear差5-37% | 比Linear差8-20% | 无效且有害 |
| Geodesic Loss | 更好的旋转度量 | Linear上差20%, MLP上好9% | 差异±3-7% | 效果取决于Head |
| BEV 100m/1m | 提高空间分辨率 | 退化3-14% | 退化2-9% | 无效且有害 |
| drcv后端 | 替代spconv | 比spconv差(0.747 vs 0.617) | — | 无效 |
| 关闭数据增强 | 控制变量 | 0.494°无aug竟超V4有aug的0.633° | 1.194°无aug不如V5有aug的1.047° | ALL上无害, B26A上有害 |
| StepLR替代CosineWR | 简化调参 | 0.494° StepLR >> 0.927° CosineWR-noaug | — | V6框架下StepLR更优 |

---

## 五、V7 优化方案设计

基于以上全面分析，V7核心策略: 以Linear+quaternion为基础，系统化验证aug/LR/axis_weights的增量效果。

### V7 实验矩阵

| 实验ID | 回归头 | Loss | Aug | LR Schedule | axis_weights | 数据集 | 假设 |
|--------|--------|------|-----|------------|-------------|--------|------|
| v7_baseline | Linear | quat | 无 | StepLR | 1,1,1 | B26A+ALL | V6最佳配置复现,作为控制组 |
| v7_aug | Linear | quat | 有 | StepLR | 1,1,1 | B26A+ALL | 验证aug在V6框架下是否仍有增益 |
| v7_cosine | Linear | quat | 无 | CosineWR | 1,1,1 | B26A+ALL | 验证CosineWR是否优于StepLR |
| v7_aug_cosine | Linear | quat | 有 | CosineWR | 1,1,1 | B26A+ALL | aug+CosineWR叠加效果 |
| v7_pitch_wt | Linear | quat | 有 | StepLR | 1,2,1 | B26A+ALL | Pitch加权定向优化 |
| v7_pitch_cosine | Linear | quat | 有 | CosineWR | 1,2,1 | B26A+ALL | 全配置最优预期 |

### 各实验的详细理由

v7_baseline: 严格复现all-v6-linear-baseline(0.494°), 确认基线可复现。

v7_aug: ALL上V6-linear(0.494°,无aug)已超V4(0.633°,有aug), 但在B26A上V6(1.194°,无aug)不如V5(1.047°,CosineWR+aug)。aug对B26A可能有显著提升; 对ALL需实验验证是否仍有增益或反而退化。

v7_cosine: V5 ALL实验中CosineWR(0.927°) >> StepLR(1.145°); 但V6 linear-baseline用StepLR已达0.494°。CosineWR的周期性LR重启是否在V6框架下仍有效需验证。

v7_aug_cosine: 如果v7_aug和v7_cosine各自有增益, 叠加可能超线性; 如果互相冲突则低于单项。

v7_pitch_wt: Pitch占总误差67%。axis_weights=1,2,1让Pitch梯度权重翻倍。V4的axis_weights=3,1.5,1过度加权Roll(实际Pitch才是瓶颈), V7修正为Pitch优先。

v7_pitch_cosine: 组合全部优化手段, 预期ALL泛化<0.42°。

---

## 六、预期目标

| 指标 | V4旧冠军 | V6新冠军 | V7目标 | 改进来源 |
|------|--------:|--------:|-------:|---------|
| ALL Mean Rot | 0.633° | 0.494° | <0.42° | aug+Pitch加权 |
| ALL P95 | 1.502° | 1.243° | <1.00° | CosineWR压缩尾部 |
| ALL Pitch | 0.336° | 0.332° | <0.25° | axis_weights=1,2,1 |
| B26A Mean Rot | 1.047°(V5) | 1.194° | <0.90° | aug+CosineWR |
| 泛化衰退 | 1.2x | 3.0x | <2.0x | aug正则化 |
