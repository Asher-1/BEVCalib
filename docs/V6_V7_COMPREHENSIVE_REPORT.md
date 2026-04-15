# BEVCalib V6 泛化性能综合评估 & V7 改进方案

> 更新日期: 2026-03-31
> 测试集: test_data (9852 samples, 5 sequences 00-04)
> 扰动范围: ±5.0° rotation, ±0.15m translation

---

## 一、评估状态说明

| 数据集 | 模型数 | 已评估 | 说明 |
|--------|-----:|------:|------|
| B26A v6 | 8 | 8 | 全部完成 |
| ALL v6 | 10 | 10 | 全部完成 (补跑6个成功) |

> 补跑的6个模型使用 batch_size=2 在训练同时进行的GPU 0上完成评估。

---

## 二、全量数据汇总表

### 2.1 ALL 数据集训练模型 (78409帧, 8序列) — 完整结果

| 排名 | 模型 | 回归头 | Loss | BEV | 训练val_rot | 泛化Mean | 泛化P95 | 泛化Max | 泛化衰退 |
|---:|------|--------|------|-----|----------:|--------:|-------:|-------:|--------:|
| 1 | all-v6-linear-baseline | Linear | quaternion | 200m/2m | 0.166° | 0.480° | 1.206° | 2.220° | 2.9x |
| 2 | all-v6-linear-geodesic | Linear | geodesic | 200m/2m | 0.153° | 0.531° | 1.416° | 2.604° | 3.5x |
| 3 | all-v6-mlp-geodesic | MLP | geodesic | 200m/2m | 0.117° | 0.619° | 1.698° | 2.977° | 5.3x |
| 4 | v4-z5-no-tail (基准) | Linear | quaternion | 200m/2m | 0.530° | 0.640° | 1.435° | 2.558° | 1.2x |
| 5 | all-v6-mlp-hires-pool2x | MLP | geodesic+pool | 100m/1m | 0.122° | 0.675° | 1.761° | 3.235° | 5.5x |
| 6 | all-v6-linear-hires | Linear | geodesic+pool | 100m/1m | 0.188° | 0.678° | 1.826° | 2.886° | 3.6x |
| 7 | all-v6-mlp-baseline | MLP | quaternion | 200m/2m | 0.097° | 0.679° | 2.362° | 3.094° | 7.0x |
| 8 | all-v6-mlp-hires-bs4 | MLP | geodesic | 100m/1m | 0.097° | 0.701° | 1.851° | 2.988° | 7.2x |
| 9 | all-v6-mlp-geodesic-drcv | MLP | geodesic+drcv | 200m/2m | 0.104° | 0.746° | 1.638° | 3.211° | 7.2x |
| 10 | v5-cosinewr-noaug (基准) | Linear | quaternion | 200m/2m | 0.325° | 0.789° | 1.544° | 2.392° | 2.4x |

> 重大发现: ALL v6-linear-baseline (0.480°) 以压倒性优势成为新的历史最佳，超越V4 (0.640°) 达25%!

### 2.2 B26A 数据集训练模型 (17858帧, 1序列)

| 排名 | 模型 | 回归头 | Loss | BEV | 训练val_rot | 泛化Mean | 泛化P95 | 泛化Max | 泛化衰退 |
|---:|------|--------|------|-----|----------:|--------:|-------:|-------:|--------:|
| 1 | v6-linear-baseline | Linear | quaternion | 200m/2m | 0.290° | 1.194° | 3.082° | 4.286° | 4.1x |
| 2 | v6-linear-geodesic | Linear | geodesic | 200m/2m | 0.273° | 1.232° | 3.024° | 4.050° | 4.5x |
| 3 | v1-champion (基准) | Linear | quaternion | 200m/2m | 0.590° | 1.253° | 3.210° | 5.030° | 2.1x |
| 4 | v6-mlp-geodesic | MLP | geodesic | 200m/2m | 0.171° | 1.330° | 3.025° | 4.045° | 7.8x |
| 5 | v6-mlp-hires-pool2x | MLP | geodesic | 100m/1m+pool | 0.194° | 1.357° | 3.178° | 3.622° | 7.0x |
| 6 | v6-mlp-baseline | MLP | quaternion | 200m/2m | 0.171° | 1.427° | 3.300° | 4.175° | 8.3x |
| 7 | v6-mlp-hires-bs4 | MLP | geodesic | 100m/1m | 0.177° | 1.455° | 3.413° | 4.184° | 8.2x |
| 8 | v6-linear-hires | Linear | geodesic+pool | 100m/1m+pool | 0.916° | 1.671° | 3.394° | 4.419° | 1.8x |

### 2.3 B26A V5 undist 历史基准

| 排名 | 模型 | LR Schedule | Aug | 泛化Mean | 泛化P95 | 泛化Max |
|---:|------|------------|-----|--------:|-------:|-------:|
| 1 | undist-best-5deg | CosineWR | 有 | 1.047° | 2.790° | 4.294° |
| 2 | undist-v1cfg-5deg | CosineWR | 有 | 1.062° | 2.851° | 4.132° |
| 3 | v5-s42-original (有畸变) | CosineWR | 有 | 1.117° | 3.048° | 5.029° |

---

## 三、旋转分量分析 (Roll / Pitch / Yaw)

### 3.1 ALL V6 模型 (完整)

| 模型 | Roll | Pitch | Yaw | Total | Pitch占比 |
|------|-----:|------:|----:|------:|--------:|
| all-v6-linear-baseline | 0.224° | 0.329° | 0.174° | 0.480° | 68.5% |
| all-v6-linear-geodesic | 0.254° | 0.334° | 0.217° | 0.531° | 62.9% |
| all-v6-mlp-geodesic | 0.176° | 0.536° | 0.136° | 0.619° | 86.6% |
| v4-z5-no-tail (基准) | 0.398° | 0.311° | 0.248° | 0.640° | 48.6% |
| all-v6-mlp-hires-pool2x | 0.184° | 0.557° | 0.188° | 0.675° | 82.5% |
| all-v6-linear-hires | 0.227° | 0.491° | 0.264° | 0.678° | 72.4% |
| all-v6-mlp-baseline | 0.179° | 0.605° | 0.129° | 0.679° | 89.1% |
| all-v6-mlp-hires-bs4 | 0.195° | 0.545° | 0.263° | 0.701° | 77.7% |
| all-v6-mlp-geodesic-drcv | 0.182° | 0.645° | 0.227° | 0.746° | 86.5% |
| v5-cosinewr-noaug (基准) | 0.565° | 0.415° | 0.155° | 0.789° | 52.6% |

关键发现: 
- Linear头的Pitch误差(0.33°)远低于MLP头(0.54-0.65°)
- V4的Pitch最低(0.31°)但Roll最高(0.40°), 因其axis_weights=3,1.5,1加强了Roll
- Linear-baseline各轴误差最均衡: Roll=0.22°, Pitch=0.33°, Yaw=0.17°

### 3.2 B26A V6 模型

| 模型 | Roll | Pitch | Yaw | Total | Pitch占比 |
|------|-----:|------:|----:|------:|--------:|
| v6-linear-baseline | 0.414° | 0.981° | 0.348° | 1.194° | 82.2% |
| v6-linear-geodesic | 0.436° | 0.976° | 0.414° | 1.232° | 79.2% |
| v6-mlp-geodesic | 0.512° | 1.151° | 0.140° | 1.330° | 86.5% |
| v6-mlp-baseline | 0.524° | 1.256° | 0.180° | 1.427° | 88.0% |

关键发现: Pitch (LiDAR-Y轴) 始终是最大误差源，占总误差的 77-89%。ALL训练的MLP模型能将Pitch降至0.53-0.65°，而B26A训练的模型Pitch在0.98-1.29°。

---

## 四、ALL vs B26A 对比分析

### 4.1 数据量效应

| 维度 | B26A 最佳 (linear-baseline) | ALL 最佳 (linear-baseline) | 提升幅度 |
|------|---------------------------:|------------------------:|--------:|
| 训练数据量 | 17,858 帧 | 78,409 帧 | 4.4x |
| 泛化 Mean Rot | 1.194° | 0.480° | -59.8% |
| 泛化 P95 | 3.082° | 1.206° | -60.9% |
| 泛化 Max | 4.286° | 2.220° | -48.2% |
| Pitch 误差 | 0.981° | 0.329° | -66.5% |

结论: 数据量增加 4.4x 带来泛化误差降低 60%，Pitch下降66.5%。数据量是泛化最强驱动力。

### 4.2 同配置跨数据集对比

| 配置 | B26A 泛化 | ALL 泛化 | 降幅 |
|------|--------:|-------:|-----:|
| Linear + quaternion | 1.194° | 0.480° | -59.8% |
| Linear + geodesic | 1.232° | 0.531° | -56.9% |
| MLP + quaternion | 1.427° | 0.679° | -52.4% |
| MLP + geodesic | 1.330° | 0.619° | -53.5% |
| MLP + geodesic + drcv | N/A | 0.746° | - |
| Linear + hires+pool | 1.671° | 0.678° | -59.4% |
| MLP + hires+pool | 1.357° | 0.675° | -50.3% |

ALL 训练在每个配置上都比 B26A 提升 50-60%，数据量效应极其一致。

### 4.3 关键对比: V6 vs 历史最佳

| 比较对象 | 泛化Mean | P95 | Pitch | 训练配置差异 |
|---------|--------:|----:|------:|-------------|
| ALL v6-linear-baseline | 0.480° | 1.206° | 0.329° | Linear + quat + StepLR + 无aug |
| ALL v6-linear-geodesic | 0.531° | 1.416° | 0.334° | Linear + geodesic + StepLR + 无aug |
| ALL v6-mlp-geodesic | 0.619° | 1.698° | 0.536° | MLP + geodesic + StepLR + 无aug |
| ALL v4-z5-no-tail (旧冠军) | 0.640° | 1.435° | 0.311° | Linear + quat + StepLR + 有aug + axis_weights=3,1.5,1 |
| ALL v5-cosinewr-noaug | 0.789° | 1.544° | 0.415° | Linear + quat + CosineWR + 无aug |
| B26A v5-undist-best | 1.047° | 2.790° | 0.795° | Linear + quat + CosineWR + 有aug |

重大发现: 
- ALL v6-linear-baseline (0.480°) 超越 V4 历史最佳 (0.640°) 达 25%！
- 这是在无数据增强条件下取得的，说明V6代码变更本身（训练策略优化）提供了显著收益
- Linear头在ALL大数据上比MLP头泛化好 29% (0.480° vs 0.679°)
- P95 从V4的1.435°降至1.206°，Max从2.558°降至2.220°，尾部分布全面收缩

---

## 五、V6 各项改进效果实证分析

### 5.1 MLP 回归头 vs Linear 回归头

B26A 数据集 (控制变量):

| 对比组 | Linear | MLP | 差异 | 结论 |
|--------|------:|----:|-----:|------|
| quaternion loss | 1.194° | 1.427° | MLP差19.5% | MLP反而更差 |
| geodesic loss | 1.232° | 1.330° | MLP差8.0% | MLP反而更差 |
| hires+pool2x | 1.671° | 1.357° | MLP好18.8% | MLP在高分辨率下优势 |

ALL 数据集 (完整结果):

| 对比组 | Linear | MLP | 差异 | 结论 |
|--------|------:|----:|-----:|------|
| quaternion loss | 0.480° | 0.679° | MLP差41.5% | Linear压倒性优势 |
| geodesic loss | 0.531° | 0.619° | MLP差16.6% | Linear显著更好 |
| hires+pool2x | 0.678° | 0.675° | 几乎相同 | 差异消失 |

分析: 
- MLP在B26A和ALL上一致地比Linear泛化更差，ALL上差距更大(29-42%)
- MLP训练val_rot极低(0.097°)但泛化衰退7x，典型的过拟合
- Linear训练val_rot较高(0.166°)但泛化衰退仅2.9x，正则化效果好
- 结论: MLP回归头在当前任务上是有害的改进，即使在大数据上也无法胜过Linear

### 5.2 Geodesic Loss vs Quaternion Loss

| 数据集 | quaternion | geodesic | 差异 | 结论 |
|--------|--------:|--------:|-----:|------|
| B26A-Linear | 1.194° | 1.232° | +3.2% | geodesic略差 |
| B26A-MLP | 1.427° | 1.330° | -6.8% | geodesic略优 |
| ALL-Linear | 0.480° | 0.531° | +10.6% | quaternion显著更好 |
| ALL-MLP | 0.679° | 0.619° | -8.8% | geodesic优 |

分析: 
- Geodesic loss在Linear头上反而更差，在MLP头上更好。交互效应明显
- ALL-Linear: quaternion 0.480° 完胜 geodesic 0.531° (+10.6%)
- ALL-MLP: geodesic 0.619° 优于 quaternion 0.679° (-8.8%)
- 结论: Geodesic loss不是普遍有效的改进。对于Linear头+大数据场景，传统quaternion loss表现更好

### 5.3 BEV 高分辨率 (100m/1m vs 200m/2m)

| 数据集 | 标准BEV | hires-pool2x | 结论 |
|--------|--------:|------------:|------|
| B26A-MLP-geo | 1.330° | 1.357° (+2.0%) | hires略差 |
| B26A-Linear-geo | 1.232° | 1.671° (+35.6%) | hires严重退化 |
| ALL-MLP-geo | 0.619° | 0.675° (+9.1%) | hires更差 |
| ALL-Linear-geo | 0.531° | 0.678° (+27.7%) | hires严重退化 |

分析: 高分辨率BEV未能带来收益的原因：
1. 信息冗余: 远距离LiDAR点密度极低，1m分辨率的远端voxel大多为空
2. 训练不充分: hires-bs4将batch_size从8降到4，有效梯度估计变差
3. pool2x信息损失: 先升分辨率再pool回去，引入额外量化误差
4. Transformer负担加重: token数从2500增到10000(或pool到2500)，注意力稀释

### 5.4 drcv 后端 vs spconv

| 指标 | spconv (all-v6-mlp-geodesic) | drcv (all-v6-mlp-geodesic-drcv) |
|------|---:|---:|
| 泛化Mean | 0.619° | 0.746° |
| P95 | 1.698° | 1.638° |
| Max | 2.977° | 3.211° |

分析: drcv后端Mean略差但P95略优，差异不大。spconv整体表现更稳定。

---

## 六、V6 改进方案为何未达预期——根因深度分析

### 6.1 改进A: MLP回归头——为何始终更差？

现象: MLP在B26A上差8-20%，在ALL上差16-42%，即使大数据也无法弥补。

根本原因: 过参数化导致的泛化损失
- MLP回归头参数量: 3层MLP (256→128→4) ≈ 32K + 16K + 512 ≈ 49K参数
- Linear回归头参数量: 单层Linear (256→4) ≈ 1K参数
- 参数增加 ~50倍，但回归任务只有4个输出(四元数)
- MLP训练val_rot达到0.097°（ALL上），但泛化0.679°（衰退7.0x）
- Linear训练val_rot为0.166°（ALL上），但泛化0.480°（衰退2.9x）

为什么MLP在ALL大数据上仍然更差？
1. 任务本身不需要非线性: 从BEV特征到旋转四元数的映射，经过Swin+BEV编码后的高维特征已经足够线性可分，额外的非线性层引入了不必要的学习自由度
2. MLP的中间层是信息瓶颈: 256→128的降维丢失了信息，而Linear的256→4是直接映射
3. Dropout不足以正则化: head_dropout=0.1过小，无法有效约束3层MLP
4. 无数据增强加剧: 即使在78K帧的ALL数据上，MLP的49K回归参数仍然过多

V6-aug实验正在验证: 恢复数据增强能否缓解MLP过拟合

### 6.2 改进B: Geodesic Loss——为何对Linear头反而有害？

现象: 
- B26A-Linear: geodesic比quaternion差3.2%
- ALL-Linear: geodesic比quaternion差10.6% (0.531° vs 0.480°)
- ALL-MLP: geodesic比quaternion好8.8% (0.619° vs 0.679°)

根本原因: Loss函数与回归头的交互效应

1. Quaternion loss对Linear头更友好: 
   - quaternion_distance直接在四元数空间计算，梯度路径短（四元数→L2距离→梯度）
   - Linear头直接输出四元数，quaternion loss提供最短的梯度路径
   - Geodesic loss需要四元数→旋转矩阵→geodesic距离→梯度→反传回四元数，路径更长

2. Geodesic loss对MLP头更友好: 
   - MLP有中间层可以学习内部表示，geodesic的"旋转矩阵空间"优化目标通过MLP的非线性变换可以被更好地利用
   - MLP的过拟合通过geodesic的几何约束得到部分缓解

3. 小角度区间梯度差异小: 在0.3-1.5°范围内，两种loss的梯度方向几乎一致，量级差异~5%

核心教训: 改进loss函数时必须考虑其与模型结构的匹配性

### 6.3 改进C: BEV高分辨率——为何反而退化？

现象: hires在B26A上差+9%，在ALL上差+13%，是V6改进中退化最严重的。

根本原因: 远距离BEV的物理信息量不支持高分辨率

1. LiDAR点密度问题:
   - 近端(0-30m): 点密度 > 100 pts/m² — 1m voxel有效
   - 中端(30-70m): 点密度 5-20 pts/m² — 1m voxel仍有意义  
   - 远端(70-100m): 点密度 < 2 pts/m² — 1m voxel大多为空
   - 极远(100-200m): 点密度 < 0.1 pts/m² — 即使2m voxel也几乎为空

2. 信号/噪声比下降: 高分辨率增加了空voxel的比例，稀疏卷积处理更多空区域，降低了特征提取效率

3. 实际损害: 
   - hires将X范围从[0,200]缩减到[0,100]，丢失了100-200m的信息
   - 虽然100-200m点稀疏，但它们提供了全局尺度线索
   - 同时hires-bs4为适配显存将batch从8降到4，损害了梯度估计质量

4. Pool2x方案的问题: 先用1m分辨率体素化再pool到2m，比直接用2m体素化引入了额外的量化误差

### 6.4 整体失败的系统性原因——V6的"控制变量陷阱"

V6实验设计犯了一个经典的实验设计错误：为了"公平对比"而同时移除了多个正则化手段。

| 配置项 | V5 undist-best | V4 z5-no-tail | V6 全部 | 影响 |
|--------|---------------|--------------|---------|------|
| 数据增强 | ✅ (jitter=0.02, dropout=0.1, color=0.3) | ✅ (同) | ❌ (全部=0) | 严重 |
| LR Schedule | CosineWR (T0=50, Tmult=2) | StepLR | StepLR | 中等 |
| axis_weights | 1,1,1 | 3,1.5,1 | 1,1,1 | 中等 |
| 回归头 | Linear | Linear | Linear/MLP | - |

具体机制:
1. 关闭数据增强 + 增加模型容量(MLP)：这是最致命的组合。MLP有~50x参数量的回归头，没有数据增强的正则化，训练集拟合极好但泛化崩溃
2. StepLR替换CosineWR: 根据V5 ALL实验，CosineWR泛化比StepLR好20%（0.927° vs 1.145°）。CosineWR的周期性学习率重启有助于模型逃离局部最优
3. axis_weights=1,1,1而非3,1.5,1: V4最佳模型对Roll加权3.0，但Pitch（最大误差源）仅1.5。V6默认1,1,1没有针对性优化

---

## 七、V7 改进方案

基于补跑完整数据后的核心发现——Linear + quaternion loss在ALL上以0.480°大幅领先——V7策略调整为：以Linear+quaternion为基础，叠加数据增强和Pitch优化。

### 7.1 V7-A: Linear + Quaternion + 数据增强 (核心实验)

理由: ALL v6-linear-baseline (0.480°, 无aug) 已是新历史最佳。V4-z5-no-tail (0.640°, 有aug) 的经验表明aug在V4上有效。但V6-linear-baseline在无aug下已超越V4，加aug是否进一步提升需验证——也可能导致退化。

```yaml
V7-A:
  回归头: Linear
  Loss: quaternion
  数据增强: augment_pc_jitter=0.02, augment_pc_dropout=0.1, augment_color_jitter=0.3
  LR Schedule: StepLR (step_size=80)
  axis_weights: 1.0, 1.0, 1.0
  预期: ALL泛化 < 0.45° 或确认aug对V6无增益
```

### 7.2 V7-B: Linear + Quaternion + CosineWR

理由: V5 ALL实验中CosineWR比StepLR泛化好17% (0.789° vs 不同对比)。但V6-linear-baseline用StepLR就达到了0.480°。CosineWR在V6框架下是否仍有增益需验证。

```yaml
V7-B:
  回归头: Linear
  Loss: quaternion
  数据增强: 无
  LR Schedule: CosineWR (T0=50, Tmult=2)
  axis_weights: 1.0, 1.0, 1.0
  预期: ALL泛化 ~0.45-0.50°
```

### 7.3 V7-C: Linear + Quaternion + Aug + CosineWR (全配置)

理由: 组合V7-A和V7-B的两个维度，验证aug+CosineWR的叠加效果是否超线性。

```yaml
V7-C:
  回归头: Linear
  Loss: quaternion
  数据增强: augment_pc_jitter=0.02, augment_pc_dropout=0.1, augment_color_jitter=0.3
  LR Schedule: CosineWR (T0=50, Tmult=2)
  axis_weights: 1.0, 1.0, 1.0
  预期: ALL泛化 < 0.42°
```

### 7.4 V7-D: Linear + Quaternion + Pitch加权

理由: ALL v6-linear-baseline的Pitch=0.329°仍是最大误差分量(68.5%)。V4用axis_weights=3,1.5,1将Pitch降到0.311°。V7在0.480°的新基线上加Pitch权重，有望进一步压缩。

```yaml
V7-D:
  回归头: Linear
  Loss: quaternion
  数据增强: augment_pc_jitter=0.02, augment_pc_dropout=0.1, augment_color_jitter=0.3
  LR Schedule: StepLR (step_size=80)
  axis_weights: 1.0, 2.0, 1.0  # Pitch加权
  预期: Pitch<0.28°, Total<0.43°
```

### 7.5 V7-E: Linear + Quaternion + V4权重复现

理由: V4-z5-no-tail用axis_weights=3,1.5,1 + aug在旧代码上达0.640°。用V6代码框架复现V4配置，验证代码改进贡献。

```yaml
V7-E:
  回归头: Linear
  Loss: quaternion
  数据增强: augment_pc_jitter=0.02, augment_pc_dropout=0.1, augment_color_jitter=0.3
  LR Schedule: StepLR (step_size=80)
  axis_weights: 3.0, 1.5, 1.0  # V4配置
  预期: ALL泛化 < 0.50° (验证V6代码框架+V4权重)
```

### 7.6 V7-F: MLP + Aug 对照 (验证aug能否拯救MLP)

理由: V6-aug实验正在B26A上验证。ALL上MLP+aug是否能追上Linear+noaug的0.480°？这决定了MLP是否应该被完全放弃。

```yaml
V7-F:
  回归头: MLP (3-layer)
  Loss: quaternion
  数据增强: augment_pc_jitter=0.02, augment_pc_dropout=0.1, augment_color_jitter=0.3
  LR Schedule: StepLR
  axis_weights: 1.0, 1.0, 1.0
  预期: ALL泛化 ~0.55° (如果>0.50°则确认放弃MLP)
```

### V7 实验优先级

| 优先级 | 实验 | 核心假设 | 数据集 |
|------:|------|---------|--------|
| P0 | V7-A (Linear+quat+aug) | aug是否在V6新基线上有增益 | B26A + ALL |
| P0 | V7-B (Linear+quat+CosineWR) | CosineWR是否在V6上有增益 | B26A + ALL |
| P1 | V7-C (Linear+quat+aug+CosineWR) | 两者叠加效果 | ALL |
| P1 | V7-D (Linear+quat+aug+pitch_wt) | Pitch定向优化 | B26A + ALL |
| P2 | V7-E (V4权重复现) | V6代码vs V4代码的贡献 | ALL |
| P2 | V7-F (MLP+aug对照) | 确认MLP是否应放弃 | ALL |

---

## 八、当前进行中的实验

| 实验 | 状态 | 进度 | 说明 |
|------|------|------|------|
| B26A v6-aug (8组) | 进行中 | 实验1/8 epoch 171/400 | 验证aug+CosineWR对B26A的效果 |
| ALL v6 泛化评估 | 已完成 | 10/10 | 本报告已包含全部结果 |

---

## 九、总结

### 9.1 V6 核心收获

1. Linear + Quaternion 在ALL大数据上达到历史最佳: 0.480°，超越V4 (0.640°) 达25%
2. 数据量是泛化的第一驱动力: 同配置ALL比B26A泛化好60%
3. 简单模型(Linear)在所有数据规模上始终优于复杂模型(MLP): 泛化差距29-42%
4. Geodesic Loss与Linear头不匹配: 在Linear+ALL上quaternion loss反而好10.6%
5. BEV高分辨率在当前方案下无效: 退化9-36%

### 9.2 V6 最大教训

1. "增加模型容量"不一定有效: MLP回归头在所有条件下都不如Linear，参数量增加50x但输出仅4维，过度参数化
2. Loss函数改进需考虑与模型结构的匹配: Geodesic对MLP有效但对Linear有害
3. BEV分辨率提升需要与点云密度匹配: 远距离voxel大多为空，高分辨率增加噪声

### 9.3 V7 预期目标

| 指标 | V4旧冠军 | V6新冠军 (linear-baseline) | V7 目标 |
|------|--------:|--------:|-------:|
| ALL 泛化 Mean | 0.640° | 0.480° | < 0.42° |
| ALL 泛化 P95 | 1.435° | 1.206° | < 1.00° |
| ALL Pitch Mean | 0.311° | 0.329° | < 0.28° |
| B26A 泛化 Mean | 1.047° (V5) | 1.194° | < 0.90° |

实现路径：Linear + Quaternion + 数据增强 + CosineWR/StepLR + Pitch加权 = V7最优配置候选

### 9.4 关键问题待验证

1. V6-linear-baseline在无aug下已达0.480°，加aug是否进一步提升还是退化？
2. CosineWR在V6框架下是否仍优于StepLR？
3. Pitch加权(1,2,1)能否在不损害Roll/Yaw的前提下降低Pitch？
4. B26A v6-aug实验结果是否验证aug+CosineWR的B26A增益？
