# V39 优化方案对比：参数解冻策略

## 参数量分析

### 当前状态（完全frozen）
```
Total: 41.7M
Trainable: 1.3M (3.13%)
  ├─ proj_branch cross-attn: 1.24M
  └─ fusion_head: 66.8K
```

### 方案A：解冻PointGPT最后2层（推荐⭐）
```
Trainable: 3-4M (8-10%)
优势：
  - 点云编码器能适应标定任务
  - 训练成本可控
  - Jacobian预期提升最明显
劣势：
  - 需要修改代码支持层级freeze
```

### 方案B：解冻DINOv2最后1层+PointGPT最后2层
```
Trainable: 6-8M (15-20%)
优势：
  - 两个encoder都能fine-tune
  - 理论上泛化能力最强
劣势：
  - 训练时间增加30-50%
  - 过拟合风险
  - batch_size可能需要降低
```

### 方案C：保持完全frozen
```
Trainable: 1.3M (3.13%)
优势：
  - 训练最快
  - batch_size可以最大（40+）
  - 代码无需修改
劣势：
  - Jacobian可能难以>0.85
  - 泛化到±10°可能不足
```

## 推荐决策树

```
如果Jacobian是核心目标（>0.85）:
  └─> 方案A（解冻PointGPT最后2层）
      时间成本：+20% 训练时间
      预期：Jacobian 0.7-0.9
      
如果时间紧迫（快速验证）:
  └─> 方案C（保持frozen）+ 增大batch_size到40
      时间成本：-20% 训练时间
      预期：Jacobian 0.6-0.8
      
如果追求极致泛化:
  └─> 方案B（双encoder部分解冻）
      时间成本：+50% 训练时间
      预期：Jacobian 0.8-0.95
```

## 实施细节

### 方案A实施（推荐）

需要修改代码添加`pointgpt_freeze_layers`支持：

```python
# pointgpt_wrapper.py 或 hybrid_triple_calib.py
self.pointgpt_encoder = PointGPTEncoder(
    checkpoint_path=pointgpt_ckpt,
    config_path=pointgpt_config,
    max_depth=pointgpt_max_depth,
    freeze=True,
    freeze_layers=18,  # 新增参数：冻结前18层
)
```

### 方案C实施（快速）

无需改代码，直接修改yaml：

```yaml
batch_size: 40  # 24 → 40
learning_rate: 6.0e-4  # 按√batch_size缩放：4e-4 * √(40/24)
```
