# 修复与调试工具 (Utilities)

数据修复和问题诊断工具集。

---

## 📋 工具列表

### `fix_calib_tr_inversion.py` - 修复Tr矩阵反向问题

自动检测并修复标定文件中Tr矩阵的坐标系反向问题。

**问题描述**:
- Tr矩阵应表示 Velodyne → Camera 变换
- 有时会错误地存储为 Camera → Velodyne
- 导致点云投影完全错位

**检测方法**:
- 检查旋转矩阵行列式
- det(R) ≈ -1.0 表示坐标系反向
- det(R) ≈ 1.0 为正常

**使用方法**:
```bash
# 检查并修复
python tools/utils/fix_calib_tr_inversion.py \
    --dataset_root /path/to/dataset \
    --fix

# 仅检查不修复
python tools/utils/fix_calib_tr_inversion.py \
    --dataset_root /path/to/dataset \
    --check_only
```

**输出示例**:
```
检查序列 00:
  当前Tr矩阵行列式: -0.999
  ❌ 检测到坐标系反向！
  
修复序列 00:
  原始Tr矩阵已备份: calib.txt.bak
  ✅ Tr矩阵已修复
  新行列式: 1.000

验证修复结果:
  ✅ 投影效果正常
```

**安全措施**:
- 自动备份原始标定文件（.bak）
- 修复前生成测试投影
- 修复后再次验证投影

---

### `debug_undistortion.py` - 调试点云去畸变

对比和调试C++与Python点云去畸变算法实现。

**功能**:
- 运行C++和Python去畸变
- 对比输出结果
- 可视化差异
- 性能基准测试

**使用方法**:
```bash
# 对比去畸变算法
python tools/utils/debug_undistortion.py \
    --input_pc pointcloud.bin \
    --poses poses.txt \
    --output_dir debug_undistort/

# 性能测试
python tools/utils/debug_undistortion.py \
    --input_pc pointcloud.bin \
    --poses poses.txt \
    --benchmark \
    --iterations 100
```

**输出内容**:
```
debug_undistort/
├── original.png              # 原始点云可视化
├── undistorted_cpp.png       # C++去畸变结果
├── undistorted_python.png    # Python去畸变结果
├── difference.png            # 差异可视化
├── statistics.txt            # 统计信息
└── benchmark.json            # 性能数据
```

**统计指标**:
- 点位置平均误差
- 最大误差
- 误差标准差
- 执行时间对比

---

## 🎯 使用场景

### 场景1: 修复投影错位问题

```bash
# 1. 检测问题
python tools/validation/check_projection_headless.py \
    --dataset_root dataset/ --sequence 00 --frame 0 \
    --output before_fix.png

# 2. 如发现投影完全错位，修复Tr矩阵
python tools/utils/fix_calib_tr_inversion.py \
    --dataset_root dataset/ --fix

# 3. 验证修复效果
python tools/validation/check_projection_headless.py \
    --dataset_root dataset/ --sequence 00 --frame 0 \
    --output after_fix.png

# 4. 对比修复前后
compare before_fix.png after_fix.png
```

### 场景2: 批量修复数据集

```bash
# 批量检查所有序列
python tools/utils/fix_calib_tr_inversion.py \
    --dataset_root dataset/ \
    --check_only \
    > tr_check_report.txt

# 批量修复
python tools/utils/fix_calib_tr_inversion.py \
    --dataset_root dataset/ \
    --fix \
    --all_sequences

# 完整验证
python tools/validation/validate_dataset.py full dataset/ \
    --output-dir validation_after_fix/ --full
```

### 场景3: 调试去畸变算法

```bash
# 对比C++和Python实现
python tools/utils/debug_undistortion.py \
    --dataset_root dataset/ \
    --sequence 00 \
    --frame 100 \
    --output_dir undistort_debug/

# 查看差异
cat undistort_debug/statistics.txt
```

### 场景4: 性能优化

```bash
# 基准测试
python tools/utils/debug_undistortion.py \
    --input_pc dataset/sequences/00/velodyne/000100.bin \
    --poses dataset/sequences/00/poses.txt \
    --benchmark \
    --iterations 100 \
    --output benchmark_results.json

# 分析结果
python -m json.tool benchmark_results.json
```

---

## 🔧 问题诊断流程

### 诊断流程图

```
数据问题
    ↓
1. 基础验证
   python tools/validation/validate_dataset.py summary dataset/
    ↓
2. 格式检查
   python tools/validation/validate_kitti_odometry.py ...
    ↓
3. Tr矩阵检查
   python tools/validation/verify_dataset_tr_fix.py ...
    ↓
4. 投影测试
   python tools/validation/check_projection_headless.py ...
    ↓
5. 问题定位
    ├─ 投影错位 → fix_calib_tr_inversion.py
    ├─ 去畸变错误 → debug_undistortion.py
    └─ 其他问题 → 查看日志和文档
```

### 常见问题诊断

**问题1: 点云投影完全错位**
```bash
# 症状: 点云与图像完全不重合
# 原因: Tr矩阵坐标系反向
# 解决:
python tools/utils/fix_calib_tr_inversion.py \
    --dataset_root dataset/ --fix
```

**问题2: 投影部分偏移**
```bash
# 症状: 点云与图像大致对齐但有偏移
# 原因: 标定精度问题或坐标系定义错误
# 诊断:
python tools/validation/verify_dataset_tr_fix.py \
    --dataset_root dataset/ --verbose

# 可能需要重新标定
```

**问题3: 运动物体拖尾**
```bash
# 症状: 运动物体点云有拖尾效应
# 原因: 点云去畸变问题
# 调试:
python tools/utils/debug_undistortion.py \
    --dataset_root dataset/ --sequence 00 --frame 100 \
    --output_dir debug/
```

**问题4: 旋转矩阵不正交**
```bash
# 症状: Tr矩阵验证显示正交性误差大
# 原因: 矩阵损坏或精度丢失
# 检查:
python tools/validation/verify_dataset_tr_fix.py \
    --dataset_root dataset/ --detailed

# 可能需要从原始标定重新生成
```

---

## ⚠️ 注意事项

### 1. 修复工具的安全性

**自动备份**:
- 所有修复工具会自动备份原始文件
- 备份文件命名: `原文件名.bak`
- 修复失败时可手动恢复

**验证修复**:
- 修复后自动运行验证
- 生成修复前后对比图
- 建议人工确认修复效果

### 2. 批量操作风险

批量修复前请：
1. 先在单个序列上测试
2. 确认修复逻辑正确
3. 备份整个数据集
4. 分批次处理大数据集

### 3. 性能考虑

**调试工具开销**:
- `debug_undistortion.py` 会运行多次算法
- 基准测试模式更慢
- 大点云文件需要更多时间

**建议**:
- 使用小数据集进行调试
- 性能测试选择代表性帧
- 避免在生产环境频繁运行

---

## 💡 最佳实践

### 1. 修复前务必备份

```bash
# 完整备份数据集
cp -r dataset/ dataset_backup/

# 或仅备份标定文件
find dataset/ -name "calib.txt" -exec cp {} {}.backup \;
```

### 2. 渐进式验证

```bash
# 1. 快速检查
python tools/utils/fix_calib_tr_inversion.py \
    --dataset_root dataset/ --check_only

# 2. 单序列测试修复
python tools/utils/fix_calib_tr_inversion.py \
    --dataset_root dataset/ --sequence 00 --fix

# 3. 验证修复效果
python tools/validation/check_projection_headless.py \
    --dataset_root dataset/ --sequence 00 --frame 0 \
    --output test.png

# 4. 确认无误后批量修复
python tools/utils/fix_calib_tr_inversion.py \
    --dataset_root dataset/ --fix --all_sequences
```

### 3. 问题追踪

```bash
# 记录修复过程
python tools/utils/fix_calib_tr_inversion.py \
    --dataset_root dataset/ --fix 2>&1 | tee fix_log.txt

# 保存修复报告
python tools/validation/validate_dataset.py full dataset/ \
    --output-dir validation_after_fix/
```

---

## 🔗 相关文档

- [主文档](../README.md)
- [验证工具文档](../validation/README.md)
- [数据准备文档](../preparation/README.md)

---

## 📞 技术支持

如遇到无法解决的问题，请提供：
1. 问题描述
2. 数据集信息（序列数、帧数等）
3. 验证工具输出
4. 修复工具日志
5. 示例投影图（修复前后）

---

**最后更新**: 2026-03-01
