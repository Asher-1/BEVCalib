# 工具重组迁移指南

本指南帮助您从旧的扁平结构迁移到新的分类目录结构。

**更新日期**: 2026-03-01

---

## 📋 重组概述

### 变更原因

- **便于查找**: 按功能分类，快速定位工具
- **降低复杂度**: 每个目录职责单一，易于理解
- **便于维护**: 相关工具集中管理，减少耦合
- **模块化**: 各工具独立开发和测试

### 新目录结构

```
tools/
├── README.md                    # 主文档
├── docs/                        # 📚 文档
│   ├── QUICK_START.md
│   ├── ARCHITECTURE.md
│   ├── VALIDATION_MODES.md
│   └── MIGRATION_GUIDE.md      # 本文档
├── preparation/                 # 📊 数据准备
├── validation/                  # ✅ 验证工具
├── visualization/               # 🎨 可视化
├── analysis/                    # 📈 分析工具
├── utils/                       # 🔧 修复调试
└── scripts/                     # 🔄 Shell脚本
```

---

## 🔄 路径映射表

### 验证工具 (validation/)

| 旧路径 | 新路径 |
|--------|--------|
| `tools/validate_dataset.py` | `tools/validation/validate_dataset.py` |
| `tools/validate_kitti_odometry.py` | `tools/validation/validate_kitti_odometry.py` |
| `tools/verify_dataset_tr_fix.py` | `tools/validation/verify_dataset_tr_fix.py` |
| `tools/comprehensive_projection_validation.py` | `tools/validation/comprehensive_projection_validation.py` |
| `tools/check_projection_headless.py` | `tools/validation/check_projection_headless.py` |
| `tools/show_dataset_summary.py` | `tools/validation/show_dataset_summary.py` |

### 可视化工具 (visualization/)

| 旧路径 | 新路径 |
|--------|--------|
| `tools/visualize_projection.py` | `tools/visualization/visualize_projection.py` |
| `tools/view_pointcloud.py` | `tools/visualization/view_pointcloud.py` |
| `tools/visualize_kitti_structure.py` | `tools/visualization/visualize_kitti_structure.py` |
| `tools/batch_generate_projections.py` | `tools/visualization/batch_generate_projections.py` |

### 数据准备工具 (preparation/)

| 旧路径 | 新路径 |
|--------|--------|
| `tools/prepare_custom_dataset.py` | `tools/preparation/prepare_custom_dataset.py` |
| `tools/batch_prepare_trips.py` | `tools/preparation/batch_prepare_trips.py` |

### 分析工具 (analysis/)

| 旧路径 | 新路径 |
|--------|--------|
| `tools/analyze_perturbation_training.py` | `tools/analysis/analyze_perturbation_training.py` |

### 修复调试工具 (utils/)

| 旧路径 | 新路径 |
|--------|--------|
| `tools/fix_calib_tr_inversion.py` | `tools/utils/fix_calib_tr_inversion.py` |
| `tools/debug_undistortion.py` | `tools/utils/debug_undistortion.py` |

### Shell脚本 (scripts/)

| 旧路径 | 新路径 |
|--------|--------|
| `tools/monitor_batch_processing.sh` | `tools/scripts/monitor_batch_processing.sh` |
| `tools/stop_batch_processing.sh` | `tools/scripts/stop_batch_processing.sh` |

### 文档 (docs/)

| 旧路径 | 新路径 |
|--------|--------|
| `tools/QUICK_START.md` | `tools/docs/QUICK_START.md` |
| `tools/ARCHITECTURE.md` | `tools/docs/ARCHITECTURE.md` |
| `tools/VALIDATION_MODES.md` | `tools/docs/VALIDATION_MODES.md` |

---

## 🚀 快速迁移

### 方式1: 使用新路径（推荐）

直接使用新的分类路径：

```bash
# 旧命令
python tools/validate_dataset.py summary dataset/

# 新命令
python tools/validation/validate_dataset.py summary dataset/
```

### 方式2: 创建软链接（兼容性）

如果有大量脚本依赖旧路径，可以创建软链接：

```bash
cd /path/to/BEVCalib/tools

# 验证工具
ln -s validation/validate_dataset.py validate_dataset.py
ln -s validation/validate_kitti_odometry.py validate_kitti_odometry.py
ln -s validation/check_projection_headless.py check_projection_headless.py

# 可视化工具
ln -s visualization/visualize_projection.py visualize_projection.py
ln -s visualization/view_pointcloud.py view_pointcloud.py

# 数据准备
ln -s preparation/prepare_custom_dataset.py prepare_custom_dataset.py

# ... 其他工具类似
```

**注意**: 软链接仅作为过渡方案，建议尽快迁移到新路径。

### 方式3: 批量更新脚本

使用以下脚本批量更新您的代码：

```bash
#!/bin/bash
# update_tool_paths.sh

# 定义替换规则
declare -A PATH_MAP=(
    ["tools/validate_dataset.py"]="tools/validation/validate_dataset.py"
    ["tools/visualize_projection.py"]="tools/visualization/visualize_projection.py"
    ["tools/prepare_custom_dataset.py"]="tools/preparation/prepare_custom_dataset.py"
    # 添加其他映射...
)

# 查找并替换
for old_path in "${!PATH_MAP[@]}"; do
    new_path="${PATH_MAP[$old_path]}"
    echo "替换: $old_path -> $new_path"
    
    # 在所有.py和.sh文件中替换
    find . -type f \( -name "*.py" -o -name "*.sh" \) -exec \
        sed -i "s|$old_path|$new_path|g" {} +
done

echo "路径更新完成！"
```

---

## 📝 常见迁移场景

### 场景1: 验证脚本

**旧代码**:
```bash
#!/bin/bash
python tools/validate_dataset.py summary dataset/
python tools/validate_dataset.py full dataset/ --output validation/
```

**新代码**:
```bash
#!/bin/bash
python tools/validation/validate_dataset.py summary dataset/
python tools/validation/validate_dataset.py full dataset/ --output validation/
```

### 场景2: Python导入

**旧代码**:
```python
import sys
from pathlib import Path

# 添加tools目录
sys.path.insert(0, str(Path(__file__).parent / 'tools'))

from validate_kitti_odometry import KITTIOdometryValidator
```

**新代码**:
```python
import sys
from pathlib import Path

# 添加validation目录
sys.path.insert(0, str(Path(__file__).parent / 'tools' / 'validation'))

from validate_kitti_odometry import KITTIOdometryValidator
```

### 场景3: 训练脚本中的验证

**旧代码**:
```python
import subprocess

def validate_dataset(dataset_path):
    cmd = ['python', 'tools/validate_dataset.py', 'summary', dataset_path]
    subprocess.run(cmd, check=True)
```

**新代码**:
```python
import subprocess

def validate_dataset(dataset_path):
    cmd = ['python', 'tools/validation/validate_dataset.py', 'summary', dataset_path]
    subprocess.run(cmd, check=True)
```

### 场景4: Makefile

**旧Makefile**:
```makefile
validate:
	python tools/validate_dataset.py full dataset/ --output validation/

visualize:
	python tools/visualize_projection.py --dataset_root dataset/ --sequence 00
```

**新Makefile**:
```makefile
validate:
	python tools/validation/validate_dataset.py full dataset/ --output validation/

visualize:
	python tools/visualization/visualize_projection.py --dataset_root dataset/ --sequence 00
```

---

## ⚠️ 注意事项

### 1. 导入路径变更

**影响范围**:
- Python脚本中的 `import` 语句
- `subprocess` 调用的脚本路径
- `sys.path` 修改

**检查方法**:
```bash
# 查找所有可能受影响的导入
grep -r "from validate_kitti" your_project/
grep -r "import validate_" your_project/
grep -r "tools/validate" your_project/
```

### 2. 相对路径问题

如果您的脚本使用相对路径调用工具，需要更新：

```python
# 旧代码（假设在BEVCalib/根目录）
subprocess.run(['python', 'tools/validate_dataset.py', ...])

# 新代码
subprocess.run(['python', 'tools/validation/validate_dataset.py', ...])
```

### 3. 文档链接

如果您有自己的文档引用工具路径，也需要更新：

```markdown
<!-- 旧链接 -->
详见 [validate_dataset.py](../tools/validate_dataset.py)

<!-- 新链接 -->
详见 [validate_dataset.py](../tools/validation/validate_dataset.py)
```

---

## ✅ 迁移检查清单

完成迁移后，请检查：

- [ ] 所有Python脚本中的导入语句已更新
- [ ] 所有Shell脚本中的路径已更新
- [ ] subprocess调用的路径已更新
- [ ] Makefile中的路径已更新
- [ ] 文档中的链接已更新
- [ ] 运行测试确保功能正常：
  ```bash
  python tools/validation/validate_dataset.py summary test_dataset/
  python tools/visualization/visualize_projection.py --dataset_root test_dataset/ --sequence 00
  ```

---

## 🔧 故障排除

### 问题1: 模块导入失败

**错误信息**:
```
ModuleNotFoundError: No module named 'validate_kitti_odometry'
```

**解决方法**:
```python
# 检查sys.path是否包含正确的目录
import sys
print(sys.path)

# 确保添加了正确的路径
sys.path.insert(0, 'tools/validation')
```

### 问题2: 文件未找到

**错误信息**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'tools/validate_dataset.py'
```

**解决方法**:
```bash
# 检查文件是否存在
ls -la tools/validation/validate_dataset.py

# 更新脚本中的路径
sed -i 's|tools/validate_dataset.py|tools/validation/validate_dataset.py|g' your_script.sh
```

### 问题3: 相对路径失效

**问题描述**: 脚本在不同目录下运行时找不到工具

**解决方法**: 使用绝对路径或基于脚本位置的相对路径
```python
from pathlib import Path

# 获取BEVCalib根目录
BEVCALIB_ROOT = Path(__file__).parent.parent  # 假设脚本在BEVCalib/scripts/
TOOLS_DIR = BEVCALIB_ROOT / 'tools' / 'validation'

# 使用绝对路径
cmd = ['python', str(TOOLS_DIR / 'validate_dataset.py'), ...]
```

---

## 💡 最佳实践建议

### 1. 逐步迁移

不要一次性修改所有代码，建议分阶段迁移：

1. **阶段1**: 创建软链接，保持兼容性
2. **阶段2**: 更新核心脚本和文档
3. **阶段3**: 更新边缘脚本和工具
4. **阶段4**: 移除软链接，完全迁移

### 2. 保持向后兼容

如果工具被外部项目使用，考虑：

```python
# 在工具脚本开头添加兼容性检查
import warnings
import sys
from pathlib import Path

# 检测是否从旧路径调用
script_path = Path(__file__)
if 'validation' not in script_path.parts:
    warnings.warn(
        "此工具已迁移到 tools/validation/ 目录。"
        "请更新您的脚本使用新路径。"
        "旧路径支持将在未来版本中移除。",
        DeprecationWarning
    )
```

### 3. 更新CI/CD

如果使用CI/CD，更新相关配置：

```yaml
# .github/workflows/validation.yml
- name: Validate Dataset
  run: python tools/validation/validate_dataset.py summary test_data/
```

---

## 📚 相关资源

- [主文档](../README.md) - 工具集完整文档
- [快速开始](QUICK_START.md) - 快速上手指南
- [架构说明](ARCHITECTURE.md) - 设计理念
- [验证模式](VALIDATION_MODES.md) - 验证工具详解

---

## 🤝 获取帮助

如果在迁移过程中遇到问题：

1. 查看相关目录的 README.md
2. 检查工具的 `--help` 输出
3. 查阅本迁移指南
4. 联系维护团队

---

**最后更新**: 2026-03-01  
**版本**: 1.0.0
