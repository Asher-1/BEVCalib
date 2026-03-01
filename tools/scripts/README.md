# Shell 脚本工具 (Shell Scripts)

批处理管理和监控脚本集。

---

## 📋 脚本列表

### `monitor_batch_processing.sh` - 监控批处理任务

实时监控正在运行的批处理任务状态。

**功能**:
- 显示任务进度
- 监控资源使用
- 实时日志输出
- 异常自动告警

**使用方法**:
```bash
# 基本监控
bash tools/scripts/monitor_batch_processing.sh

# 监控特定任务
bash tools/scripts/monitor_batch_processing.sh --job_id 12345

# 持续监控（每5秒刷新）
bash tools/scripts/monitor_batch_processing.sh --interval 5
```

**显示内容**:
```
╔══════════════════════════════════════════╗
║      批处理任务监控                      ║
╚══════════════════════════════════════════╝

任务ID: 12345
状态: 运行中 ✓
进度: 45/120 序列 (37.5%)

资源使用:
  CPU: 87%
  内存: 23.5 GB / 64 GB
  GPU: 2x V100 (92%, 88%)
  磁盘: 1.2 TB / 2.0 TB

当前处理:
  序列: 05
  帧: 1523/3831
  预计剩余时间: 2h 15m

最近日志:
  [11:23:45] 完成序列04验证
  [11:23:47] 开始序列05验证
  [11:23:50] Tr矩阵: OK
```

**监控选项**:
- `--interval N`: 刷新间隔（秒）
- `--log_tail N`: 显示最后N行日志
- `--alert`: 启用告警（任务失败时）
- `--export FILE`: 导出监控数据到文件

---

### `stop_batch_processing.sh` - 停止批处理任务

安全地停止正在运行的批处理任务。

**功能**:
- 优雅停止（完成当前任务）
- 强制停止（立即终止）
- 清理临时文件
- 保存中间结果

**使用方法**:
```bash
# 优雅停止（推荐）
bash tools/scripts/stop_batch_processing.sh --graceful

# 强制停止
bash tools/scripts/stop_batch_processing.sh --force

# 停止特定任务
bash tools/scripts/stop_batch_processing.sh --job_id 12345

# 停止所有批处理任务
bash tools/scripts/stop_batch_processing.sh --all
```

**停止流程**:
```
优雅停止模式:
1. 发送停止信号
2. 等待当前任务完成
3. 保存已完成部分结果
4. 清理临时文件
5. 生成中断报告

强制停止模式:
1. 立即终止所有进程
2. 尝试保存中间结果
3. 清理临时文件
4. 标记任务状态为"已中断"
```

**输出示例**:
```
停止批处理任务: 12345

[1/5] 发送停止信号...              ✓
[2/5] 等待当前任务完成...          ✓
      (序列05处理中，预计30秒)
[3/5] 保存中间结果...              ✓
      已完成: 序列00-04 (45帧)
[4/5] 清理临时文件...              ✓
[5/5] 生成中断报告...              ✓

任务已安全停止。
中断报告: batch_interrupt_report_12345.txt
```

---

## 🎯 使用场景

### 场景1: 启动并监控长时间批处理

```bash
# Terminal 1: 启动批处理
python tools/preparation/batch_prepare_trips.py \
    --source_dir trips/ \
    --output_dir dataset/ \
    --workers 8 \
    > batch.log 2>&1 &

# 保存任务ID
export BATCH_PID=$!

# Terminal 2: 监控进度
bash tools/scripts/monitor_batch_processing.sh \
    --job_id $BATCH_PID --interval 10
```

### 场景2: 任务异常时停止

```bash
# 发现问题，优雅停止
bash tools/scripts/stop_batch_processing.sh --graceful

# 查看已完成部分
ls -lh dataset/sequences/

# 检查中断报告
cat batch_interrupt_report_*.txt
```

### 场景3: 多任务管理

```bash
# 启动多个批处理任务
python tools/preparation/batch_prepare_trips.py \
    --source_dir trips_set1/ --output_dir dataset1/ &
PID1=$!

python tools/preparation/batch_prepare_trips.py \
    --source_dir trips_set2/ --output_dir dataset2/ &
PID2=$!

# 监控所有任务
bash tools/scripts/monitor_batch_processing.sh --all

# 停止特定任务
bash tools/scripts/stop_batch_processing.sh --job_id $PID1 --graceful
```

### 场景4: 自动化工作流

```bash
#!/bin/bash
# auto_process.sh

# 启动批处理
python tools/preparation/batch_prepare_trips.py \
    --source_dir trips/ \
    --output_dir dataset/ \
    --workers 8 &
BATCH_PID=$!

# 后台监控
bash tools/scripts/monitor_batch_processing.sh \
    --job_id $BATCH_PID \
    --interval 60 \
    --export monitoring_log.txt &

# 等待完成
wait $BATCH_PID

# 自动验证
python tools/validation/validate_dataset.py full dataset/ \
    --output-dir validation/ --full

echo "处理完成！"
```

---

## 📊 监控指标说明

### 系统资源

**CPU使用率**:
- 正常范围: 70-95%
- < 50%: 可能IO瓶颈
- > 98%: 考虑减少并行度

**内存使用**:
- 正常范围: 根据数据集大小
- 接近满载: 减少 `--workers`
- 内存泄漏: 检查脚本

**GPU使用**:
- 理想: > 90%
- < 50%: 数据加载瓶颈
- 波动大: 批大小调整

### 任务进度

**处理速度**:
- 帧/秒
- 序列/小时
- 预计完成时间

**错误率**:
- 正常: < 1%
- 1-5%: 检查数据质量
- > 5%: 严重问题，建议停止检查

---

## ⚠️ 注意事项

### 1. 停止时机选择

**优雅停止适用于**:
- 发现配置错误
- 需要调整参数
- 系统资源不足
- 有足够时间等待

**强制停止适用于**:
- 任务卡死无响应
- 紧急情况需要释放资源
- 优雅停止失败
- 进程异常

### 2. 中间结果处理

**优雅停止后**:
- 已完成的序列可以保留
- 进行中的序列可能不完整
- 检查并清理不完整数据

**强制停止后**:
- 全面检查数据完整性
- 重新验证已生成数据
- 可能需要重新处理部分序列

### 3. 资源清理

停止任务后检查：
```bash
# 检查孤立进程
ps aux | grep -E "python|batch"

# 检查临时文件
find /tmp -name "*batch*" -mtime -1

# 清理GPU内存
nvidia-smi

# 检查磁盘空间
df -h
```

---

## 💡 最佳实践

### 1. 使用tmux/screen进行持久监控

```bash
# 创建tmux会话
tmux new -s batch_monitor

# 在tmux中启动监控
bash tools/scripts/monitor_batch_processing.sh --interval 30

# 分离会话: Ctrl+B, D
# 重新连接: tmux attach -t batch_monitor
```

### 2. 日志记录

```bash
# 完整日志记录
python tools/preparation/batch_prepare_trips.py \
    --source_dir trips/ --output_dir dataset/ \
    2>&1 | tee -a batch_$(date +%Y%m%d_%H%M%S).log
```

### 3. 错误告警

```bash
# 添加告警脚本
bash tools/scripts/monitor_batch_processing.sh \
    --alert \
    --alert_email your@email.com \
    --alert_threshold 5  # 5%错误率触发
```

### 4. 定期检查点

```bash
# 配置自动检查点
python tools/preparation/batch_prepare_trips.py \
    --source_dir trips/ \
    --output_dir dataset/ \
    --checkpoint_interval 100  # 每100帧保存一次
```

---

## 🔧 故障排除

### 监控脚本无响应

```bash
# 检查脚本权限
chmod +x tools/scripts/monitor_batch_processing.sh

# 检查依赖
which watch
which ps
which awk
```

### 停止脚本失败

```bash
# 手动查找并终止进程
ps aux | grep batch_prepare

# 强制终止
kill -9 <PID>

# 清理僵尸进程
ps aux | grep defunct
```

### 资源监控不准确

```bash
# 安装htop（更精确的资源监控）
sudo apt-get install htop

# 使用htop手动监控
htop -p <PID>
```

---

## 🔗 相关文档

- [主文档](../README.md)
- [数据准备文档](../preparation/README.md)
- [验证工具文档](../validation/README.md)

---

## 📝 脚本开发指南

如需开发新的批处理脚本，建议：

1. **遵循命名规范**: `动词_操作对象.sh`
2. **添加帮助信息**: `--help` 选项
3. **错误处理**: 检查返回值，提供清晰错误信息
4. **日志输出**: 使用时间戳和日志级别
5. **资源清理**: 使用trap捕获退出信号

示例模板：
```bash
#!/bin/bash
set -euo pipefail  # 严格模式

# 帮助信息
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -h, --help     显示帮助"
    echo "  -i, --input    输入目录"
    exit 1
}

# 清理函数
cleanup() {
    echo "清理中..."
    # 清理代码
}
trap cleanup EXIT

# 主逻辑
main() {
    # 脚本内容
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        -i|--input) INPUT="$2"; shift 2 ;;
        *) echo "未知选项: $1"; usage ;;
    esac
done

main
```

---

**最后更新**: 2026-03-01
