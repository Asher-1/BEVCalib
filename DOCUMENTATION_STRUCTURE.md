# BEVCalib 文档结构说明

## 📚 文档分类

### 1. 快速开始类
- **README_TRAINING_SCRIPTS.md** ⭐ 推荐首读
  - 简洁的脚本使用说明
  - 3 分钟上手
  - 包含所有脚本的基本用法

- **QUICK_START_TRAINING.md** 
  - 详细的快速开始指南
  - 包含前提条件、监控、故障排查
  - 10 分钟完整流程

### 2. 详细参考类
- **TRAINING_GUIDE.md** ⭐ 参数调优必读
  - 详细的训练参数说明
  - 不同场景配置建议
  - 参数调优指南
  - 需要更新：添加新脚本说明

- **TRAINING_REFACTOR_SUMMARY.md**
  - 脚本重构的完整总结
  - 技术细节和改进对比
  - 适合了解重构背景

### 3. 快速参考类
- **SCRIPT_CHANGES_SUMMARY.txt**
  - 文本格式的快速参考
  - 命令速查
  - 适合打印或快速查找

### 4. 日志说明类
- **logs/README.md**
  - 日志目录结构说明
  - TensorBoard 使用

## 🎯 推荐阅读顺序

### 新手入门
1. README_TRAINING_SCRIPTS.md (3 分钟)
2. QUICK_START_TRAINING.md (10 分钟)
3. 实际操作：bash start_training.sh all v1

### 参数调优
1. TRAINING_GUIDE.md
2. 根据训练效果调整参数
3. 参考 TRAINING_REFACTOR_SUMMARY.md 了解日志结构

### 快速查找
- SCRIPT_CHANGES_SUMMARY.txt

## 📋 文档状态

| 文档 | 状态 | 说明 |
|------|------|------|
| README_TRAINING_SCRIPTS.md | ✅ 最新 | 2026-03-01 创建 |
| QUICK_START_TRAINING.md | ✅ 最新 | 2026-03-01 创建 |
| TRAINING_REFACTOR_SUMMARY.md | ✅ 最新 | 2026-03-01 创建 |
| SCRIPT_CHANGES_SUMMARY.txt | ✅ 最新 | 2026-03-01 创建 |
| logs/README.md | ✅ 最新 | 2026-03-01 创建 |
| TRAINING_GUIDE.md | ⚠️ 需更新 | 2026-01-30 创建，需添加新脚本说明 |
| stop_training.sh | ⚠️ 需更新 | 需支持 train_universal.sh |

## 🔄 需要的更新

### 1. stop_training.sh
- 添加对 train_universal.sh 的支持
- 更新进程查找逻辑

### 2. TRAINING_GUIDE.md
- 在开头添加新脚本的说明和链接
- 保持详细的参数调优内容

## 📌 文档定位

- **README_TRAINING_SCRIPTS.md**: 入口文档，最简洁
- **QUICK_START_TRAINING.md**: 完整的新手指南
- **TRAINING_GUIDE.md**: 参数调优圣经
- **TRAINING_REFACTOR_SUMMARY.md**: 技术文档，了解重构
- **SCRIPT_CHANGES_SUMMARY.txt**: 命令速查表

