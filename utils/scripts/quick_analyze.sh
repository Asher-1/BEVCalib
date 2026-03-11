#!/bin/bash
# 快速分析脚本 - 提供常用分析场景的快捷命令

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."  # Go to BEVCalib root

show_help() {
    cat << EOF
BEVCalib 快速分析工具

用法:
    bash quick_analyze.sh <scenario> [options]

场景选项:
    5deg        - 分析5度扰动实验 (z=1, z=5, z=10)
    10deg       - 分析10度rotation实验 (z=1, z=5, z=10)
    custom      - 使用自定义配置文件

选项:
    --skip-test    - 跳过测试评估，读取已有结果
    --only-train   - 只分析训练数据
    --config FILE  - 指定配置文件（custom场景必需）

示例:
    # 分析5deg实验（读取已有测试结果）
    bash quick_analyze.sh 5deg --skip-test
    
    # 分析10deg实验（只看训练数据）
    bash quick_analyze.sh 10deg --only-train
    
    # 使用自定义配置
    bash quick_analyze.sh custom --config my_config.yaml

EOF
}

if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

SCENARIO=$1
shift

case $SCENARIO in
    5deg)
        echo "分析5deg扰动实验..."
        python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config.yaml "$@"
        ;;
    
    10deg|rotation)
        echo "分析10deg rotation实验..."
        python utils/scripts/analyze_experiments.py --config utils/configs/experiment_config_rotation.yaml "$@"
        ;;
    
    custom)
        echo "使用自定义配置..."
        python utils/scripts/analyze_experiments.py "$@"
        ;;
    
    -h|--help|help)
        show_help
        ;;
    
    *)
        echo "✗ 未知场景: $SCENARIO"
        echo ""
        show_help
        exit 1
        ;;
esac
