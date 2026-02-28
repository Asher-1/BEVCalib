#!/bin/bash
# Trip 数据下载工具使用示例

echo "========================================"
echo "Trip 数据下载工具 - 使用示例"
echo "========================================"
echo ""

# 切换到脚本目录
cd "$(dirname "$0")"

echo "当前工作目录: $(pwd)"
echo ""

# 示例 1: 查看帮助
echo "示例 1: 查看帮助信息"
echo "命令: python3 download_trips.py --help"
echo "----------------------------------------"
# python3 download_trips.py --help
echo ""

# 示例 2: 检查 trip 是否存在
echo "示例 2: 检查 trip 是否存在（不下载）"
echo "命令: python3 download_trips.py --check-only YR-P789-19_20260213_012754"
echo "----------------------------------------"
# python3 download_trips.py --check-only YR-P789-19_20260213_012754
echo ""

# 示例 3: 下载单个 trip（选择性下载）
echo "示例 3: 下载单个 trip（只下载 bags/important 和 configs）"
echo "命令: python3 download_trips.py YR-P789-19_20260213_012754"
echo "----------------------------------------"
# 取消注释以执行下载
# python3 download_trips.py YR-P789-19_20260213_012754
echo "（已注释，取消注释以执行）"
echo ""

# 示例 4: 下载完整 trip
echo "示例 4: 下载完整 trip（所有数据）"
echo "命令: python3 download_trips.py --full YR-P789-19_20260213_012754"
echo "----------------------------------------"
# 取消注释以执行下载
# python3 download_trips.py --full YR-P789-19_20260213_012754
echo "（已注释，取消注释以执行）"
echo ""

# 示例 5: 从文件批量下载
echo "示例 5: 从文件批量下载"
echo "----------------------------------------"
echo "步骤 1: 创建 trip 列表文件"
cat > trips_example.txt << 'EOF'
# 示例 trip 列表
YR-P789-19_20260213_012754
YR-B26A1-1_20251117_031232
EOF
echo "已创建: trips_example.txt"
echo ""
echo "步骤 2: 执行下载"
echo "命令: python3 download_trips.py --file trips_example.txt"
# 取消注释以执行下载
# python3 download_trips.py --file trips_example.txt
echo "（已注释，取消注释以执行）"
echo ""

# 示例 6: 下载多个 trips
echo "示例 6: 下载多个 trips"
echo "命令: python3 download_trips.py YR-P789-19_20260213_012754 YR-B26A1-1_20251117_031232"
echo "----------------------------------------"
# 取消注释以执行下载
# python3 download_trips.py YR-P789-19_20260213_012754 YR-B26A1-1_20251117_031232
echo "（已注释，取消注释以执行）"
echo ""

# 示例 7: 指定输出目录
echo "示例 7: 指定自定义输出目录"
echo "命令: python3 download_trips.py -o /tmp/my_trips YR-P789-19_20260213_012754"
echo "----------------------------------------"
# 取消注释以执行下载
# python3 download_trips.py -o /tmp/my_trips YR-P789-19_20260213_012754
echo "（已注释，取消注释以执行）"
echo ""

echo "========================================"
echo "提示："
echo "1. 取消相应命令的注释（删除开头的 #）即可执行"
echo "2. 默认下载到: ./trips/ 目录"
echo "3. 日志文件: ./download_trips_YYYYMMDD_HHMMSS.log"
echo "4. 更多帮助: python3 download_trips.py --help"
echo "========================================"
