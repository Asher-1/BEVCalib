#!/bin/bash
# ==============================================================================
# 仅运行图像 Resize - 快捷脚本
# ==============================================================================
# 
# 用途: 当数据已经通过 prepare_custom_dataset.py 生成后，只需要 resize 图像
# 
# 用法:
#   ./run_resize_only.sh <dataset_root> [width] [height] [workers]
# 
# 示例:
#   ./run_resize_only.sh /data/bevcalib_training_data
#   ./run_resize_only.sh /data/bevcalib_training_data 640 360
#   ./run_resize_only.sh /data/bevcalib_training_data 640 360 64
# 
# ==============================================================================

set -e

# ========== 参数解析 ==========
DATASET_ROOT="$1"
RESIZE_WIDTH="${2:-640}"
RESIZE_HEIGHT="${3:-360}"
WORKERS="${4:-32}"
QUALITY="${5:-95}"

# ========== 颜色定义 ==========
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ========== 参数验证 ==========
if [ -z "$DATASET_ROOT" ]; then
    echo -e "${RED}错误: 必须指定数据集根目录${NC}"
    echo ""
    echo "用法: $0 <dataset_root> [width] [height] [workers] [quality]"
    echo ""
    echo "示例:"
    echo "  $0 /data/bevcalib_training_data"
    echo "  $0 /data/bevcalib_training_data 640 360"
    echo "  $0 /data/bevcalib_training_data 640 360 64 95"
    echo ""
    echo "参数:"
    echo "  dataset_root  - 数据集根目录（必需）"
    echo "  width         - 目标宽度（默认: 640）"
    echo "  height        - 目标高度（默认: 360）"
    echo "  workers       - 并行工作进程（默认: 32）"
    echo "  quality       - JPEG质量 0-100（默认: 95）"
    exit 1
fi

if [ ! -d "$DATASET_ROOT" ]; then
    echo -e "${RED}错误: 数据集目录不存在: ${DATASET_ROOT}${NC}"
    exit 1
fi

SEQ_DIR="${DATASET_ROOT}/sequences"
if [ ! -d "$SEQ_DIR" ]; then
    echo -e "${RED}错误: 未找到 sequences 目录: ${SEQ_DIR}${NC}"
    echo "请先运行 prepare_custom_dataset.py 或 batch_prepare_trips.py"
    exit 1
fi

# ========== 脚本路径 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESIZE_SCRIPT="${SCRIPT_DIR}/resize_images.py"

if [ ! -f "$RESIZE_SCRIPT" ]; then
    echo -e "${RED}错误: 未找到 resize 脚本: ${RESIZE_SCRIPT}${NC}"
    exit 1
fi

# ========== 打印配置 ==========
echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                           图像 Resize 预处理                                  ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${BLUE}配置信息:${NC}"
echo "  数据集目录:    ${DATASET_ROOT}"
echo "  目标尺寸:      ${RESIZE_WIDTH}×${RESIZE_HEIGHT}"
echo "  并行工作进程:  ${WORKERS}"
echo "  JPEG质量:      ${QUALITY}"
echo "  开始时间:      $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ========== 统计图像数量 ==========
IMAGE_COUNT=$(find "$SEQ_DIR" -name "*.png" -path "*/image_2/*" 2>/dev/null | wc -l)
echo -e "${BLUE}扫描结果:${NC}"
echo "  找到 ${IMAGE_COUNT} 张图像需要 resize"
echo ""

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  未找到需要 resize 的图像${NC}"
    echo "  请确认 sequences/*/image_2/ 目录中有 PNG 图像"
    exit 1
fi

# ========== 运行 Resize ==========
echo "═══════════════════════════════════════════════════════════════════════════════"
echo -e "${GREEN}开始 Resize 处理${NC}"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

START_TIME=$(date +%s)

python3 "${RESIZE_SCRIPT}" \
    --dataset_root "${DATASET_ROOT}" \
    --width "${RESIZE_WIDTH}" \
    --height "${RESIZE_HEIGHT}" \
    --workers "${WORKERS}" \
    --quality "${QUALITY}"

EXIT_CODE=$?

ELAPSED=$(($(date +%s) - START_TIME))

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ 图像 Resize 完成！${NC}"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo -e "${BLUE}时间统计:${NC}"
    echo "  总耗时: ${ELAPSED}秒"
    if [ "$IMAGE_COUNT" -gt 0 ]; then
        RATE=$((IMAGE_COUNT / (ELAPSED + 1)))
        echo "  处理速度: ${RATE} 张/秒"
    fi
    echo ""
    echo -e "${BLUE}输出位置:${NC}"
    echo "  数据集根目录: ${DATASET_ROOT}"
    echo "  原始图像: sequences/*/image_2/"
    echo "  Resize图像: sequences/*/image_2_${RESIZE_WIDTH}x${RESIZE_HEIGHT}/"
    echo ""
    echo -e "${BLUE}下一步:${NC}"
    echo "  使用以下参数启动训练："
    echo ""
    echo -e "  ${YELLOW}python train.py \\${NC}"
    echo -e "  ${YELLOW}    --data_root ${DATASET_ROOT} \\${NC}"
    echo -e "  ${YELLOW}    --img_H ${RESIZE_HEIGHT} \\${NC}"
    echo -e "  ${YELLOW}    --img_W ${RESIZE_WIDTH}${NC}"
    echo ""
else
    echo -e "${RED}❌ 图像 Resize 失败${NC}"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    exit $EXIT_CODE
fi
