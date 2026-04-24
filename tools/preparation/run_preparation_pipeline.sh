#!/bin/bash
# ==============================================================================
# 完整数据准备流水线 - Shell 版本
# ==============================================================================
# 
# 功能：
#   步骤1: 运行 batch_prepare_trips.py 生成原始数据（PNG图像 + 点云）
#   步骤2: 运行 resize_images.py 生成训练用的 resize 图像（JPEG）
# 
# 用法示例：
#   # 批量模式：处理 trips_dir 下所有 trip 子目录
#   ./run_preparation_pipeline.sh /path/to/trips_dir /data/output 640 360
#   ./run_preparation_pipeline.sh /path/to/trips_dir /data/output 640 360 traffic_2 10.0 -j 4
#
#   # 单 trip 模式：直接指定一个 trip 行程目录（含 bags/ 和 configs/）
#   ./run_preparation_pipeline.sh /path/to/trips_dir/YR-C061-9_xxx /data/output 640 360
# 
# 参数：
#   $1: trips 根目录（包含多个 trip 子目录）或单个 trip 目录（含 bags/ 和 configs/）
#   $2: 输出目录
#   $3: resize 宽度（可选，默认640）
#   $4: resize 高度（可选，默认360）
#   $5: 相机名称（可选，默认traffic_2）
#   $6: 目标帧率（可选，默认10.0）
#   --force-config: 强制使用lidars.cfg外参（可出现在任意位置）
#   -j N: 并行处理trip数量（可出现在任意位置，默认1=串行）
#   --sequence-id N: 单trip模式的sequence编号（可选，默认0）
#   --start-sequence N: 批量模式的起始sequence编号（可选，默认0）
#   --resize-only: 跳过数据准备，只运行 resize（此时 $1 为数据集根目录）
#   --skip-resize: 跳过 resize 步骤，只运行数据准备
#   --dry-run: 预览模式，只打印将执行的命令，不实际运行
# 
# 作者: AI Assistant
# 日期: 2026-02-28
# ==============================================================================

set -e  # 遇到错误立即退出

# ========== 解析标志参数 ==========
FORCE_CONFIG=""
PARALLEL_JOBS=""
SEQUENCE_ID="0"
START_SEQUENCE=""
RESIZE_ONLY=false
SKIP_RESIZE=false
DRY_RUN=false
POSITIONAL_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --force-config)
            FORCE_CONFIG="--force-config"
            shift
            ;;
        -j)
            PARALLEL_JOBS="-j $2"
            shift 2
            ;;
        --sequence-id)
            SEQUENCE_ID="$2"
            shift 2
            ;;
        --start-sequence)
            START_SEQUENCE="$2"
            shift 2
            ;;
        --resize-only)
            RESIZE_ONLY=true
            shift
            ;;
        --skip-resize)
            SKIP_RESIZE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# ========== 互斥标志检查 ==========
if [ "$RESIZE_ONLY" = true ] && [ "$SKIP_RESIZE" = true ]; then
    echo -e "${RED}错误: --resize-only 和 --skip-resize 不能同时使用${NC}"
    exit 1
fi

# ========== 参数解析 ==========
TRIPS_DIR="${POSITIONAL_ARGS[0]}"
OUTPUT_DIR="${POSITIONAL_ARGS[1]}"
RESIZE_WIDTH="${POSITIONAL_ARGS[2]:-640}"      # 默认640
RESIZE_HEIGHT="${POSITIONAL_ARGS[3]:-360}"     # 默认360
CAMERA_NAME="${POSITIONAL_ARGS[4]:-traffic_2}" # 默认traffic_2
TARGET_FPS="${POSITIONAL_ARGS[5]:-10.0}"       # 默认10.0

# ========== 脚本目录 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_PREPARE_SCRIPT="${SCRIPT_DIR}/batch_prepare_trips.py"
SINGLE_PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_custom_dataset.py"
RESIZE_SCRIPT="${SCRIPT_DIR}/resize_images.py"

# ========== 颜色定义 ==========
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========== 参数验证 ==========
if [ "$RESIZE_ONLY" = true ]; then
    if [ -z "${POSITIONAL_ARGS[0]}" ]; then
        echo -e "${RED}错误: --resize-only 模式需要指定数据集根目录${NC}"
        echo "用法: $0 --resize-only <dataset_root> [width] [height]"
        exit 1
    fi
    OUTPUT_DIR="${POSITIONAL_ARGS[0]}"
    RESIZE_WIDTH="${POSITIONAL_ARGS[1]:-640}"
    RESIZE_HEIGHT="${POSITIONAL_ARGS[2]:-360}"
elif [ -z "${POSITIONAL_ARGS[0]}" ] || [ -z "${POSITIONAL_ARGS[1]}" ]; then
    echo -e "${RED}错误: 参数不足${NC}"
    echo ""
    echo "用法: $0 <input_dir> <output_dir> [width] [height] [camera_name] [fps] [--force-config]"
    echo ""
    echo "示例:"
    echo "  # 批量模式：处理 trips_dir 下所有 trip"
    echo "  $0 /path/to/trips /data/output 640 360"
    echo "  $0 /path/to/trips /data/output 640 360 traffic_2 10.0 --force-config -j 2"
    echo ""
    echo "  # 单 trip 模式：直接指定一个 trip 目录"
    echo "  $0 /path/to/trips/YR-C061-9_xxx /data/output 640 360"
    echo ""
    echo "  # 仅 resize 模式：数据已准备好，只做图像缩放"
    echo "  $0 --resize-only /data/output 640 360"
    echo ""
    echo "  # 跳过 resize：只做数据准备"
    echo "  $0 --skip-resize /path/to/trips /data/output"
    echo ""
    echo "参数:"
    echo "  input_dir        - trips 根目录或单个 trip 目录（含 bags/ 和 configs/）"
    echo "  output_dir       - 输出目录"
    echo "  width            - resize 宽度（默认: 640）"
    echo "  height           - resize 高度（默认: 360）"
    echo "  camera_name      - 相机名称（默认: traffic_2）"
    echo "  fps              - 目标帧率（默认: 10.0）"
    echo "  --force-config   - 强制使用lidars.cfg外参替代bag外参"
    echo "  -j N             - 并行处理trip数量（默认: 1=串行）"
    echo "  --sequence-id N  - 单trip模式的sequence编号（默认: 0）"
    echo "  --resize-only    - 跳过数据准备，只运行 resize"
    echo "  --skip-resize    - 跳过 resize，只运行数据准备"
    echo "  --dry-run        - 预览模式，只打印将执行的命令，不实际运行"
    exit 1
fi

if [ "$RESIZE_ONLY" = false ]; then
    if [ ! -d "$TRIPS_DIR" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo -e "${YELLOW}⚠️  [DRY-RUN] 输入目录不存在: ${TRIPS_DIR}（dry-run 模式继续）${NC}"
        else
            echo -e "${RED}错误: 输入目录不存在: ${TRIPS_DIR}${NC}"
            exit 1
        fi
    fi
fi

# ========== 检测输入模式：单 trip 还是 trips 根目录 ==========
SINGLE_TRIP_MODE=false
if [ "$RESIZE_ONLY" = false ] && [ -d "${TRIPS_DIR}/configs" ] && [ -d "${TRIPS_DIR}/bags" ]; then
    SINGLE_TRIP_MODE=true
fi

if [ "$RESIZE_ONLY" = false ]; then
    if [ "$SINGLE_TRIP_MODE" = true ]; then
        if [ ! -f "$SINGLE_PREPARE_SCRIPT" ]; then
            if [ "$DRY_RUN" = true ]; then
                echo -e "${YELLOW}⚠️  [DRY-RUN] 未找到数据准备脚本: ${SINGLE_PREPARE_SCRIPT}${NC}"
            else
                echo -e "${RED}错误: 未找到数据准备脚本: ${SINGLE_PREPARE_SCRIPT}${NC}"
                exit 1
            fi
        fi
    else
        if [ ! -f "$BATCH_PREPARE_SCRIPT" ]; then
            if [ "$DRY_RUN" = true ]; then
                echo -e "${YELLOW}⚠️  [DRY-RUN] 未找到批量准备脚本: ${BATCH_PREPARE_SCRIPT}${NC}"
            else
                echo -e "${RED}错误: 未找到批量准备脚本: ${BATCH_PREPARE_SCRIPT}${NC}"
                exit 1
            fi
        fi
    fi
fi

if [ "$SKIP_RESIZE" = false ] && [ ! -f "$RESIZE_SCRIPT" ]; then
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}⚠️  [DRY-RUN] 未找到 resize 脚本: ${RESIZE_SCRIPT}${NC}"
    else
        echo -e "${RED}错误: 未找到 resize 脚本: ${RESIZE_SCRIPT}${NC}"
        exit 1
    fi
fi

# ========== 单 trip 模式：定位 bag 目录 ==========
if [ "$SINGLE_TRIP_MODE" = true ]; then
    TRIP_CONFIG_DIR="${TRIPS_DIR}/configs"
    # 按优先级查找 bag 目录
    if [ -d "${TRIPS_DIR}/bags/important" ]; then
        TRIP_BAG_DIR="${TRIPS_DIR}/bags/important"
    elif [ -d "${TRIPS_DIR}/bags/unimportant" ]; then
        TRIP_BAG_DIR="${TRIPS_DIR}/bags/unimportant"
    else
        TRIP_BAG_DIR="${TRIPS_DIR}/bags"
    fi
fi

# ========== 打印配置 ==========
echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         完整数据准备流水线                                   ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${BLUE}配置信息:${NC}"
if [ "$RESIZE_ONLY" = true ]; then
    echo -e "  运行模式:      ${YELLOW}仅 Resize${NC}"
    echo "  数据集目录:    ${OUTPUT_DIR}"
    echo "  Resize 尺寸:   ${RESIZE_WIDTH}×${RESIZE_HEIGHT}"
elif [ "$SKIP_RESIZE" = true ]; then
    echo -e "  运行模式:      ${YELLOW}仅数据准备（跳过 Resize）${NC}"
    if [ "$SINGLE_TRIP_MODE" = true ]; then
        echo -e "  处理模式:      ${YELLOW}单 trip${NC}"
        echo "  Trip 目录:     ${TRIPS_DIR}"
        echo "  Bag 目录:      ${TRIP_BAG_DIR}"
        echo "  Config 目录:   ${TRIP_CONFIG_DIR}"
        echo "  Sequence ID:   ${SEQUENCE_ID}"
    else
        echo "  处理模式:      批量"
        echo "  Trips 目录:    ${TRIPS_DIR}"
        if [ -n "$START_SEQUENCE" ]; then
            echo -e "  起始Sequence:  ${YELLOW}${START_SEQUENCE}${NC}"
        fi
    fi
    echo "  输出目录:      ${OUTPUT_DIR}"
    echo "  相机名称:      ${CAMERA_NAME}"
    echo "  目标帧率:      ${TARGET_FPS}"
else
    if [ "$SINGLE_TRIP_MODE" = true ]; then
        echo -e "  处理模式:      ${YELLOW}单 trip${NC}"
        echo "  Trip 目录:     ${TRIPS_DIR}"
        echo "  Bag 目录:      ${TRIP_BAG_DIR}"
        echo "  Config 目录:   ${TRIP_CONFIG_DIR}"
        echo "  Sequence ID:   ${SEQUENCE_ID}"
    else
        echo "  处理模式:      批量"
        echo "  Trips 目录:    ${TRIPS_DIR}"
        if [ -n "$START_SEQUENCE" ]; then
            echo -e "  起始Sequence:  ${YELLOW}${START_SEQUENCE}${NC}"
        fi
    fi
    echo "  输出目录:      ${OUTPUT_DIR}"
    echo "  相机名称:      ${CAMERA_NAME}"
    echo "  目标帧率:      ${TARGET_FPS}"
    echo "  Resize 尺寸:   ${RESIZE_WIDTH}×${RESIZE_HEIGHT}"
fi
if [ -n "$PARALLEL_JOBS" ] && [ "$SINGLE_TRIP_MODE" = false ]; then
    echo -e "  并行处理:      ${YELLOW}${PARALLEL_JOBS}${NC}"
fi
if [ -n "$FORCE_CONFIG" ]; then
    echo -e "  ${YELLOW}⚠️  强制使用lidars.cfg外参（忽略bag中的lidar外参）${NC}"
fi
if [ "$DRY_RUN" = true ]; then
    echo -e "  ${YELLOW}🔍 DRY-RUN 模式：只打印命令，不实际执行${NC}"
fi
echo "  开始时间:      $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

START_TIME=$(date +%s)
PREPARE_TIME=0
RESIZE_TIME=0

# ========== 步骤 1: 数据准备 ==========
if [ "$RESIZE_ONLY" = true ]; then
    echo -e "${YELLOW}跳过步骤 1: --resize-only 模式，直接进入 resize${NC}"
    echo ""
else
    STEP_LABEL="1/2"
    if [ "$SKIP_RESIZE" = true ]; then
        STEP_LABEL="1/1"
    fi
    echo "═══════════════════════════════════════════════════════════════════════════════"
    if [ "$SINGLE_TRIP_MODE" = true ]; then
        echo -e "${GREEN}步骤 ${STEP_LABEL}: 单 trip 数据准备（生成原始图像和点云）${NC}"
    else
        echo -e "${GREEN}步骤 ${STEP_LABEL}: 批量数据准备（生成原始图像和点云）${NC}"
    fi
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo ""

    PREPARE_START=$(date +%s)

    if [ "$SINGLE_TRIP_MODE" = true ]; then
        FORCE_CONFIG_SINGLE=""
        if [ -n "$FORCE_CONFIG" ]; then
            FORCE_CONFIG_SINGLE="--force-config"
        fi

        if [ "$DRY_RUN" = true ]; then
            echo -e "${YELLOW}[DRY-RUN] 将执行:${NC}"
            echo "  python3 ${SINGLE_PREPARE_SCRIPT}" \
                "--bag_dir ${TRIP_BAG_DIR}" \
                "--config_dir ${TRIP_CONFIG_DIR}" \
                "--output_dir ${OUTPUT_DIR}" \
                "--sequence_id ${SEQUENCE_ID}" \
                "--camera_name ${CAMERA_NAME}" \
                "--target_fps ${TARGET_FPS}" \
                ${FORCE_CONFIG_SINGLE}
            echo ""
        else
            python3 "${SINGLE_PREPARE_SCRIPT}" \
                --bag_dir "${TRIP_BAG_DIR}" \
                --config_dir "${TRIP_CONFIG_DIR}" \
                --output_dir "${OUTPUT_DIR}" \
                --sequence_id "${SEQUENCE_ID}" \
                --camera_name "${CAMERA_NAME}" \
                --target_fps "${TARGET_FPS}" \
                ${FORCE_CONFIG_SINGLE}
        fi
    else
        START_SEQ_ARG=""
        if [ -n "$START_SEQUENCE" ]; then
            START_SEQ_ARG="--start_sequence ${START_SEQUENCE}"
        fi

        if [ "$DRY_RUN" = true ]; then
            echo -e "${YELLOW}[DRY-RUN] 将执行:${NC}"
            echo "  python3 ${BATCH_PREPARE_SCRIPT}" \
                "--trips_dir ${TRIPS_DIR}" \
                "--output_dir ${OUTPUT_DIR}" \
                "--camera_name ${CAMERA_NAME}" \
                "--target_fps ${TARGET_FPS}" \
                ${FORCE_CONFIG} \
                ${PARALLEL_JOBS} \
                ${START_SEQ_ARG}
            echo ""
        else
            python3 "${BATCH_PREPARE_SCRIPT}" \
                --trips_dir "${TRIPS_DIR}" \
                --output_dir "${OUTPUT_DIR}" \
                --camera_name "${CAMERA_NAME}" \
                --target_fps "${TARGET_FPS}" \
                ${FORCE_CONFIG} \
                ${PARALLEL_JOBS} \
                ${START_SEQ_ARG}
        fi
    fi

    if [ "$DRY_RUN" = false ]; then
        if [ $? -ne 0 ]; then
            echo -e "\n${RED}❌ 数据准备失败${NC}"
            exit 1
        fi

        PREPARE_TIME=$(($(date +%s) - PREPARE_START))
        echo -e "\n${GREEN}✅ 数据准备完成${NC} (耗时: ${PREPARE_TIME}秒)"
    fi
fi

# ========== 步骤 2: 图像 Resize ==========
if [ "$SKIP_RESIZE" = true ]; then
    echo ""
    echo -e "${YELLOW}跳过步骤 2: --skip-resize 模式${NC}"
else
    STEP_LABEL="2/2"
    if [ "$RESIZE_ONLY" = true ]; then
        STEP_LABEL="1/1"
    fi
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo -e "${GREEN}步骤 ${STEP_LABEL}: 图像 Resize（生成训练用的 resize 图像）${NC}"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN] 将执行:${NC}"
        echo "  python3 ${RESIZE_SCRIPT}" \
            "--dataset_root ${OUTPUT_DIR}" \
            "--width ${RESIZE_WIDTH}" \
            "--height ${RESIZE_HEIGHT}" \
            "--workers 32" \
            "--quality 95"
        echo ""
        echo -e "${YELLOW}[DRY-RUN] 将在 ${OUTPUT_DIR}/sequences/*/image_2/ 中搜索 PNG 图像${NC}"
    else
        SEQ_DIR="${OUTPUT_DIR}/sequences"
        if [ ! -d "$SEQ_DIR" ]; then
            echo -e "${RED}❌ 未找到 sequences 目录: ${SEQ_DIR}${NC}"
            exit 1
        fi

        IMAGE_COUNT=$(find "$SEQ_DIR" -name "*.png" -path "*/image_2/*" | wc -l)
        echo "找到 ${IMAGE_COUNT} 张图像需要 resize"

        if [ "$IMAGE_COUNT" -eq 0 ]; then
            echo -e "${YELLOW}⚠️  未找到需要 resize 的图像，跳过 resize 步骤${NC}"
        else
            RESIZE_START=$(date +%s)

            python3 "${RESIZE_SCRIPT}" \
                --dataset_root "${OUTPUT_DIR}" \
                --width "${RESIZE_WIDTH}" \
                --height "${RESIZE_HEIGHT}" \
                --workers 32 \
                --quality 95

            if [ $? -ne 0 ]; then
                echo -e "\n${RED}❌ 图像 resize 失败${NC}"
                exit 1
            fi

            RESIZE_TIME=$(($(date +%s) - RESIZE_START))
            echo -e "\n${GREEN}✅ 图像 resize 完成${NC} (耗时: ${RESIZE_TIME}秒)"
        fi
    fi
fi

# ========== 完成总结 ==========
TOTAL_TIME=$(($(date +%s) - START_TIME))

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}🔍 DRY-RUN 预览完成（未执行任何操作）${NC}"
elif [ "$RESIZE_ONLY" = true ]; then
    echo -e "${GREEN}🎉 Resize 完成！${NC}"
elif [ "$SKIP_RESIZE" = true ]; then
    echo -e "${GREEN}🎉 数据准备完成！${NC}"
else
    echo -e "${GREEN}🎉 完整数据准备流程完成！${NC}"
fi
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
if [ "$DRY_RUN" = false ]; then
    echo -e "${BLUE}时间统计:${NC}"
    if [ "$RESIZE_ONLY" = false ] && [ "$PREPARE_TIME" -gt 0 ]; then
        echo "  数据准备: ${PREPARE_TIME}秒"
    fi
    if [ "$SKIP_RESIZE" = false ] && [ "$RESIZE_TIME" -gt 0 ]; then
        echo "  图像 Resize: ${RESIZE_TIME}秒"
    fi
    echo "  总耗时: ${TOTAL_TIME}秒"
    echo ""
fi
echo -e "${BLUE}输出位置:${NC}"
echo "  数据集根目录: ${OUTPUT_DIR}"
if [ "$RESIZE_ONLY" = false ]; then
    echo "  原始图像: sequences/*/image_2/ (PNG格式)"
    echo "  点云数据: sequences/*/velodyne/"
fi
if [ "$SKIP_RESIZE" = false ]; then
    echo "  Resize图像: sequences/*/image_2_${RESIZE_WIDTH}x${RESIZE_HEIGHT}/ (JPEG格式)"
fi
echo ""
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VALIDATE_SCRIPT="${SCRIPT_DIR}/../validation/validate_dataset.py"

echo -e "${BLUE}下一步:${NC}"
_NEXT_STEP=1

if [ "$SKIP_RESIZE" = true ]; then
    echo "  ${_NEXT_STEP}. Resize 图像："
    echo -e "  ${YELLOW}bash $(basename "$0") --resize-only ${OUTPUT_DIR} ${RESIZE_WIDTH} ${RESIZE_HEIGHT}${NC}"
    echo ""
    _NEXT_STEP=$(( _NEXT_STEP + 1 ))
fi

echo "  ${_NEXT_STEP}. 验证数据集完整性："
VALIDATE_OUTPUT="${OUTPUT_DIR}/validation_results"
echo ""
echo -e "  快速验证 (~17秒):"
echo -e "  ${YELLOW}python ${VALIDATE_SCRIPT} quick ${OUTPUT_DIR} \\${NC}"
echo -e "  ${YELLOW}    --output-dir ${VALIDATE_OUTPUT}${NC}"
echo ""
echo -e "  完整验证 (~15分钟):"
echo -e "  ${YELLOW}python ${VALIDATE_SCRIPT} full ${OUTPUT_DIR} \\${NC}"
echo -e "  ${YELLOW}    --output-dir ${VALIDATE_OUTPUT}${NC}"
echo ""
_NEXT_STEP=$(( _NEXT_STEP + 1 ))

echo "  ${_NEXT_STEP}. 验证通过后启动训练："
echo ""
echo -e "  ${YELLOW}python train.py \\${NC}"
echo -e "  ${YELLOW}    --data_root ${OUTPUT_DIR} \\${NC}"
echo -e "  ${YELLOW}    --img_H ${RESIZE_HEIGHT} \\${NC}"
echo -e "  ${YELLOW}    --img_W ${RESIZE_WIDTH}${NC}"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
