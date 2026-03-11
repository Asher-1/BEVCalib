#!/bin/bash
# 批量评估三个模型在测试数据上的泛化能力

set -e

# 激活conda环境
source /opt/conda/etc/profile.d/conda.sh
conda activate bevcalib

BASE_DIR="/mnt/drtraining/user/dahailu/code/BEVCalib"
TEST_DATA="/mnt/drtraining/user/dahailu/data/bevcalib/test_data"
LOGS_DIR="$BASE_DIR/logs/B26A"

# 模型配置：模型名称, BEV_ZBOUND_STEP, checkpoint路径
declare -A MODELS=(
    ["z1"]="20.0:model_small_5deg_v4-z1"
    ["z5"]="4.0:model_small_5deg_v4-z5"
    ["z10"]="2.0:model_small_5deg_v4-z10"
)

echo "================================================================================"
echo "Evaluating BEVCalib Models on Test Data"
echo "================================================================================"
echo ""

for model_key in z1 z5 z10; do
    IFS=':' read -r zbound_step model_dir <<< "${MODELS[$model_key]}"
    
    CKPT_PATH="$LOGS_DIR/$model_dir/B26A_scratch/checkpoint/ckpt_400.pth"
    OUTPUT_DIR="$LOGS_DIR/$model_dir/test_data_eval"
    
    if [ ! -f "$CKPT_PATH" ]; then
        echo "⚠️  Checkpoint not found: $CKPT_PATH"
        echo "    Skipping $model_dir..."
        echo ""
        continue
    fi
    
    echo "------------------------------------------------------------"
    echo "Evaluating: $model_dir"
    echo "  Checkpoint: $CKPT_PATH"
    echo "  BEV_ZBOUND_STEP: $zbound_step"
    echo "  Output: $OUTPUT_DIR"
    echo "------------------------------------------------------------"
    
    BEV_ZBOUND_STEP=$zbound_step python $BASE_DIR/evaluate_checkpoint.py \
        --ckpt_path "$CKPT_PATH" \
        --dataset_root "$TEST_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --angle_range_deg 5 \
        --trans_range 0.3 \
        --use_full_dataset \
        --max_batches 0 \
        --vis_interval 10 \
        --batch_size 4 \
        --rotation_only 1
    
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation complete for $model_dir"
        echo ""
        echo "Results summary:"
        tail -20 "$OUTPUT_DIR/extrinsics_and_errors.txt" | grep -A 10 "Average"
        echo ""
    else
        echo "❌ Evaluation failed for $model_dir"
    fi
    echo ""
done

echo "================================================================================"
echo "✓ Batch evaluation complete!"
echo "================================================================================"
