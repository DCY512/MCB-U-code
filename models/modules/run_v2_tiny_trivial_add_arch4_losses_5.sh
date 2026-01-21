#!/usr/bin/env bash
# ==============================================================================
# V2Tiny + 多增强方式 + 四架构 + 多融合 + 七损失
# ==============================================================================

set -euo pipefail

# —— 核心优化配置 ——
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export TORCH_DATALOADER_USE_SHMEM=1
export PYTORCH_USE_SHM=1
export TORCH_SHARED_MEMORY_PATH="/dev/shm"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_DATALOADER_PERSISTENT_WORKERS=1
export TORCH_DATALOADER_PREFETCH_FACTOR=2
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONWARNINGS="ignore::FutureWarning"
unset TORCH_USE_CUDA_DSA

# —— 通用训练参数 ——
BATCH_SIZE=32
EPOCHS=120
LR=8e-5
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=5
DROP_PATH=0.2
HEAD_TYPE="fpn_pan"
PATIENCE=20
NUM_WORKERS=8
SEED=42
INPUT_SIZE=224

# —— 固定组件 ——
STUDENT_MODEL="convnextv2_tiny"

# —— 架构数组 + 注意力配置数组 ——
ARCHS=(
  "Arch1_Baseline_NoAttention"
  "Arch2_PositionalKing"
  "Arch3_UltimateEfficiency"
  #"Arch4_HierarchicalExpert"
)
ATTN_CFGS=(
  '{}' 
  '{"N3":"coordatt","N4":"coordatt","N5":"coordatt"}'
  '{"N3":"eca","N4":"eca","N5":"eca"}'
  #'{"N3":"coordatt","N4":"cbam","N5":"eca"}'
)

# —— 增强模式数组 ——
AUG_MODES=( "none" "standard" "rand_aug" "trivial_aug" )

# —— 融合模式 ——
FUSES=("add" "ahcr@intra_level" "ahcr@inter_level")

# —— 预训练权重 ——
PRETRAINED_WEIGHTS="./student_weights_switch_no_head/${STUDENT_MODEL}.mapped_to_backbone.safetensors"

# —— 输出根目录 —— 除了 2的其他的都在这
BASE_OUT="./output_v2_student_loss_1232ssssss12_other"

# —— 损失列表 ——
LOSSES=("mcb_convex" "mcb" "bce" "focal" "fals" "gebce" "dals")

# —— 检查预训练权重 ——
if [ ! -f "$PRETRAINED_WEIGHTS" ]; then
  echo "❌ 找不到预训练权重：$PRETRAINED_WEIGHTS"
  exit 1
fi

echo "=================================================================="
echo "  实验集合：V2Tiny"
echo "  架构：${ARCHS[*]}"
echo "  增强方式：${AUG_MODES[*]}"
echo "  融合：${FUSES[*]}"
echo "  损失：${LOSSES[*]}"
echo "  预训练：$PRETRAINED_WEIGHTS"
echo "=================================================================="

COUNT=0
START_FROM=1  # 你可以调整从第几号实验开始

for AUG_MODE in "${AUG_MODES[@]}"; do
  for ai in "${!ARCHS[@]}"; do
    ARCH_NAME="${ARCHS[$ai]}"
    ATTN_JSON="${ATTN_CFGS[$ai]}"

    for FUSE in "${FUSES[@]}"; do
      FUSE_MODE="${FUSE%%@*}"
      AHCR_MODE="${FUSE#*@}"
      AHCR_ARG=""
      if [[ "$FUSE_MODE" == "ahcr" && "$AHCR_MODE" != "$FUSE_MODE" ]]; then
        AHCR_ARG="--ahcr_mode ${AHCR_MODE}"
      fi

      for LOSS in "${LOSSES[@]}"; do
        COUNT=$((COUNT + 1))
        if [ $COUNT -lt $START_FROM ]; then
          echo "⏭️ 跳过实验 #$COUNT: aug=${AUG_MODE} | arch=${ARCH_NAME} | fuse=${FUSE} | loss=${LOSS}"
          continue
        fi

        EXTRA_LOSS_ARGS=""
        case "$LOSS" in
          "mcb_convex")
            EXTRA_LOSS_ARGS="--mcb_tau 1.0 --mcb_momentum 0.9 --mcb_wmin 1e-3"
            ;;
          "mcb")
            EXTRA_LOSS_ARGS="--mcb_tau 1.0 --mcb_momentum 0.9"
            ;;
          "bce")
            EXTRA_LOSS_ARGS=""
            ;;
          "focal")
            EXTRA_LOSS_ARGS="--focal_gamma 2.0 --focal_alpha 0.25"
            ;;
          "fals")
            EXTRA_LOSS_ARGS="--fals_eps 0.1 --fals_gamma 2.0"
            ;;
          "gebce")
            EXTRA_LOSS_ARGS="--ge_lambda 0.1 --ge_pos_only true --ge_alpha 0.75 --ge_ema true --ge_momentum 0.9 --ge_band 0.0 --ge_trainable true"
            ;;
          "dals")
            EXTRA_LOSS_ARGS="--dals_eps 0.1 --dals_gamma 2.0"
            ;;
          *)
            echo "⚠️ 未知损失：$LOSS，跳过"; continue;;
        esac

        OUT_DIR="${BASE_OUT}/tiny_${AUG_MODE}_${FUSE}_${ARCH_NAME}_loss-${LOSS}"
        mkdir -p "$OUT_DIR"

        echo ""
        echo "──────────────────────────────────────────────────────────"
        echo "▶ 实验 #$COUNT: aug=${AUG_MODE} | arch=${ARCH_NAME} | fuse=${FUSE} | loss=${LOSS}"
        echo "  输出目录：${OUT_DIR}"
        echo "──────────────────────────────────────────────────────────"

        export OUTPUT_DIR_HINT="${OUT_DIR}"
        if ! python -u main_finetune.py \
            --aug_mode ${AUG_MODE} \
            --patience ${PATIENCE} \
            --model ${STUDENT_MODEL} \
            --finetune ${PRETRAINED_WEIGHTS} \
            --output_dir ${OUT_DIR} \
            --model_prefix '' \
            --batch_size ${BATCH_SIZE} \
            --epochs ${EPOCHS} \
            --lr ${LR} \
            --weight_decay ${WEIGHT_DECAY} \
            --warmup_epochs ${WARMUP_EPOCHS} \
            --drop_path ${DROP_PATH} \
            --input_size ${INPUT_SIZE} \
            --fuse_mode ${FUSE_MODE} \
            ${AHCR_ARG} \
            --head_type ${HEAD_TYPE} \
            --attention_config "${ATTN_JSON}" \
            --dual_view true \
            --teacher_mode false \
            --num_workers ${NUM_WORKERS} \
            --seed ${SEED} \
            --device cuda \
            --base_loss ${LOSS} \
            ${EXTRA_LOSS_ARGS} 2>&1 | tee "${OUT_DIR}/train.log"; then
          echo "❌ 失败：aug=${AUG_MODE} | arch=${ARCH_NAME} | fuse=${FUSE} | loss=${LOSS} | 已跳过"
          continue
        fi

        date +"%F %T" > "${OUT_DIR}/_FINISHED"
        echo "✅ 完成：aug=${AUG_MODE} | arch=${ARCH_NAME} | fuse=${FUSE} | loss=${LOSS}"
      done
    done
  done
done

echo ""
echo "🎉 所有增强 × 架构 × 融合 × 损失 实验已完成"
