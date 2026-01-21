#!/usr/bin/env bash
# ==============================================================================
# V2Tiny + 多增强方式 + 创新注意力 + 多损失
# 改进版：支持灵活控制实验顺序和范围
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
EPOCHS=180
#LR=8e-5
LR=1e-4
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=5
DROP_PATH=0.2
HEAD_TYPE="fpn_pan"
PATIENCE=25
NUM_WORKERS=8
SEED=42
INPUT_SIZE=224

# —— 固定组件 ——
STUDENT_MODEL="convnextv2_tiny"

# —— 架构数组 + 注意力配置数组（创新注意力） ——
ARCHS=(
  #"Arch1_Baseline_NoAttention"
  #"Arch4_HierarchicalExpert"
  "8_CoordAtt"
)
ATTN_CFGS=(
  #'{}'
  #'{"N3":"coordatt_u","N4":"cbam","N5":"eca"}'
  '{"N3":"coordatt_u"}'
)

# —— 增强模式数组 ——
AUG_MODES=("conditional")

# —— 融合模式 ——
FUSES=("add")

# —— 损失列表 ——
LOSSES=( "mcb_convex" ) #"mcb" "mlsm" "bce"

# —— 预训练权重 ——
PRETRAINED_WEIGHTS="./student_weights_switch_no_head/${STUDENT_MODEL}.mapped_to_backbone.safetensors"

# —— 运行次数和输出根目录模板 ——
RUN_TIMES=5
BASE_OUT_TEMPLATE="./shishi1"

# —— 实验控制参数 ——
START_FROM=1         # 从第几个实验开始  
RUN_SPECIFIC=""       # 运行特定实验
RUN_COUNT=0           # 运行多少个实验（0表示全部）
SKIP_COMPLETED=true   # 是否跳过已完成的实验

# ==============================================================================
# 解析命令行参数
# ==============================================================================
while [[ $# -gt 0 ]]; do
  case $1 in
    --start-from)
      START_FROM="$2"
      shift 2
      ;;
    --run-specific)
      RUN_SPECIFIC="$2"
      shift 2
      ;;
    --run-count)
      RUN_COUNT="$2"
      shift 2
      ;;
    --skip-completed)
      SKIP_COMPLETED=true
      shift
      ;;
    --no-skip-completed)
      SKIP_COMPLETED=false
      shift
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# —— 检查预训练权重 ——
if [ ! -f "$PRETRAINED_WEIGHTS" ]; then
  echo "❌ 找不到预训练权重：$PRETRAINED_WEIGHTS"
  exit 1
fi

# ==============================================================================
# 主运行循环
# ==============================================================================
for ((RUN_TIME=1; RUN_TIME<=RUN_TIMES; RUN_TIME++)); do
  echo ""
  echo "=================================================================="
  echo "  第 $RUN_TIME/$RUN_TIMES 次运行"
  echo "=================================================================="
  
  # —— 设置当前运行的输出根目录 ——
  BASE_OUT="${BASE_OUT_TEMPLATE}_${RUN_TIME}"
  
  # ==============================================================================
  # 实验配置生成
  # ==============================================================================
  declare -a EXPERIMENTS
  declare -A EXPERIMENT_CONFIGS
  
  COUNT=0
  for AUG_MODE in "${AUG_MODES[@]}"; do
    for ai in "${!ARCHS[@]}"; do
      ARCH_NAME="${ARCHS[$ai]}"
      ATTN_JSON="${ATTN_CFGS[$ai]}"
      for FUSE in "${FUSES[@]}"; do
        for LOSS in "${LOSSES[@]}"; do
          COUNT=$((COUNT + 1))
          
          # 构建实验配置
          EXP_KEY="$COUNT"
          EXPERIMENTS+=("$EXP_KEY")
          
          # 存储实验配置
          EXPERIMENT_CONFIGS["${EXP_KEY}_aug"]="$AUG_MODE"
          EXPERIMENT_CONFIGS["${EXP_KEY}_arch"]="$ARCH_NAME"
          EXPERIMENT_CONFIGS["${EXP_KEY}_attn"]="$ATTN_JSON"
          EXPERIMENT_CONFIGS["${EXP_KEY}_fuse"]="$FUSE"
          EXPERIMENT_CONFIGS["${EXP_KEY}_loss"]="$LOSS"
          
          # 构建输出目录
          OUT_DIR="${BASE_OUT}/tiny_${AUG_MODE}_${FUSE}_${ARCH_NAME}_loss-${LOSS}"
          EXPERIMENT_CONFIGS["${EXP_KEY}_out"]="$OUT_DIR"
        done
      done
    done
  done
  
  TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}
  
  echo "  实验配置总览 - 第 $RUN_TIME 次运行"
  echo "=================================================================="
  echo "  总实验数: $TOTAL_EXPERIMENTS"
  echo "  起始位置: $START_FROM"
  echo "  特定实验: ${RUN_SPECIFIC:-全部}"
  echo "  运行数量: ${RUN_COUNT:-全部}"
  echo "  跳过已完成: $SKIP_COMPLETED"
  echo "  增强方式: ${AUG_MODES[*]}"
  echo "  架构: ${ARCHS[*]}"
  echo "  融合: ${FUSES[*]}"
  echo "  损失: ${LOSSES[*]}"
  echo "  输出目录: $BASE_OUT"
  echo "=================================================================="
  
  # ==============================================================================
  # 实验选择逻辑
  # ==============================================================================
  declare -a SELECTED_EXPERIMENTS
  
  if [ -n "$RUN_SPECIFIC" ]; then
    # 解析特定实验编号
    IFS=',' read -ra PARTS <<< "$RUN_SPECIFIC"
    for part in "${PARTS[@]}"; do
      if [[ $part == *"-"* ]]; then
        # 范围选择，如 1-5
        IFS='-' read -ra RANGE <<< "$part"
        start=${RANGE[0]}
        end=${RANGE[1]}
        for ((i=start; i<=end; i++)); do
          if [ $i -ge 1 ] && [ $i -le $TOTAL_EXPERIMENTS ]; then
            SELECTED_EXPERIMENTS+=("$i")
          fi
        done
      else
        # 单个实验编号
        if [ $part -ge 1 ] && [ $part -le $TOTAL_EXPERIMENTS ]; then
          SELECTED_EXPERIMENTS+=("$part")
        fi
      fi
    done
  else
    # 顺序运行
    for ((i=START_FROM; i<=TOTAL_EXPERIMENTS; i++)); do
      SELECTED_EXPERIMENTS+=("$i")
      
      # 如果设置了运行数量限制
      if [ $RUN_COUNT -gt 0 ] && [ ${#SELECTED_EXPERIMENTS[@]} -ge $RUN_COUNT ]; then
        break
      fi
    done
  fi
  
  # 去重和排序
  IFS=$'\n' SELECTED_EXPERIMENTS=($(sort -nu <<< "${SELECTED_EXPERIMENTS[*]}"))
  unset IFS
  
  echo "✅ 第 $RUN_TIME 次运行已选择 ${#SELECTED_EXPERIMENTS[@]} 个实验: ${SELECTED_EXPERIMENTS[*]}"
  echo ""
  
  # ==============================================================================
  # 实验执行
  # ==============================================================================
  CURRENT=0
  TOTAL_SELECTED=${#SELECTED_EXPERIMENTS[@]}
  
  for EXP_NUM in "${SELECTED_EXPERIMENTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # 获取实验配置
    AUG_MODE="${EXPERIMENT_CONFIGS["${EXP_NUM}_aug"]}"
    ARCH_NAME="${EXPERIMENT_CONFIGS["${EXP_NUM}_arch"]}"
    ATTN_JSON="${EXPERIMENT_CONFIGS["${EXP_NUM}_attn"]}"
    FUSE="${EXPERIMENT_CONFIGS["${EXP_NUM}_fuse"]}"
    LOSS="${EXPERIMENT_CONFIGS["${EXP_NUM}_loss"]}"
    OUT_DIR="${EXPERIMENT_CONFIGS["${EXP_NUM}_out"]}"
    
    # 检查是否已完成
    if [ "$SKIP_COMPLETED" = true ] && [ -f "${OUT_DIR}/_FINISHED" ]; then
      echo "⏭️ 第 $RUN_TIME 次运行跳过已完成实验 #$EXP_NUM/$TOTAL_SELECTED: aug=${AUG_MODE} | arch=${ARCH_NAME} | fuse=${FUSE} | loss=${LOSS}"
      continue
    fi
    
    # 准备融合参数
    FUSE_MODE="${FUSE%%@*}"
    AHCR_MODE="${FUSE#*@}"
    AHCR_ARG=""
    if [[ "$FUSE_MODE" == "ahcr" && "$AHCR_MODE" != "$FUSE_MODE" ]]; then
      AHCR_ARG="--ahcr_mode ${AHCR_MODE}"
    fi
    
    # 准备损失参数
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
      "mlsm")
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
        echo "⚠️ 未知损失：$LOSS，跳过"
        continue
        ;;
    esac
    
    mkdir -p "$OUT_DIR"
    
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "▶ 第 $RUN_TIME 次运行 - 实验 #$CURRENT/$TOTAL_SELECTED (全局 #$EXP_NUM)"
    echo "  配置: aug=${AUG_MODE} | arch=${ARCH_NAME} | fuse=${FUSE} | loss=${LOSS}"
    echo "  输出: ${OUT_DIR}"
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
      echo "❌ 第 $RUN_TIME 次运行实验失败: #$EXP_NUM"
      continue
    fi
    
    date +"%F %T" > "${OUT_DIR}/_FINISHED"
    echo "✅ 完成第 $RUN_TIME 次运行实验 #$CURRENT/$TOTAL_SELECTED (全局 #$EXP_NUM)"
  done
  
  echo ""
  echo "🎉 第 $RUN_TIME 次运行所有选择的实验已完成"
  echo "   总计: $TOTAL_SELECTED 个实验"
  echo "   输出目录: $BASE_OUT"
  
  # 清理变量，为下一次运行做准备
  unset EXPERIMENTS
  unset EXPERIMENT_CONFIGS
  unset SELECTED_EXPERIMENTS
  
done

echo ""
echo "================================================================"
echo "🎉 所有 $RUN_TIMES 次运行已完成！"
echo "   输出目录:"
for ((RUN_TIME=1; RUN_TIME<=RUN_TIMES; RUN_TIME++)); do
  echo "   ${BASE_OUT_TEMPLATE}_${RUN_TIME}"
done
echo "================================================================"