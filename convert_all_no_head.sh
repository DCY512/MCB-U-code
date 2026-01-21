#!/usr/bin/env bash
# ==============================================================================
#  全学生模型权重 “去头” 转换脚本 (V1 & V2 兼容)
# ==============================================================================

set -e

# --- 配置 ---
# 包含您所有原始权重文件的目录
SOURCE_DIR="student_weights"
# 输出转换后“纯净”权重的目录
DEST_DIR="student_weights_switch_no_head"
# 您的映射脚本路径
MAP_SCRIPT="tools/map_official_v2_weights.py"

# --- 模型列表 ---
# 定义所有需要转换的模型的基础名称和它们的原始后缀
MODEL_SPECS=(
    "convnext_tiny_1k_224_ema.pth"
    "convnext_small_1k_224_ema.pth"
    "convnextv2_femto_1k_224_ema.pt"
    "convnextv2_nano_1k_224_ema.pt"
    "convnextv2_pico_1k_224_ema.pt"
    "convnextv2_tiny_1k_224_ema.pt"
)

echo "🚀 开始 “去头” 转换所有学生模型权重..."

# 确保目标目录存在
mkdir -p "${DEST_DIR}"

# --- 循环转换 ---
for FILENAME in "${MODEL_SPECS[@]}"; do
    
    # 构造输入文件的完整路径
    INPUT_PATH="${SOURCE_DIR}/${FILENAME}"
    
    # --- 智能构造输出文件名 ---
    # 1. 移除原始后缀 (.pth 或 .pt)
    BASE_NAME=${FILENAME%.*}
    # 2. 移除尾部的 "_ema" (如果存在)
    BASE_NAME=${BASE_NAME%_ema}
    # 3. 移除分辨率 "_1k_224" (如果存在)
    BASE_NAME=${BASE_NAME%_1k_224}
    # 4. 构建最终的输出文件名
    OUTPUT_NAME="${BASE_NAME}.mapped_to_backbone.safetensors"
    OUTPUT_PATH="${DEST_DIR}/${OUTPUT_NAME}"
    
    echo "--------------------------------------------------"
    
    # 检查原始文件是否存在
    if [ ! -f "$INPUT_PATH" ]; then
        echo "⚠️  警告: 未找到原始权重文件: ${INPUT_PATH}"
        echo "   跳过 ${FILENAME}。"
        continue
    fi

    echo "🔥 正在转换: ${FILENAME}"
    echo "  - 输入: ${INPUT_PATH}"
    echo "  - 输出: ${OUTPUT_PATH}"

    # 执行转换命令
    python "${MAP_SCRIPT}" \
      --input "${INPUT_PATH}" \
      --output "${OUTPUT_PATH}"
      
    echo "✅ ${FILENAME} “去头” 转换成功!"
done

echo "--------------------------------------------------"
echo "🎉 所有学生模型权重均已成功“去头”并保存至 ${DEST_DIR} 目录！"