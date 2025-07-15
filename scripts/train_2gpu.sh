#!/bin/bash

# 定义超参数 (内存优化版本)
L_IN=48
STRIDE=12
LLM_LAYERS=3
BATCH_SIZE=2
LR=1e-4
ACCUMULATION_STEPS=6
EPOCHS=50
PATIENCE=20
NUM_WORKERS=8



# 创建运行名称
TIMESTAMP=$(date +'%Y%m%d-%H%M')
RUN_NAME="L${L_IN}_S${STRIDE}_B${BATCH_SIZE}_LR${LR}_LLM${LLM_LAYERS}_${TIMESTAMP}"

# 创建日志目录
mkdir -p logs

echo "Starting training run: ${RUN_NAME}"
echo "Logging to: logs/${RUN_NAME}.log"
echo "Effective batch size: $((BATCH_SIZE * ACCUMULATION_STEPS * 2))"

# 启动训练 (2 GPU版本)
torchrun --nproc_per_node=2 train.py \
    --L_in $L_IN \
    --train_stride $STRIDE \
    --llm_layers $LLM_LAYERS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --accumulation_steps $ACCUMULATION_STEPS \
    --weight_decay 0.01 \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --num_workers $NUM_WORKERS 2>&1 | tee "logs/${RUN_NAME}.log"

echo "Training finished for run: ${RUN_NAME}" 