#!/bin/bash

# 定义超参数
L_IN=336
STRIDE=3       # 或者从6开始
LLM_LAYERS=6   # 或者从3开始
BATCH_SIZE=8   # 或者根据显存调整
LR=5e-5        # 一个更安全的起点
ACCUMULATION_STEPS=1 # 如果BATCH_SIZE已经够大，可以不用累积
EPOCHS=50
PATIENCE=15

# 创建运行名称
TIMESTAMP=$(date +'%Y%m%d-%H%M')
RUN_NAME="L${L_IN}_S${STRIDE}_B${BATCH_SIZE}_LR${LR}_LLM${LLM_LAYERS}_${TIMESTAMP}"

# 创建日志目录
mkdir -p logs

echo "Starting training run: ${RUN_NAME}"
echo "Logging to: logs/${RUN_NAME}.log"

# 启动训练
torchrun --nproc_per_node=4 train.py \
    --L_in $L_IN \
    --train_stride $STRIDE \
    --llm_layers $LLM_LAYERS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --accumulation_steps $ACCUMULATION_STEPS \
    --weight_decay 0.01 \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --num_workers 32 2>&1 | tee "logs/${RUN_NAME}.log"

echo "Training finished for run: ${RUN_NAME}" 