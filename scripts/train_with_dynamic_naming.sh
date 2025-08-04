#!/bin/bash

# 设置微信推送Token (请替换为您的完整token)
export WECHAT_BOT_TOKEN="eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjE5MDQyMSwidXVpZCI6IjMyNGQwZTk0LTE0MmMtNDJlZC05Y2U2LTA1ZThiNzc4M2QzMiIsImlzX2FkbWluIjpmYWxzZSwiYmFja3N0YWdlX3JvbGUiOiIiLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.MaS1BOQtTmip9pitMl-hl9AGO_yc6y7QPItfwoxM-iZNulmE5FsL7LOnjVbdhx8xjIJ4qKCONnE9IzEGVb8qEQ" # 替换成您的完整token

# 定义超参数 (新实验配置 - V8.0)
L_IN=336
STRIDE=3       # 保持不变
LLM_LAYERS=9   # 增加容量
BATCH_SIZE=4   # 或者根据显存调整
LR=5e-5        # 保持不变
DROPOUT_RATE=0.2  # 增加正则化
ACCUMULATION_STEPS=2 # 如果BATCH_SIZE已经够大，可以不用累积
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
    --dropout_rate $DROPOUT_RATE \
    --accumulation_steps $ACCUMULATION_STEPS \
    --weight_decay 0.01 \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --num_workers 32 2>&1 | tee "logs/${RUN_NAME}.log"

echo "Training finished for run: ${RUN_NAME}" 