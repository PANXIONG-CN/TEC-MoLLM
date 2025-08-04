#!/bin/bash

# 设置微信推送Token (请替换为您的完整token)

#!/bin/bash

# --- WandB & 通知配置 ---
# 1. WandB API Key (推荐)
#    请确保已设置此环境变量。如果没有，脚本会发出警告。
#    您可以在 shell 中执行: export WANDB_API_KEY="YOUR_KEY_HERE"
export WANDB_API_KEY="946d5c9d77aa906f738ec843d4256380b4a819f8" 
#    或者，您也可以在执行此脚本前运行 `wandb login`。
if [ -z "$WANDB_API_KEY" ]; then
    echo "警告: 环境变量 WANDB_API_KEY 未设置。将尝试使用缓存的登录信息。"
    echo "推荐设置API Key以实现无交互运行: export WANDB_API_KEY=..."
fi

# 2. WandB 项目和实体（用户名/团队名）

export WANDB_PROJECT="TEC-MoLLM-Project"
export WANDB_ENTITY="xiongpan-tsinghua-university" 

# 3. 微信通知Token
#    重要：请确保这是从AutoDL官网复制的完整、无省略号的Token
export WECHAT_BOT_TOKEN="eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjE5MDQyMSwidXVpZCI6IjMyNGQwZTk0LTE0MmMtNDJlZC05Y2U2LTA1ZThiNzc4M2QzMiIsImlzX2FkbWluIjpmYWxzZSwiYmFja3N0YWdlX3JvbGUiOiIiLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.MaS1BOQtTmip9pitMl-hl9AGO_yc6y7QPItfwoxM-iZNulmE5FsL7LOnjVbdhx8xjIJ4qKCONnE9IzEGVb8qEQ" # 替换成您的完整token

# 定义超参数
L_IN=48
STRIDE=12
LLM_LAYERS=3
BATCH_SIZE=2
LR=1e-4
DROPOUT_RATE=0.2
ACCUMULATION_STEPS=6
EPOCHS=10
PATIENCE=5
NUM_WORKERS=8

# 创建运行名称
TIMESTAMP=$(date +'%Y%m%d-%H%M')
RUN_NAME="L${L_IN}_S${STRIDE}_B${BATCH_SIZE}_LR${LR}_LLM${LLM_LAYERS}_${TIMESTAMP}"

# 创建目录
mkdir -p logs checkpoints

echo "Starting training run: ${RUN_NAME}"
echo "Logging to: logs/${RUN_NAME}.log"
echo "Effective batch size: $((BATCH_SIZE * ACCUMULATION_STEPS * 2))"

# 启动训练
torchrun --nproc_per_node=2 train.py \
    --run_name $RUN_NAME \
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
    --num_workers $NUM_WORKERS 2>&1 | tee "logs/${RUN_NAME}.log"

echo "Training finished for run: ${RUN_NAME}"