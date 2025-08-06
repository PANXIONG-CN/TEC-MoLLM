#!/bin/bash

#!/bin/bash

# --- WandB & 微信推送配置 ---
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

# 3. ServerChan 微信推送配置
#    已在代码中集成 ServerChan 推送，无需额外环境变量配置
#    SendKey: SCT291707Tq6qEBYkKMcRIdczO5LM6Qp1U
echo "✅ ServerChan 微信推送已配置完成"
echo "   训练过程中会自动发送微信通知"

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