# 动态运行命名功能使用说明

## 功能概述

根据方案要求，现在训练脚本会自动生成唯一的实验名称，并用于：
- Checkpoint文件命名
- 日志文件命名
- 实验跟踪和管理

## 命名规则

运行名称格式：`L{L_in}_S{train_stride}_B{batch_size}_LR{lr}_LLM{llm_layers}_{timestamp}`

示例：`L48_S3_B2_LR1e-4_LLM3_20241215-1430`

包含的参数：
- L_in: 输入序列长度
- train_stride: 训练数据采样步长
- batch_size: 批次大小
- lr: 学习率
- llm_layers: LLM层数
- timestamp: 时间戳 (YYYYMMDD-HHMM)

## 使用方法

### 方法1: 使用Shell脚本（推荐）

```bash
# 4 GPU版本
./scripts/train_with_dynamic_naming.sh

# 2 GPU版本（内存优化）
./scripts/train_2gpu.sh
```

Shell脚本会自动：
1. 生成运行名称
2. 创建日志目录
3. 启动训练并保存日志到 `logs/{run_name}.log`

### 方法2: 直接运行Python脚本

```bash
torchrun --nproc_per_node=2 train.py \
    --L_in 48 \
    --train_stride 3 \
    --batch_size 2 \
    --lr 1e-4 \
    --llm_layers 3
```

Python脚本会自动在内部生成运行名称。

## 文件输出

训练完成后，会生成以下文件：

1. **Checkpoint文件**: `checkpoints/best_model_{run_name}.pth`
2. **日志文件**: `logs/{run_name}.log` (仅使用shell脚本时)

## 测试/评估

### 使用最新的checkpoint

```bash
python test.py --model_checkpoint latest
```

这会自动查找最新的checkpoint文件。

### 使用特定的checkpoint

```bash
python test.py --model_checkpoint checkpoints/best_model_L48_S3_B2_LR1e-4_LLM3_20241215-1430.pth
```

## 自定义参数

可以通过修改shell脚本中的变量来调整超参数：

```bash
# 修改 scripts/train_2gpu.sh 中的参数
L_IN=336       # 改变输入长度
STRIDE=12      # 改变采样步长
BATCH_SIZE=4   # 改变批次大小
LR=5e-5        # 改变学习率
```

## 优势

1. **唯一性**: 每次运行都有唯一标识，避免文件覆盖
2. **可追溯**: 从文件名即可了解训练配置
3. **组织性**: 便于管理多个实验
4. **兼容性**: 支持分布式训练和旧版checkpoint格式 