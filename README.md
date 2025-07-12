# `TEC-MoLLM`: 融合GNN与LLM的电离层TEC时空预测模型

**版本**: 1.0
**项目阶段**: v0.1 - 模型实现与验证

这是一个融合了图神经网络 (GNN) 和大型语言模型 (LLM) 的混合架构，旨在对全球电离层总电子含量 (TEC) 进行高精度的时空预测。

---

## 核心特性

- **混合模型架构**: 结合 GNN (`GATv2Conv`) 捕捉空间依赖性，多尺度一维卷积提取时间序列特征，并利用预训练LLM (`GPT-2`) 进行深度序列推理。
- **高效参数微调**: 采用 LoRA (Low-Rank Adaptation) 技术对 GPT-2 进行微调，仅训练少量参数即可达到优异性能。
- **时空嵌入**: 为每个节点（地理位置）、时刻、年份和季节生成可学习的嵌入向量，为模型提供丰富的时空上下文。
- **分布式训练**: 内置支持使用 `torchrun` (DistributedDataParallel) 进行多GPU训练，显著提升训练速度。
- **性能优化**: 使用混合精度训练 (torch.autocast) 和梯度累积等技术，以在有限的显存下支持大模型和更大的有效批量大小。

---

## 项目结构

```
.
├── checkpoints/         # 存放训练好的模型权重
├── data/
│   ├── raw/             # 存放原始 HDF5 数据
│   └── processed/       # 存放预处理后的数据 (scaler, graph, .pt文件)
├── scripts/
│   └── preprocess.py    # 离线数据预处理脚本
├── src/
│   ├── data/            # 数据加载器和数据集定义
│   ├── evaluation/      # 评估指标计算
│   ├── features/        # 特征工程
│   ├── graph/           # 图构建
│   ├── model/           # 模型定义 (TEC-MoLLM及子模块)
│   └── models/          # 基线模型定义
├── test.py              # 模型评估脚本
└── train.py             # 模型训练脚本
```

---

## 运行流程

### 步骤 0: 环境设置

1.  **克隆仓库**:
    ```bash
    git clone https://github.com/your-username/TEC-MoLLM.git
    cd TEC-MoLLM
    ```
2.  **创建Conda环境**:
    项目依赖`PyTorch` (含CUDA), `transformers`, `scikit-learn`, `pandas`, `h5py`, `peft`等。建议创建一个新的Conda环境。
    ```bash
    conda create -n tecgpt_env python=3.9
    conda activate tecgpt_env
    pip install torch numpy pandas h5py scikit-learn tqdm joblib peft transformers einops torch_geometric
    ```
3.  **准备数据**:
    将原始的 `CRIM_SW2hr_AI_v1.2` HDF5 数据文件（2013年至2025年）放入 `data/raw/` 目录下。

### 步骤 1: 数据预处理

运行离线预处理脚本。该脚本会完成所有必需的数据处理步骤，包括：
- 加载2013-2025年的HDF5数据。
- 按年份划分训练集、验证集和测试集。
- 创建特征 (`X`) 和目标 (`Y`)。
- 分别为特征和目标创建并保存标准化 `scaler`。
- 构建并保存图邻接矩阵。
- 将处理好的数据集保存为 `.pt` 文件。

```bash
python scripts/preprocess.py
```

### 步骤 2: 模型训练

使用 `torchrun` 启动分布式训练。脚本提供了丰富的命令行参数以调整训练配置。

**训练命令示例 (使用2个GPU):**
```bash
# WORLD_SIZE 是你的GPU数量
export WORLD_SIZE=2

# 启动分布式训练
torchrun --standalone --nproc_per_node=$WORLD_SIZE train.py \
    --L_in 48 \
    --L_out 12 \
    --batch_size 2 \
    --accumulation_steps 6 \
    --epochs 50 \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --patience 20 \
    --num_workers 8
```
- 有效批量大小 (Effective Batch Size) = `batch_size` × `accumulation_steps` × `WORLD_SIZE` = 2 × 6 × 2 = 24。
- 最佳模型将根据验证集损失保存在 `checkpoints/best_model.pth`。

### 步骤 3: 模型评估

训练完成后，使用 `test.py` 脚本评估模型在测试集上的性能，并与基线模型进行对比。

```bash
python test.py \
    --model_checkpoint checkpoints/best_model.pth \
    --batch_size 16 \
    --L_in 48 \
    --L_out 12
```
- 脚本会输出详细的逐预测时步指标，并将结果保存到 `results/evaluation_results.csv` 和 `results/evaluation_summary.txt`。

---

## 模型架构详解

### 数据流

```mermaid
graph TD
    A[Input Data<br/>(B, 48, N, 10)] --> B(Spatio-Temporal<br/>Embedding Fusion);
    C[Adjacency Matrix A] --> D{GNN Encoder (GATv2)};
    B --> D;
    D --> E[Temporal ConvEmbedder<br/>(Multi-Scale Conv)];
    E --> F(Latent Patching<br/>12 -> 3 Tokens);
    F --> G{GPT-2 Backbone<br/>(3 Layers w/ LoRA)};
    G --> H[Prediction Head (MLP)];
    H --> I[Output<br/>(B, N, 12)];
```

### 核心模块

1.  **`SpatioTemporalEmbedding`**: 为输入数据添加可学习的嵌入。输入特征从6维扩展至 `6 + d_emb` 维。
    - **节点嵌入**: `nn.Embedding(num_nodes=2911, ...)`
    - **时间嵌入**: 包含4个部分：时刻 (`tod`), 年中日 (`doy`), **年份索引 (`year`)**, 和 **季节 (`season`)**。
2.  **`SpatialEncoder`**: 在每个时间步上，利用 `GATv2Conv` 聚合空间信息。
3.  **`TemporalEncoder`**: 使用多尺度卷积 (`strides=[2, 2]`) 将长度为`48`的序列压缩至`12`，然后通过`LatentPatching` (`patch_size=4`)生成`3`个Tokens。
4.  **`LLMBackbone`**:
    - 使用预训练 `gpt2` 模型的前3层。
    - 采用 **LoRA** 微调，配置为 `r=32`, `lora_alpha=64`, `target_modules=['c_attn']`。
5.  **`PredictionHead`**: 将LLM输出的序列特征解码为最终预测结果。这是一个**双层MLP**（Linear -> GELU -> Dropout -> Linear），而非简单的线性层，具有更强的拟合能力。

---

## 数据与特征

- **数据源**: `CRIM_SW2hr_AI_v1.2` (HDF5格式), **2013-2025年**。
- **数据划分 (按时序)**:
    - **训练集**: 2013 - 2021 (9年)
    - **验证集**: 2022 - 2023 (2年)
    - **测试集**: 2024 - 2025 (2年)
- **输入特征 `X` (10维)**:
    - `TEC` (1维)
    - `space_weather_indices` (5维): AE, Dst, F10.7, Kp, ap
    - `SpatioTemporalEmbedding` (4维, 来自`d_emb`参数，嵌入后与原始特征拼接)
- **目标 `Y`**: 未来 **12个时间步** (24小时) 的TEC值。
- **滑动窗口**:
    - **输入序列长度 `L_in`**: **48** (4天)
    - **输出序列长度 `L_out`**: **12** (1天)
- **图结构**: 基于地理位置（纬度和经度）构建邻接矩阵，节点间距小于 **150km** 的被视为相连。

---

## 评估与基线

- **评估指标**: `MAE` (平均绝对误差), `RMSE` (均方根误差), `R^2 Score` (决定系数), `Pearsonr` (皮尔逊相关系数)。所有指标均在**逆标准化后**的真实值上计算，并报告所有预测时步的平均值和逐时步值。
- **基线模型**: `test.py` 中实现的基线为 **历史均值 (Historical Average)**，它使用输入窗口内 (`L_in=48`) 的TEC时间序列平均值作为未来12个时间步的预测值。

---

## 引用

*(如果您在研究中使用本模型或代码库，请按如下方式引用...)* 