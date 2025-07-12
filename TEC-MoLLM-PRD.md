### **项目研究文档 (PRD): `TEC-MoLLM` 时空预测模型**

**文档版本**: 1.0
**项目代号**: `TEC-MoLLM`
**阶段**: v0.1 - 沙盒验证 (Sandbox Validation)
**目标**: 验证融合GNN与预训练LLM进行电离层TEC时空预测的技术可行性。
**环境**: : 已有conda环境 `tecgpt_env`，请在这个环境上修改

---
### **1. 数据集与预处理 (Dataset and Preprocessing)**

#### **1.1. 源数据描述**

*   **数据集**: `/home/panxiong/TEC-MoLLM/data/raw/CRIM_SW2hr_AI_v1.2_2014_DataDrivenRange_CN.hdf5和`/home/panxiong/TEC-MoLLM/data/raw/CRIM_SW2hr_AI_v1.2_2015_DataDrivenRange_CN.hdf5
*   **格式**: HDF5
*   **核心内容**:
    *   `ionosphere/TEC`: `(N_times, 41, 71)` - 垂直总电子含量 (TECU)
    *   `space_weather_indices`: 5个关键指数 (Kp, Dst, ap, F10.7, AE)，形状 `(N_times,)`
    *   `coordinates`: `latitude(41)`, `longitude(71)`, `time`, `hour`, `day_of_year` 等
*   **时间分辨率**: 2小时 (12个数据点/天)
*   **空间分辨率**: 1°x1° 格网 (41x71 = 2911个节点)

#### **1.2. 本阶段使用的数据子集**

*   **目的**: 用于快速原型开发和初步验证。
*   **时间范围**: **2014-01-01T00:00:00Z 至 2015-12-31T22:00:00Z**。
*   **数据划分 (严格按时序)**:
    *   **训练集 (Training Set)**: 2014年数据 (共 `365 * 12 = 4380` 个时间步)。
    *   **验证集 (Validation Set)**: 2015年1月-6月数据 (约 `181 * 12 = 2172` 个时间步)。
    *   **测试集 (Test Set)**: 2015年7月-12月数据 (约 `184 * 12 = 2208` 个时间步)。

#### **1.3. 特征工程 (Feature Engineering)**

1.  **输入特征 `X`**:
    *   **目标**: 构建一个多通道的时空张量。
    *   **步骤**:
        a. 读取 `ionosphere/TEC`，作为基础特征通道。
        b. 读取 `space_weather_indices` 下的5个指数。注意Kp_Index 需要scale_factor变化
        c. 将每个一维的指数时间序列（shape: `(N_times,)`）扩展为三维张量（shape: `(N_times, 41, 71)`），使其在空间维度上广播。
        d. 将TEC张量和5个广播后的指数张量在新的特征维度上进行堆叠 (`np.stack` 或 `torch.stack`)。
    *   **最终 `X` 形状**: `(N_times, 41, 71, 6)`，其中6为特征数 (1个TEC + 5个指数)。

2.  **目标 `Y`**:
    *   **目标**: 预测未来24小时（12个时间步）的TEC值。
    *   **形状**: `(N_times, 41, 71, 12)`。

3.  **时间编码特征**:
    *   从 `coordinates` 组中提取 `hour` 和 `day_of_year`。这些将作为输入，送入模型中的`Embedding`层。

4.  **空间图结构**:
    *   **节点 (Nodes)**: 2911个格点。
    *   **邻接矩阵 `A`**:
        a. 根据 `latitude` 和 `longitude` 计算所有节点对之间的地理距离（Haversine公式）。
        b. 定义一个距离阈值 `d_threshold`（例如150km）。
        c. 如果两节点间距离小于`d_threshold`，则在邻接矩阵中对应位置设为1，否则为0。
        d. 对邻接矩阵进行归一化处理（例如，对称归一化：`D^(-1/2) * A * D^(-1/2)`）。
        e. 该矩阵构建一次后，保存为文件供模型加载。

#### **1.4. 数据标准化与样本生成**

1.  **标准化 (Standardization)**:
    *   在**训练集**上计算输入特征 `X` 中每个通道（TEC和5个指数）的均值和标准差。
    *   使用该均值和标准差对整个数据集（训练、验证、测试）的 `X` 进行Z-Score标准化。
    *   保存该`scaler`对象，用于后续结果的逆标准化。

2.  **滑动窗口采样 (Sliding Window Sampling)**:
    *   **输入序列长度 `L_in`**: 336 (28天)
    *   **输出序列长度 `L_out`**: 12 (1天)
    *   在标准化后的 `X` 和 `Y` 上进行滑动，生成 `(X_sample, Y_sample)` 对。每个 `X_sample` 形状为 `(336, 41, 71, 6)`，`Y_sample` 形状为 `(12, 41, 71, 1)`。

---

### **2. 模型架构与实现 (Model Architecture & Implementation)**

#### **2.1. 模型总体设计 (`TEC-MoLLM`)**
本模型是一个融合了图神经网络、多尺度时间卷积和预训练语言模型的混合架构，用于处理和预测格网化时空序列数据。

#### **2.2. 模块化分解**

1.  **时空嵌入模块 (`SpatioTemporalEmbedding`)**:
    *   **功能**: 为输入数据提供时空上下文。
    *   **组件**:
        *   `NodeEmbedding`: `nn.Embedding(num_nodes=2911, embedding_dim=D_emb)`，为每个格点生成可学习的空间向量。
        *   `TimeOfDayEmbedding`: `nn.Embedding(num_embeddings=12, embedding_dim=D_emb)`，为一天中的12个时刻生成时间向量。
        *   `DayOfYearEmbedding`: `nn.Embedding(num_embeddings=366, embedding_dim=D_emb)`，为一年中的日期生成季节性/年度性向量。
    *   **输出**: 与输入数据融合的时空特征。

2.  **空间特征提取模块 (`SpatialEncoder`)**:
    *   **功能**: 在每个时间步上，利用图结构聚合空间信息。
    *   **组件**: `torch_geometric.nn.GATv2Conv` (Graph Attention Network v2)。
    *   **输入**: 节点特征 `(B*L_in, N_nodes, D_in)` 和 `edge_index`。
    *   **输出**: 经过空间信息交互后的节点特征 `(B*L_in, N_nodes, D_out)`。

3.  **时间特征提取模块 (`TemporalEncoder`)**:
    *   **功能**: 从长序列中提取多尺度的动态时序特征。**(此部分核心逻辑借鉴自SeisMoLLM)**
    *   **组件**:
        *   **多尺度卷积嵌入器 (`MultiScaleConvEmbedder`)**:
            *   采用堆叠的`Multi_Scale_Conv_Block`。
            *   **配置**: `conv_strides=[2, 2, 1, 1]`，实现4倍下采样，将长度`336`的序列压缩至`84`。
        *   **隐式分块 (`LatentPatching`)**:
            *   在长度为`84`的特征序列上操作。
            *   **配置**: `patch_size=4`，生成`21`个Tokens。
            *   每个Token的维度将被`rearrange`至`768`，以匹配GPT-2的输入。

4.  **序列推理模块 (`LLMBackbone`)**:
    *   **功能**: 对经过时空编码和特征提取的Token序列进行深度序列建模。
    *   **组件**:
        *   **GPT-2**: 使用`transformers`库加载预训练的`gpt2`模型，截取前3层。
        *   **PEFT策略**: 采用**LoRA (Low-Rank Adaptation)**。
            *   `peft.LoraConfig`配置: `r=16`, `lora_alpha=32`, `target_modules=['q_attn', 'c_attn']`。
            *   使用`peft.get_peft_model`包装GPT-2模型。
            *   **训练参数**: 冻结GPT原始权重，仅训练LoRA参数、LayerNorm层和位置编码。

5.  **预测头模块 (`PredictionHead`)**:
    *   **功能**: 将LLM输出的序列特征解码为最终的预测结果。
    *   **组件**: 一个或多个`nn.Linear`层。
    *   **输入**: LLM输出的 `(B*N_nodes, 21, 768)` 特征序列。
    *   **处理**: 将输入展平 (`flatten`)，然后通过线性层映射到`L_out=12`的维度。
    *   **输出**: 预测结果 `(B, N_nodes, 12)`，需`reshape`。

#### **2.3. 数据流 (Data Flow)**

```mermaid
graph TD
    A[Input Data<br/>(B, 336, N, 6)] --> B(Spatio-Temporal<br/>Embedding Fusion);
    C[Adjacency Matrix A] --> D{GNN Encoder};
    B --> D;
    D --> E[Temporal ConvEmbedder<br/>(SeisMoLLM Style)];
    E --> F(Latent Patching<br/>84 -> 21 Tokens);
    F --> G{GPT-2 Backbone<br/>(with LoRA)};
    G --> H[Prediction Head];
    H --> I[Output<br/>(B, N, 12)];
```

---

### **3. 基线模型库 (Baseline Model Zoo)**

本框架将支持以下可扩展的基线模型库，用于性能对比。

| 模型类别     | 模型名称   | 核心技术           | 备注                 |
| :----------- | :--------- | :----------------- | :------------------- |
| **经典时序** | `HA`       | Historical Average | 性能下限             |
|              | `SARIMA`   | Seasonal ARIMA     | 经典统计模型         |
| **新增模型** | *(可扩展)* | -                  | *(可添加其他新模型)* |

---

### **4. 评估指标 (Evaluation Metrics)**

将采用以下指标全面评估模型性能。所有指标均在逆标准化后的真实值和预测值上计算。

*   **`MAE` (Mean Absolute Error)**: `mean(|y_true - y_pred|)`。核心评估指标。
*   **`RMSE` (Root Mean Squared Error)**: `sqrt(mean((y_true - y_pred)^2))`。对大误差更敏感。
*   **`R^2` (Coefficient of Determination)**: R-squared score。衡量模型对数据方差的解释程度。
*   **`Correlation Coefficient` (Pearson r)**: `Cov(y_true, y_pred) / (std(y_true) * std(y_pred))`。衡量预测值与真实值的线性相关性。

评估将在每个预测时间步（horizon 1 to 12）上独立进行，并报告所有时间步的**平均值**。 