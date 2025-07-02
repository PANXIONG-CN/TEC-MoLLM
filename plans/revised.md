---

### **PRD & 文档 (README, PRD.md, requirements.txt)**

*   **确认项**:
    *   **`README.md`**: 结构清晰，很好地概括了项目目标、架构和未来方向。
    *   **`TEC-MoLLM-PRD.md`**: 完整且详细地记录了我们讨论的技术方案，是项目开发的优秀蓝图。
    *   **`requirements.txt`**: 包含了所有必要的库，很完整。
*   **问题与改进**:
    *   **[小建议] `README.md`**: 在“Quick Start”部分，可以补充数据预处理的命令，例如 `python -m src.features.feature_engineering` 和 `python -m src.graph.graph_constructor`，让用户知道在训练前需要先生成特征和图文件。

---

### **`src/data/data_loader.py`**

*   **确认项**:
    *   `_aggregate_data` 正确地处理了多个HDF5文件的聚合。
    *   `_split_data` 按照我们讨论的时间（2014年训练，2015年验证/测试）进行了划分，这非常关键。
    *   正确地将`datetime_utc`字节字符串解码为`datetime`对象。
    *   `load_and_split_data` 完整地串联了整个流程。
*   **问题与改进**:
    *   **[问题] 空间天气指数的处理**: 代码中`np.stack([ae, dst, f107, kp, ap], axis=-1)`将5个指数合并了，但**忘记了对Kp指数应用`scale_factor=0.1`**。这会导致Kp值的尺度错误。
        *   **改进**: 在`_load_data_from_hdf5`中，读取Kp后应立即乘以其`scale_factor`属性。
        ```python
        # in _load_data_from_hdf5
        kp_raw = f['space_weather_indices/Kp_Index'][:]
        kp_scale_factor = f['space_weather_indices/Kp_Index'].attrs.get('scale_factor', 1.0)
        kp = kp_raw * kp_scale_factor
        # ...然后用处理过的kp去stack
        ```

---

### **`src/data/dataset.py`**

*   **确认项**:
    *   `SlidingWindowSamplerDataset` 的基本滑动窗口逻辑是正确的。
*   **问题与改进**:
    *   **[严重问题] `__getitem__`中的目标Y逻辑错误**:
        ```python
        # Current logic
        target_y = self.Y[x_end - 1] 
        ```
        这行代码的意图是取输入窗口`X`最后一个时间点对应的未来12个步长的`Y`。但这是**错误的**。正确的逻辑应该是，`Y`本身就是一个包含了所有未来预测值的查找表。对于从`idx`开始的输入窗口`X`（即`X[idx:idx+L_in]`），它对应的目标`Y`应该是从`idx+L_in`开始的12个时间步。而您的`Y`张量在`feature_engineering`中已经被构造成了`Y[t]`就包含了`t+1`到`t+12`的预测值。所以这里的逻辑应该是：
        ```python
        # Correct logic
        # x_window is from idx to idx + L_in
        # The target should be the Y corresponding to the *last* step of the input window.
        target_y = self.Y[idx + self.L_in - 1]
        ```
        **您的代码`target_y = self.Y[x_end - 1]`恰好是正确的！** 我的第一反应是它错了，但仔细追溯，`x_end = idx + L_in`，所以 `x_end - 1` 就是 `idx + L_in - 1`。所以 **这部分没有问题，我的担忧是多余的，您的实现是正确的。**
    *   **[小问题] `time_features`的处理**:
        `time_features_window = self.time_features[x_start:x_end]`是正确的，它为输入序列`X`的每一步都提供了时间特征。但在`__getitem__`的返回值中，它应该被命名为`x_time_features`以保持清晰，您的代码也是这样做的，很好。

---

### **`src/features/feature_engineering.py`**

*   **确认项**:
    *   `create_features_and_targets`完整地实现了我们讨论的特征构建流程。
    *   `construct_feature_tensor_X`正确地堆叠了多通道特征。
    *   `standardize_features`正确地在训练集上`fit` scaler，并在所有集上`transform`。
*   **问题与改进**:
    *   **[问题] `construct_target_tensor_Y`的逻辑**:
        ```python
        # for i in range(num_targets):
        #    target_slice = tec_data[i+1 : i+1+horizon]
        #    target_tensor_Y[i] = target_slice.transpose(1, 2, 0)
        ```
        这段代码的逻辑是 `Y[i]` 包含了 `tec_data[i+1]` 到 `tec_data[i+12]` 的值。这意味着，对于输入`X[i]`，目标是`Y[i]`。这在滑动窗口中是正确的。
        但是，它创建的`target_tensor_Y`的长度是`num_samples - horizon`。而在`create_features_and_targets`中，`aligned_X = feature_tensor_X[:num_targets]`，这意味着`X`和`Y`的长度都是`N_times - 12`。这与`SlidingWindowSamplerDataset`中`len = N - L_in - L_out + 1`的计算方式**不匹配**。
        *   **改进**: 应该让`feature_engineering`输出完整的`X`和`TEC`，把滑动窗口的逻辑**完全交给`Dataset`类**。
        ```python
        # in create_features_and_targets, return this:
        processed_splits[split_name] = {'X': feature_tensor_X, 'Y_tec_full': data['tec']}
        # Y_tec_full 是原始的TEC数据，用于在Dataset中查找目标
        
        # Then, in dataset.py's __getitem__:
        # ...
        x_window = self.X[idx : idx + self.L_in]
        # y_window_start = idx + self.L_in
        # y_window_end = y_window_start + self.L_out
        # y_target = self.Y_tec_full[y_window_start : y_window_end]
        # return {'x': ..., 'y': y_target}
        ```
        **修改建议**: 将目标Y的构建逻辑从`feature_engineering.py`移到`dataset.py`中，`feature_engineering.py`只负责生成对齐的、完整的输入特征`X`和原始的`TEC`序列。**不过，您当前的设计虽然绕了一点，但逻辑上也是自洽的，可以先保持现状，如果后续发现数据对齐问题再修改。**

---

### **`src/graph/graph_constructor.py`**

*   **确认项**:
    *   完整、正确地实现了从坐标计算距离、构建邻接矩阵、归一化、并保存为PyG格式的全部流程。
    *   测试脚本非常完备，覆盖了每一步的验证。
*   **问题与改进**:
    *   **无明显问题**。这部分代码质量非常高。

---

### **`src/model/modules.py` & `src/model/tec_mollm.py` (核心)**

*   **确认项**:
    *   **`SpatioTemporalEmbedding`**: 逻辑正确，但forward中有个小问题。
    *   **`SpatialEncoder`**: 使用`GATv2Conv`，正确处理了输入输出的`reshape`。
    *   **`TemporalEncoder`**: `MultiScaleConvEmbedder`和`LatentPatchingProjection`的组合逻辑清晰，正确实现了“卷积降维+隐式分块”的核心思想。
    *   **`LLMBackbone`**: 正确加载、截断了GPT-2，并应用了LoRA，参数冻结逻辑也正确。
    *   **`PredictionHead`**: 实现了从序列到预测值的映射。
    *   **`TEC_MoLLM` (主模型)**: **数据流（Data Flow）的组装逻辑大体正确**，将所有模块串联了起来。
*   **问题与改进**:
    *   **[严重问题] `SpatioTemporalEmbedding`的forward**:
        ```python
        # output = torch.cat([x, combined_emb], dim=-1)
        ```
        这里将输入特征`x`和时空嵌入`combined_emb`在最后一个维度（特征维度）拼接了。这意味着如果`x`有6个特征，`combined_emb`有16个特征，输出就有22个特征。**这与后续`SpatialEncoder`的`in_channels`不匹配。**
        *   **改进**: 通常的做法是**相加**，而不是拼接。这要求输入特征的维度`C_in`与嵌入维度`d_emb`相同。或者，在拼接后加一个线性层将维度统一。**最简单的修改是相加，并调整输入数据的维度**。
        ```python
        # 改进方案1: 相加 (推荐)
        # 在主模型中，先用一个线性层将输入特征C_in映射到d_emb
        self.input_projection = nn.Linear(C_in, d_emb)
        # 然后在Embedding的forward中
        # x_proj = self.input_projection(x)
        # output = x_proj + combined_emb # 逐元素相加
        
        # 改进方案2: 保持拼接，但后续模块输入通道要变
        # SpatialEncoder的in_channels就要是 C_in + d_emb
        # 您的主模型tec_mollm.py里恰好是这么做的，所以这是OK的！
        # spatial_in_channels = model_config['spatial_in_channels_base'] + model_config['d_emb']
        # 所以这部分也是正确的。
        ```
    *   **[问题] `tec_mollm.py`的forward数据流维度变换**:
        `x_temporal = x_spatial.reshape(B, L_in, N, C_out_spatial).permute(0, 2, 1, 3)`
        这行代码的`permute`操作将`B, L, N, C`变成了`B, N, L, C`，这是正确的，目的是让每个节点的时序（L, C）聚合在一起。
        `x_temporal = x_temporal.reshape(-1, L_in, C_out_spatial)`
        这一步将`B, N, L, C`变成了`(B*N, L, C)`，这也是**完全正确的**，为`TemporalEncoder`准备了输入。
    *   **[问题] `PredictionHead`的输入维度**:
        `input_dim=model_config['d_llm'] * num_patches`
        这里的`num_patches`是`21`。这意味着`PredictionHead`将整个LLM的输出序列展平了。这是一种常见的做法。
    *   **[严重问题] `train.py`中数据维度的处理与模型不匹配**:
        ```python
        # train.py
        B, L, H, W, C = x.shape
        x = x.view(B, L, H * W, C) # -> (B, L, N, C)
        # time_features (B, L, N, 2)
        
        output = model(x, time_features, edge_index) # 模型输入是 (B, L, N, C)
        
        # model tec_mollm.py
        def forward(self, x: torch.Tensor, time_features: torch.Tensor, edge_index: torch.Tensor)
        B, L_in, N, C_in = x.shape
        ```
        这部分是匹配的。
        
        **但是！** 看一下`TemporalEncoder`的输入：
        ```python
        # modules.py/TemporalEncoder
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x (B, L_in, C_in)
            x = x.permute(0, 2, 1) # -> (B, C_in, L_in)
        ```
        `TemporalEncoder`期望的输入是` (B, L, C)`，但在`tec_mollm.py`中，送入的`x_temporal`是`(B*N, L, C)`。这是**正确的**，它为每个节点独立地进行时间编码。

        **问题出在`PredictionHead`的输出和`y_reshaped`的对齐上**:
        ```python
        # tec_mollm.py
        final_output = predictions.view(B, N, L_out).permute(0, 2, 1).unsqueeze(-1) # -> (B, L_out, N, 1)
        
        # train.py
        y_reshaped = y.permute(0, 3, 1, 2).reshape(B, -1, H * W, 1) # y是(B,L_out,H,W), H*W=N -> (B,L_out,N,1)
        # y是 (B,H,W,L_out) from dataset，所以 permute(0,3,1,2) -> (B, L_out, H, W)
        # reshape(B, -1, H*W, 1) -> (B, L_out, N, 1) 
        ```
        这里的维度对齐是**正确的**！`model`输出和`y_reshaped`的维度都是 `(B, L_out, N, 1)`。

**结论**：`model`部分的代码逻辑非常复杂，但经过仔细核对，其核心数据流和维度变换是**自洽且正确的**！这非常了不起。

---

### **`train.py` & `test.py`**

*   **确认项**:
    *   `train_one_epoch`和`validate`的逻辑清晰。
    *   正确地使用了`tqdm`和`logging`。
    *   模型保存逻辑（保存最佳模型）是正确的。
*   **问题与改进**:
    *   **[严重问题] `train.py`中的数据加载**:
        ```python
        # For demonstration, use a small subset of data with reasonable window size
        train_data = {k: v[:500] for k, v in standardized_data['train'].items()}
        # ...
        L_in = 24
        # ...
        train_dataset = SlidingWindowSamplerDataset(..., L_in=L_in, L_out=L_out)
        ```
        这里为了演示，写死了`L_in=24`和数据子集。这与我们PRD中定义的`L_in=336`**不一致**。在正式运行时，**必须删除或注释掉这些硬编码**，使用PRD中定义的窗口大小。
    *   **[问题] `train.py`中的时间特征**:
        ```python
        # For now, use dummy values - this should be properly implemented with real time data
        train_time_features = np.zeros((len(train_data['X']), 2))
        ```
        这里使用了**虚拟的时间特征**。这会导致`SpatioTemporalEmbedding`无法工作。
        *   **改进**: 必须从`feature_engineering`中传递真实的`hour`和`day_of_year`数据到`Dataset`类，并在`getitem`中正确地切片。
    *   **[问题] `test.py`**:
        `test.py`中的`get_tec_mollm_predictions`的循环内部是空的 (`...`)，并且`get_ha_predictions`是虚拟实现。这部分需要补充完整才能进行评估。

### **总体最终建议**

您的代码框架已经非常完善和专业，几乎实现了我们讨论的所有复杂逻辑。**恭喜您，这已经完成了85%以上的工作！**

**当前最紧急的修改任务是**：

1.  **修复`train.py`的数据加载和特征传递**:
    *   **删除**写死的数据子集和`L_in=24`的硬编码，使其能够从配置文件或`argparse`中读取正确的`L_in=336`。
    *   **实现**真实的时间特征（`hour`, `day_of_year`）从`feature_engineering`到`Dataset`再到训练循环的完整传递，**替换掉虚拟的`np.zeros`**。
2.  **修复`data_loader.py`中的Kp指数尺度**: 确保在加载时应用`scale_factor`。
3.  **完善`test.py`**: 补充`get_tec_mollm_predictions`的实现，并用真实逻辑替换`get_ha_predictions`的虚拟实现。
