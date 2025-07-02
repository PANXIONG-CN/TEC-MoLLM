下面是详细的核对和最后的微调建议。

---

### **`src/data/data_loader.py`**

* **状态**: **已修复** :+1:

* **核对**:

  ```python
  # Handle Kp Index with proper scale_factor
  kp_raw = f['space_weather_indices/Kp_Index'][:]
  kp_scale_factor = f['space_weather_indices/Kp_Index'].attrs.get('scale_factor', 1.0)
  kp = kp_raw * kp_scale_factor
  logging.info(f"Applied scale_factor {kp_scale_factor} to Kp_Index")
  ```

* **评价**: 完美！您已经正确地读取了`scale_factor`属性并应用到了`Kp_Index`上。这确保了所有空间天气指数在进入模型前都具有正确的物理尺度。这个问题已经解决。

---

### **`src/features/feature_engineering.py`**

* **状态**: **已修复** :+1:

* **核对**:

  ```python
  def extract_time_features(time_data: pd.DatetimeIndex) -> np.ndarray:
      # ...
  
  def create_features_and_targets(...):
      # ...
      # Extract real time features from datetime data
      time_features = extract_time_features(data['time'])
      # ...
      aligned_time_features = time_features[:num_targets]
      # ...
      processed_splits[split_name] = {
          'X': aligned_X, 
          'Y': target_tensor_Y,
          'time_features': aligned_time_features  # <---
      }
  ```

* **评价**: 非常好！您添加了`extract_time_features`函数，并将其集成到了主流程中。现在，处理后的数据字典中包含了真实的`time_features`，解决了之前使用虚拟数据的问题。

---

### **`src/data/dataset.py`**

* **状态**: **逻辑有待商榷，但目前可以工作**

* **核对**:

  ```python
  # __init__
  self.num_samples = max(0, len(X) - self.L_in + 1)
  if len(Y) < self.num_samples:
      self.num_samples = len(Y)
  
  # __getitem__
  target_y = self.Y[idx]
  ```

* **评价**: 您这里的实现非常巧妙，通过在`__init__`中用`Y`的长度来对齐`num_samples`，并直接在`__getitem__`中用`idx`索引`Y`，绕过了`feature_engineering.py`中`Y`被截断的问题。**这个逻辑是自洽的，可以让代码跑起来，所以暂时可以保持不动。**

  *   **长期建议**: 更“标准”的做法是在`feature_engineering`中不处理`Y`，只输出完整的`X`和`TEC`，然后在`dataset.py`中进行`Y`的切片。`Y_start = idx + L_in`, `Y_end = Y_start + L_out`。但目前您的实现也能工作，可以优先推进。

* **[小问题] `target_y = self.Y[idx]` 的对齐**

  *   在 `feature_engineering.py` 中 `target_tensor_Y[i]` 对应的是 `tec_data[i+1 : i+1+horizon]`。
  *   在 `dataset.py` 中，`x_window` 从 `idx` 开始。那么 `x_window` 的最后一个点是 `idx + L_in - 1`。
  *   按照时序预测的定义，这个`x_window`应该用来预测从`idx + L_in`开始的序列。
  *   而`Y[idx]`实际上是 `tec_data[idx+1]` 到 `tec_data[idx+12]`。
  *   这意味着您的模型在用 `X[idx : idx+336]` 来预测 `Y[idx]`，即 `TEC[idx+1 : idx+13]`。这在时间上是错位的。
  *   **正确的目标应该是 `Y[idx + L_in - 1]`**。

* **[必须修改]**:

  ```python
  # in src/data/dataset.py, __getitem__
  # ...
  x_end = idx + self.L_in
  x_window = self.X[idx:x_end]
  time_features_window = self.time_features[idx:x_end]
  
  # The target `Y` was pre-calculated. `Y[t]` contains the future values for `X[t]`.
  # Our input window ends at time `t = idx + L_in - 1`.
  # Therefore, the corresponding target is at this index.
  target_y = self.Y[idx + self.L_in - 1] # <--- 关键修改！
  
  return {
      'x': torch.tensor(x_window, dtype=torch.float32),
      'y': torch.tensor(target_y, dtype=torch.float32),
      'x_time_features': torch.tensor(time_features_window, dtype=torch.float32)
  }
  ```

  **这个对齐问题非常关键，必须修复，否则模型学到的是错误的时序关系。**

---

### **`train.py`**

* **状态**: **已修复大部分问题，配置更灵活** :+1:

* **核对**:

  ```python
  def parse_args():
      # ...
  
  def main():
      args = parse_args()
      # ...
      if args.use_subset:
          # ...
      else:
          train_data = standardized_data['train']
          val_data = standardized_data['val']
  
      train_time_features = processed_data['train']['time_features']
      # ...
  ```

* **评价**: 优秀！您引入了`argparse`来管理超参数和配置，如`L_in`和`use_subset`。并且正确地从`processed_data`中加载了真实的`time_features`。这解决了之前硬编码和虚拟数据的所有问题。现在的训练脚本非常灵活和健壮。

---

### **`test.py`**

*   **状态**: **仍有待实现**
*   **核对**: `get_tec_mollm_predictions`的循环体依然是空的，`get_ha_predictions`依然是虚拟实现。
*   **评价**: 这部分符合预期，因为我们的重点是先跑通训练。在训练出第一个可用模型后，再来完善这部分是合理的。
*   **下一步**: 在训练出`best_model.pth`后，您需要：
    1.  补充`get_tec_mollm_predictions`的逻辑，使其能正确地进行模型前向传播并收集结果。
    2.  在`src/models/baselines.py`中完整实现`HistoricalAverage`的`fit`和`predict`，然后在`test.py`中加载并调用它。

---

### **`src/model/modules.py` & `tec_mollm.py`**

* **状态**: **逻辑正确，但`SpatioTemporalEmbedding`有小问题**

* **核对**:

  ```python
  # src/model/modules.py -> SpatioTemporalEmbedding.forward
  output = torch.cat([x, combined_emb], dim=-1)
  ```

  这个拼接操作没有问题，因为在主模型`tec_mollm.py`中，`SpatialEncoder`的输入通道数已经正确地考虑了拼接后的维度(`spatial_in_channels = model_config['spatial_in_channels_base'] + model_config['d_emb']`)。

  但是，`tod_embedding`的`num_embeddings`被硬编码为`24`。

  ```python
  self.tod_embedding = nn.Embedding(num_embeddings=24, embedding_dim=d_emb)
  ```

  而您的数据是2小时间隔，一天只有12个时间点（0, 2, ..., 22）。输入给它的索引最大是22，如果直接用`tod_indices`作为索引，会超出范围。

* **[必须修改]**:

  1. **修改`SpatioTemporalEmbedding`**: `tod_embedding`的`num_embeddings`应该设为`12`。

     ```python
     # in src/model/modules.py
     self.tod_embedding = nn.Embedding(num_embeddings=12, embedding_dim=d_emb) # 2-hour slots
     ```

  2. **修改输入给`tod_embedding`的索引**: 在`forward`中，输入的`time_features`中的小时值（0, 2, ..., 22）需要除以2，才能得到正确的索引（0, 1, ..., 11）。

     ```python
     # in src/model/modules.py -> SpatioTemporalEmbedding.forward
     tod_indices = (time_features[..., 0] / 2).long() # <-- 关键修改：将小时值映射到0-11
     doy_indices = time_features[..., 1].long()
     tod_emb = self.tod_embedding(tod_indices)
     ```

     （*注：您的`extract_time_features`里是将hour/24做了归一化，传给embedding层需要的是整数索引，这里需要确认一下最终传给embedding层的是什么*）。
     **假设`feature_engineering`中`extract_time_features`返回的是未归一化的整数hour和day_of_year，那么上述修改是必须的。**

### **最终结论和行动计划**

您的代码已经非常接近成功运行第一版“沙盒”实验了。

**请按以下优先级进行最后的修改：**

1.  **【最高优先级】修复 `src/data/dataset.py` 中的目标`Y`索引错误**：
    *   将 `target_y = self.Y[idx]` 修改为 `target_y = self.Y[idx + self.L_in - 1]`。
    *   **这是逻辑上最关键的错误，不修复的话模型会学到完全错误的东西。**

2.  **【高优先级】修复 `src/model/modules.py` 中的时间嵌入**:
    *   将`TimeOfDayEmbedding`的`num_embeddings`从`24`改为`12`。
    *   确保传入的`hour`索引被正确地处理（例如`/2`）以匹配`0-11`的范围。