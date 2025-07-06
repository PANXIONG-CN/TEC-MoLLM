
#### **Phase 1: 代码修改**

**任务 1.1: 修复 `PredictionHead` 模块中的系统性偏差**

*   **文件路径**: `src/model/modules.py`
*   **目标模块**: `class PredictionHead`
*   **问题描述**: 当前 `PredictionHead` 在输出层使用 `nn.ReLU()` 激活函数。然而，模型的训练目标 `Y` 经过 `StandardScaler` 归一化后，其分布以0为中心，包含大量负值。`ReLU` 会将所有预测的负值截断为0，导致模型无法学习低于均值的目标，从而产生巨大的、系统性的预测误差（MAE/RMSE）。
*   **修改指令**:
    1.  定位到 `PredictionHead` 类的 `__init__` 方法。
    2.  删除或注释掉 `self.activation = nn.ReLU()` 这一行。
    3.  定位到 `forward` 方法。
    4.  删除或注释掉 `output = self.activation(output)` 这一行。
*   **代码修改 (上下文展示)**:

    ```python
    # File: src/model/modules.py
    
    class PredictionHead(nn.Module):
        """
        Maps features from the LLM backbone to the final prediction sequence.
        """
        def __init__(self, input_dim: int, output_dim: int):
            """
            Args:
                input_dim (int): The flattened input dimension from the LLM (e.g., 21 * 768).
                output_dim (int): The output prediction dimension (e.g., 12).
            """
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
            # --- START MODIFICATION 1.1.1 ---
            # REMOVED: self.activation = nn.ReLU() 
            # REASON: The model must predict standardized values which can be negative.
            #         ReLU introduces a positive bias, crippling regression performance.
            # --- END MODIFICATION 1.1.1 ---
            logging.info(f"PredictionHead initialized with input_dim={input_dim}, output_dim={output_dim}")
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Processes the input tensor to produce the final output.
            ...
            """
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
            
            output = self.fc(x)
            
            # --- START MODIFICATION 1.1.2 ---
            # REMOVED: output = self.activation(output)
            # --- END MODIFICATION 1.1.2 ---
            
            return output
    ```

**任务 1.2: 纠正 `target_scaler` 的拟合数据源**

*   **文件路径**: `scripts/preprocess.py`
*   **目标函数**: `main()`
*   **问题描述**: 当前的预处理脚本使用输入特征 `X` 中的TEC通道来拟合 `target_scaler`。正确的做法是，`target_scaler` 应该在它将要变换的数据，即目标 `Y` 的所有值上进行拟合，以获得最准确的均值和标准差。
*   **修改指令**:
    1.  定位到 `main` 函数中标记为 "3. 为目标(Y)创建并保存独立的scaler" 的代码块。
    2.  找到加载训练集数据的行。将 `processed_splits['train']['X'][:, :, :, 0:1]` 替换为 `processed_splits['train']['Y']`。
    3.  由于 `Y` 的形状是 `(N, H, W, L_out)`，需要将其展平为 `(-1, 1)` 的二维数组以供 `scaler.fit()` 使用。
*   **代码修改 (上下文展示)**:

    ```python
    # File: scripts/preprocess.py
    
    def main():
        # ... (previous code)
    
        # --- 3. 为目标(Y)创建并保存独立的scaler ---
        target_scaler_path = os.path.join(output_dir, 'target_scaler.joblib')
        target_scaler = StandardScaler()
        
        # --- START MODIFICATION 1.2.1 ---
        # OLD LOGIC:
        # train_tec_data = processed_splits['train']['X'][:, :, :, 0:1]
        # train_tec_reshaped = train_tec_data.reshape(-1, 1)
    
        # NEW LOGIC:
        # REASON: The target scaler must be fitted on the actual target data (Y)
        #         to correctly capture its statistical properties for normalization.
        train_y_data = processed_splits['train']['Y']
        train_tec_reshaped = train_y_data.reshape(-1, 1)
        # --- END MODIFICATION 1.2.1 ---
        
        target_scaler.fit(train_tec_reshaped)
        joblib.dump(target_scaler, target_scaler_path)
        logging.info(f"目标scaler已在训练集上拟合并保存至: {target_scaler_path}")
        
        # ... (rest of the code)
    ```

**任务 1.3: 引入梯度裁剪以增强训练稳定性**

*   **文件路径**: `train.py`
*   **目标函数**: `train_one_epoch()`
*   **问题描述**: 训练日志中的 `overflow` 警告表明模型在训练中发生了梯度爆炸，导致权重和输出值失控。引入梯度裁剪是防止此问题的标准且有效的方法。
*   **修改指令**:
    1.  定位到 `train_one_epoch` 函数内的训练循环中。
    2.  在 `scaler.scale(loss).backward()` 之后和 `scaler.step(optimizer)` 之前，插入梯度裁剪逻辑。
    3.  **关键步骤**: 必须先调用 `scaler.unscale_(optimizer)` 来反缩放梯度，然后再对原始梯度的范数进行裁剪。
*   **代码修改 (上下文展示)**:

    ```python
    # File: train.py
    
    def train_one_epoch(...):
        # ... (loop starts)
        for i, batch in enumerate(progress_bar):
            # ... (forward pass and loss calculation)
            
            # Scale loss and perform backward pass
            scaler.scale(loss).backward()
            
            # --- START MODIFICATION 1.3.1 ---
            # REASON: Prevents gradient explosion, which is the root cause of the
            #         'overflow' warning and numerical instability during training.
            #         The scaler.unscale_ is necessary before clipping in mixed-precision.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # --- END MODIFICATION 1.3.1 ---
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        # ... (rest of the function)
    ```

---

