

#### **Phase 1: 综合代码修改**

**任务 1.1: 【稳定 · 根本性修复】在验证/测试阶段禁用混合精度**

* **文件路径**: `train.py`

* **目标函数**: `validate()`

* **问题描述**: `float16`的数值范围（上限约6.5e4）是导致逆变换后`overflow`的根本原因。在验证阶段，数值稳定性比速度更重要，应使用`float32`。

* **修改指令**: 在 `validate()` 函数中，**完全移除** `with torch.autocast(...)` 上下文管理器。

* **代码修改 (上下文展示)**:

  ```python
  # File: train.py
  
  def validate(model, dataloader, loss_fn, device, edge_index, target_scaler_path, rank=0):
      # ... (function start)
      with torch.no_grad():
          progress_bar = tqdm(dataloader, desc="Validating") if rank == 0 else dataloader
          for batch in progress_bar:
              # ... (data to device)
              
              # --- START MODIFICATION 1.1.1 ---
              # REASON: Using float32 for validation is crucial for numerical stability.
              #         The float16 range is too small to handle inverse-transformed
              #         values, which was the root cause of the 'overflow' warning.
              # REMOVED: with torch.autocast(device_type='cuda', dtype=torch.float16):
              output = model(x, time_features, edge_index)
              y_reshaped = y.permute(0, 3, 1, 2).reshape(B, -1, H * W, 1)
              loss = loss_fn(output, y_reshaped)
              # --- END MODIFICATION 1.1.1 ---
              
              total_loss += loss.item()
              all_preds.append(output.cpu().numpy())
              all_trues.append(y_reshaped.cpu().numpy())
      # ... (rest of the function)
  ```

**任务 1.2: 【稳定】切换到更鲁棒的损失函数**

* **文件路径**: `train.py`

* **目标函数**: `main()`

* **问题描述**: `HuberLoss` 对异常值不敏感，能提供比`MSELoss`更稳定的梯度，有助于缓解数值问题并直接优化与MAE更相关的目标。

* **修改指令**: 将 `nn.MSELoss()` 替换为 `nn.HuberLoss(delta=1.0)`。

* **代码修改 (上下文展示)**:

  ```python
  # File: train.py, in main()
  # --- START MODIFICATION 1.2.1 ---
  # REASON: HuberLoss is more robust to outliers than MSELoss, stabilizing training.
  # OLD: loss_fn = nn.MSELoss()
  # NEW:
  loss_fn = nn.HuberLoss(delta=1.0)
  # --- END MODIFICATION 1.2.1 ---
  ```

**任务 1.3: 【稳定】使用更智能的学习率调度器**

* **文件路径**: `train.py`

* **目标函数**: `main()`

* **问题描述**: `CosineAnnealingLR`可能会在训练后期将学习率降得过低。`ReduceLROnPlateau`能根据验证损失的表现自适应地降低学习率，更适合在模型性能饱和时进行调整。

* **修改指令**:

  1.  将`CosineAnnealingLR`替换为`ReduceLROnPlateau`，并设置`patience`和`factor`。
  2.  在训练循环的 epoch 结束处，使用`val_loss`来调用`scheduler.step()`。

* **代码修改 (上下文展示)**:

  ```python
  # File: train.py, in main()
  
  # --- START MODIFICATION 1.3.1 ---
  # REASON: ReduceLROnPlateau provides adaptive learning rate adjustment based on
  #         validation performance, which is more robust than a fixed cosine schedule.
  # OLD: scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(...)
  # NEW:
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-7)
  # --- END MODIFICATION 1.3.1 ---
  
  # In training loop...
  try:
      for epoch in range(args.epochs):
          # ...
          val_loss, val_metrics = validate(...)
  
          # --- START MODIFICATION 1.3.2 ---
          # Use validation loss to drive the scheduler
          scheduler.step(val_loss)
          if rank == 0:
              # Note: get_last_lr() for ReduceLROnPlateau shows the *current* optimizer LR
              logging.info(f"LR scheduler checked. Current LR: {optimizer.param_groups[0]['lr']:.2e}")
          # --- END MODIFICATION 1.3.2 ---
          
          if rank == 0:
              # ... (logging and saving logic)
  ```

**任务 1.4: 【突破】提升模型容量以学习更复杂的模式**

* **文件路径**: `src/model/modules.py`

* **目标模块**: `class LLMBackbone`

* **问题描述**: 当前LoRA秩`r=16`可能限制了模型性能。增加`rank`是提升性能上限的关键。

* **修改指令**: 在`LoraConfig`中，将`r`提升至`32`，`lora_alpha`提升至`64`。

* **代码修改 (上下文展示)**:

  ```python
  # File: src/model/modules.py, in LLMBackbone.__init__()
  # --- START MODIFICATION 1.4.1 ---
  # REASON: Increasing LoRA rank enhances model capacity to learn complex patterns.
  lora_config = LoraConfig(
      r=32,             # OLD: 16
      lora_alpha=64,    # OLD: 32
      target_modules=["c_attn"],
      lora_dropout=0.1,
      bias="none"
  )
  # --- END MODIFICATION 1.4.1 ---
  ```

**任务 1.5: 【突破】引入Dropout以增强模型泛化能力**

* **文件路径**: `src/model/tec_mollm.py`

* **目标函数**: `forward()`

* **修改指令**: 在`LLMBackbone`和`PredictionHead`之间加入`Dropout`层。

* **代码修改 (上下文展示)**:

  ```python
  # File: src/model/tec_mollm.py, in TEC_MoLLM.forward()
  # --- START MODIFICATION 1.5.1 ---
  # REASON: Adds regularization to prevent overfitting and improve generalization.
  x_llm = self.llm_backbone(...)
  x_llm = torch.nn.functional.dropout(x_llm, p=0.1, training=self.training)
  predictions = self.prediction_head(x_llm)
  # --- END MODIFICATION 1.5.1 ---
  ```

**任务 1.6: 【稳定 · 终极保险】在评估阶段增加数值钳制**

* **文件路径**: `src/evaluation/metrics.py`

* **目标函数**: `evaluate_metrics()`

* **问题描述**: 作为最终的防御性措施，确保所有参与指标计算的预测值都严格处于物理合理的范围内，为评估流程提供100%的数值稳定性保障。

* **修改指令**: 在`evaluate_metrics()`函数中，对反归一化后的`y_pred_unscaled`使用`np.clip()`进行钳制。

* **代码修改 (上下文展示)**:

  ```python
  # File: src/evaluation/metrics.py
  
  def evaluate_metrics(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, scaler) -> dict:
      # ... (inverse transform code)
      y_true_unscaled = y_true_unscaled_reshaped.reshape(original_true_shape)
      y_pred_unscaled = y_pred_unscaled_reshaped.reshape(original_pred_shape)
  
      # --- START MODIFICATION 1.6.1 (Re-added for ultimate stability) ---
      # REASON: Adds a final safeguard to prevent any possible numerical issues
      #         by ensuring all predicted values are within a physically
      #         plausible range for TEC before metric calculation.
      TEC_MIN, TEC_MAX = 0, 200  # Define physical bounds for TEC in TECU
      y_pred_unscaled = np.clip(y_pred_unscaled, TEC_MIN, TEC_MAX)
      # --- END MODIFICATION 1.6.1 ---
  
      if y_true_unscaled.ndim > 2:
      # ... (rest of the function)
  ```



---