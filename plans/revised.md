#### **Phase 1: 关键代码修改**

**任务 1.1: 移除 `torch.compile` 以消除兼容性警告和潜在开销**

* **文件路径**: `train.py`

* **目标函数**: `main()`

* **问题描述**: `torch.compile()` 与项目中使用的 `transformers` 版本存在兼容性问题，导致编译失败并回退到Eager模式。这不仅无法带来性能提升，还会增加启动时间和日志噪音。

* **修改指令**:

  1.  在 `main()` 函数中，定位到模型编译相关的代码块。
  2.  删除或注释掉 `model = torch.compile(model)` 这一行以及相关的日志打印。

* **代码修改 (上下文展示)**:

  ```python
  # File: train.py
  
  def main():
      # ... (previous code)
      
      # --- Model, Optimizer, Loss ---
      model = TEC_MoLLM(model_config).to(device)
      
      # --- Performance Optimizations ---
      # --- START MODIFICATION 1.1.1 ---
      # REMOVED: torch.compile() due to compatibility issues with transformers 4.41+
      # REASON: This call fails with "torch.compiler has no attribute 'is_compiling'"
      #         and falls back to eager mode, adding overhead without performance gains.
      #         Disabling it simplifies debugging and ensures stable execution.
      # if rank == 0:
      #     logging.info("Compiling the model with torch.compile()...")
      # model = torch.compile(model)
      # if rank == 0:
      #     logging.info("Model compiled successfully.")
      # --- END MODIFICATION 1.1.1 ---
  
      # 2. Initialize Gradient Scaler for mixed-precision training
      scaler = torch.cuda.amp.GradScaler()
  
      # ... (rest of the function)
  ```

**任务 1.2: 修正 `Dataset` 样本对齐逻辑以避免数据丢失**

* **文件路径**: `src/data/dataset.py`

* **目标模块**: `class SlidingWindowSamplerDataset`

* **问题描述**: 当前计算样本总数的逻辑 `len(X) - L_in + 1` 未考虑未来标签 `L_out` 的长度，导致最后 `L_out - 1` 个时间步的数据虽然有输入窗口，但没有完整的未来标签，实际上无法用于训练。

* **修改指令**:

  1.  在 `SlidingWindowSamplerDataset` 的 `__init__` 方法中，找到计算 `self.num_samples` 的行。
  2.  将计算公式修改为 `len(self.X) - self.L_in - self.L_out + 1`，确保每个样本都有足够的输入和输出序列长度。

* **代码修改 (上下文展示)**:

  ```python
  # File: src/data/dataset.py
  
  class SlidingWindowSamplerDataset(Dataset):
      def __init__(self, data_path: str, mode: str, L_in: int = 336, L_out: int = 12):
          # ... (previous code)
          
          # --- START MODIFICATION 1.2.1 ---
          # REASON: The original calculation did not account for the output window length (L_out),
          #         leading to index errors or using incomplete targets at the end of the dataset.
          # OLD: self.num_samples = max(0, len(self.X) - self.L_in + 1)
          # NEW:
          self.num_samples = max(0, len(self.X) - self.L_in - self.L_out + 1)
          # --- END MODIFICATION 1.2.1 ---
          
          if self.num_samples <= 0:
              # ... (rest of the function)
  ```

**任务 1.3: 引入学习率调度器以动态调整学习率，辅助稳定训练**

* **文件路径**: `train.py`

* **目标函数**: `main()`

* **问题描述**: 恒定的学习率可能在训练后期过大，导致震荡或无法收敛。引入学习率调度器可以根据训练进程动态调整学习率，有助于模型跳出局部最优并更精细地收敛。

* **修改指令**:

  1.  在 `main()` 函数中，定位到 `optimizer` 的定义之后。
  2.  实例化一个学习率调度器，推荐使用 `torch.optim.lr_scheduler.CosineAnnealingLR`，它能平滑地降低学习率。
  3.  在训练循环的**每个 epoch 结束之后**，调用 `scheduler.step()` 来更新学习率。

* **代码修改 (上下文展示)**:

  ```python
  # File: train.py
  
  def main():
      # ... (previous code)
  
      optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
      loss_fn = nn.MSELoss()
  
      # --- START MODIFICATION 1.3.1 ---
      # REASON: Adds dynamic learning rate adjustment, which helps stabilize training
      #         and allows for better convergence in later epochs. CosineAnnealingLR
      #         is a robust choice for many deep learning tasks.
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
      # --- END MODIFICATION 1.3.1 ---
  
      # --- Training Loop ---
      # ...
      try:
          for epoch in range(args.epochs):
              # ... (train_one_epoch and validate calls)
              
              # --- START MODIFICATION 1.3.2 ---
              # Step the scheduler after each epoch
              scheduler.step()
              if rank == 0:
                  logging.info(f"LR scheduler stepped. New LR: {scheduler.get_last_lr()[0]:.2e}")
              # --- END MODIFICATION 1.3.2 ---
  
              if rank == 0:
                  # ... (logging and saving logic)
  ```

  *注：`CosineAnnealingLR`比`CosineAnnealingWarmRestarts`更简单，适合初步修复。*

**任务 1.4: 增强评估函数的鲁棒性**

* **文件路径**: `src/evaluation/metrics.py`

* **目标函数**: `evaluate_metrics` 和 `evaluate_horizons`

* **问题描述**:

  1.  `Pearson R` 的计算方式在当前输入维度下存在逻辑冗余。
  2.  `overflow` 警告表明有非有限值（`Inf`/`NaN`）传入，导致计算崩溃。需要增加防御性检查。

* **修改指令**:

  1.  在 `evaluate_metrics` 函数中，修改 `pearsonr` 的计算，直接对展平后的一维数组进行。
  2.  在 `evaluate_horizons` 函数的开头，对传入的 `y_pred_horizons_scaled` 进行 `np.isfinite` 检查。如果发现非有限值，打印警告并将其替换为0，以保证后续计算能继续进行。

* **代码修改 (上下文展示)**:

  ```python
  # File: src/evaluation/metrics.py
  
  def evaluate_metrics(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, scaler) -> dict:
      # ... (code for inverse transform)
  
      # --- START MODIFICATION 1.4.1 ---
      # REASON: The original Pearson R calculation was flawed for single-feature predictions.
      #         This version correctly calculates it on the flattened data arrays.
      # OLD Pearson R block:
      # pearson_coeffs = []
      # for i in range(y_true_unscaled.shape[1]): ...
      # avg_pearson_r = np.mean(pearson_coeffs)
  
      # NEW Pearson R calculation:
      # Ensure arrays are 1D for pearsonr
      y_true_flat = y_true_unscaled.ravel()
      y_pred_flat = y_pred_unscaled.ravel()
      if np.std(y_true_flat) > 0 and np.std(y_pred_flat) > 0:
          avg_pearson_r = pearsonr(y_true_flat, y_pred_flat)[0]
      else:
          avg_pearson_r = 0.0
      # --- END MODIFICATION 1.4.1 ---
  
      metrics = {
          'mae': mae,
          'rmse': rmse,
          'r2_score': r2,
          'pearson_r': avg_pearson_r
      }
      return metrics
  
  def evaluate_horizons(y_true_horizons_scaled: np.ndarray, y_pred_horizons_scaled: np.ndarray, target_scaler_path: str = None) -> dict:
      logging.info("Evaluating metrics across all horizons...")
  
      # --- START MODIFICATION 1.4.2 ---
      # REASON: Adds a defensive check to handle non-finite values (inf/nan) that
      #         were causing the 'overflow' warning. This prevents crashes and
      #         provides clear logging when numerical instability occurs.
      if not np.all(np.isfinite(y_pred_horizons_scaled)):
          num_non_finite = np.sum(~np.isfinite(y_pred_horizons_scaled))
          logging.warning(
              f"Overflow Guard: Found {num_non_finite} non-finite values in predictions. "
              "Clamping them to 0 for metric calculation."
          )
          y_pred_horizons_scaled = np.nan_to_num(y_pred_horizons_scaled, nan=0.0, posinf=0.0, neginf=0.0)
      # --- END MODIFICATION 1.4.2 ---
  
      # Load scaler if provided
      # ... (rest of the function)
  ```

---
