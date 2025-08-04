import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
import pandas as pd
from tqdm import tqdm
import argparse
import joblib
from sklearn.preprocessing import StandardScaler

from src.data.dataset import SlidingWindowSamplerDataset
from src.model.tec_mollm import TEC_MoLLM
from src.evaluation.metrics import evaluate_horizons

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- 一次性加载scaler缓存 ---
_cached_target_scaler = None
_cached_feature_scaler = None
_cached_tec_feature_scaler = None
_cached_scaler_paths = {}


def get_cached_scalers(target_scaler_path):
    """缓存scaler，避免重复I/O"""
    global _cached_target_scaler, _cached_feature_scaler, _cached_tec_feature_scaler, _cached_scaler_paths

    if target_scaler_path not in _cached_scaler_paths:
        try:
            # 加载target_scaler
            _cached_target_scaler = joblib.load(target_scaler_path)

            # 加载feature_scaler
            feature_scaler_path = target_scaler_path.replace("target_scaler.joblib", "scaler.joblib")
            _cached_feature_scaler = joblib.load(feature_scaler_path)

            # 创建TEC特征专用scaler
            from sklearn.preprocessing import StandardScaler

            _cached_tec_feature_scaler = StandardScaler()
            _cached_tec_feature_scaler.mean_ = np.array([_cached_feature_scaler.mean_[0]])
            _cached_tec_feature_scaler.scale_ = np.array([_cached_feature_scaler.scale_[0]])

            _cached_scaler_paths[target_scaler_path] = True
            logging.info(f"Cached scalers loaded from {target_scaler_path}")

        except Exception as e:
            logging.error(f"Failed to load scalers: {e}")
            return None, None, None

    return _cached_target_scaler, _cached_feature_scaler, _cached_tec_feature_scaler


def inverse_transform_output(tensor_scaled: torch.Tensor, scaler) -> torch.Tensor:
    """
    辅助函数：将标准化的张量通过scaler进行逆变换

    Args:
        tensor_scaled (torch.Tensor): 标准化的张量，形状为 (B, ..., C) 或 (B, ...)
        scaler: 已拟合的StandardScaler

    Returns:
        torch.Tensor: 逆变换后的张量，保持原始形状
    """
    original_shape = tensor_scaled.shape

    # 如果是单特征张量，需要添加特征维度
    if len(original_shape) == 3:  # (B, H, W) -> (B, H, W, 1)
        tensor_reshaped = tensor_scaled.reshape(-1, 1)
    else:  # (B, H, W, C) -> (B*H*W*C, 1) for single feature
        tensor_reshaped = tensor_scaled.reshape(-1, 1)

    # 转换为numpy进行逆变换
    tensor_numpy = tensor_reshaped.cpu().numpy()
    tensor_unscaled_numpy = scaler.inverse_transform(tensor_numpy)

    # 转换回tensor并恢复原始形状
    tensor_unscaled = torch.from_numpy(tensor_unscaled_numpy).to(tensor_scaled.device)
    tensor_unscaled = tensor_unscaled.reshape(original_shape)

    return tensor_unscaled


def transform_output(tensor_unscaled: torch.Tensor, scaler) -> torch.Tensor:
    """
    辅助函数：将物理值张量通过scaler进行标准化

    Args:
        tensor_unscaled (torch.Tensor): 物理值张量
        scaler: 已拟合的StandardScaler

    Returns:
        torch.Tensor: 标准化后的张量，保持原始形状
    """
    original_shape = tensor_unscaled.shape

    # 重塑为scaler期望的形状
    if len(original_shape) == 3:  # (B, H, W) -> (B, H, W, 1)
        tensor_reshaped = tensor_unscaled.reshape(-1, 1)
    else:  # (B, H, W, C) -> (B*H*W*C, 1) for single feature
        tensor_reshaped = tensor_unscaled.reshape(-1, 1)

    # 转换为numpy进行标准化
    tensor_numpy = tensor_reshaped.cpu().numpy()
    tensor_scaled_numpy = scaler.transform(tensor_numpy)

    # 转换回tensor并恢复原始形状
    tensor_scaled = torch.from_numpy(tensor_scaled_numpy).to(tensor_unscaled.device)
    tensor_scaled = tensor_scaled.reshape(original_shape)

    return tensor_scaled


def get_tec_mollm_predictions(model, dataloader, device, edge_index, target_scaler_path):
    """获取TEC-MoLLM模型的预测结果"""
    model.eval()
    all_preds = []
    all_trues = []

    # --- 使用缓存的scaler，避免重复I/O ---
    target_scaler, feature_scaler, tec_feature_scaler = get_cached_scalers(target_scaler_path)
    if target_scaler is None:
        logging.error("Failed to load scalers")
        return np.array([]), np.array([])
    # --- END MODIFICATION ---

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="TEC-MoLLM Inference"):
            x = batch["x"].to(device)
            y_residual_scaled = batch["y"].to(device)
            time_features = batch["x_time_features"].to(device)

            # 与train.py中validate函数相同的reshape逻辑
            B, L, H, W, C = x.shape
            x = x.view(B, L, H * W, C)
            # time_features shape should be (B, L, N, 2), so we need to expand the spatial dimension
            if time_features.numel() > 0:
                time_features = time_features.unsqueeze(-2).expand(B, L, H * W, -1)  # (B, L, N, 2)

            # 1. Get model's residual prediction (scaled)
            output_residual_scaled = model(x, time_features, edge_index)
            # Reshape target to match output: (B, H, W, L_out) -> (B, L_out, H*W, 1)
            y_reshaped = y_residual_scaled.permute(0, 3, 1, 2).reshape(B, -1, H * W, 1)

            # --- START MODIFICATION: Reconstruct to absolute values and re-scale ---
            # 2. Get persistence baseline (this is the last known TEC value from input X)
            # It's the first channel of the last time step. Shape: (B, H*W, 1)
            persistence_baseline_from_x_scaled = x[:, -1, :, 0].unsqueeze(-1)  # (B, H*W, 1)

            # 3. Inverse transform all parts to get physical values
            # Note: output is (B, L_out, H*W, 1), need to handle properly
            output_residual_unscaled = inverse_transform_output(output_residual_scaled, target_scaler)
            y_residual_unscaled = inverse_transform_output(y_reshaped, target_scaler)

            # Baseline is from X, so use the correct TEC feature scaler
            persistence_baseline_unscaled = inverse_transform_output(persistence_baseline_from_x_scaled, tec_feature_scaler)

            # 4. Reconstruct absolute physical values
            # Expand baseline to match the prediction horizon
            L_out = output_residual_unscaled.shape[1]
            persistence_baseline_expanded = persistence_baseline_unscaled.unsqueeze(1).expand(-1, L_out, -1, -1)

            output_absolute_unscaled = persistence_baseline_expanded + output_residual_unscaled
            target_absolute_unscaled = persistence_baseline_expanded + y_residual_unscaled

            # 5. Re-scale the absolute values to feed into metrics.py
            # Use tec_feature_scaler for absolute values to maintain consistency
            output_final_scaled = transform_output(output_absolute_unscaled, tec_feature_scaler)
            target_final_scaled = transform_output(target_absolute_unscaled, tec_feature_scaler)

            all_preds.append(output_final_scaled.cpu().numpy())
            all_trues.append(target_final_scaled.cpu().numpy())
            # --- END MODIFICATION ---

    return np.concatenate(all_trues, axis=0), np.concatenate(all_preds, axis=0)


def get_baseline_predictions(test_dataset, L_in, L_out):
    """生成简单的基线预测（残差形式）"""
    logging.info("生成持久化模型基线预测（残差形式）...")

    predictions = []

    # 遍历测试数据集样本
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        x_window = sample["x"].numpy()  # (L_in, H, W, C)

        # --- START MODIFICATION: Generate residual baseline predictions ---
        # REASON: Since we're now predicting residuals, the persistence model baseline
        #         should predict zero residuals (future = current, so residual = 0)

        # For persistence model in residual space, the prediction is always 0
        # because persistence assumes future values equal current values
        # So: residual = future - current = current - current = 0

        H, W = x_window.shape[1], x_window.shape[2]
        pred_residual = np.zeros((L_out, H, W, 1))  # All zeros for residual prediction

        # 转换为期望的格式 (H, W, L_out)
        pred_reshaped = pred_residual.transpose(1, 2, 0, 3).squeeze(-1)  # (H, W, L_out)
        predictions.append(pred_reshaped)
        # --- END MODIFICATION ---

    # 转换为最终格式 (num_samples, H, W, L_out)
    predictions = np.array(predictions)

    return predictions


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """查找最新的checkpoint文件"""
    import glob

    pattern = os.path.join(checkpoint_dir, "best_model_*.pth")
    checkpoint_files = glob.glob(pattern)

    if not checkpoint_files:
        # 如果没有找到动态命名的文件，尝试默认文件
        default_path = os.path.join(checkpoint_dir, "best_model.pth")
        if os.path.exists(default_path):
            return default_path
        else:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    # 按修改时间排序，返回最新的
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return latest_checkpoint


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估TEC-MoLLM和基线模型")

    # 数据配置
    parser.add_argument("--L_in", type=int, default=48, help="输入序列长度")
    parser.add_argument("--L_out", type=int, default=12, help="输出序列长度")

    # 模型配置
    parser.add_argument("--d_emb", type=int, default=16, help="嵌入维度")
    parser.add_argument("--llm_layers", type=int, default=3, help="LLM层数")

    # 文件路径
    parser.add_argument("--target_scaler_path", type=str, default="data/processed/target_scaler.joblib")
    parser.add_argument("--graph_path", type=str, default="data/processed/graph_A.pt")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help='Path to model checkpoint. Use "latest" to auto-find the most recent checkpoint.',
    )
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=16)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"使用设备: {device}")
    logging.info(f"测试配置: L_in={args.L_in}, L_out={args.L_out}")

    # --- 验证预处理数据文件存在 ---
    data_dir = "data/processed"
    if not os.path.exists(args.target_scaler_path):
        logging.error(f"Target scaler not found at {args.target_scaler_path}. Please run preprocessing script first.")
        return

    logging.info("加载预处理好的测试数据...")

    # 创建测试数据集（使用预处理好的数据）
    test_dataset = SlidingWindowSamplerDataset(data_path=data_dir, mode="test", L_in=args.L_in, L_out=args.L_out)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logging.info(f"测试数据集大小: {len(test_dataset)} 样本")

    # --- 模型配置 ---
    # 计算卷积后的时间序列长度
    conv_output_len = args.L_in // (2 * 2)  # 两个stride-2的卷积
    patch_len = 4
    num_patches = conv_output_len // patch_len

    if conv_output_len % patch_len != 0:
        patch_len = 2 if conv_output_len % 2 == 0 else 1
        num_patches = conv_output_len // patch_len
        logging.warning(f"调整patch_len为{patch_len}以适应conv_output_len {conv_output_len}")

    model_config = {
        "num_nodes": 2911,
        "d_emb": args.d_emb,
        "spatial_in_channels_base": 6,
        "spatial_out_channels": 11,
        "spatial_heads": 2,
        "temporal_channel_list": [64, 128],  # 修改为11*2=22，与输入维度一致用于残差连接
        "temporal_strides": [2, 2],
        "patch_len": patch_len,
        "d_llm": 768,
        "llm_layers": args.llm_layers,
        "prediction_horizon": args.L_out,
        "temporal_seq_len": args.L_in,
        "num_years": 13,
    }

    # --- 加载模型和获取预测 ---
    results = {}

    # TEC-MoLLM预测
    logging.info("加载TEC-MoLLM模型...")
    model = TEC_MoLLM(model_config).to(device)

    # --- START MODIFICATION: Support dynamic checkpoint naming ---
    # 处理"latest"选项或直接使用指定的checkpoint路径
    if args.model_checkpoint == "latest":
        checkpoint_path = find_latest_checkpoint()
        logging.info(f"自动找到最新checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = args.model_checkpoint
        logging.info(f"使用指定checkpoint: {checkpoint_path}")

    # 加载模型权重（处理DDP和torch.compile保存的模型）
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # --- END MODIFICATION ---

    # 创建一个新的state_dict来存储修复后的键
    new_state_dict = {}
    for k, v in checkpoint.items():
        # 移除 'module.' 前缀 (来自DDP)
        if k.startswith("module."):
            k = k[len("module.") :]
        # 移除 '_orig_mod.' 前缀 (来自torch.compile)
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        new_state_dict[k] = v

    # 使用修复后的state_dict加载模型
    model.load_state_dict(new_state_dict)

    edge_index = torch.load(args.graph_path, weights_only=False)["edge_index"].to(device)

    logging.info("获取TEC-MoLLM预测结果...")
    y_true, y_pred_mollm = get_tec_mollm_predictions(model, test_loader, device, edge_index, args.target_scaler_path)

    # 使用已缓存的scaler（在get_tec_mollm_predictions中已加载）
    target_scaler, feature_scaler, tec_feature_scaler = get_cached_scalers(args.target_scaler_path)

    # 基线预测
    logging.info("生成基线预测...")
    y_pred_ha = get_baseline_predictions(test_dataset, args.L_in, args.L_out)

    # --- START MODIFICATION: Process baseline predictions consistently ---
    # REASON: Need to ensure baseline predictions are in the same scale as TEC-MoLLM predictions
    #         for fair comparison. Since TEC-MoLLM predictions are already processed to absolute
    #         values and re-scaled, we need to do the same for baseline.

    # Need to align dimensions: y_pred_ha is currently residuals (zeros)
    # We need to convert it to absolute values by adding persistence baseline
    logging.info("处理基线预测以保持一致性...")

    # 使用已缓存的scaler
    target_scaler, _, _ = get_cached_scalers(args.target_scaler_path)

    baseline_absolute_predictions = []
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        x_window = sample["x"].numpy()  # (L_in, H, W, C)

        # Get the last known TEC value (persistence baseline)
        last_tec_value = x_window[-1, :, :, 0]  # (H, W)

        # Create persistence prediction (repeat last value for L_out steps)
        persistence_pred = np.tile(last_tec_value[:, :, np.newaxis], (1, 1, args.L_out))  # (H, W, L_out)

        baseline_absolute_predictions.append(persistence_pred)

    y_pred_ha_absolute = np.array(baseline_absolute_predictions)  # (num_samples, H, W, L_out)

    # Convert to same format as TEC-MoLLM predictions: (num_samples, L_out, H*W, 1)
    B, H, W, L_out = y_pred_ha_absolute.shape
    y_pred_ha_reshaped = y_pred_ha_absolute.transpose(0, 3, 1, 2).reshape(B, L_out, H * W, 1)

    # Apply TEC feature scaler normalization to match TEC-MoLLM prediction format
    y_pred_ha_scaled = transform_output(torch.from_numpy(y_pred_ha_reshaped).float(), tec_feature_scaler).numpy()
    # --- END MODIFICATION ---

    # 确保基线预测的样本数与真实值匹配
    min_samples = min(len(y_true), len(y_pred_ha_scaled))
    y_true_matched = y_true[:min_samples]
    y_pred_ha_matched = y_pred_ha_scaled[:min_samples]

    # --- 评估指标 ---
    # Save temporary TEC scaler for evaluation
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix="_tec_scaler.joblib") as temp_file:
        temp_scaler_path = temp_file.name
        joblib.dump(tec_feature_scaler, temp_scaler_path)

    logging.info("评估TEC-MoLLM模型...")
    results["TEC-MoLLM"] = evaluate_horizons(y_true_matched, y_pred_mollm[:min_samples], temp_scaler_path)

    logging.info("评估历史平均基线...")
    results["HistoricalAverage"] = evaluate_horizons(y_true_matched, y_pred_ha_matched, temp_scaler_path)

    # Clean up temporary file
    os.unlink(temp_scaler_path)

    # --- 格式化和保存结果 ---
    results_df = pd.DataFrame(results).T

    # 添加详细的结果展示
    logging.info("=" * 80)
    logging.info("最终评估结果")
    logging.info("=" * 80)

    for model_name, metrics in results.items():
        logging.info(f"\n📊 {model_name} 模型结果:")
        logging.info(f"  平均MAE:  {metrics['mae_avg']:.6f}")
        logging.info(f"  平均RMSE: {metrics['rmse_avg']:.6f}")
        logging.info(f"  平均R²:   {metrics['r2_score_avg']:.6f}")
        logging.info(f"  平均Pearson R: {metrics['pearson_r_avg']:.6f}")

        if "mae_by_horizon" in metrics:
            logging.info("  各预测时步详细指标:")
            for h in range(len(metrics["mae_by_horizon"])):
                logging.info(
                    f"    时步{h+1:2d}: MAE={metrics['mae_by_horizon'][h]:.6f}, "
                    f"RMSE={metrics['rmse_by_horizon'][h]:.6f}, "
                    f"R²={metrics['r2_by_horizon'][h]:.6f}, "
                    f"Pearson R={metrics['pearson_by_horizon'][h]:.6f}"
                )

    # 计算改进百分比
    if "TEC-MoLLM" in results and "HistoricalAverage" in results:
        logging.info("\n🎯 TEC-MoLLM相对于历史平均的改进:")
        tec_metrics = results["TEC-MoLLM"]
        ha_metrics = results["HistoricalAverage"]

        mae_improvement = ((ha_metrics["mae_avg"] - tec_metrics["mae_avg"]) / ha_metrics["mae_avg"]) * 100
        rmse_improvement = ((ha_metrics["rmse_avg"] - tec_metrics["rmse_avg"]) / ha_metrics["rmse_avg"]) * 100
        r2_improvement = ((tec_metrics["r2_score_avg"] - ha_metrics["r2_score_avg"]) / abs(ha_metrics["r2_score_avg"])) * 100
        pearson_improvement = ((tec_metrics["pearson_r_avg"] - ha_metrics["pearson_r_avg"]) / ha_metrics["pearson_r_avg"]) * 100

        logging.info(f"  MAE改进:     {mae_improvement:+.2f}%")
        logging.info(f"  RMSE改进:    {rmse_improvement:+.2f}%")
        logging.info(f"  R²改进:      {r2_improvement:+.2f}%")
        logging.info(f"  Pearson R改进: {pearson_improvement:+.2f}%")

    logging.info("=" * 80)

    # 保存结果
    output_path = os.path.join(args.output_dir, "evaluation_results.csv")
    results_df.to_csv(output_path)
    logging.info(f"详细结果已保存到: {output_path}")

    # 保存简要结果摘要
    summary_path = os.path.join(args.output_dir, "evaluation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("TEC-MoLLM模型评估结果摘要\n")
        f.write("=" * 50 + "\n\n")

        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  平均MAE:  {metrics['mae_avg']:.6f}\n")
            f.write(f"  平均RMSE: {metrics['rmse_avg']:.6f}\n")
            f.write(f"  平均R²:   {metrics['r2_score_avg']:.6f}\n")
            f.write(f"  平均Pearson R: {metrics['pearson_r_avg']:.6f}\n\n")

    logging.info(f"结果摘要已保存到: {summary_path}")


if __name__ == "__main__":
    main()
