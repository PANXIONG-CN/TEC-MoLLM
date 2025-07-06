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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_tec_mollm_predictions(model, dataloader, device, edge_index):
    """获取TEC-MoLLM模型的预测结果"""
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="TEC-MoLLM Inference"):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            time_features = batch['x_time_features'].to(device)
            
            # 与train.py中validate函数相同的reshape逻辑
            B, L, H, W, C = x.shape
            x = x.view(B, L, H * W, C)
            # time_features shape should be (B, L, N, 2), so we need to expand the spatial dimension
            if time_features.numel() > 0:
                time_features = time_features.unsqueeze(-2).expand(B, L, H * W, -1)  # (B, L, N, 2)
            
            output = model(x, time_features, edge_index)
            # Reshape target to match output: (B, H, W, L_out) -> (B, L_out, H*W, 1)
            y_reshaped = y.permute(0, 3, 1, 2).reshape(B, -1, H * W, 1)
            
            all_preds.append(output.cpu().numpy())
            all_trues.append(y_reshaped.cpu().numpy())
    
    return np.concatenate(all_trues, axis=0), np.concatenate(all_preds, axis=0)

def get_baseline_predictions(test_dataset, L_in, L_out):
    """生成简单的基线预测"""
    logging.info("生成历史平均基线预测...")
    
    predictions = []
    
    # 遍历测试数据集样本
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        x_window = sample['x'].numpy()  # (L_in, H, W, C)
        
        # 计算历史平均值 (对时间维度求平均)
        # 只使用TEC特征（第0个特征）进行预测
        historical_avg = np.mean(x_window[:, :, :, 0:1], axis=0, keepdims=True)  # (1, H, W, 1)
        
        # 重复L_out次作为未来预测
        pred = np.repeat(historical_avg, L_out, axis=0)  # (L_out, H, W, 1)
        
        # 转换为期望的格式 (H, W, L_out)
        pred_reshaped = pred.transpose(1, 2, 0, 3).squeeze(-1)  # (H, W, L_out)
        predictions.append(pred_reshaped)
    
    # 转换为最终格式 (num_samples, H, W, L_out)
    predictions = np.array(predictions)
    
    return predictions

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估TEC-MoLLM和基线模型")
    
    # 数据配置
    parser.add_argument('--L_in', type=int, default=48, help='输入序列长度')
    parser.add_argument('--L_out', type=int, default=12, help='输出序列长度')
    
    # 模型配置
    parser.add_argument('--d_emb', type=int, default=16, help='嵌入维度')
    parser.add_argument('--llm_layers', type=int, default=3, help='LLM层数')
    
    # 文件路径
    parser.add_argument('--target_scaler_path', type=str, default='data/processed/target_scaler.joblib')
    parser.add_argument('--graph_path', type=str, default='data/processed/graph_A.pt')
    parser.add_argument('--model_checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=16)
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"使用设备: {device}")
    logging.info(f"测试配置: L_in={args.L_in}, L_out={args.L_out}")
    
    # --- 验证预处理数据文件存在 ---
    data_dir = 'data/processed'
    if not os.path.exists(args.target_scaler_path):
        logging.error(f"Target scaler not found at {args.target_scaler_path}. Please run preprocessing script first.")
        return
    
    logging.info("加载预处理好的测试数据...")
    
    # 创建测试数据集（使用预处理好的数据）
    test_dataset = SlidingWindowSamplerDataset(
        data_path=data_dir, 
        mode='test',
        L_in=args.L_in, 
        L_out=args.L_out
    )
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
        "num_nodes": 2911, "d_emb": args.d_emb, "spatial_in_channels_base": 6,
        "spatial_out_channels": 32, "spatial_heads": 2, "temporal_channel_list": [64, 128],
        "temporal_strides": [2, 2], "patch_len": patch_len, "d_llm": 768, 
        "llm_layers": args.llm_layers, "prediction_horizon": args.L_out, 
        "temporal_seq_len": args.L_in
    }
    
    # --- 加载模型和获取预测 ---
    results = {}
    
    # TEC-MoLLM预测
    logging.info("加载TEC-MoLLM模型...")
    model = TEC_MoLLM(model_config).to(device)
    
    # 加载模型权重（处理DDP和torch.compile保存的模型）
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    
    # 创建一个新的state_dict来存储修复后的键
    new_state_dict = {}
    for k, v in checkpoint.items():
        # 移除 'module.' 前缀 (来自DDP)
        if k.startswith('module.'):
            k = k[len('module.'):]
        # 移除 '_orig_mod.' 前缀 (来自torch.compile)
        if k.startswith('_orig_mod.'):
            k = k[len('_orig_mod.'):]
        new_state_dict[k] = v
    
    # 使用修复后的state_dict加载模型
    model.load_state_dict(new_state_dict)
    
    edge_index = torch.load(args.graph_path)['edge_index'].to(device)
    
    logging.info("获取TEC-MoLLM预测结果...")
    y_true, y_pred_mollm = get_tec_mollm_predictions(model, test_loader, device, edge_index)
    
    # 基线预测
    logging.info("生成基线预测...")
    y_pred_ha = get_baseline_predictions(test_dataset, args.L_in, args.L_out)
    
    # 需要将基线预测重塑为与y_true相同的格式
    # y_true: (num_samples, L_out, H*W, 1)
    # y_pred_ha: (num_samples, H, W, L_out)
    B, H, W, L_out = y_pred_ha.shape
    y_pred_ha_reshaped = y_pred_ha.transpose(0, 3, 1, 2).reshape(B, L_out, H*W, 1)
    
    # 确保基线预测的样本数与真实值匹配
    min_samples = min(len(y_true), len(y_pred_ha_reshaped))
    y_true_matched = y_true[:min_samples]
    y_pred_ha_matched = y_pred_ha_reshaped[:min_samples]
    
    # --- 评估指标 ---
    logging.info("评估TEC-MoLLM模型...")
    results['TEC-MoLLM'] = evaluate_horizons(y_true_matched, y_pred_mollm[:min_samples], args.target_scaler_path)
    
    logging.info("评估历史平均基线...")
    results['HistoricalAverage'] = evaluate_horizons(y_true_matched, y_pred_ha_matched, args.target_scaler_path)

    # --- 格式化和保存结果 ---
    results_df = pd.DataFrame(results).T
    
    # 添加详细的结果展示
    logging.info("="*80)
    logging.info("最终评估结果")
    logging.info("="*80)
    
    for model_name, metrics in results.items():
        logging.info(f"\n📊 {model_name} 模型结果:")
        logging.info(f"  平均MAE:  {metrics['mae_avg']:.6f}")
        logging.info(f"  平均RMSE: {metrics['rmse_avg']:.6f}")
        logging.info(f"  平均R²:   {metrics['r2_score_avg']:.6f}")
        logging.info(f"  平均Pearson R: {metrics['pearson_r_avg']:.6f}")
        
        if 'mae_by_horizon' in metrics:
            logging.info("  各预测时步详细指标:")
            for h in range(len(metrics['mae_by_horizon'])):
                logging.info(f"    时步{h+1:2d}: MAE={metrics['mae_by_horizon'][h]:.6f}, "
                           f"RMSE={metrics['rmse_by_horizon'][h]:.6f}, "
                           f"R²={metrics['r2_by_horizon'][h]:.6f}, "
                           f"Pearson R={metrics['pearson_by_horizon'][h]:.6f}")
    
    # 计算改进百分比
    if 'TEC-MoLLM' in results and 'HistoricalAverage' in results:
        logging.info("\n🎯 TEC-MoLLM相对于历史平均的改进:")
        tec_metrics = results['TEC-MoLLM']
        ha_metrics = results['HistoricalAverage']
        
        mae_improvement = ((ha_metrics['mae_avg'] - tec_metrics['mae_avg']) / ha_metrics['mae_avg']) * 100
        rmse_improvement = ((ha_metrics['rmse_avg'] - tec_metrics['rmse_avg']) / ha_metrics['rmse_avg']) * 100
        r2_improvement = ((tec_metrics['r2_score_avg'] - ha_metrics['r2_score_avg']) / abs(ha_metrics['r2_score_avg'])) * 100
        pearson_improvement = ((tec_metrics['pearson_r_avg'] - ha_metrics['pearson_r_avg']) / ha_metrics['pearson_r_avg']) * 100
        
        logging.info(f"  MAE改进:     {mae_improvement:+.2f}%")
        logging.info(f"  RMSE改进:    {rmse_improvement:+.2f}%") 
        logging.info(f"  R²改进:      {r2_improvement:+.2f}%")
        logging.info(f"  Pearson R改进: {pearson_improvement:+.2f}%")
    
    logging.info("="*80)
    
    # 保存结果
    output_path = os.path.join(args.output_dir, "evaluation_results.csv")
    results_df.to_csv(output_path)
    logging.info(f"详细结果已保存到: {output_path}")
    
    # 保存简要结果摘要
    summary_path = os.path.join(args.output_dir, "evaluation_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("TEC-MoLLM模型评估结果摘要\n")
        f.write("="*50 + "\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  平均MAE:  {metrics['mae_avg']:.6f}\n")
            f.write(f"  平均RMSE: {metrics['rmse_avg']:.6f}\n")
            f.write(f"  平均R²:   {metrics['r2_score_avg']:.6f}\n")
            f.write(f"  平均Pearson R: {metrics['pearson_r_avg']:.6f}\n\n")
    
    logging.info(f"结果摘要已保存到: {summary_path}")

if __name__ == '__main__':
    main() 