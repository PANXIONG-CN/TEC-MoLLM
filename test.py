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
from src.features.feature_engineering import create_features_and_targets, standardize_features
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

def get_baseline_predictions(test_data, L_in, L_out):
    """生成简单的基线预测"""
    logging.info("生成历史平均基线预测...")
    
    # 获取测试数据形状
    B, H, W, features = test_data['X'].shape
    
    # 简单的历史平均：使用输入序列最后一个时间步的平均值作为所有未来预测
    # 这里我们需要创建滑动窗口样本来匹配数据格式
    predictions = []
    
    # 遍历可能的滑动窗口
    for i in range(len(test_data['X']) - L_in - L_out + 1):
        # 取输入序列的历史数据 (L_in, H, W, features)
        input_seq = test_data['X'][i:i+L_in]
        
        # 计算历史平均值 (对时间维度求平均)
        # 只使用TEC特征（第0个特征）进行预测
        historical_avg = np.mean(input_seq[:, :, :, 0:1], axis=0, keepdims=True)  # (1, H, W, 1)
        
        # 重复L_out次作为未来预测
        pred = np.repeat(historical_avg, L_out, axis=0)  # (L_out, H, W, 1)
        predictions.append(pred)
    
    # 转换为与目标相同的格式 (num_samples, H, W, L_out)
    predictions = np.array(predictions)  # (num_samples, L_out, H, W, 1)
    predictions = predictions.transpose(0, 2, 3, 1, 4).squeeze(-1)  # (num_samples, H, W, L_out)
    
    return predictions

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估TEC-MoLLM和基线模型")
    
    # 数据配置
    parser.add_argument('--L_in', type=int, default=48, help='输入序列长度')
    parser.add_argument('--L_out', type=int, default=12, help='输出序列长度')
    parser.add_argument('--data_files', nargs='+', default=[
        'data/raw/CRIM_SW2hr_AI_v1.2_2014_DataDrivenRange_CN.hdf5',
        'data/raw/CRIM_SW2hr_AI_v1.2_2015_DataDrivenRange_CN.hdf5'
    ])
    
    # 模型配置
    parser.add_argument('--d_emb', type=int, default=16, help='嵌入维度')
    parser.add_argument('--llm_layers', type=int, default=3, help='LLM层数')
    
    # 文件路径
    parser.add_argument('--scaler_path', type=str, default='data/processed/scaler.joblib')
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
    
    # --- 加载和预处理数据 ---
    logging.info("加载和预处理数据...")
    processed_data = create_features_and_targets(args.data_files)
    standardized_data, _ = standardize_features(processed_data, scaler_path=args.scaler_path)
    
    # 创建目标数据的独立缩放器（如果不存在）
    if not os.path.exists(args.target_scaler_path):
        logging.info("创建目标数据缩放器...")
        target_scaler = StandardScaler()
        train_tec_data = processed_data['train']['X'][:, :, :, 0:1]
        train_tec_reshaped = train_tec_data.reshape(-1, 1)
        target_scaler.fit(train_tec_reshaped)
        joblib.dump(target_scaler, args.target_scaler_path)
    
    # 使用真实的时间特征
    test_time_features = processed_data['test']['time_features']
    
    # 创建测试数据集
    test_dataset = SlidingWindowSamplerDataset(
        standardized_data['test']['X'], 
        standardized_data['test']['Y'], 
        test_time_features, 
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
    
    # 加载模型权重（处理DDP保存的模型）
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'module.' in list(checkpoint.keys())[0]:
        # 如果是DDP保存的模型，去除'module.'前缀
        new_checkpoint = {}
        for k, v in checkpoint.items():
            new_checkpoint[k.replace('module.', '')] = v
        checkpoint = new_checkpoint
    
    model.load_state_dict(checkpoint)
    edge_index = torch.load(args.graph_path)['edge_index'].to(device)
    
    logging.info("获取TEC-MoLLM预测结果...")
    y_true, y_pred_mollm = get_tec_mollm_predictions(model, test_loader, device, edge_index)
    
    # 基线预测
    logging.info("生成基线预测...")
    y_pred_ha = get_baseline_predictions(standardized_data['test'], args.L_in, args.L_out)
    
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