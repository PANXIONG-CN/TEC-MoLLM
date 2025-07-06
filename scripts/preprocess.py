import torch
import numpy as np
import logging
import os
import joblib
from sklearn.preprocessing import StandardScaler

# 为脚本添加项目根目录到sys.path，以便能够导入src中的模块
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.feature_engineering import create_features_and_targets, standardize_features

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    主函数，执行离线数据预处理的完整流程。
    """
    logging.info("--- 开始执行离线数据预处理脚本 ---")

    # 定义输入和输出路径
    # 根据PRD和现有代码，我们处理2014和2015年的数据
    file_paths = [
        'data/raw/CRIM_SW2hr_AI_v1.2_2014_DataDrivenRange_CN.hdf5',
        'data/raw/CRIM_SW2hr_AI_v1.2_2015_DataDrivenRange_CN.hdf5'
    ]
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"输出目录 '{output_dir}' 已确认存在。")

    # --- 1. 创建特征和目标 ---
    # 这个函数内部已经包含了数据加载、拆分和特征工程
    # PRD中默认的L_out是12，这里保持一致
    processed_splits = create_features_and_targets(file_paths, horizon=12)
    if not processed_splits:
        logging.error("创建特征和目标失败。正在中止脚本。")
        return

    # --- 2. 标准化特征 (X) ---
    # 这个函数会拟合并保存scaler.joblib，然后返回标准化后的数据
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    standardized_splits, _ = standardize_features(processed_splits, scaler_path=scaler_path)

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

    # --- 4. 保存处理好的数据为 .pt 文件 ---
    for split_name in ['train', 'val', 'test']:
        logging.info(f"--- 正在处理和保存 '{split_name}' 数据集 ---")
        
        # 从standardized_splits获取标准化的X
        # 从processed_splits获取原始的Y和time_features
        # 注意：standardized_splits只包含X和Y，所以time_features从原始的processed_splits中获取
        standardized_x = standardized_splits[split_name]['X']
        original_y = processed_splits[split_name]['Y']
        time_features = processed_splits[split_name]['time_features']
        
        # --- 新增：对Y进行标准化 ---
        logging.info(f"对 '{split_name}' 的Y进行标准化...")
        # Y的形状是 (N, H, W, L_out)，scaler期望 (n_samples, n_features)
        # target_scaler是为单特征TEC拟合的，所以reshape为 (N*H*W*L_out, 1)
        y_reshaped = original_y.reshape(-1, 1) 
        y_scaled_reshaped = target_scaler.transform(y_reshaped)
        y_scaled = y_scaled_reshaped.reshape(original_y.shape)
        logging.info(f"'{split_name}' 的Y标准化完成。")
        # --- 新增结束 ---
        
        # 将numpy数组转换为torch张量
        x_tensor = torch.from_numpy(standardized_x).float()
        y_tensor = torch.from_numpy(y_scaled).float()  # <--- Y现在是标准化尺度
        tf_tensor = torch.from_numpy(time_features).float()

        logging.info(f"'{split_name}' 张量形状: X={x_tensor.shape}, Y={y_tensor.shape}, time_features={tf_tensor.shape}")

        # 组合成一个字典
        data_to_save = {
            'X': x_tensor,
            'Y': y_tensor,
            'time_features': tf_tensor
        }
        
        # 定义输出文件路径
        output_filepath = os.path.join(output_dir, f"{split_name}_set.pt")
        
        # 保存到文件
        torch.save(data_to_save, output_filepath)
        logging.info(f"'{split_name}' 数据集已成功保存至: {output_filepath}")

    logging.info("--- 离线数据预处理脚本执行完毕 ---")

if __name__ == '__main__':
    main() 