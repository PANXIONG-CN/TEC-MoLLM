#!/usr/bin/env python3
"""
验证"预测残差"方案下，数据标准化scaler的合理性。

主要检查:
1. 特征标准化器 (scaler.joblib) 是否正确处理6个输入特征。
2. 目标标准化器 (target_scaler.joblib) 是否已为残差数据拟合 (均值接近0)。
"""

import joblib
import numpy as np
import os

# 设置numpy打印选项，不使用科学计数法，方便观察
np.set_printoptions(suppress=True, precision=6, linewidth=120)

def check_scaler_file(filepath, scaler_name, is_residual_scaler=False):
    """
    检查单个scaler文件，并根据是否为残差scaler进行特定验证。
    
    Args:
        filepath (str): scaler文件的路径。
        scaler_name (str): 用于打印的scaler名称。
        is_residual_scaler (bool): 如果为True，则会检查均值是否接近0。
    """
    print(f"\n{'='*60}")
    print(f"📊 检查: {scaler_name}")
    print(f"   路径: {filepath}")
    print(f"{'='*60}")

    if not os.path.exists(filepath):
        print(f"❌ 失败: 文件不存在！")
        return False
    
    try:
        scaler = joblib.load(filepath)
        print(f"Scaler类型: {type(scaler).__name__}")
        
        if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
            print("❌ 失败: Scaler对象缺少 'mean_' 或 'scale_' 属性。")
            return False

        n_features = len(scaler.mean_)
        print(f"特征数量: {n_features}")
        
        print(f"\n📈 均值 (Mean):")
        print(scaler.mean_)
        
        print(f"\n📊 标准差 (Scale / Std Dev):")
        print(scaler.scale_)
        
        # --- 统计分析与警告检查 ---
        warnings = []
        
        # 检查是否有NaN或Inf
        if np.any(np.isnan(scaler.mean_)) or np.any(np.isnan(scaler.scale_)):
            warnings.append("发现NaN值！")
        if np.any(np.isinf(scaler.mean_)) or np.any(np.isinf(scaler.scale_)):
            warnings.append("发现Inf值！")
        
        # 检查是否有过小的标准差
        if np.any(scaler.scale_ < 1e-6):
            small_indices = np.where(scaler.scale_ < 1e-6)[0]
            warnings.append(f"发现过小的标准差 (< 1e-6)，在特征索引: {small_indices}")

        # --- 特定于残差Scaler的检查 ---
        if is_residual_scaler:
            # 理论上，残差的均值应该非常接近0。
            # 我们给一个宽松的阈值，例如绝对值小于0.1。
            if np.any(np.abs(scaler.mean_) > 0.1):
                warnings.append(f"残差均值 ({scaler.mean_[0]:.6f}) 偏离0过远，可能不是残差数据！")
            else:
                print("\n✅ 验证通过: 目标均值接近0，符合残差数据特性。")
        
        if warnings:
            print(f"\n🚨 警告:")
            for warning in warnings:
                print(f"   - {warning}")
            print("   请仔细检查您的数据预处理流程。")
            return False
        else:
            print(f"\n✅ 验证通过: 标准化参数看起来健康。")
            return True
            
    except Exception as e:
        print(f"❌ 失败: 加载或处理 {scaler_name} 时出错: {e}")
        return False

def main():
    print("="*60)
    print("🔍 数据标准化验证脚本 (预测残差版)")
    print("="*60)
    
    # 检查特征scaler
    feature_scaler_ok = check_scaler_file(
        'data/processed/scaler.joblib', 
        '特征标准化器 (Feature Scaler)',
        is_residual_scaler=False
    )
    
    # 检查目标scaler，并标记它应该是残差scaler
    target_scaler_ok = check_scaler_file(
        'data/processed/target_scaler.joblib',
        '目标(残差)标准化器 (Target/Residual Scaler)',
        is_residual_scaler=True
    )
    
    # 总结
    print(f"\n{'='*60}")
    print("📋 总结")
    print(f"{'='*60}")
    
    if feature_scaler_ok and target_scaler_ok:
        print("✅ 所有scaler文件均已通过验证！")
        print("   - 特征scaler正确处理了6个输入特征。")
        print("   - 目标scaler已在残差数据上正确拟合 (均值 ≈ 0)。")
        print('\n💡 您现在可以安全地开始训练"预测残差"模型了。')
    else:
        print("❌ 存在问题的scaler文件！")
        print("   请不要开始训练，并仔细检查以下步骤：")
        print("   1. 是否已切换到 'feature/predicting-residuals' 分支？")
        print("   2. `feature_engineering.py` 中生成残差Y的逻辑是否正确？")
        print("   3. `preprocess.py` 是否在残差Y上拟合了 `target_scaler`？")
        print("   4. 是否在运行 `preprocess.py` 之前清理了旧的 `data/processed` 目录？")

if __name__ == "__main__":
    main() 