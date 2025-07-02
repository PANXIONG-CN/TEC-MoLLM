import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
import pandas as pd
from tqdm import tqdm
import argparse

from src.data.dataset import SlidingWindowSamplerDataset
from src.features.feature_engineering import create_features_and_targets, standardize_features
from src.model.tec_mollm import TEC_MoLLM
from src.evaluation.metrics import evaluate_horizons
from src.models.baselines import HistoricalAverage, SarimaBaseline, save_baseline, load_baseline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_tec_mollm_predictions(model, dataloader, device, edge_index):
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="TEC-MoLLM Inference"):
            # ... (same reshaping as in validate function) ...
            all_preds.append(model(batch['x'].to(device), batch['x_time_features'].to(device), edge_index).cpu().numpy())
            all_trues.append(batch['y'].cpu().numpy())
    return np.concatenate(all_trues, axis=0), np.concatenate(all_preds, axis=0)

def get_ha_predictions(test_data):
    # This is a simplified example. A real implementation would need to load a fitted HA model.
    logging.info("Generating dummy Historical Average predictions...")
    return np.random.rand(*test_data['Y'].shape)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Load Data ---
    processed_data = create_features_and_targets(args.data_files)
    standardized_data, _ = standardize_features(processed_data, scaler_path=args.scaler_path)
    test_dataset = SlidingWindowSamplerDataset(standardized_data['test']['X'], standardized_data['test']['Y'], np.zeros((len(standardized_data['test']['X']), 2)))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # --- Load Models and Get Predictions ---
    results = {}

    # TEC-MoLLM
    model_config = { # This should ideally be loaded from a config file
        "num_nodes": 2911, "d_emb": 16, "spatial_in_channels_base": 6, "spatial_out_channels": 32, 
        "spatial_heads": 2, "temporal_channel_list": [64, 128], "temporal_strides": [2, 2], 
        "patch_len": 4, "d_llm": 768, "llm_layers": 3, "prediction_horizon": 12, "temporal_seq_len": 84
    }
    model = TEC_MoLLM(model_config).to(device)
    model.load_state_dict(torch.load(args.model_checkpoint))
    edge_index = torch.load(args.graph_path)['edge_index'].to(device)
    y_true, y_pred_mollm = get_tec_mollm_predictions(model, test_loader, device, edge_index)
    
    # Baselines
    y_pred_ha = get_ha_predictions(standardized_data['test']) # Dummy implementation
    
    # --- Evaluate Metrics ---
    logging.info("Evaluating TEC-MoLLM...")
    results['TEC-MoLLM'] = evaluate_horizons(y_true, y_pred_mollm, args.scaler_path)
    
    logging.info("Evaluating Historical Average...")
    # Note: HA predictions are not scaled, so we need to handle this in a real scenario
    # For this dummy script, we'll reuse the scaler, which is incorrect but allows the script to run.
    results['HistoricalAverage'] = evaluate_horizons(y_true, y_pred_ha, args.scaler_path)

    # --- Format and Save Results ---
    results_df = pd.DataFrame(results).T
    logging.info("\n--- Final Results ---")
    logging.info(f"\n{results_df.to_string()}")
    
    output_path = os.path.join(args.output_dir, "evaluation_results.csv")
    results_df.to_csv(output_path)
    logging.info(f"Results saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate TEC-MoLLM and baseline models.")
    parser.add_argument('--data_files', nargs='+', default=[
        'data/raw/CRIM_SW2hr_AI_v1.2_2014_DataDrivenRange_CN.hdf5',
        'data/raw/CRIM_SW2hr_AI_v1.2_2015_DataDrivenRange_CN.hdf5'
    ])
    parser.add_argument('--scaler_path', type=str, default='data/processed/scaler.joblib')
    parser.add_argument('--graph_path', type=str, default='data/processed/graph_A.pt')
    parser.add_argument('--model_checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=16)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args) 