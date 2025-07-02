import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
import joblib
from tqdm import tqdm

from src.data.dataset import SlidingWindowSamplerDataset
from src.features.feature_engineering import create_features_and_targets, standardize_features
from src.model.tec_mollm import TEC_MoLLM
from src.evaluation.metrics import evaluate_horizons

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, edge_index):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        time_features = batch['x_time_features'].to(device)
        
        # Reshape for model input
        B, L, H, W, C = x.shape
        x = x.view(B, L, H * W, C)
        # time_features shape should be (B, L, N, 2), so we need to expand the spatial dimension
        if time_features.numel() > 0:
            time_features = time_features.unsqueeze(-2).expand(B, L, H * W, -1)  # (B, L, N, 2)

        output = model(x, time_features, edge_index)
        # Reshape target to match output: (B, H, W, L_out) -> (B, L_out, H*W, 1)
        y_reshaped = y.permute(0, 3, 1, 2).reshape(B, -1, H * W, 1)
        loss = loss_fn(output, y_reshaped)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, loss_fn, device, edge_index, scaler_path, target_scaler_path):
    model.eval()
    total_loss = 0
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            time_features = batch['x_time_features'].to(device)

            B, L, H, W, C = x.shape
            x = x.view(B, L, H * W, C)
            # time_features shape should be (B, L, N, 2), so we need to expand the spatial dimension
            if time_features.numel() > 0:
                time_features = time_features.unsqueeze(-2).expand(B, L, H * W, -1)  # (B, L, N, 2)
            
            output = model(x, time_features, edge_index)
            # Reshape target to match output: (B, H, W, L_out) -> (B, L_out, H*W, 1)
            y_reshaped = y.permute(0, 3, 1, 2).reshape(B, -1, H * W, 1)
            loss = loss_fn(output, y_reshaped)
            total_loss += loss.item()
            
            all_preds.append(output.cpu().numpy())
            all_trues.append(y_reshaped.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    
    # Concatenate all batches
    y_true_horizons = np.concatenate(all_trues, axis=0)
    y_pred_horizons = np.concatenate(all_preds, axis=0)
    
    metrics = evaluate_horizons(y_true_horizons, y_pred_horizons, target_scaler_path)
    return avg_loss, metrics

def main():
    # --- Config ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    file_paths = [
        'data/raw/CRIM_SW2hr_AI_v1.2_2014_DataDrivenRange_CN.hdf5',
        'data/raw/CRIM_SW2hr_AI_v1.2_2015_DataDrivenRange_CN.hdf5'
    ]
    scaler_path = 'data/processed/scaler.joblib'
    graph_path = 'data/processed/graph_A.pt'
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_config = {
        "num_nodes": 2911, "d_emb": 16, "spatial_in_channels_base": 6,
        "spatial_out_channels": 32, "spatial_heads": 2, "temporal_channel_list": [64, 128],
        "temporal_strides": [2, 2], "patch_len": 2, "d_llm": 768, "llm_layers": 3,  # Changed patch_len from 4 to 2
        "prediction_horizon": 12, "temporal_seq_len": 24  # Updated to match L_in
    }
    
    # --- Data Loading and Processing ---
    processed_data = create_features_and_targets(file_paths)
    standardized_data, _ = standardize_features(processed_data, scaler_path=scaler_path)
    
    # Create separate scaler for TEC target data (only first feature is TEC)
    target_scaler_path = 'data/processed/target_scaler.joblib'
    from sklearn.preprocessing import StandardScaler
    target_scaler = StandardScaler()
    # Extract only TEC data (first feature) from training data for target scaling
    train_tec_data = processed_data['train']['X'][:, :, :, 0:1]  # Only TEC feature
    train_tec_reshaped = train_tec_data.reshape(-1, 1)
    target_scaler.fit(train_tec_reshaped)
    joblib.dump(target_scaler, target_scaler_path)
    edge_index = torch.load(graph_path)['edge_index'].to(device)
    
    # For demonstration, use a small subset of data with reasonable window size
    train_data = {k: v[:500] for k, v in standardized_data['train'].items()}
    val_data = {k: v[:250] for k, v in standardized_data['val'].items()}

    # Use smaller window sizes that make sense for our data size
    L_in = 24  # 24 time steps (2 days if 2-hour intervals)
    L_out = 12  # 12 time steps prediction horizon
    
    # Create proper time features with (hour_of_day, day_of_year) for each time step
    # For now, use dummy values - this should be properly implemented with real time data
    train_time_features = np.zeros((len(train_data['X']), 2))  # (N, 2)
    val_time_features = np.zeros((len(val_data['X']), 2))      # (N, 2)
    
    train_dataset = SlidingWindowSamplerDataset(train_data['X'], train_data['Y'], train_time_features, L_in=L_in, L_out=L_out)
    val_dataset = SlidingWindowSamplerDataset(val_data['X'], val_data['Y'], val_time_features, L_in=L_in, L_out=L_out)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # --- Model, Optimizer, Loss ---
    model = TEC_MoLLM(model_config).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    # --- Training Loop ---
    best_val_loss = float('inf')
    num_epochs = 3
    
    for epoch in range(num_epochs):
        logging.info(f"--- Epoch {epoch+1}/{num_epochs} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, edge_index)
        val_loss, val_metrics = validate(model, val_loader, loss_fn, device, edge_index, scaler_path, target_scaler_path)
        
        logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logging.info(f"Val Metrics: {val_metrics}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"New best model saved to {checkpoint_path}")

if __name__ == '__main__':
    main() 