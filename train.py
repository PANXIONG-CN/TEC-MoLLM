import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
import joblib
import argparse
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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train TEC-MoLLM model')
    
    # Data configuration
    parser.add_argument('--L_in', type=int, default=336, help='Input sequence length (default: 336 as per PRD)')
    parser.add_argument('--L_out', type=int, default=12, help='Output sequence length (prediction horizon)')
    parser.add_argument('--use_subset', action='store_true', help='Use small data subset for quick testing')
    parser.add_argument('--subset_size', type=int, default=500, help='Size of data subset for training')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    # Model configuration
    parser.add_argument('--d_emb', type=int, default=16, help='Embedding dimension')
    parser.add_argument('--llm_layers', type=int, default=3, help='Number of LLM layers')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # --- Config ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Training configuration: L_in={args.L_in}, L_out={args.L_out}, epochs={args.epochs}")
    
    file_paths = [
        'data/raw/CRIM_SW2hr_AI_v1.2_2014_DataDrivenRange_CN.hdf5',
        'data/raw/CRIM_SW2hr_AI_v1.2_2015_DataDrivenRange_CN.hdf5'
    ]
    scaler_path = 'data/processed/scaler.joblib'
    graph_path = 'data/processed/graph_A.pt'
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Calculate temporal sequence length after convolutions based on L_in
    # L_in -> L_in//2 -> L_in//4 (with strides [2, 2])
    conv_output_len = args.L_in // (2 * 2)  # After two stride-2 convolutions
    patch_len = 4  # Use patch_len=4 for proper LLM input
    num_patches = conv_output_len // patch_len
    
    # Ensure patch_len divides evenly into conv_output_len
    if conv_output_len % patch_len != 0:
        # Adjust patch_len to fit
        patch_len = 2 if conv_output_len % 2 == 0 else 1
        num_patches = conv_output_len // patch_len
        logging.warning(f"Adjusted patch_len to {patch_len} to fit conv_output_len {conv_output_len}")
    
    model_config = {
        "num_nodes": 2911, "d_emb": args.d_emb, "spatial_in_channels_base": 6,
        "spatial_out_channels": 32, "spatial_heads": 2, "temporal_channel_list": [64, 128],
        "temporal_strides": [2, 2], "patch_len": patch_len, "d_llm": 768, "llm_layers": args.llm_layers,
        "prediction_horizon": args.L_out, "temporal_seq_len": args.L_in
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
    
    # Use full dataset or subset based on arguments
    if args.use_subset:
        logging.info(f"Using data subset of size {args.subset_size} for quick testing")
        train_data = {k: v[:args.subset_size] for k, v in standardized_data['train'].items()}
        val_data = {k: v[:args.subset_size//2] for k, v in standardized_data['val'].items()}
    else:
        logging.info("Using full dataset for training")
        train_data = standardized_data['train']
        val_data = standardized_data['val']
    
    # Use real time features extracted from datetime data
    train_time_features = processed_data['train']['time_features']
    val_time_features = processed_data['val']['time_features']
    
    # Apply subset if needed
    if args.use_subset:
        train_time_features = train_time_features[:args.subset_size]
        val_time_features = val_time_features[:args.subset_size//2]
    
    train_dataset = SlidingWindowSamplerDataset(train_data['X'], train_data['Y'], train_time_features, L_in=args.L_in, L_out=args.L_out)
    val_dataset = SlidingWindowSamplerDataset(val_data['X'], val_data['Y'], val_time_features, L_in=args.L_in, L_out=args.L_out)
    
    logging.info(f"Training dataset size: {len(train_dataset)} samples")
    logging.info(f"Validation dataset size: {len(val_dataset)} samples")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # --- Model, Optimizer, Loss ---
    model = TEC_MoLLM(model_config).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = nn.MSELoss()
    
    # --- Training Loop ---
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
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