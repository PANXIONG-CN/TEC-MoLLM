import torch
import torch._dynamo
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import logging
import os
import joblib
import argparse
from tqdm import tqdm

from src.data.dataset import SlidingWindowSamplerDataset
# The following imports are now handled by the offline preprocess.py script
# from src.features.feature_engineering import create_features_and_targets, standardize_features
from src.model.tec_mollm import TEC_MoLLM
from src.evaluation.metrics import evaluate_horizons

# Suppress torch.compile backend errors and fallback to eager execution
torch._dynamo.config.suppress_errors = True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, edge_index, scaler, rank=0, accumulation_steps=1):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training") if rank == 0 else dataloader
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(progress_bar):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        time_features = batch['x_time_features'].to(device)
        
        B, L, H, W, C = x.shape
        x = x.view(B, L, H * W, C)
        if time_features.numel() > 0:
            time_features = time_features.unsqueeze(-2).expand(B, L, H * W, -1)

        # Use autocast for mixed-precision training
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(x, time_features, edge_index)
            y_reshaped = y.permute(0, 3, 1, 2).reshape(B, -1, H * W, 1)
            loss = loss_fn(output, y_reshaped)
            loss = loss / accumulation_steps
        
        # Scale loss and perform backward pass
        scaler.scale(loss).backward()
        
        # --- START MODIFICATION 1.3.1 ---
        # REASON: Prevents gradient explosion, which is the root cause of the
        #         'overflow' warning and numerical instability during training.
        #         The scaler.unscale_ is necessary before clipping in mixed-precision.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # --- END MODIFICATION 1.3.1 ---
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    if (len(dataloader)) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, loss_fn, device, edge_index, target_scaler_path, rank=0):
    model.eval()
    total_loss = 0
    all_preds = []
    all_trues = []
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating") if rank == 0 else dataloader
        for batch in progress_bar:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            time_features = batch['x_time_features'].to(device)

            B, L, H, W, C = x.shape
            x = x.view(B, L, H * W, C)
            if time_features.numel() > 0:
                time_features = time_features.unsqueeze(-2).expand(B, L, H * W, -1)
            
            # Use autocast in validation for consistency
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model(x, time_features, edge_index)
                y_reshaped = y.permute(0, 3, 1, 2).reshape(B, -1, H * W, 1)
                loss = loss_fn(output, y_reshaped)
            
            total_loss += loss.item()
            
            all_preds.append(output.cpu().numpy())
            all_trues.append(y_reshaped.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    
    # Concatenate all batches
    y_true_scaled = np.concatenate(all_trues, axis=0)
    y_pred_scaled = np.concatenate(all_preds, axis=0)
    
    # 调用新的评估函数，传入scaler路径进行逆变换
    metrics = evaluate_horizons(y_true_scaled, y_pred_scaled, target_scaler_path)
    return avg_loss, metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train TEC-MoLLM model')
    
    # Data configuration
    parser.add_argument('--L_in', type=int, default=48, help='Input sequence length (default: 48, adjusted to be feasible)')
    parser.add_argument('--L_out', type=int, default=12, help='Output sequence length (prediction horizon)')
    parser.add_argument('--use_subset', action='store_true', help='Use small data subset for quick testing (ignored with preprocessed data)')
    parser.add_argument('--subset_size', type=int, default=500, help='Size of data subset for training')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16, to be used with gradient accumulation)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Minimum change to qualify as improvement')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps for handling large sequences')
    
    # Model configuration
    parser.add_argument('--d_emb', type=int, default=16, help='Embedding dimension')
    parser.add_argument('--llm_layers', type=int, default=3, help='Number of LLM layers')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # --- Distributed Setup ---
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Only log from rank 0 to avoid duplicate messages
    if rank == 0:
        logging.info(f"Using device: {device}")
        logging.info(f"Distributed training: rank {rank}/{world_size}")
        logging.info(f"Training configuration: L_in={args.L_in}, L_out={args.L_out}, epochs={args.epochs}")
    
    # --- Define paths ---
    data_dir = 'data/processed'
    graph_path = os.path.join(data_dir, 'graph_A.pt')
    target_scaler_path = os.path.join(data_dir, 'target_scaler.joblib')
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
    
    # --- Data Loading and Processing (Simplified) ---
    # The complex data loading and feature engineering is now done offline.
    # We just need to instantiate the new Dataset class.
    
    if rank == 0:
        logging.info("Loading pre-processed data...")

    # 验证target_scaler文件存在（用于后续评估）
    if not os.path.exists(target_scaler_path):
        logging.error(f"Target scaler not found at {target_scaler_path}. Cannot proceed.")
        cleanup_distributed()
        return
    elif rank == 0:
        logging.info(f"Target scaler path verified: {target_scaler_path}")

    train_dataset = SlidingWindowSamplerDataset(data_path=data_dir, mode='train', L_in=args.L_in, L_out=args.L_out)
    val_dataset = SlidingWindowSamplerDataset(data_path=data_dir, mode='val', L_in=args.L_in, L_out=args.L_out)

    edge_index = torch.load(graph_path)['edge_index'].to(device)

    # Subset logic is removed as it's better handled during offline preprocessing if needed.
    # If required, a separate set of smaller .pt files should be created.
    if args.use_subset:
        if rank == 0:
            logging.warning("Warning: --use_subset is ignored with the new pre-processed data pipeline.")
            logging.warning("For testing, create smaller .pt files with the preprocessing script.")

    if rank == 0:
        logging.info(f"Training dataset size: {len(train_dataset)} samples")
        logging.info(f"Validation dataset size: {len(val_dataset)} samples")
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    
    # --- DataLoader Optimization ---
    # As per PRD, optimize DataLoader parameters for H20 GPU.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=4,          # PRD: Set to 16, reduced to 4 to avoid shared memory issues
        pin_memory=True,         # PRD: Set to True
        prefetch_factor=4        # PRD: Set to 4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        sampler=val_sampler,
        num_workers=4,          # PRD: Set to 16, reduced to 4 to avoid shared memory issues
        pin_memory=True,         # PRD: Set to True
        prefetch_factor=4        # PRD: Set to 4
    )
    
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

    # Wrap model for distributed training
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = nn.MSELoss()

    # --- START MODIFICATION 1.3.1 ---
    # REASON: Adds dynamic learning rate adjustment, which helps stabilize training
    #         and allows for better convergence in later epochs. CosineAnnealingLR
    #         is a robust choice for many deep learning tasks.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    # --- END MODIFICATION 1.3.1 ---
    
    # --- Training Loop ---
    best_val_loss = float('inf')
    patience_counter = 0
    
    try:
        for epoch in range(args.epochs):
            # Set epoch for distributed sampler
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            if rank == 0:
                logging.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
            
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, edge_index, scaler, rank, args.accumulation_steps)
            val_loss, val_metrics = validate(model, val_loader, loss_fn, device, edge_index, target_scaler_path, rank)
            
            # --- START MODIFICATION 1.3.2 ---
            # Step the scheduler after each epoch
            scheduler.step()
            if rank == 0:
                logging.info(f"LR scheduler stepped. New LR: {scheduler.get_last_lr()[0]:.2e}")
            # --- END MODIFICATION 1.3.2 ---
            
            if rank == 0:
                # 每个epoch都显示基本信息
                logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
                # 每10个epoch或最后一个epoch显示详细指标
                if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                    logging.info("="*80)
                    logging.info(f"DETAILED METRICS - Epoch {epoch+1}/{args.epochs}")
                    logging.info("="*80)
                    logging.info(f"Training Loss: {train_loss:.6f}")
                    logging.info(f"Validation Loss: {val_loss:.6f}")
                    logging.info("Validation Metrics by Horizon:")
                    logging.info(f"  - MAE Average: {val_metrics['mae_avg']:.6f}")
                    logging.info(f"  - RMSE Average: {val_metrics['rmse_avg']:.6f}")
                    logging.info(f"  - R² Score Average: {val_metrics['r2_score_avg']:.6f}")
                    logging.info(f"  - Pearson R Average: {val_metrics['pearson_r_avg']:.6f}")
                    
                    # 如果val_metrics包含各个horizon的详细指标，也打印出来
                    if 'mae_by_horizon' in val_metrics:
                        logging.info("MAE by Horizon:")
                        for h, mae in enumerate(val_metrics['mae_by_horizon'], 1):
                            logging.info(f"  Horizon {h:2d}: {mae:.6f}")
                    
                    if 'rmse_by_horizon' in val_metrics:
                        logging.info("RMSE by Horizon:")
                        for h, rmse in enumerate(val_metrics['rmse_by_horizon'], 1):
                            logging.info(f"  Horizon {h:2d}: {rmse:.6f}")
                            
                    if 'r2_by_horizon' in val_metrics:
                        logging.info("R² Score by Horizon:")
                        for h, r2 in enumerate(val_metrics['r2_by_horizon'], 1):
                            logging.info(f"  Horizon {h:2d}: {r2:.6f}")
                            
                    if 'pearson_by_horizon' in val_metrics:
                        logging.info("Pearson R by Horizon:")
                        for h, pearson in enumerate(val_metrics['pearson_by_horizon'], 1):
                            logging.info(f"  Horizon {h:2d}: {pearson:.6f}")
                    
                    logging.info(f"Best Validation Loss So Far: {best_val_loss:.6f}")
                    logging.info("="*80)
                else:
                    # 简单显示关键指标
                    logging.info(f"Val R²: {val_metrics['r2_score_avg']:.4f} | Pearson R: {val_metrics['pearson_r_avg']:.4f}")
                
                # Early stopping and model saving logic
                if val_loss < best_val_loss - args.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                    # Save the unwrapped model state for DDP
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), checkpoint_path)
                    logging.info(f"🎉 New best model saved to {checkpoint_path} (Val Loss: {val_loss:.6f})")
                else:
                    patience_counter += 1
                    logging.info(f"No improvement for {patience_counter}/{args.patience} epochs")
                
                # Early stopping check
                if patience_counter >= args.patience:
                    logging.info(f"🛑 Early stopping triggered after {epoch+1} epochs (patience: {args.patience})")
                    logging.info(f"Best validation loss: {best_val_loss:.6f}")
                    break
    
    finally:
        cleanup_distributed()

if __name__ == '__main__':
    main() 