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
import matplotlib.pyplot as plt

from src.data.dataset import SlidingWindowSamplerDataset
from src.model.tec_mollm import TEC_MoLLM

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

def parse_args():
    """Parse command line arguments for LR finder."""
    parser = argparse.ArgumentParser(description='Learning Rate Range Test for TEC-MoLLM')
    
    # Data configuration (copied from train.py)
    parser.add_argument('--L_in', type=int, default=48, help='Input sequence length')
    parser.add_argument('--L_out', type=int, default=12, help='Output sequence length')
    
    # Model configuration (copied from train.py)
    parser.add_argument('--d_emb', type=int, default=16, help='Embedding dimension')
    parser.add_argument('--llm_layers', type=int, default=3, help='Number of LLM layers')

    # LR Finder configuration
    parser.add_argument('--min_lr', type=float, default=1e-8, help='Minimum learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-1, help='Maximum learning rate')
    parser.add_argument('--num_iter', type=int, default=100, help='Number of iterations for the test')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')

    return parser.parse_args()

def find_lr(args):
    """Main function to run the LR range test."""
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        logging.info("--- Starting Learning Rate Range Test ---")
        logging.info(f"Using device: {device}, Rank: {rank}/{world_size}")
        logging.info(f"Test Configuration: min_lr={args.min_lr}, max_lr={args.max_lr}, num_iter={args.num_iter}")

    # --- Setup paths and model config (same as train.py) ---
    data_dir = 'data/processed'
    graph_path = os.path.join(data_dir, 'graph_A.pt')
    target_scaler_path = os.path.join(data_dir, 'target_scaler.joblib')
    os.makedirs(args.output_dir, exist_ok=True)

    conv_output_len = args.L_in // 4
    patch_len = 4 if conv_output_len % 4 == 0 else (2 if conv_output_len % 2 == 0 else 1)
    
    model_config = {
        "num_nodes": 2911, "d_emb": args.d_emb, "spatial_in_channels_base": 6,
        "spatial_out_channels": 32, "spatial_heads": 2, "temporal_channel_list": [64, 128],
        "temporal_strides": [2, 2], "patch_len": patch_len, "d_llm": 768, "llm_layers": args.llm_layers,
        "prediction_horizon": args.L_out, "temporal_seq_len": args.L_in
    }
    
    # --- Load Data (same as train.py) ---
    if rank == 0: logging.info("Loading data...")
    try:
        target_scaler = joblib.load(target_scaler_path)
    except FileNotFoundError:
        logging.error(f"Target scaler not found at {target_scaler_path}.")
        cleanup_distributed()
        return

    train_dataset = SlidingWindowSamplerDataset(data_path=data_dir, mode='train', L_in=args.L_in, L_out=args.L_out, target_scaler=target_scaler)
    edge_index = torch.load(graph_path)['edge_index'].to(device)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=4, pin_memory=True, prefetch_factor=4
    )
    
    # --- Setup Model, Optimizer, Loss (same as train.py but optimizer LR is placeholder) ---
    model = TEC_MoLLM(model_config).to(device)
    model = torch.compile(model)
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.min_lr)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    # --- LR Range Test Logic ---
    lrs = np.logspace(np.log10(args.min_lr), np.log10(args.max_lr), args.num_iter)
    losses = []
    
    model.train()
    data_iter = iter(train_loader)
    
    progress_bar = tqdm(range(args.num_iter), desc="Finding LR", disable=(rank!=0))
    for i in progress_bar:
        # Get next batch, reset iterator if needed
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Update LR for this step
        for param_group in optimizer.param_groups:
            param_group['lr'] = lrs[i]

        # Forward and backward pass
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        time_features = batch['x_time_features'].to(device)

        B, L, H, W, C = x.shape
        x = x.view(B, L, H * W, C)
        if time_features.numel() > 0:
            time_features = time_features.unsqueeze(-2).expand(B, L, H * W, -1)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(x, time_features, edge_index)
            y_reshaped = y.permute(0, 3, 1, 2).reshape(B, -1, H * W, 1)
            loss = loss_fn(output, y_reshaped)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        progress_bar.set_postfix(lr=lrs[i], loss=loss.item())

        # Stop if loss explodes
        if loss.item() > 4 * (losses[0] if losses else 1.0) and i > 10:
            if rank == 0: logging.info("Loss exploded, stopping test.")
            break
    
    # Only rank 0 saves the plot
    if rank == 0:
        # Trim the lists if the loop broke early
        lrs = lrs[:len(losses)]
        
        logging.info("LR range test finished. Plotting results...")
        plt.figure()
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Range Test')
        
        plot_path = os.path.join(args.output_dir, 'lr_finder_plot.png')
        plt.savefig(plot_path)
        logging.info(f"Plot saved to {plot_path}")
        
        # Save raw data
        data_path = os.path.join(args.output_dir, 'lr_finder_data.npz')
        np.savez(data_path, lrs=lrs, losses=losses)
        logging.info(f"Raw data saved to {data_path}")

        # Suggest best LR
        try:
            # Find the point with the steepest gradient
            # We ignore the first few and last few points for stability
            skip_start = 10
            skip_end = 5
            if len(losses) > skip_start + skip_end:
                losses_smoothed = np.convolve(losses, np.ones(5)/5, mode='valid')
                lrs_smoothed = lrs[len(lrs) - len(losses_smoothed):]

                # Use numpy gradient on log-loss
                grads = np.gradient(np.log10(losses_smoothed))
                
                # Find the minimum gradient (steepest descent)
                best_idx = np.argmin(grads[skip_start:-skip_end]) + skip_start
                best_lr = lrs_smoothed[best_idx]
                logging.info(f"📈 Suggested LR (steepest descent): {best_lr:.2e}")
        except Exception as e:
            logging.warning(f"Could not automatically suggest a learning rate: {e}")


    cleanup_distributed()

if __name__ == '__main__':
    args = parse_args()
    find_lr(args) 