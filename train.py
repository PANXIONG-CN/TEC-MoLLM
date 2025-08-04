# train.py


# --- 在 any import 之前调用 ---
def suppress_other_ranks_logs():
    import logging, torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
        root.addHandler(logging.NullHandler())


suppress_other_ranks_logs()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
import logging
import os
import argparse
import joblib
import wandb
from datetime import datetime
import torch.multiprocessing as mp

# 导入您的项目模块
from src.data.dataset import SlidingWindowSamplerDataset
from src.data.collate import GraphCollator
from src.model.tec_mollm import TEC_MoLLM
from src.evaluation.metrics import evaluate_horizons
from src.utils.notifications import send_wechat_notification

if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)


# --- DDP辅助函数 (与之前版本相同) ---
def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def suppress_non_main_logs():
    if not is_main_process():
        logging.getLogger().handlers.clear()
        logging.getLogger().setLevel(logging.WARNING)


def setup_logging(run_name):
    if is_main_process():
        log_file_path = f"logs/{run_name}.log"
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - INFO - %(message)s", handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()]
        )


# --- WandB元数据辅助函数 ---
def get_file_metadata(file_path):
    """获取文件的元数据"""
    if not os.path.exists(file_path):
        return None
    stat = os.stat(file_path)
    return {"path": file_path, "size_bytes": stat.st_size, "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat()}


# --- 一次性加载scaler缓存 ---
_cached_scaler = None
_cached_scaler_path = None


def get_cached_scaler(target_scaler_path):
    """缓存scaler，避免每个epoch反复I/O"""
    global _cached_scaler, _cached_scaler_path

    if _cached_scaler is None or _cached_scaler_path != target_scaler_path:
        try:
            _cached_scaler = joblib.load(target_scaler_path)
            _cached_scaler_path = target_scaler_path
            logging.info(f"Cached scaler loaded from {target_scaler_path}")
        except Exception as e:
            logging.error(f"Failed to load scaler from {target_scaler_path}: {e}")
            return None

    return _cached_scaler


# --- 验证函数 (与之前版本相同) ---
def validate(model, dataloader, criterion, device, target_scaler_path):
    model.eval()
    all_preds_cpu = []
    all_trues_cpu = []
    total_loss = 0.0

    temp_scaler_path = f"/tmp/tmp_{os.getpid()}_tec_scaler.joblib"
    if is_main_process():
        try:
            # 使用缓存的scaler，避免重复I/O
            target_scaler = get_cached_scaler(target_scaler_path)
            if target_scaler is None:
                dist.barrier()
                return float("inf"), None
            joblib.dump(target_scaler, temp_scaler_path)
        except Exception as e:
            logging.error(f"Failed to create temp scaler file: {e}")
            dist.barrier()
            return float("inf"), None
    dist.barrier()

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            time_features = batch["x_time_features"].to(device, non_blocking=True)
            graph_data = batch["graph_data"]
            edge_index = graph_data["edge_index"].to(device, non_blocking=True)
            edge_weight = graph_data["edge_weight"].to(device, non_blocking=True)

            B, L, H, W, C = x.shape
            x_reshaped = x.view(B, L, H * W, C)
            time_features_expanded = time_features.unsqueeze(-2).expand(B, L, H * W, -1) if time_features.numel() > 0 else time_features

            output = model(x_reshaped, time_features_expanded, edge_index, edge_weight)
            y_reshaped = y.permute(0, 3, 1, 2).reshape(B, -1, H * W, 1)

            loss = criterion(output, y_reshaped)
            total_loss += loss.item()

            all_preds_cpu.append(output.cpu())
            all_trues_cpu.append(y_reshaped.cpu())

    total_loss_tensor = torch.tensor(total_loss, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    num_batches_total = len(dataloader) * get_world_size()
    avg_batch_loss = total_loss_tensor.item() / num_batches_total

    local_preds = torch.cat(all_preds_cpu, dim=0)
    local_trues = torch.cat(all_trues_cpu, dim=0)

    # 将CPU张量移到GPU进行分布式聚合（NCCL后端要求）
    local_preds_gpu = local_preds.to(device)
    local_trues_gpu = local_trues.to(device)
    gathered_preds_list = [torch.zeros_like(local_preds_gpu) for _ in range(get_world_size())]
    gathered_trues_list = [torch.zeros_like(local_trues_gpu) for _ in range(get_world_size())]
    dist.all_gather(gathered_preds_list, local_preds_gpu)
    dist.all_gather(gathered_trues_list, local_trues_gpu)

    # 将结果移回CPU进行后续处理
    gathered_preds_list = [pred.cpu() for pred in gathered_preds_list]
    gathered_trues_list = [true.cpu() for true in gathered_trues_list]

    metrics = None
    if is_main_process():
        final_preds = torch.cat(gathered_preds_list, dim=0).numpy()
        final_trues = torch.cat(gathered_trues_list, dim=0).numpy()
        try:
            metrics = evaluate_horizons(final_trues, final_preds, temp_scaler_path)
        except Exception as e:
            logging.error(f"Metric evaluation failed: {e}")
        finally:
            if os.path.exists(temp_scaler_path):
                os.remove(temp_scaler_path)

    dist.barrier()
    return avg_batch_loss, metrics


def main(args):
    setup_ddp()
    suppress_non_main_logs()
    setup_logging(args.run_name)

    device = torch.device(f"cuda:{get_rank()}")

    if is_main_process():
        # 初始化WandB
        try:
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "TEC-MoLLM-Default-Project"),
                entity=os.getenv("WANDB_ENTITY"),
                name=args.run_name,
                config=vars(args),
            )
        except Exception as e:
            logging.warning(f"Could not initialize WandB: {e}. Training will continue without it.")

        logging.info(f"===== Starting Run: {args.run_name} =====")
        logging.info(f"Distributed training on {get_world_size()} GPUs.")

    ## MODIFICATION FOR WANDB: Log dataset artifact metadata
    if is_main_process() and wandb.run:
        logging.info("Logging dataset artifact metadata to WandB...")
        dataset_artifact = wandb.Artifact(
            "processed-tec-dataset",
            type="dataset",
            description="Pre-processed TEC data splits, scalers, and graph file.",
            metadata={
                "L_in": args.L_in,
                "L_out": args.L_out,
                "train_set": get_file_metadata("data/processed/train_set.pt"),
                "val_set": get_file_metadata("data/processed/val_set.pt"),
                "test_set": get_file_metadata("data/processed/test_set.pt"),
                "feature_scaler": get_file_metadata("data/processed/scaler.joblib"),
                "target_scaler": get_file_metadata("data/processed/target_scaler.joblib"),
                "graph_A": get_file_metadata("data/processed/graph_A.pt"),
            },
        )
        wandb.log_artifact(dataset_artifact)

    # --- 数据加载 ---
    if is_main_process():
        logging.info("Loading pre-processed data...")
    train_dataset = SlidingWindowSamplerDataset(data_path="data/processed", mode="train", L_in=args.L_in, L_out=args.L_out, stride=args.train_stride)
    val_dataset = SlidingWindowSamplerDataset(data_path="data/processed", mode="val", L_in=args.L_in, L_out=args.L_out, stride=1)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # ① 载入 (CPU) graph data once
    graph_data = torch.load("data/processed/graph_A.pt", weights_only=False)  # 仅CPU

    # ② 顶层可 picklable collate
    collator = GraphCollator(graph_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
        persistent_workers=True,  # 保证整个训练周期都不再spawn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
        persistent_workers=True,  # 避免验证时重新spawn worker
    )

    # --- 模型等初始化 ---
    if is_main_process():
        eff_batch_size = args.batch_size * args.accumulation_steps * get_world_size()
        logging.info(f"Effective batch size: {eff_batch_size}")

    # 模型配置... (与之前版本相同)
    conv_output_len = args.L_in // 4
    patch_len = 4 if conv_output_len % 4 == 0 else 2
    model_config = {
        "num_nodes": 2911,
        "d_emb": 16,
        "spatial_in_channels_base": 6,
        "spatial_out_channels": 11,
        "spatial_heads": 2,
        "temporal_channel_list": [64, 128],
        "temporal_strides": [2, 2],
        "patch_len": patch_len,
        "d_llm": 768,
        "llm_layers": args.llm_layers,
        "prediction_horizon": args.L_out,
        "temporal_seq_len": args.L_in,
        "num_years": 13,
        "dropout_rate": args.dropout_rate,
    }
    model = TEC_MoLLM(model_config).to(device)
    model = DDP(model, device_ids=[get_rank()], find_unused_parameters=False)

    if is_main_process() and wandb.run:
        wandb.watch(model, log="all", log_freq=100)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    # 使用兼容的amp接口
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float("inf")
    patience_counter = 0

    if is_main_process():
        send_wechat_notification(f"🚀 训练开始: {args.run_name}", f"Epochs: {args.epochs}, LR: {args.lr}")

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0.0
        optimizer.zero_grad()
        log_interval = max(1, len(train_loader) // 5)

        for i, batch in enumerate(train_loader):
            # 训练步... (与之前版本相同)
            with torch.cuda.amp.autocast():
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                time_features = batch["x_time_features"].to(device, non_blocking=True)
                graph_data = batch["graph_data"]
                edge_index = graph_data["edge_index"].to(device, non_blocking=True)
                edge_weight = graph_data["edge_weight"].to(device, non_blocking=True)
                B, L, H, W, C = x.shape
                x_reshaped = x.view(B, L, H * W, C)
                time_features_expanded = time_features.unsqueeze(-2).expand(B, L, H * W, -1) if time_features.numel() > 0 else time_features
                output = model(x_reshaped, time_features_expanded, edge_index, edge_weight)
                y_reshaped = y.permute(0, 3, 1, 2).reshape(B, -1, H * W, 1)
                loss = criterion(output, y_reshaped)
                loss = loss / args.accumulation_steps
            scaler.scale(loss).backward()
            train_loss += loss.item() * args.accumulation_steps
            if (i + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)  # 在裁剪前unscale梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if is_main_process() and (i + 1) % log_interval == 0:
                logging.info(f"Epoch {epoch+1} | Progress: {((i+1)*100)/len(train_loader):.0f}% | Avg Batch Loss: {train_loss/(i+1):.4f}")

        val_loss, val_metrics = validate(model, val_loader, criterion, device, "data/processed/target_scaler.joblib")

        if is_main_process():
            avg_epoch_train_loss = train_loss / len(train_loader)
            logging.info(f"--- Epoch {epoch+1} Summary ---")
            logging.info(f"Train Loss: {avg_epoch_train_loss:.4f} | Val Loss: {val_loss:.4f}")

            log_data = {"epoch": epoch + 1, "train_loss": avg_epoch_train_loss, "val_loss": val_loss}
            if val_metrics:
                logging.info(
                    f"Val MAE: {val_metrics['mae_avg']:.4f} | Val R²: {val_metrics['r2_score_avg']:.4f} | Pearson R: {val_metrics['pearson_r_avg']:.4f}"
                )
                log_data.update(
                    {
                        "val_mae_avg": val_metrics["mae_avg"],
                        "val_r2_avg": val_metrics["r2_score_avg"],
                        "val_pearson_avg": val_metrics["pearson_r_avg"],
                    }
                )
            if wandb.run:
                wandb.log(log_data)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                checkpoint_path = f"checkpoints/best_model_{args.run_name}.pth"
                torch.save(model.module.state_dict(), checkpoint_path)
                logging.info(f"🎉 New best model saved to {checkpoint_path} (Val Loss: {val_loss:.6f})")

                ## MODIFICATION FOR WANDB: Log checkpoint artifact metadata
                if wandb.run and val_metrics:
                    model_artifact = wandb.Artifact(
                        f"model-{args.run_name}",
                        type="model",
                        description=f"Best model from run {args.run_name} at epoch {epoch+1}",
                        metadata={
                            "epoch": epoch + 1,
                            "val_loss": best_val_loss,
                            "checkpoint_file": get_file_metadata(checkpoint_path),
                            **val_metrics,
                        },
                    )
                    # artifact.add_file() is NOT called, so the file is not uploaded.
                    wandb.log_artifact(model_artifact, aliases=["best", f"epoch-{epoch+1}"])

                send_wechat_notification(
                    f"🎉 新最佳模型: {args.run_name}",
                    f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, R2: {val_metrics.get('r2_score_avg', 'N/A') if val_metrics else 'N/A':.4f}",
                )
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logging.info("Early stopping triggered.")
                    break

    if is_main_process():
        if wandb.run:
            wandb.finish()
        send_wechat_notification(f"✅ 训练结束: {args.run_name}", f"Best Val Loss: {best_val_loss:.4f}")
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TEC-MoLLM Distributed Training")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--L_in", type=int, default=48)
    parser.add_argument("--L_out", type=int, default=12)
    parser.add_argument("--train_stride", type=int, default=12)
    parser.add_argument("--llm_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--accumulation_steps", type=int, default=6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    # DDP setup is now managed within main
    main(args)
