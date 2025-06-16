import os
import time
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import shutil
import argparse
from pathlib import Path
from datetime import datetime

from utils.train_utils import set_seed, train_one_epoch, evaluate
from utils.checkpoint_utils import save_checkpoint
from utils.model_utils import get_model_profile
from utils.log_utils import setup_logger
from datasets.get_dataset import get_dataset
from models.encoders.get_encoder import get_encoder
from models.heads.get_head import get_head
from models.losses.get_loss import get_loss
from configs.load_config import load_config

def main(config_name):
    config, config_path = load_config(config_name)

    # Timestamped experiment folder
    exp_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config_path.stem
    exp_dir = Path("experiments") / f"{exp_name}_{exp_time}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    shutil.copy(config_path, exp_dir / "config.yaml")

    # Setup logger
    logger = setup_logger(exp_dir / "log.txt")
    logger.info(f"Experiment: {exp_name}")

    seed = config.get("seed", 42)
    set_seed(seed)

    device = torch.device(config.get("device", "cuda"))
    
    # Dataset
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)
    
    train_set = get_dataset(config, "train_dataset")
    test_set, rotation_vote, num_votes = get_dataset(config, "test_dataset")
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    # Model
    encoder = get_encoder(config).to(device)
    head = get_head(config).to(device)
    
    # Model profile
    num_points = getattr(train_set, "num_points", train_set[0][0].shape[0])
    
    dummy_input = torch.rand(1, 3, num_points).float().to(device)
    flops, params = get_model_profile(torch.nn.Sequential(encoder, head), dummy_input)
    logger.info(f"Model Params: {params:,} | FLOPs: {flops / 1e6:.2f} MFLOPs")

    # Loss
    loss_fn = get_loss(config)

    # Optimizer + Scheduler
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)
    
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    # Training loop
    num_epochs = config.get("num_epochs", 200)
    
    best_acc = 0.0
    best_epoch = -1

    window_size = 5
    acc_window = []
    best_window_mean = -1
    converged_epoch = -1
    epoch_times, epoch_mems = [], []
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_acc, train_time, train_mem = train_one_epoch(
            encoder, head, train_loader, loss_fn, optimizer, scheduler, device, logger
        )
        test_loss, test_acc, test_ma, test_time, test_mem, _ = evaluate(
            encoder, head, test_loader, loss_fn, device, logger, rotation_vote, num_votes
        )
        
        epoch_times.append(train_time)
        epoch_mems.append(train_mem)
        
        # Update window
        acc_window.append(test_acc)
        if len(acc_window) > window_size:
            acc_window.pop(0)

        if len(acc_window) == window_size:
            window_mean = sum(acc_window) / window_size
            if window_mean > best_window_mean:
                best_window_mean = window_mean
                best_acc = max(acc_window)
                best_acc_idx = acc_window.index(best_acc)
                best_epoch = (epoch + 1 - window_size) + best_acc_idx
                converged_epoch = epoch + 1
                converged_time = sum(epoch_times)
                converged_mem = max(epoch_mems)
                
                save_checkpoint(encoder, head, optimizer, scheduler, epoch, best_acc, exp_dir, is_best=True)
    
    save_checkpoint(encoder, head, optimizer, scheduler, epoch, best_acc, exp_dir, is_best=False)
    
    logger.info(f"Best Test Acc: {best_acc:.2f}% at Epoch {best_epoch}")
    logger.info(f"[Converged] Epoch: {converged_epoch} | Mean OA: {best_window_mean:.2f}% | Time: {converged_time:.2f}s | Mem: {converged_mem:.1f}MB")

    logger.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with YAML config.")
    parser.add_argument(
        "-cfg", "--config",
        type=str,
        required=True,
        help="Name of the config file under /code/configs/"
    )
    args = parser.parse_args()
    main(args.config)
