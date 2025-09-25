import os
import shutil
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from timm.scheduler import CosineLRScheduler

from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.log_utils import setup_logger
from utils.model_utils import get_model_profile
from utils.optim_utils import get_optimizer, get_scheduler
from utils.train_utils import set_seed, pretrain_one_epoch, pretrain_evaluate
from datasets.get_dataset import get_dataset
from models.encoders.get_encoder import get_encoder
from models.heads.get_head import get_head
from models.losses.get_loss import get_loss


def run_pretraining(rank, world_size, local_rank, config, config_path, device, use_ddp, exp_name=None, ckpt_type="best"):
    seed = config.get("seed", 42)
    set_seed(seed)

    # Setup experiment directory and logger
    is_resumed = "experiments" in str(config_path)
    if rank == 0:
        if is_resumed:  # config_path = code/experiments/exp_name/config.yaml
            exp_dir = config_path.parent  
            logger = setup_logger(exp_dir / "log.txt")
            logger.info(f"\n[Resume] Continuing training")
        else:  # config_path = code/configs/config.yaml
            exp_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = config_path.stem  # config_name
            exp_dir = config_path.parents[1] / "experiments" / f"{exp_name}_{exp_time}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(config_path, exp_dir / "config.yaml")
            
            logger = setup_logger(exp_dir / "log.txt")
            logger.info(f"Experiment: {exp_name}")
    else:
        logger = None

    # Dataset
    batch_size = config.get("batch_size", 128)
    num_workers = config.get("num_workers", 4)
    
    train_set = get_dataset(config, "train")
    val_set = get_dataset(config, "val")

    if use_ddp:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, sampler=train_sampler,
        num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    # Model
    encoder = get_encoder(config).to(device)
    head = get_head(config).to(device)  # Here head is decoder for pretraining
    model = torch.nn.Sequential(encoder, head).to(device)
    
    if use_ddp:
        encoder = DistributedDataParallel(encoder, device_ids=[local_rank])
        head = DistributedDataParallel(head, device_ids=[local_rank])

    # Model Profile
    if rank == 0:
        if isinstance(train_set[0][0], tuple): # for modelnet_mae_render
            model = RenderMAE(encoder, head)
            num_points = getattr(train_set, "num_points", train_set[0][0][0].shape[0])
            dummy_input = torch.rand(1, num_points, 3).float().to(device)
        else:
            num_points = getattr(train_set, "num_points", train_set[0][0].shape[0])
            dummy_input = torch.rand(1, num_points, 3).float().to(device)
        flops, params = get_model_profile(model, dummy_input)
        logger.info(f"Model Params: {params / 1e6:.2f} M | FLOPs: {flops / 1e9:.2f} G")

    # Loss
    loss_fn = get_loss(config)

    # Optimizer + Scheduler
    named_params = list(model.named_parameters())
    optimizer = get_optimizer(config, named_params)
    scheduler = get_scheduler(config, optimizer)

    # Resume logic
    start_epoch = 0
    best_loss = float("inf")

    ckpt_path = exp_dir / f"checkpoint_{ckpt_type}.pth"
    if is_resumed:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        start_epoch, best_loss = load_checkpoint(ckpt_path, device, encoder, head, optimizer, scheduler)
        if rank == 0:
            logger.info(f"Resumed from checkpoint at epoch {start_epoch}")
            
    # Training loop
    sum_time = 0.0
    converged_epoch, converged_time = -1, 0.0

    epochs = config.get("epochs", 300)    

    for epoch in range(start_epoch, start_epoch + epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{start_epoch + epochs}")

        train_loss, train_time, train_mem = pretrain_one_epoch(
            epoch, encoder, head, train_loader, loss_fn, optimizer, scheduler, device, logger if rank == 0 else None
        )
        val_loss, val_time, val_mem = pretrain_evaluate(
            encoder, head, val_loader, loss_fn, device, logger if rank == 0 else None
        )

        if rank == 0:
            sum_time += train_time

            if val_loss < best_loss:
                best_loss = val_loss
                converged_epoch = epoch + 1
                converged_time = sum_time
                save_checkpoint(encoder, head, optimizer, scheduler, epoch, best_loss, exp_dir, is_best=True)

    if rank == 0:
        save_checkpoint(encoder, head, optimizer, scheduler, epoch, best_loss, exp_dir, is_best=False)
        logger.info(f"[Converged] Epoch: {converged_epoch} | Best Val Loss: {best_loss:.2f}% | Time: {converged_time:.2f}s")
        logger.info("Pretraining complete.")