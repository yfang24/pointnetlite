import os
import shutil
import torch
import time
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from sklearn.metrics import confusion_matrix

from utils.checkpoint_utils import save_checkpoint
from utils.model_utils import get_model_profile
from utils.log_utils import setup_logger
from datasets.get_dataset import get_dataset
from models.encoders.get_encoder import get_encoder
from models.heads.get_head import get_head
from models.losses.get_loss import get_loss


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(encoder, head, dataloader, loss_fn, optimizer, scheduler, device, logger=None):
    encoder.train()
    head.train()
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        inputs, targets = batch[0].float().permute(0, 2, 1).to(device), batch[1].long().to(device)

        optimizer.zero_grad()
        embeddings = encoder(inputs)
        outputs = head(embeddings)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)

    scheduler.step()
    
    end_time = time.time()
    mem_used = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    epoch_time = end_time - start_time
    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    
    if logger:
        logger.info(f"[Train] Acc: {acc:.2f}% | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s | Mem: {mem_used:.1f} MB")
    return avg_loss, acc, epoch_time, mem_used


def evaluate(encoder, head, dataloader, loss_fn, device, logger=None, rotation_vote=False, num_votes=None):
    encoder.eval()
    head.eval()
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            inputs, targets = batch[0].float().permute(0, 2, 1).to(device), batch[1].long().to(device)

            embeddings = encoder(inputs)
            outputs = head(embeddings)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total += inputs.size(0)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    end_time = time.time()
    mem_used = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    epoch_time = end_time - start_time
       
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)    
    
    if rotation_vote:
        K = num_votes
        N = total
        assert N % K == 0, "Dataset size must be divisible by number of rotations"
        B = N // K

        outputs_all = all_preds.reshape(B, K)  # (B, K)
        labels_all = all_targets.reshape(B, K)  # (B, K)
        true_labels = labels_all[:, 0]  # all rotations share label

        # Majority voting
        voted_preds = []
        for preds in outputs_all:
            voted_label = np.bincount(preds).argmax()
            voted_preds.append(voted_label)
        voted_preds = np.array(voted_preds)

        correct = (voted_preds == true_labels).sum()
        total = B
        acc = 100.0 * correct / total
        avg_loss = total_loss / N
        cm = confusion_matrix(true_labels, voted_preds)
    else:
        acc = 100.0 * correct / total
        avg_loss = total_loss / total
        cm = confusion_matrix(all_targets, all_preds)
    
    # Mean Class Accuracy (mA)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    mean_acc = 100.0 * np.nanmean(per_class_acc)
        
    if logger:
        logger.info(f"[Eval]  OA: {acc:.2f}% | mA: {mean_acc:.2f}% | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s | Mem: {mem_used:.1f} MB")
    
    return avg_loss, acc, mean_acc, epoch_time, mem_used, cm


def run_training(rank, world_size, local_rank, config, config_path, device, use_ddp):
    seed = config.get("seed", 42)
    set_seed(seed)

    # Timestamped experiment folder and logger (only once)
    is_resumed = "experiments" in str(config_path)
    if rank == 0:
        if is_resumed:
            exp_dir = config_path.parent
            logger = setup_logger(exp_dir / "log.txt")
            logger.info(f"\n[Resume] Continuing training")
        else:
            exp_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = config_path.stem
            exp_dir = Path("experiments") / f"{exp_name}_{exp_time}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(config_path, exp_dir / "config.yaml")
            
            logger = setup_logger(exp_dir / "log.txt")
            logger.info(f"Experiment: {exp_name}")
    else:
        logger = None
        exp_dir = None

    # Dataset
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)

    train_set = get_dataset(config, "train_dataset")
    test_set, rotation_vote, num_votes = get_dataset(config, "test_dataset")

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
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    # Model
    encoder = get_encoder(config).to(device)
    head = get_head(config).to(device)
    
    if use_ddp:
        encoder = DistributedDataParallel(encoder, device_ids=[local_rank])
        head = DistributedDataParallel(head, device_ids=[local_rank])
        
    # Model Profile
    if rank == 0:
        num_points = getattr(train_set, "num_points", train_set[0][0].shape[0])
        dummy_input = torch.rand(1, 3, num_points).float().to(device)
        
        profile_encoder = encoder.module if isinstance(encoder, DistributedDataParallel) else encoder
        profile_head = head.module if isinstance(head, DistributedDataParallel) else head
        
        flops, params = get_model_profile(torch.nn.Sequential(profile_encoder, profile_head), dummy_input)
        # flops, params = get_model_profile(torch.nn.Sequential(encoder, head), dummy_input)
        logger.info(f"Model Params: {params:,} | FLOPs: {flops / 1e6:.2f} MFLOPs")      

    # Loss
    loss_fn = get_loss(config)
    
    # Optimizer + Scheduler
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)
    
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # Training Loop
    epochs = config.get("epochs", 200)
    
    best_acc = 0.0
    best_epoch = -1
    
    window_size = 5
    acc_window = []
    best_window_mean = -1
    converged_epoch = -1
    epoch_times, epoch_mems = [], []

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}")

        train_loss, train_acc, train_time, train_mem = train_one_epoch(
            encoder, head, train_loader, loss_fn, optimizer, scheduler, device, logger if rank == 0 else None
        )
        test_loss, test_acc, test_ma, test_time, test_mem, _ = evaluate(
            encoder, head, test_loader, loss_fn, device, logger if rank == 0 else None, rotation_vote, num_votes
        )

        if rank == 0:
            epoch_times.append(train_time)
            epoch_mems.append(train_mem)

            # Sliding window
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

    if rank == 0:
        save_checkpoint(encoder, head, optimizer, scheduler, epoch, best_acc, exp_dir, is_best=False)
        logger.info(f"Best Test Acc: {best_acc:.2f}% at Epoch {best_epoch}")
        logger.info(f"[Converged] Epoch: {converged_epoch} | Mean OA: {best_window_mean:.2f}% | Time: {converged_time:.2f}s | Mem: {converged_mem:.1f}MB")
        logger.info("Training complete.")