import os
import shutil
import torch
import time
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from sklearn.metrics import confusion_matrix

from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.model_utils import get_model_profile
from utils.log_utils import setup_logger
from datasets.get_dataset import get_dataset
from models.encoders.get_encoder import get_encoder
from models.heads.get_head import get_head
from models.losses.get_loss import get_loss


#=====================
# utils
#=====================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_optimizer(config, named_params): # named_params: list of (name, param) tuples, e.g., list(model.named_parameters())
    opt_cfg = config.get("optimizer", None)

    if opt_cfg is None:
        return torch.optim.Adam(
            [p for _, p in named_params if p.requires_grad],
            lr=0.001, weight_decay=0.0001
        )

    opt_type = opt_cfg.get("name", "adam")
    opt_args = opt_cfg.get("args", {})    
            
    if opt_type == "adamw":
        # AdamW decouple weight decay
        def add_weight_decay(named_params, weight_decay=1e-5, skip_list=()):
            decay, no_decay = [], []
            for name, param in named_params:
                if not param.requires_grad:
                    continue
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.0},
                {'params': decay, 'weight_decay': weight_decay}
            ]
        
        param_groups = add_weight_decay(named_params, opt_args.get("weight_decay", 1e-5))
        return torch.optim.AdamW(param_groups, **opt_args)
    else:
        params = [p for _, p in named_params if p.requires_grad]
        if opt_type == "adam":
            return torch.optim.Adam(params, **opt_args)
        elif opt_type == "sgd":
            return torch.optim.SGD(params, nesterov=True, momentum=0.9, **opt_args)
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")        

def get_scheduler(config, optimizer):
    sched_cfg = config.get("scheduler", None)

    if sched_cfg is None:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    sched_type = sched_cfg.get("name", "steplr")
    sched_args = sched_cfg.get("args", {})

    if sched_type == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, **sched_args)
    elif sched_type == "coslr":
        return CosineLRScheduler(
            optimizer,
            t_initial=sched_args.get("epochs", config.get("epochs", 200)),
            lr_min=1e-6,
            cycle_mul=1,
            warmup_lr_init=1e-6,
            warmup_t=sched_args.get("initial_epochs", 10),
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")


#=====================
# train
#=====================
def train_one_epoch(epoch, encoder, head, dataloader, loss_fn, optimizer, scheduler, device, logger=None):
    encoder.train()
    head.train()
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        if isinstance(batch[0], tuple): # for modelnet_mae_render
            # Unpack list of 3-tuples
            vis_pts       = torch.stack([x[0] for x in batch[0]]).float().to(device)
            mask_pts      = torch.stack([x[1] for x in batch[0]]).float().to(device)
            reflected_pts = torch.stack([x[2] for x in batch[0]]).float().to(device)
            inputs = tuple(vis_pts, mask_pts, reflected_pts)
        else:
            inputs = batch[0].float().to(device)  # (B, N, 3)
        targets = batch[1].long().to(device)

        optimizer.zero_grad()
        outputs = head(encoder(inputs))
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)

    if isinstance(scheduler, CosineLRScheduler):
        scheduler.step(epoch)
    else:
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
            if isinstance(batch[0], tuple): # for modelnet_mae_render
                # Unpack list of 3-tuples
                vis_pts       = torch.stack([x[0] for x in batch[0]]).float().to(device)
                mask_pts      = torch.stack([x[1] for x in batch[0]]).float().to(device)
                reflected_pts = torch.stack([x[2] for x in batch[0]]).float().to(device)
                inputs = tuple(vis_pts, mask_pts, reflected_pts)
            else:
                inputs = batch[0].float().to(device)  # (B, N, 3)
            targets = batch[1].long().to(device)

            outputs = head(encoder(inputs))
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


def run_training(rank, world_size, local_rank, config, config_path, device, use_ddp, exp_name=None, ckpt_type="best"):
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
    model = torch.nn.Sequential(encoder, head).to(device)
    
    if use_ddp:
        encoder = DistributedDataParallel(encoder, device_ids=[local_rank])
        head = DistributedDataParallel(head, device_ids=[local_rank])
        
    # Model Profile
    if rank == 0:
        if isinstance(train_set[0][0], tuple): # for modelnet_mae_render
            num_points = getattr(train_set, "num_points", train_set[0][0][0].shape[0])
            dummy_input = torch.rand(1, num_points, 3).float().to(device)
            dummy_input = tuple(dummy_input.clone() for _ in train_set[0][0])
        else:
            num_points = getattr(train_set, "num_points", train_set[0][0].shape[0])
            dummy_input = torch.rand(1, num_points, 3).float().to(device)
        flops, params = get_model_profile(model, dummy_input)    
        logger.info(f"Model Params: {params:,} | FLOPs: {flops / 1e6:.2f} MFLOPs")      

    # Loss
    loss_fn = get_loss(config)
    
    # Optimizer + Scheduler
    named_params = list(model.named_parameters())
    optimizer = get_optimizer(config, named_params)
    scheduler = get_scheduler(config, optimizer)

    # Resume logic
    start_epoch = 0
    
    best_acc = 0.0
    best_epoch = -1
    
    window_size = 5
    acc_window = []
    best_window_mean = -1
    converged_epoch = -1
    epoch_times, epoch_mems = [], []

    ckpt_path = exp_dir / f"checkpoint_{ckpt_type}.pth"
    if is_resumed:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        start_epoch, best_acc = load_checkpoint(ckpt_path, device, encoder, head, optimizer, scheduler)
        if rank == 0:
            logger.info(f"Resumed from checkpoint at epoch {start_epoch}")
    
    # Training Loop
    epochs = config.get("epochs", 200)

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}")

        train_loss, train_acc, train_time, train_mem = train_one_epoch(
            epoch, encoder, head, train_loader, loss_fn, optimizer, scheduler, device, logger
        )
        test_loss, test_acc, test_ma, test_time, test_mem, _ = evaluate(
            encoder, head, test_loader, loss_fn, device, logger, rotation_vote, num_votes
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
        
        
#=====================
# pretrain
# diff from train: no acc; only loss; using val_set to eval
#=====================
def pretrain_one_epoch(epoch, encoder, head, dataloader, loss_fn, optimizer, scheduler, device, logger=None):
    encoder.train()
    head.train()
    
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

    total_loss = 0.0
    total = 0

    for batch in tqdm(dataloader, desc="Pretrain", leave=False):
        if isinstance(batch[0], tuple): # for modelnet_mae_render
            # Unpack list of 3-tuples
            vis_pts       = torch.stack([x[0] for x in batch[0]]).float().to(device)
            mask_pts      = torch.stack([x[1] for x in batch[0]]).float().to(device)
            reflected_pts = torch.stack([x[2] for x in batch[0]]).float().to(device)
            inputs = tuple(vis_pts, mask_pts, reflected_pts)
        else:
            inputs = batch[0].float().to(device)  # (B, N, 3)
        targets = batch[1].long().to(device)       

        optimizer.zero_grad()
        pred, target = head(encoder(inputs))

        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)

    if isinstance(scheduler, CosineLRScheduler):
        scheduler.step(epoch)
    else:
        scheduler.step()

    end_time = time.time()
    mem_used = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    epoch_time = end_time - start_time
    avg_loss = total_loss / total

    if logger:
        logger.info(f"[Pretrain] Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s | Mem: {mem_used:.1f} MB")

    return avg_loss, epoch_time, mem_used

def pretrain_evaluate(encoder, head, dataloader, loss_fn, device, logger=None):
    encoder.eval()
    head.eval()
    
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val", leave=False):            
            if isinstance(batch[0], tuple): # for modelnet_mae_render
                # Unpack list of 3-tuples
                vis_pts       = torch.stack([x[0] for x in batch[0]]).float().to(device)
                mask_pts      = torch.stack([x[1] for x in batch[0]]).float().to(device)
                reflected_pts = torch.stack([x[2] for x in batch[0]]).float().to(device)
                inputs = tuple(vis_pts, mask_pts, reflected_pts)
            else:
                inputs = batch[0].float().to(device)  # (B, N, 3)
            targets = batch[1].long().to(device)

            pred, target = head(encoder(inputs))

            loss = loss_fn(pred, target)
            total_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

    end_time = time.time()
    mem_used = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    epoch_time = end_time - start_time
    avg_loss = total_loss / total

    if logger:
        logger.info(f"[Val] Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s | Mem: {mem_used:.1f} MB")

    return avg_loss, epoch_time, mem_used


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
    
    train_set = get_dataset(config, "train_dataset")
    val_set = get_dataset(config, "val_dataset")

    if use_ddp:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, sampler=train_sampler,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
            num_points = getattr(train_set, "num_points", train_set[0][0][0].shape[0])
            dummy_input = torch.rand(1, num_points, 3).float().to(device)
            dummy_input = tuple(dummy_input.clone() for _ in train_set[0][0])
        else:
            num_points = getattr(train_set, "num_points", train_set[0][0].shape[0])
            dummy_input = torch.rand(1, num_points, 3).float().to(device)
        flops, params = get_model_profile(model, dummy_input)
        logger.info(f"Model Params: {params:,} | FLOPs: {flops / 1e6:.2f} MFLOPs")

    # Loss
    loss_fn = get_loss(config)

    # Optimizer + Scheduler
    named_params = list(model.named_parameters())
    optimizer = get_optimizer(config, named_params)
    scheduler = get_scheduler(config, optimizer)

    # Resume logic
    start_epoch = 0

    best_loss = float("inf")
    best_epoch = -1
    epoch_times, epoch_mems = [], []

    ckpt_path = exp_dir / f"checkpoint_{ckpt_type}.pth"
    if is_resumed:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        start_epoch, best_acc = load_checkpoint(ckpt_path, device, encoder, head, optimizer, scheduler)
        if rank == 0:
            logger.info(f"Resumed from checkpoint at epoch {start_epoch}")
            
    # Training loop
    epochs = config.get("epochs", 300)    

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}")

        train_loss, train_time, train_mem = pretrain_one_epoch(
            epoch, encoder, head, train_loader, loss_fn, optimizer, scheduler, device, logger if rank == 0 else None
        )
        val_loss, val_time, val_mem = pretrain_evaluate(
            encoder, head, val_loader, loss_fn, device, logger if rank == 0 else None
        )

        if rank == 0:
            epoch_times.append(train_time)
            epoch_mems.append(train_mem)
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch + 1
                converged_time = sum(epoch_times)
                converged_mem = max(epoch_mems)
                save_checkpoint(encoder, head, optimizer, scheduler, epoch, best_loss, exp_dir, is_best=True)

    if rank == 0:
        save_checkpoint(encoder, head, optimizer, scheduler, epoch, best_loss, exp_dir, is_best=False)
        logger.info(f"Best Val Loss: {best_loss:.4f} at Epoch {best_epoch}")
        logger.info(f"[Converged] Epoch: {best_epoch} | Loss: {best_loss:.4f} | Time: {converged_time:.2f}s | Mem: {converged_mem:.1f}MB")
        logger.info("Pretraining complete.")


#=====================
# finetune
#=====================
def run_finetuning(rank, world_size, local_rank, config, config_path, device, use_ddp, exp_name, ckpt_type="best"):
    seed = config.get("seed", 42)
    set_seed(seed)

    # Setup experiment directory and logger
    # config_path = code/configs/config.yaml
    ckpt_dir = config_path.parents[1] / "experiments" / exp_name
    if rank == 0:        
        exp_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = config_path.parents[1] / "experiments" / f"{config_path.stem}_{exp_time}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_path, exp_dir / "config.yaml")

        logger = setup_logger(exp_dir / "log.txt")
        logger.info(f"\n[Finetune] Loading model from {exp_name} using checkpoint_{ckpt_type}")
    else:
        logger = None

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
    model = torch.nn.Sequential(encoder, head).to(device)
    
    if use_ddp:
        encoder = DistributedDataParallel(encoder, device_ids=[local_rank])
        head = DistributedDataParallel(head, device_ids=[local_rank])
        
    # Model Profile
    if rank == 0:
        if isinstance(train_set[0][0], tuple): # for modelnet_mae_render
            num_points = getattr(train_set, "num_points", train_set[0][0][0].shape[0])
            dummy_input = torch.rand(1, num_points, 3).float().to(device)
            dummy_input = tuple(dummy_input.clone() for _ in train_set[0][0])
        else:
            num_points = getattr(train_set, "num_points", train_set[0][0].shape[0])
            dummy_input = torch.rand(1, num_points, 3).float().to(device)
        flops, params = get_model_profile(model, dummy_input)    
        logger.info(f"Model Params: {params:,} | FLOPs: {flops / 1e6:.2f} MFLOPs")      

    # Loss
    loss_fn = get_loss(config)
    
    # Optimizer + Scheduler
    named_params = list(model.named_parameters())
    optimizer = get_optimizer(config, named_params)
    scheduler = get_scheduler(config, optimizer)

    # Loading pretrained model   
    ckpt_path = ckpt_dir / f"checkpoint_{ckpt_type}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    _ = load_checkpoint(ckpt_path, device, encoder=encoder)
    
    # Training Loop
    best_acc = 0.0
    best_epoch = -1
    
    window_size = 5
    acc_window = []
    best_window_mean = -1
    converged_epoch = -1
    epoch_times, epoch_mems = [], []
    
    epochs = config.get("epochs", 200)

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}")

        train_loss, train_acc, train_time, train_mem = train_one_epoch(
            epoch, encoder, head, train_loader, loss_fn, optimizer, scheduler, device, logger
        )
        test_loss, test_acc, test_ma, test_time, test_mem, _ = evaluate(
            encoder, head, test_loader, loss_fn, device, logger, rotation_vote, num_votes
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
        logger.info("Finetuning complete.")
