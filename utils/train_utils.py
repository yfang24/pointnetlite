import os
import shutil
import torch
import time
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
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
# train utils
#=====================

def set_seed(seed):
    o3d.utility.random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Make CUDA deterministic (may slow down training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False    

#=====================
# train loops
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
        inputs = batch[0].float().to(device)  # (B, N, 3)                 
        targets = batch[1].long().to(device)

        optimizer.zero_grad()        
        outputs = head(encoder(inputs))
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs   # PointNetClsHead returns (logits, trans_feat)
        preds = logits.argmax(dim=1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    if isinstance(scheduler, CosineLRScheduler):
        scheduler.step(epoch)
    else:
        scheduler.step()
    
    end_time = time.time()
    epoch_time = end_time - start_time

    mem_used = torch.cuda.max_memory_allocated(device) / 1024**2  # MB    

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
            inputs = batch[0].float().to(device)  # (B, N, 3)
            targets = batch[1].long().to(device)

            outputs = head(encoder(inputs))
            loss = loss_fn(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs   # PointNetClsHead returns (logits, trans_feat)
            preds = logits.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    end_time = time.time()
    epoch_time = end_time - start_time

    mem_used = torch.cuda.max_memory_allocated(device) / 1024**2  # MB    
       
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
        inputs = batch[0].float().to(device)  # (B, N, 3)
        targets = batch[1].long().to(device)       

        optimizer.zero_grad()
        pred, target = head(encoder(inputs))

        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)

    if isinstance(scheduler, CosineLRScheduler):
        scheduler.step(epoch)
    else:
        scheduler.step()

    end_time = time.time()
    epoch_time = end_time - start_time

    mem_used = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    
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
        for batch in tqdm(dataloader, desc="Eval", leave=False):           
            inputs = batch[0].float().to(device)  # (B, N, 3)
            targets = batch[1].long().to(device)       
    
            pred, target = head(encoder(inputs))

            loss = loss_fn(pred, target)
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)

    end_time = time.time()
    epoch_time = end_time - start_time

    mem_used = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    
    avg_loss = total_loss / total

    if logger:
        logger.info(f"[Eval] Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s | Mem: {mem_used:.1f} MB")

    return avg_loss, epoch_time, mem_used
