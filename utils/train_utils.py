import torch
import time
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

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
