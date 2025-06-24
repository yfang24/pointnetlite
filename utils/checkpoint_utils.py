import torch
from pathlib import Path

from utils.model_utils import unwrap_model

def save_checkpoint(encoder, head, optimizer, scheduler, epoch, best_acc, exp_dir, is_best=False):
    checkpoint = {
        'encoder': unwrap_model(encoder).state_dict(),
        'head': unwrap_model(head).state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch + 1,
        'best_acc': best_acc
    }

    last_path = Path(exp_dir) / "checkpoint_last.pth"
    torch.save(checkpoint, last_path)

    if is_best:
        best_path = Path(exp_dir) / "checkpoint_best.pth"
        torch.save(checkpoint, best_path)
    
def load_checkpoint(ckpt_path, device, encoder=None, head=None, optimizer=None, scheduler=None):
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    if encoder is not None:
        unwrap_model(encoder).load_state_dict(checkpoint["encoder"])
    if head is not None:
        unwrap_model(head).load_state_dict(checkpoint["head"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_acc = checkpoint.get("best_acc", 0.0)

    return start_epoch, best_acc
