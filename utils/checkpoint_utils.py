import torch
from pathlib import Path

from utils.model_utils import unwrap_model

def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, exp_dir, is_best=False):
    checkpoint = {
        'model_state_dict': unwrap_model(model).state_dict(),
        'optimizer_state_dict': unwrap_model(optimizer).state_dict(),
        'scheduler_state_dict': unwrap_model(scheduler).state_dict(),
        'epoch': epoch + 1,
        'best_acc': best_acc
    }

    last_path = Path(exp_dir) / "checkpoint_last.pth"
    torch.save(checkpoint, last_path)

    if is_best:
        best_path = Path(exp_dir) / "checkpoint_best.pth"
        torch.save(checkpoint, best_path)

def load_checkpoint(model, optimizer, scheduler, path, device='cuda'):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0) + 1
    best_acc = checkpoint.get('best_acc', 0.0)

    return start_epoch, best_acc
