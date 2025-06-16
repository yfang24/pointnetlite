import torch
from pathlib import Path

def save_checkpoint(encoder, head, optimizer, scheduler, epoch, best_acc, exp_dir, is_best=False):
    checkpoint = {
        'encoder_state_dict': encoder.state_dict(),
        'head_state_dict': head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch + 1,
        'best_acc': best_acc
    }

    last_path = Path(exp_dir) / "checkpoint_last.pth"
    torch.save(checkpoint, last_path)

    if is_best:
        best_path = Path(exp_dir) / "checkpoint_best.pth"
        torch.save(checkpoint, best_path)

def load_checkpoint(encoder, head, optimizer, scheduler, path, device='cuda'):
    checkpoint = torch.load(path, map_location=device)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    head.load_state_dict(checkpoint['head_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0) + 1
    best_acc = checkpoint.get('best_acc', 0.0)

    return start_epoch, best_acc
