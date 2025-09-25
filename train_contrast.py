import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import Sequential, CrossEntropyLoss
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import time
import logging

from datasets.modelnet_render import ModelNetRender
from datasets.modelnet import ModelNet
from datasets.scanobjectnn import ScanObjectNN
from datasets.wrappers.aug_wrapper import AugWrapper
from models.encoders.pointnetlite_encoder import PointNetLiteEncoder
from models.encoders.pointmae_encoder import PointMAEEncoder   # Teacher encoder
from models.heads.pointnet_cls_head import PointNetClsHead
from models.heads.pointmae_cls_head import PointMAEClsHead
from models.modules.builders import build_fc_layers
import utils.pcd_utils as pcd_utils
from utils.checkpoint_utils import load_checkpoint

# ==================== Config ====================
PROJ_ROOT = Path("/mmfs1/projects/smartlab/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 11
num_views = 3
epochs = 200
batch_size = 32

lambda_obj = 0.1
lambda_cls = 0.1

ckpt_name = "exp_mn11_3render_maepretrain_20250821_005500"

# ==================== Logging ====================
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

# ==================== Dataset ====================
train_dataset = AugWrapper(ModelNetRender(
    root_dir=PROJ_ROOT / "data/modelnet40_manually_aligned",
    class_map=PROJ_ROOT / "code/configs/class_map_modelnet11.json",
    split="train",
    num_views=num_views
))

class ObjLabelWrapper(Dataset):
    def __init__(self, base_dataset, num_views):
        self.base_dataset = base_dataset
        self.num_views = num_views

        # Precompute object IDs for every sample index
        self.obj_labels = [i // num_views for i in range(len(base_dataset))]

    def __getitem__(self, idx):
        data, class_label = self.base_dataset[idx]
        obj_label = self.obj_labels[idx]
        return data, (class_label, obj_label)

    def __len__(self):
        return len(self.base_dataset)
wrapped_dataset = ObjLabelWrapper(train_dataset, num_views=num_views)

train_dataloader = DataLoader(
    wrapped_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

test_dataset = ScanObjectNN(
    root_dir=PROJ_ROOT / "data/scanobjectnn/main_split_nobg",
    class_map=PROJ_ROOT / "code/configs/class_map_scanobjectnn11.json",
    split="test"
)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ==================== Modules ====================
# encoder = PointNetLiteEncoder().to(device)
# head = PointNetClsHead(out_dim=num_classes).to(device)

encoder = PointMAEEncoder(noaug=True).to(device)
head = PointMAEClsHead(out_dim=num_classes).to(device)
# load pretrained weights for teacher
ckpt_dir = PROJ_ROOT / f"code/experiments/{ckpt_name}"
ckpt_path = ckpt_dir / f"checkpoint_best.pth"
if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
_ = load_checkpoint(ckpt_path, device, encoder=encoder)

# ==================== Loss & Optim ====================
ce_loss = CrossEntropyLoss()

def contrastive_loss(z, labels, tau=0.1):
    # 1) Normalize embeddings for cosine similarity
    z = F.normalize(z, dim=1)                # (N, D)

    # 2) Similarity matrix (cosine similarities)
    sim = torch.matmul(z, z.T) / tau         # (N, N)

    # 3) Mask to ignore self-comparisons
    mask = torch.eye(z.size(0), dtype=torch.bool, device=z.device)

    # 4) Positive mask: same label but not itself
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask

    # 5) Log-softmax over similarities
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # (N, N)

    # 6) Average log-probability of positives
    pos_counts = pos_mask.sum(1)                               # (N,)
    safe_pos_counts = pos_counts + (pos_counts == 0).float()   # avoid /0
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / safe_pos_counts

    # 7) Final loss: negative log-likelihood over all valid samples
    valid = pos_counts > 0
    if valid.any():
        return -mean_log_prob_pos[valid].mean()
    else:
        return torch.tensor(0.0, device=z.device, requires_grad=True)


optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(head.parameters()),
    lr=1e-3, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

# ==================== Training ====================
for epoch in range(epochs+100):
    encoder.train()
    total_loss = 0.0

    for partials, (labels, obj_labels) in tqdm(train_dataloader, desc=f"Train {epoch+1}", leave=False):
        partials = partials.float().to(device)            # (B, N, 3)
        labels = labels.long().to(device)                # (B,)

        # Forward through encoder
        # z = encoder(partials)   # (B, D)
        z = encoder(partials).max(dim=1)[0]   # pointmae

        # Contrastive loss
        loss = contrastive_loss(z, labels)  # ClsCL only

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / len(train_dataloader.dataset)
    logger.info(f"[Pretrain] Epoch {epoch+1}: ClsCL={avg_loss:.4f}")

best_acc = 0.0
for epoch in range(epochs):
    encoder.train()
    head.train()

    total_loss, total_ce, total_obj, total_cls = 0.0, 0.0, 0.0, 0.0
    correct, total = 0, 0

    for partials, (labels, obj_labels) in tqdm(train_dataloader, desc=f"Train {epoch+1}", leave=False):
        partials = partials.float().to(device)            # (B, N, 3)
        labels = labels.long().to(device)                # (B,)
        obj_labels = obj_labels.long().to(device) 
 
        B = partials.size(0)
        
        # ---- Forward ----
        z = encoder(partials)       # (B, D)
        logits = head(z)            # (B, num_classes)

        # ---- Losses ----
        # CE classification
        loss_ce = ce_loss(logits, labels)

        # Object-level contrastive (each sample is its own "object")
        # loss_obj = contrastive_loss(z, obj_labels)

        # # Class-level contrastive (positives = same class)
        # loss_cls = contrastive_loss(z, labels)

        # Total loss
        # loss = loss_ce + lambda_obj * loss_obj + lambda_cls * loss_cls
        loss = loss_ce # + lambda_cls * loss_cls

        # ---- Backward ----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- Logging stats ----
        total_loss += loss.item() * labels.size(0)
        total_ce   += loss_ce.item() * labels.size(0)
        # total_obj  += loss_obj.item() * labels.size(0)
        # total_cls  += loss_cls.item() * labels.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    scheduler.step() 

    acc = 100.0 * correct / total
    avg_loss = total_loss / total
    avg_ce   = total_ce   / total
    # avg_obj  = total_obj  / total
    # avg_cls  = total_cls  / total

    logger.info(f"[Train] Epoch {epoch+1}: Acc={acc:.2f}% | "
          f"Total={avg_loss:.4f} | CE={avg_ce:.4f} | "
        #   f"Obj={avg_obj:.4f} | Cls={avg_cls:.4f}"
          )
         
    # ---- Evaluation ----
    encoder.eval()
    head.eval()

    val_correct, val_total, val_total_loss = 0, 0, 0.0
    with torch.no_grad():
        for pcs, labels in tqdm(test_dataloader, desc="Eval", leave=False):
            pcs, labels = pcs.float().to(device), labels.long().to(device)

            f_s = encoder(pcs)
            l_s = head(f_s)

            preds = l_s.argmax(dim=1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

            loss = ce_loss(l_s, labels)
            val_total_loss += loss.item() * labels.size(0)

    val_acc = 100.0 * val_correct / val_total
    val_avg_loss = val_total_loss / val_total
    logger.info(f"[Eval] Epoch {epoch+1}: Acc={val_acc:.2f}% Loss={val_avg_loss:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch
logger.info(f"Best Acc={best_acc:.2f}% at Epoch {best_epoch}")
