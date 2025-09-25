import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import time
import logging

import open3d as o3d
from pytorch3d.structures import Meshes
# from pytorch3d.loss import point_face_distance # point_mesh_face_distance

from datasets.modelnet_render import ModelNetRender
from datasets.modelnet_mesh import ModelNetMesh
from datasets.modelnet import ModelNet
from datasets.scanobjectnn import ScanObjectNN
from datasets.wrappers.aug_wrapper import AugWrapper
from models.encoders.pointnetlite_encoder import PointNetLiteEncoder
from models.encoders.pointmae_encoder import PointMAEEncoder
from models.heads.pointnet_cls_head import PointNetClsHead
from models.heads.pointmae_cls_head import PointMAEClsHead
from models.modules.builders import build_fc_layers
import utils.pcd_utils as pcd_utils
import utils.mesh_utils as mesh_utils
from utils.checkpoint_utils import load_checkpoint

# ==================== Config ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_dim = 1024
num_classes = 11
num_views = 3
epochs = 200
batch_size = 32

# ==================== Logging ====================
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

# ==================== Dataset ====================
train_dataset = AugWrapper(ModelNet(
    root="modelnet40_manually_aligned",
    class_map="modelnet11",
    split="train"
)
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

test_dataset = ScanObjectNN(
    root="main_split_nobg",
    class_map="scanobjectnn11",
    normalize=True,
    split="test"
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    drop_last=False
)

# ==================== Modules ====================
encoder = PointNetLiteEncoder().to(device)
head = PointNetClsHead(out_dim=num_classes).to(device)


# ==================== Loss & Optim ====================
ce_loss = CrossEntropyLoss()

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(head.parameters()),
    lr=1e-3, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

best_acc = 0.0
for epoch in range(epochs):
    encoder.train()
    head.train()

    correct, total, total_loss = 0, 0, 0.0

    for points, labels in tqdm(train_dataloader, desc=f"Train {epoch+1}", leave=False):
        points = points.float().to(device)            # (B, N, 3)
        labels = labels.long().to(device)                # (B,)

        # ---- Clear gradient buffer ----
        optimizer.zero_grad()

        # ---- Forward ----
        logits = head(encoder(points))  # (B, num_classes)

        # ---- Losses ----
        loss = ce_loss(logits, labels)

        # ---- Backward ----
        loss.backward()
        optimizer.step()

        # ---- Logging stats ----
        total_loss += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    scheduler.step() 

    acc = 100.0 * correct / total
    avg_loss = total_loss / total

    logger.info(f"[Train] Epoch {epoch+1}: Acc={acc:.2f}% | Loss={avg_loss:.4f}")
         
    # ---- Evaluation ----
    encoder.eval()
    head.eval()

    val_correct, val_total, val_total_loss = 0, 0, 0.0
    with torch.no_grad():
        for pcs, labels in tqdm(test_dataloader, desc="Eval", leave=False):
            pcs, labels = pcs.float().to(device), labels.long().to(device)

            f_s = encoder(pcs, dataset_type="scan")
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
logger.info(f"[Converged] Best Acc={best_acc:.2f}% at Epoch {best_epoch}")
