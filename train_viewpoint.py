import os
import torch
from torch.utils.data import DataLoader
from torch.nn import Sequential, CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import time
import logging

from datasets.modelnet_mesh import ModelNetMesh
from datasets.scanobjectnn import ScanObjectNN
from models.modules.viewpoint_learner import ViewpointLearner
from models.encoders.pointnetlite_encoder import PointNetLiteEncoder
from models.heads.pointnet_cls_head import PointNetClsHead
import utils.mesh_utils as mesh_utils
import utils.pcd_utils as pcd_utils
from pytorch3d.structures import Meshes

PROJ_ROOT = Path("/mmfs1/projects/smartlab/")

# ==================== Setup ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 11
num_views = 4
num_points = 1024
lambda_repel = 0.05
epochs = 200
batch_size = 32

# ==================== Logging ====================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = PROJ_ROOT / f"code/experiments/train_viewpoint_{timestamp}"
exp_dir.mkdir(parents=True, exist_ok=True)

log_path = exp_dir / "log.txt"
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())  # Also print to stdout

# ==================== Dataset ====================
train_dataset = ModelNetMesh(
    root_dir=PROJ_ROOT / "data/modelnet40_manually_aligned",
    class_map=PROJ_ROOT / "code/configs/class_map_modelnet11.json",
    split="train",
    use_cache=True
)
def collate_mesh(batch):
    verts_batch, faces_batch, labels_batch = zip(*batch)
    return list(verts_batch), list(faces_batch), torch.tensor(labels_batch)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size//num_views, shuffle=True, num_workers=4, drop_last=True, collate_fn=collate_mesh)

test_dataset = ScanObjectNN(
    root_dir=PROJ_ROOT / "data/scanobjectnn/main_split_nobg",
    class_map=PROJ_ROOT / "code/configs/class_map_scanobjectnn11.json",
    split="test",
    normalize=True,
    use_cache=True
)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ==================== Modules ====================
viewpoint_learner = ViewpointLearner(num_classes=num_classes, num_views=num_views).to(device)
encoder = PointNetLiteEncoder().to(device)
head = PointNetClsHead(out_dim=num_classes).to(device)
model = Sequential(encoder, head).to(device)
loss_fn = CrossEntropyLoss()

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(viewpoint_learner.parameters()),
    lr=1e-3, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

# ==================== Training ====================
best_acc = 0.0
for epoch in range(epochs):
    model.train()
    viewpoint_learner.train()

    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

    correct, total, total_loss = 0, 0, 0.0

    for verts, faces, labels in tqdm(train_dataloader, total=len(train_dataloader), desc="Train", leave=False):
        verts, faces, labels = verts.float().to(device), faces.long().to(device), labels.long().to(device)
        mesh = Meshes(verts=verts*torch.tensor([[-1, 1, -1]], device=verts.device), faces=faces)

        cam_pos = viewpoint_learner(labels)  # (B, V, 3)

        pcs = []
        pc_labels = []
    
        for b in range(verts.size(0)):
            for v in range(num_views):
                view_dir = cam_pos[b, v]  # (3,)
                pts = render_mesh_torch(meshes[b], view_dir, num_points=num_points, device=device)
                pts = pcd_utils.normalize_pcd_tensor(pts)
                pts[:, [0, 2]] *= -1
                pcs.append(pts)                 # (N, 3)
                pc_labels.append(labels[b])    # scalar
    
        pcs = torch.stack(pcs, dim=0).to(device)               # (B * V, N, 3)
        pc_labels = torch.stack(pc_labels, dim=0).to(device)   # (B * V,)

        optimizer.zero_grad()
        preds = model(pcs)
        loss = loss_fn(preds, pc_labels)
        repel = viewpoint_learner.repelling_loss()
        total_batch_loss = loss + lambda_repel * repel
        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item() * pcs.size(0)
        pred_labels = preds.argmax(dim=1)
        correct += pred_labels.eq(pc_labels).sum().item()
        total += pcs.size(0)

    scheduler.step()  
    
    acc = 100.0 * correct / total
    avg_loss = total_loss / total

    end_time = time.time()
    mem_used = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    epoch_time = end_time - start_time
    
    logger.info(f"[Train] Epoch {epoch+1}: Acc = {acc:.2f}% | Loss = {avg_loss:.4f} | Time = {epoch_time:.1f}s | Mem = {mem_used:.1f} MB")

    # ========== Evaluation ==========
    model.eval()
    viewpoint_learner.eval()

    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    
    val_correct, val_total, val_loss = 0, 0, 0.0

    for pcs, labels in tqdm(test_dataloader, total=len(test_dataloader), desc="Eval", leave=False):
        pcs, labels = pcs.float().to(device), labels.long().to(device)

        with torch.no_grad():
            preds = model(pcs)
            loss = loss_fn(preds, labels)

        val_loss += loss.item() * batch_size
        pred_labels = preds.argmax(dim=1)
        val_correct += pred_labels.eq(labels).sum().item()
        val_total += batch_size
    
    val_acc = 100.0 * val_correct / val_total
    val_avg_loss = val_loss / val_total

    end_time = time.time()
    mem_used = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    epoch_time = end_time - start_time
    
    logger.info(f"[Eval]  Acc = {acc:.2f}% | Loss = {avg_loss:.4f} | Time = {epoch_time:.1f}s | Mem = {mem_used:.1f} MB")

    if val_acc > best_acc:
        best_acc = val_acc
        ckpt_path = exp_dir / "checkpoint_best.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'viewpoint_state_dict': viewpoint_learner.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, ckpt_path)
