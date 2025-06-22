import os
import torch
from torch.utils.data import DataLoader
from torch.nn import Sequential, CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from datasets.modelnet_mesh import ModelNetMesh
from datasets.scanobjectnn import ScanObjectNN
from models.modules.viewpoint_learner import ViewpointLearner
from models.encoders.pointnetlite_encoder import PointNetLiteEncoder
from models.heads.pointnet_cls_head import PointNetClsHead
import utils.mesh_utils as mesh_utils
import utils.pcd_utils as pcd_utils
from pytorch3d.structures import Meshes

PROJ_ROOT = "/mmfs1/projects/smartlab/"

# ==================== Setup ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 11
num_views = 3
num_points = 1024
lambda_repel = 0.05
epochs = 200
batch_size = 32

# ==================== Dataset ====================
train_dataset = ModelNetMesh(
    root_dir=os.path.join(PROJ_ROOT, "data/modelnet40_manually_aligned"),
    class_map=os.path.join(PROJ_ROOT, "code/configs/class_map_modelnet11.json"),
    split="train",
    use_cache=True
)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True)

test_dataset = ScanObjectNN(
    root_dir=os.path.join(PROJ_ROOT, "data/scanobjectnn/main_split_nobg"),
    class_map=os.path.join(PROJ_ROOT, "code/configs/class_map_scanobjectnn11.json"),
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
for epoch in range(epochs):
    model.train()
    viewpoint_learner.train()

    correct, total, total_loss = 0, 0, 0.0
    pcs = []
    for verts, faces, labels in tqdm(train_dataloader, total=len(train_dataloader), desc="Train", leave=False):
        verts, faces, labels = verts[0].float().to(device), faces[0].long().to(device), labels[0].long().to(device)
        mesh = Meshes(verts=verts*torch.tensor([-1, 1, -1], device=verts.device), faces=faces)

        cam_pos = viewpoint_learner(labels)  # (V, 3)

        v_idx = torch.randint(0, num_views, (1,)).item()
        pts = mesh_utils.render_mesh_torch(mesh, cam_pos[v_idx], num_points=num_points, device=device)
        pts = pcd_utils.normalize_pcd_tensor(pts)
        pts[:, [0, 2]] *= -1  # (N, 3)
        pcs.append(pts)

        if len(pcs) == batch_size:
            pcs = torch.stack(pcs)  # (B, N, 3)
        else:
            continue

        optimizer.zero_grad()
        preds = model(pcs)
        loss = loss_fn(preds, labels)
        repel = viewpoint_learner.repelling_loss()
        total_batch_loss = loss + lambda_repel * repel
        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item() * pcs.size(0)
        pred_labels = preds.argmax(dim=1)
        correct += pred_labels.eq(labels).sum().item()
        total += pcs.size(0)

        pcs = []

    scheduler.step()

    acc = 100.0 * correct / total
    avg_loss = total_loss / total
    print(f"Epoch {epoch+1}: Acc = {acc:.2f}%, Loss = {avg_loss:.4f}")

    # ========== Evaluation ==========
    model.eval()
    viewpoint_learner.eval()

    val_correct, val_total, val_loss = 0, 0, 0.0

    for pcs, labels in tqdm(test_loader, total=len(test_dataloader), desc="Eval", leave=False):
        pcs, labels = pcs.float().to(device), labels.long().to(device)

        with torch.no_grad():
            preds = model(pcs)
            loss = loss_fn(preds, labels)

        val_loss += loss.item() * B
        pred_labels = preds.argmax(dim=1)
        val_correct += pred_labels.eq(labels).sum().item()
        val_total += B

    val_acc = 100.0 * val_correct / val_total
    val_avg_loss = val_loss / val_total
    print(f"[Eval] Acc = {val_acc:.2f}%, Loss = {val_avg_loss:.4f}")

