import os
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

# ckpt_name_teacher = "exp_mn2nn_11cls_20250910_190250"
# embed_dim = 1024

ckpt_name_teacher = "exp_mn2nn_11cls_maefinetune_20250910_233314"
ckpt_name_student = "exp_mn11_3render_maepretrain_20250910_200040"
embed_dim = 384

num_classes = 11
num_views = 3
epochs = 200
batch_size = 32
lambda_mse = 0.3
lambda_fd = 0.1  # 0.07 for pointnetlite, 0.1 for pointmae
lambda_rd = 0.1

# ==================== Logging ====================
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# exp_dir = PROJ_ROOT / f"code/experiments/train_al_{timestamp}"
# exp_dir.mkdir(parents=True, exist_ok=True)

# logging.basicConfig(filename=exp_dir / "log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()
# logger.addHandler(logging.StreamHandler())

# ==================== Dataset ====================
# student_dataset = ModelNetRender(
#     root_dir=PROJ_ROOT / "data/modelnet40_manually_aligned",
#     class_map=PROJ_ROOT / "code/configs/class_map_modelnet11.json",
# )

# teacher_dataset = ModelNet(
#     root_dir=PROJ_ROOT / "data/modelnet40_manually_aligned",
#     class_map=PROJ_ROOT / "code/configs/class_map_modelnet11.json",
# )

class ModelNetFullPartial(torch.utils.data.Dataset):
    def __init__(self, root_dir, class_map, num_views=3, split="train"):
        self.full = ModelNet(root_dir, class_map, split=split)
        self.partial = ModelNetRender(root_dir, class_map, split=split, num_views=num_views)
        self.num_views = num_views
        assert len(self.full) * num_views == len(self.partial)

    def __len__(self):
        return len(self.full)

    def __getitem__(self, idx):
        full_pc, label = self.full[idx]  # (N,3), int
        partials = []
        for j in range(self.num_views):
            part_pc, _ = self.partial[self.num_views * idx + j]
            partials.append(part_pc)
        partials = np.stack(partials, axis=0)  # (V,N,3)
        return full_pc, partials, label

train_dataset = AugWrapper(ModelNetFullPartial(PROJ_ROOT / "data/modelnet40_manually_aligned",
                                    PROJ_ROOT / "code/configs/class_map_modelnet11.json"))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_dataset = ScanObjectNN(
    root_dir=PROJ_ROOT / "data/scanobjectnn/main_split_nobg",
    class_map=PROJ_ROOT / "code/configs/class_map_scanobjectnn11.json",
    split="test"
)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ==================== Teacher / Student ====================
# Teacher: Pretrained PointMAE + classifier (frozen)
# teacher_encoder = PointNetLiteEncoder().to(device)
# teacher_head = PointNetClsHead(out_dim=num_classes).to(device)

teacher_encoder = PointMAEEncoder(mask_type='block', noaug=True).to(device)
teacher_head = PointMAEClsHead(out_dim=num_classes).to(device)

# load pretrained weights for teacher
ckpt_dir = PROJ_ROOT / f"code/experiments/{ckpt_name_teacher}"
ckpt_path = ckpt_dir / f"checkpoint_best.pth"
if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
_ = load_checkpoint(ckpt_path, device, encoder=teacher_encoder, head=teacher_head)

teacher_encoder.eval().requires_grad_(False)
teacher_head.eval().requires_grad_(False)

# Student: PointNetLite + classifier (trainable)
# student_encoder = PointNetLiteEncoder().to(device)
# student_head = PointNetClsHead(out_dim=num_classes).to(device)

# load pretrained weights for student
student_encoder = PointMAEEncoder(mask_type='rand', noaug=True).to(device)
student_head = PointMAEClsHead(out_dim=num_classes).to(device)
ckpt_dir = PROJ_ROOT / f"code/experiments/{ckpt_name_student}"
ckpt_path = ckpt_dir / f"checkpoint_best.pth"
if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
_ = load_checkpoint(ckpt_path, device, encoder=student_encoder)

# ==================== Feature Discriminator ====================
class FeatureDiscriminator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = build_fc_layers(dims=[embed_dim, 32, 1], dropout=0.5)

    def forward(self, x):
        return self.fc(x)

feat_disc = FeatureDiscriminator(embed_dim=embed_dim).to(device)

# ==================== Response Discriminator ====================
class ResponseDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = build_fc_layers(dims=[num_classes, 32, 1], dropout=0.5)

    def forward(self, x):
        return self.fc(x)

resp_disc = ResponseDiscriminator(num_classes=num_classes).to(device)

# ==================== Loss & Optim ====================
ce_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()

opt_student = Adam(
    list(student_encoder.parameters()) + list(student_head.parameters()),
    lr=1e-3, weight_decay=1e-4
)
sched_student = StepLR(opt_student, step_size=20, gamma=0.7)

opt_fd = Adam(
    feat_disc.parameters(),
    lr=1e-4, weight_decay=1e-4
)
sched_fd = StepLR(opt_fd, step_size=10, gamma=0.5) # faster decay

opt_rd = Adam(
    resp_disc.parameters(),
    lr=1e-4, weight_decay=1e-4
)
sched_rd = StepLR(opt_rd, step_size=10, gamma=0.5)

# ==================== Training Loop ====================
best_acc = 0.0
for epoch in range(epochs):
    student_encoder.train()
    student_head.train()
    feat_disc.train()
    resp_disc.train()
    
    # torch.cuda.reset_peak_memory_stats(device)
    # start_time = time.time()

    # Track loss components separately
    total, correct = 0, 0
    total_ce, total_mse, total_fd, total_adv_fd, total_rd, total_adv_rd, total_loss = 0, 0, 0, 0, 0, 0, 0

    # Iterate over unified dataset
    k = 5
    step = 0
    for full_pc, partials, labels in tqdm(train_dataloader, desc=f"Train {epoch+1}", leave=False):
        step += 1

        full_pc = full_pc.float().to(device)              # (B, N, 3)
        partials = partials.float().to(device)            # (B, V, N, 3)
        labels = labels.long().to(device)                # (B,)

        B, V, N, _ = partials.shape
        partials = partials.view(B*V, N, 3)       # flatten views: (B*V, N, 3)
        labels_rep = labels.unsqueeze(1).repeat(1, V).view(-1)  # (B*V,)

        # ---- Teacher forward (frozen) ----
        with torch.no_grad():
            f_t = teacher_encoder(full_pc)     # (B, D); (B, G, D) if pointmae
            l_t = teacher_head(f_t)            # (B, C)
            f_t = f_t.max(dim=1)[0]            # (B, D); enable if pointmae
        
        # ---- Expand teacher to match student views ----
        f_t = f_t.unsqueeze(1).repeat(1, V, 1).view(B*V, -1)   # (B*V, D)
        l_t = l_t.unsqueeze(1).repeat(1, V, 1).view(B*V, -1)   # (B*V, C)

        # =====================================================
        # 1) Update discriminators (detach student for stability)
        # =====================================================
        with torch.no_grad():
            f_s_det = student_encoder(partials)
            l_s_det = student_head(f_s_det)
            f_s_det = f_s_det.max(dim=1)[0]     # enable if pointmae

        # -- FD --        
        real_logits = feat_disc(f_t.detach())
        fake_logits = feat_disc(f_s_det.detach())
        real_targets = torch.empty_like(real_logits).uniform_(0.9, 1.0) # torch.ones_like(real_logits)
        fake_targets = torch.empty_like(fake_logits).uniform_(0.0, 0.1) # torch.zeros_like(fake_logits)
        loss_fd = (bce_loss(real_logits, real_targets) +
                bce_loss(fake_logits, fake_targets))
        if step % k == 0:
            opt_fd.zero_grad()
            loss_fd.backward()
            # clip gradients (L2 norm ≤ 1.0)
            torch.nn.utils.clip_grad_norm_(feat_disc.parameters(), max_norm=1.0)
            opt_fd.step()

        # -- RD --
        real_logits_resp = resp_disc(l_t.detach())
        fake_logits_resp = resp_disc(l_s_det.detach())        
        real_targets_resp = torch.empty_like(real_logits_resp).uniform_(0.9, 1.0) # torch.ones_like(real_logits_resp)
        fake_targets_resp = torch.empty_like(fake_logits_resp).uniform_(0.0, 0.1) # torch.zeros_like(fake_logits_resp)
        loss_rd = (bce_loss(real_logits_resp, torch.ones_like(real_logits_resp)) +
                bce_loss(fake_logits_resp, torch.zeros_like(fake_logits_resp)))
        if step % k == 0:
            opt_rd.zero_grad()
            loss_rd.backward()
            torch.nn.utils.clip_grad_norm_(resp_disc.parameters(), max_norm=1.0)
            opt_rd.step()

        # =====================================================
        # 2) Update student (new forward, discriminators fixed)
        # =====================================================
        f_s = student_encoder(partials)
        l_s = student_head(f_s)
        f_s = f_s.max(dim=1)[0]     # enable if pointmae

        loss_ce  = ce_loss(l_s, labels_rep)
        loss_mse = mse_loss(f_s, f_t)

        loss_adv_fd = bce_loss(feat_disc(f_s), torch.ones_like(feat_disc(f_s)))
        loss_adv_rd = bce_loss(resp_disc(l_s), torch.ones_like(resp_disc(l_s)))

        loss = loss_ce + lambda_mse*loss_mse + lambda_fd*loss_adv_fd # + lambda_rd*loss_adv_rd

        opt_student.zero_grad()
        loss.backward()
        opt_student.step()

        # ---- Stats ----
        pred_labels = l_s.argmax(dim=1)
        correct += pred_labels.eq(labels_rep).sum().item()
        total += labels_rep.size(0)
        total_loss += loss.item() * labels_rep.size(0)

        total_ce     += loss_ce.item()     * labels_rep.size(0)
        total_mse    += loss_mse.item()    * labels_rep.size(0)
        total_fd     += loss_fd.item()     * labels_rep.size(0)
        total_adv_fd += loss_adv_fd.item() * labels_rep.size(0)
        total_rd     += loss_rd.item()     * labels_rep.size(0)
        total_adv_rd += loss_adv_rd.item() * labels_rep.size(0)

    sched_student.step()
    sched_fd.step()
    sched_rd.step()

    train_acc = 100.0 * correct / total
    avg_loss = total_loss / total

    avg_ce     = total_ce / total
    avg_mse    = total_mse / total
    avg_fd     = total_fd / total
    avg_adv_fd = total_adv_fd / total
    avg_rd     = total_rd / total
    avg_adv_rd = total_adv_rd / total

    # epoch_time = time.time() - start_time
    # mem_used = torch.cuda.max_memory_allocated(device) / 1024**2
    # logger.info(f"[Train] Epoch {epoch+1}: Acc={train_acc:.2f}% Loss={avg_loss:.4f} "
    #             f"Time={epoch_time:.1f}s Mem={mem_used:.1f}MB")

    logger.info(
        f"[Train] Epoch {epoch+1}: "
        f"Acc={train_acc:.2f}% | Total={avg_loss:.4f} "
        f"| CE={avg_ce:.4f} | MSE={avg_mse:.4f} "
        f"| FD={avg_fd:.4f} | AdvFD={avg_adv_fd:.4f} "
        f"| RD={avg_rd:.4f} | AdvRD={avg_adv_rd:.4f} "
    )

    # ---- Evaluation ----
    student_encoder.eval()
    student_head.eval()

    val_correct, val_total, val_total_loss = 0, 0, 0.0
    with torch.no_grad():
        for pcs, labels in tqdm(test_dataloader, desc="Eval", leave=False):
            pcs, labels = pcs.float().to(device), labels.long().to(device)

            f_s = student_encoder(pcs)
            l_s = student_head(f_s)

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
    #     torch.save({
    #         "student_encoder": student_encoder.state_dict(),
    #         "student_head": student_head.state_dict(),
    #         "feat_disc": feat_disc.state_dict(),
    #         "resp_disc": resp_disc.state_dict(),
    #         "best_acc": best_acc,
    #         "epoch": epoch+1,
    #     }, exp_dir / "best.pth")
logger.info(f"Best Acc={best_acc:.2f}% at Epoch {best_epoch}")

    # if val_acc > best_acc:
    #     best_acc = val_acc

