import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

from train_rl import brute_force_align

# ==================== Config ====================
PROJ_ROOT = Path("/mmfs1/projects/smartlab/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_name = "exp_mn2nn_11cls_3render_20250910_191436"
embed_dim = 1024
num_classes = 11
batch_size = 32

# ==================== Dataset ====================
train_dataset = ModelNetMesh(
    root_dir=PROJ_ROOT / "data/modelnet40_manually_aligned",
    class_map=PROJ_ROOT / "code/configs/class_map_modelnet11.json",
    split="train"
)

test_dataset = ScanObjectNN(
    root_dir=PROJ_ROOT / "data/scanobjectnn/main_split_nobg",
    class_map=PROJ_ROOT / "code/configs/class_map_scanobjectnn11.json",
    split="test"
)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ==================== Model ====================
encoder = PointNetLiteEncoder().to(device)
head = PointNetClsHead(out_dim=num_classes).to(device)

ckpt_dir = PROJ_ROOT / f"code/experiments/{ckpt_name}"
ckpt_path = ckpt_dir / "checkpoint_best.pth"
if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
_ = load_checkpoint(ckpt_path, device, encoder=encoder, head=head)

encoder.eval()
head.eval()

# ==================== Evaluation ====================
ce_loss = nn.CrossEntropyLoss()
val_correct, val_total, val_total_loss = 0, 0, 0.0

all_logits = []
all_labels = []
all_ranks = []   # position of correct label in sorted logits

with torch.no_grad():
    for pcs, labels in tqdm(test_dataloader, desc="Test"):
        pcs, labels = pcs.float().to(device), labels.long().to(device)

        feats = encoder(pcs)
        logits = head(feats)  # (B, C)

        # Accuracy & loss
        preds = logits.argmax(dim=1)
        val_correct += preds.eq(labels).sum().item()
        val_total += labels.size(0)
        loss = ce_loss(logits, labels)
        val_total_loss += loss.item() * labels.size(0)

        # Save for analysis
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        # Compute rank of correct label for each sample
        sorted_indices = torch.argsort(logits, dim=1, descending=True)  # (B, C)
        for i in range(labels.size(0)):
            correct_label = labels[i].item()
            rank = (sorted_indices[i] == correct_label).nonzero(as_tuple=True)[0].item() + 1
            all_ranks.append(rank)

# ==================== Results ====================
val_acc = 100.0 * val_correct / val_total
val_avg_loss = val_total_loss / val_total
print(f"[Test] Acc={val_acc:.2f}% Loss={val_avg_loss:.4f}")

# Stack & save
all_logits = np.concatenate(all_logits, axis=0)  # (N, C)
all_labels = np.concatenate(all_labels, axis=0)  # (N,)
all_ranks = np.array(all_ranks)                  # (N,)

# ==================== Quick Stats ====================
# Build class names from dataset.class_map (merge multiple names per label)
inv_map = {}
for name, lab in test_dataset.class_map.items():
    inv_map.setdefault(lab, []).append(name)
class_names = ["/".join(inv_map[i]) for i in range(num_classes)]

# Confusion matrix
preds = np.argmax(all_logits, axis=1)
cm = confusion_matrix(all_labels, preds)

# Histogram of ranks (skip rank=1)
rank_bins = np.arange(2, num_classes+2)
hist_data = []
for c in range(num_classes):
    class_ranks = all_ranks[all_labels == c]
    counts, _ = np.histogram(class_ranks, bins=rank_bins)
    hist_data.append(counts)
hist_data = np.array(hist_data)

# Misclassification ranking with counts and %
mis_ranked = []
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
for i, cname in enumerate(class_names):
    mis_counts = [(class_names[j], cm[i, j], cm_norm[i, j])
                  for j in range(num_classes) if j != i]
    mis_counts.sort(key=lambda x: x[1], reverse=True)
    mis_ranked.append(mis_counts)


'''
# === Plot ===
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# (a) Confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=axes[0], xticks_rotation=45, cmap="Blues", colorbar=False)
axes[0].set_title("Confusion Matrix")

# (b) Rank histogram (≥2) with counts
bottom = np.zeros(hist_data.shape[1])
for c in range(num_classes):
    bars = axes[1].bar(
        rank_bins[:-1],
        hist_data[c],
        bottom=bottom,
        width=0.8,
        label=f"{class_names[c]} ({c})"
    )
    # Annotate counts
    for bar, count in zip(bars, hist_data[c]):
        if count > 0:
            axes[1].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_y() + bar.get_height()/2,
                str(int(count)),
                ha="center", va="center", fontsize=7, color="white"
            )
    bottom += hist_data[c]

axes[1].set_xlabel("Rank of Correct Class (≥2)")
axes[1].set_ylabel("Number of Predictions")
axes[1].set_title("Distribution of Correct Label Ranks (Errors)")
axes[1].legend(ncol=2, fontsize=7)

# (c) Misclassification map
topN = min(10, num_classes-1)  # limit columns
axes[2].imshow(np.ones((num_classes, topN)), cmap="Greys", alpha=0)  # blank grid
axes[2].set_xticks(np.arange(topN))
axes[2].set_xticklabels([f"Top{r}" for r in range(1, topN+1)], rotation=45)
axes[2].set_yticks(np.arange(num_classes))
axes[2].set_yticklabels(class_names)
axes[2].set_title(f"Top-{topN} Misclassified Targets (Count, %)")

for i in range(num_classes):
    for j in range(topN):
        pred_name, count, frac = mis_ranked[i][j]
        text = f"{pred_name}\n({count},{frac*100:.1f}%)"
        axes[2].text(j, i, text, ha="center", va="center", fontsize=7)

plt.tight_layout()
plt.show()
'''

'''
# === Viz ===
true_classes = ["cabinet", "desk", "sink", "table"]

# --- build colored pcds ---
rows_to_viz = []
row_captions = []

for cname in true_classes:
    # Find label id
    true_label = None
    for k, v in inv_map.items():
        if cname in v:
            true_label = k
            break
    if true_label is None:
        continue

    # Confusion row: sort descending by count
    cm_row = [(j, np.sum((all_labels==true_label) & (preds==j)))
              for j in range(len(class_names)) if j != true_label]
    cm_row = [(j, cnt) for j, cnt in cm_row if cnt > 0]
    cm_row.sort(key=lambda x: x[1], reverse=True)
    chosen_targets = [j for j, cnt in cm_row[:4]]

    row_samples = []

    # --- True sample (blue) ---
    correct_idx = np.where((all_labels==true_label) & (preds==true_label))[0]
    if len(correct_idx) > 0:
        pts = test_dataset[correct_idx[0]][0]
        row_samples.append(pcd_utils.init_pcd(pts, colors=np.array([0, 153, 255])))

    # --- Misclassified samples (red) ---
    for t in chosen_targets:
        mis_idx = np.where((all_labels==true_label) & (preds==t))[0]
        if len(mis_idx) > 0:
            pts = test_dataset[mis_idx[0]][0]
            row_samples.append(pcd_utils.init_pcd(pts, colors=np.array([255, 0, 0])))
        else:
            row_samples.append(pcd_utils.init_pcd(np.zeros_like(test_dataset[0][0]),
                                                  colors=np.array([128,128,128])))

    rows_to_viz.extend(row_samples)
    row_captions.append(
        f"{cname} → {', '.join(class_names[t] for t in chosen_targets)}"
    )

print(f"[Visualization] {len(row_captions)} rows, each with 1 true (blue) + 4 misclassified (red)")
for cap in row_captions:
    print(" -", cap)

# --- Visualize ---
pcd_utils.viz_pcd(rows_to_viz, rows=len(row_captions))
'''


# ==================== Refinement via Registration ====================

# Collect wrong predictions
preds = np.argmax(all_logits, axis=1)
wrong_indices = np.where(preds != all_labels)[0]
rng = np.random.default_rng()
rng.shuffle(wrong_indices)
wrong_indices = wrong_indices[:4]  # pick 4 wrong predictions for demo

results = []

for idx in wrong_indices:
    pts = test_dataset[idx][0]  # partial scan
    true_lab = all_labels[idx]
    pred_lab = preds[idx]

    # --- top-4 candidate classes from logits ---
    top4 = np.argsort(all_logits[idx])[::-1][:4]

    best_score = float("inf")
    best_class = None
    best_mesh = None
    best_aligned = None

    # loop over candidate classes
    for cand in top4:
        mesh_indices = [i for i, y in enumerate(train_dataset.labels) if y == cand]
        if len(mesh_indices) == 0:
            continue
        # just pick first mesh of that class
        verts, faces, _ = train_dataset[mesh_indices[0]]

        aligned, params, score = brute_force_align(pts, verts, faces, device=device)
        if score < best_score:
            best_score = score
            best_class = cand
            best_mesh = (verts, faces)
            best_aligned = aligned

    results.append((pts, true_lab, pred_lab, best_class, best_mesh, best_aligned, best_score))

    print(f"\n[Refine] idx={idx}, "
          f"True={class_names[true_lab]}, "
          f"Pred={class_names[pred_lab]}, "
          f"Final={class_names[best_class]}, "
          f"Chamfer={best_score:.4f}")

'''
# ==================== Visualization ====================
vis_objs = []
spacing = 3
soft_red = np.array([255, 100, 100])   # partial
soft_blue = np.array([100, 150, 255])  # mesh

for i, (pts, true_lab, pred_lab, best_class, (verts, faces), aligned, score) in enumerate(results):
    row = i // 2
    col = i % 2

    mesh = mesh_utils.init_mesh(verts, faces).translate([col*spacing, -row*spacing, 0])
    pcd = pcd_utils.init_pcd(aligned, colors=soft_red).translate([col*spacing, -row*spacing, 0])

    vis_objs.extend([mesh, pcd])

    print(f"Row {row}, Col {col}: True={class_names[true_lab]}, "
          f"Pred={class_names[pred_lab]}, "
          f"BestMesh={class_names[best_class]}")

o3d.visualization.draw_geometries(
    vis_objs,
    mesh_show_wireframe=True,
    mesh_show_back_face=True
)
'''