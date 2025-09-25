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
PROJ_ROOT = Path("/mmfs1/projects/smartlab/")
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
train_dataset = ModelNetMesh(
    root_dir=PROJ_ROOT / "data/modelnet40_manually_aligned",
    class_map=PROJ_ROOT / "code/configs/class_map_modelnet11.json",
    split="train",
)

val_dataset = ModelNetRender(
    root_dir=PROJ_ROOT / "data/modelnet40_manually_aligned",
    class_map=PROJ_ROOT / "code/configs/class_map_modelnet11.json",
    split="train",
    num_views=num_views,
    viewpoint_mode="fixed"
)

test_dataset = ScanObjectNN(
    root_dir=PROJ_ROOT / "data/scanobjectnn/main_split_nobg",
    class_map=PROJ_ROOT / "code/configs/class_map_scanobjectnn11.json",
    split="test"
)

def apply_yaw_scale(points, yaw, scale):
    """Apply yaw rotation (around Y-axis) and isotropic scaling."""
    return pcd_utils.rotate_xyz(points, (0.0, yaw, 0.0)) * scale

def apply_yaw_scale_trans(points, yaw, scale, mesh_verts=None):
    pts = apply_yaw_scale(points, yaw, scale)
    if mesh_verts is not None:
        # center-align
        t = mesh_verts.mean(0) - pts.mean(0)
        pts = pts + t
    return pts


def one_way_chamfer(points, verts, faces, device='cuda'):
    """
    Compute one-way Chamfer distance: partial points -> mesh surface.
    """
    # Convert numpy -> torch
    if not torch.is_tensor(points):
        points = torch.tensor(points, dtype=torch.float32)
    if not torch.is_tensor(verts):
        verts = torch.tensor(verts, dtype=torch.float32)
    if not torch.is_tensor(faces):
        faces = torch.tensor(faces, dtype=torch.int64)

    points = points.to(device)    # (N,3)
    verts = verts.to(device)      # (V,3)
    faces = faces.to(device)      # (F,3)

    # Get triangle vertices
    tris = verts[faces]           # (F,3,3)
    a, b, c = tris[:,0], tris[:,1], tris[:,2]  # (F,3) each

    # Expand points vs faces
    P = points.unsqueeze(1)       # (N,1,3)
    A = a.unsqueeze(0)            # (1,F,3)
    B = b.unsqueeze(0)
    C = c.unsqueeze(0)

    # Compute edge vectors
    AB, AC, AP = B - A, C - A, P - A
    d1, d2 = (AP*AB).sum(-1), (AP*AC).sum(-1)

    BP = P - B
    d3, d4 = (BP*AB).sum(-1), (BP*AC).sum(-1)

    CP = P - C
    d5, d6 = (CP*AB).sum(-1), (CP*AC).sum(-1)

    # Distances squared
    dist2 = torch.full((points.shape[0], faces.shape[0]), float("inf"), device=device)

    # region A
    mask = (d1 <= 0) & (d2 <= 0)
    dist2[mask] = (AP.norm(dim=-1)**2)[mask]

    # region B
    mask = (d3 >= 0) & (d4 <= d3)
    dist2[mask] = (BP.norm(dim=-1)**2)[mask]

    # region C
    mask = (d6 >= 0) & (d5 <= d6)
    dist2[mask] = (CP.norm(dim=-1)**2)[mask]

    # edge AB
    vc = d1*d4 - d3*d2
    mask = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v = d1 / (d1 - d3 + 1e-12)
    proj = A + v.unsqueeze(-1)*AB
    dist2[mask] = ((P-proj).norm(dim=-1)**2)[mask]

    # edge AC
    vb = d5*d2 - d1*d6
    mask = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w = d2 / (d2 - d6 + 1e-12)
    proj = A + w.unsqueeze(-1)*AC
    dist2[mask] = ((P-proj).norm(dim=-1)**2)[mask]

    # edge BC
    va = d3*d6 - d5*d4
    mask = (va <= 0) & ((d4-d3) >= 0) & ((d5-d6) >= 0)
    w = (d4-d3) / ((d4-d3)+(d5-d6)+1e-12)
    proj = B + w.unsqueeze(-1)*(C-B)
    dist2[mask] = ((P-proj).norm(dim=-1)**2)[mask]

    # inside face
    denom = (va+vb+vc+1e-12)
    v = vb/denom
    w = vc/denom
    proj = A + AB*v.unsqueeze(-1) + AC*w.unsqueeze(-1)
    inside_mask = ~( (dist2 < float("inf")).any(dim=1, keepdim=True).expand_as(dist2) )
    dist2[inside_mask] = ((P-proj).norm(dim=-1)**2)[inside_mask]

    # min dist per point
    min_dists = torch.sqrt(dist2.min(dim=1).values)  # (N,)
    return min_dists.mean().item()

#     # Convert numpy -> torch tensors on device
#     points_t = torch.tensor(points, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N, 3)
#     verts_t  = torch.tensor(verts,  dtype=torch.float32, device=device).unsqueeze(0)  # (1, V, 3)
#     faces_t  = torch.tensor(faces,  dtype=torch.int64,   device=device).unsqueeze(0)  # (1, F, 3)

#     # Construct mesh
#     mesh = Meshes(verts=verts_t, faces=faces_t)

#     verts_packed = mesh.verts_packed()
#     faces_packed = mesh.faces_packed()
#     tris = verts_packed[faces_packed]  # (F, 3, 3)

#     dists2 = point_face_distance(points_t, tris.unsqueeze(0))  # (1, N, F)
#     dists, _ = torch.min(dists2, dim=-1)  # min dist per point
#     return torch.sqrt(dists).mean().detach().cpu().numpy().item()

    # # Squared distance from each point -> closest triangle
    # dists2 = point_mesh_face_distance(mesh, points_t)  # (1, N)
    
    # # Return mean distance (numpy float)
    # return torch.sqrt(dists2).mean().detach().cpu().numpy().item()

def brute_force_align(points, verts, faces,
                      yaw_res=np.pi/18,  # 10° in radians
                      scale_res=0.05,
                      s_min=0.8, s_max=1.2,
                      device="cuda"):
    """
    Brute-force search for best yaw+scale alignment of partial -> mesh.
    """
    best_score = float("inf")
    best_params = None
    best_aligned = None
    
    yaw_vals = np.arange(0, 2*np.pi, yaw_res)
    scale_vals = np.arange(s_min, s_max+1e-6, scale_res)    
    total_iters = len(yaw_vals) * len(scale_vals)

    with tqdm(total=total_iters, desc="Brute-force alignment", leave=False) as pbar:
        for yaw in yaw_vals:
            for s in scale_vals:
                aligned = apply_yaw_scale(points, yaw, s)
                # aligned = apply_yaw_scale_trans(points, yaw, s, mesh_verts=verts)
                score = one_way_chamfer(aligned, verts, faces, device=device)
                if score < best_score:
                    best_score = score
                    best_params = (yaw, round(s, 3))
                    best_aligned = aligned
                pbar.update(1)

    return best_aligned, best_params, best_score


def pick_pairs(train_dataset, val_dataset, classes, num_views, device="cuda"):
    results_same, results_diff = [], []
    rng = np.random.default_rng()

    for cls in classes:
        mesh_indices = [i for i, y in enumerate(train_dataset.labels) if y == cls]

        if len(mesh_indices) < 2:
            continue

        # ---- same object pair ----
        idx_mesh = mesh_indices[0]
        verts, faces, _ = train_dataset[idx_mesh]

        # pick one random view for this mesh
        start_idx = idx_mesh * num_views
        view_idx = rng.integers(0, num_views)
        points, _ = val_dataset[start_idx + view_idx]

        aligned, params, score = brute_force_align(points, verts, faces, device=device)
        results_same.append((verts, faces, aligned, params, score))

        # ---- different object pair ----
        idx_mesh2 = mesh_indices[1]
        verts2, faces2, _ = train_dataset[idx_mesh2]

        start_idx2 = idx_mesh2 * num_views
        view_idx2 = rng.integers(0, num_views)
        points2, _ = val_dataset[start_idx2 + view_idx2]

        aligned2, params2, score2 = brute_force_align(points2, verts2, faces2, device=device)
        results_diff.append((verts2, faces2, aligned2, params2, score2))

    return results_same, results_diff

def pick_pairs_nocorr(train_dataset, test_dataset, classes, device="cuda"):
    """
    Build mesh ↔ partial pairs when test_dataset has no view correspondence
    (e.g., ScanObjectNN).
    """
    results_same, results_diff = [], []
    rng = np.random.default_rng()

    for cls in classes:
        mesh_indices = [i for i, y in enumerate(train_dataset.labels) if y == cls]
        pcd_indices = [i for i, y in enumerate(test_dataset.labels) if y == cls]

        if len(mesh_indices) < 2 or len(pcd_indices) < 2:
            continue

        # ---- same-class pair (mesh + one partial from same class) ----
        idx_mesh = rng.choice(mesh_indices)
        verts, faces, _ = train_dataset[idx_mesh]

        idx_pcd = rng.choice(pcd_indices)
        points, _ = test_dataset[idx_pcd]

        aligned, params, score = brute_force_align(points, verts, faces, device=device)
        results_same.append((verts, faces, aligned, params, score))

        # ---- different-object pair (mesh + another partial of same class) ----
        idx_mesh2 = rng.choice(mesh_indices)
        verts2, faces2, _ = train_dataset[idx_mesh2]

        idx_pcd2 = rng.choice([i for i in pcd_indices if i != idx_pcd])
        points2, _ = test_dataset[idx_pcd2]

        aligned2, params2, score2 = brute_force_align(points2, verts2, faces2, device=device)
        results_diff.append((verts2, faces2, aligned2, params2, score2))

    return results_same, results_diff


def visualize_pairs(results_same, results_diff, spacing=3):
    vis_objs = []

    soft_red  = np.array([255, 160, 122])   # salmon
    soft_blue = np.array([135, 206, 250])   # sky blue

    # row 1 = same pairs
    for i, (verts, faces, aligned, params, score) in enumerate(results_same):
        mesh = mesh_utils.init_mesh(verts, faces)
        pcd = pcd_utils.init_pcd(aligned, colors=soft_red)
        mesh = mesh.translate([i*spacing, 0, 0])
        pcd = pcd.translate([i*spacing, 0, 0])
        vis_objs.extend([mesh, pcd])
        print(f"[SAME] yaw={params[0]:.3f} rad, scale={params[1]}, chamfer={score:.4f}")

    # row 2 = diff pairs
    for i, (verts, faces, aligned, params, score) in enumerate(results_diff):
        mesh = mesh_utils.init_mesh(verts, faces)
        pcd = pcd_utils.init_pcd(aligned, colors=soft_blue)
        mesh = mesh.translate([i*spacing, -spacing, 0])
        pcd = pcd.translate([i*spacing, -spacing, 0])
        vis_objs.extend([mesh, pcd])
        print(f"[DIFF] yaw={params[0]:.3f} rad, scale={params[1]}, chamfer={score:.4f}")

    o3d.visualization.draw_geometries(
        vis_objs, mesh_show_wireframe=True, mesh_show_back_face=True
    )

'''
# Build class names from dataset.class_map (merge multiple names per label)
inv_map = {}
for name, lab in test_dataset.class_map.items():
    inv_map.setdefault(lab, []).append(name)
class_names = ["/".join(inv_map[i]) for i in range(num_classes)]


# Randomly choose labels
chosen_labels = random.sample(set(train_dataset.labels), 3)
chosen_names = [class_names[l] for l in chosen_labels]
print("Chosen classes:", chosen_names)

# results_same, results_diff = pick_pairs(train_dataset, val_dataset, chosen_labels, num_views=num_views, device=device)
results_same, results_diff = pick_pairs_nocorr(train_dataset, test_dataset, chosen_labels, device=device)

print("\n")
visualize_pairs(results_same, results_diff)


'''