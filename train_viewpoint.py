import os
import torch
from torch.utils.data import DataLoader
from pytorch3d.structures import Meshes

from dataset.modelnet_mesh import ModelNetMesh
from models.modules.viewpoint_learner import ViewpointLearner
import utils.mesh_utils as mesh_utils
import utils.pcd_utils as pcd_utils

# Setup
device = torch.device('cuda')
num_classes = 11
num_views = 3
num_points = 1024
lambda_repel = 0.05

# Dataset
DATA_ROOT = "/your_path/data/modelnet40_manually_aligned"
CLASS_MAP = "/your_path/code/configs/class_map_modelnet11.json"

train_set = ModelNetMesh(root_dir=DATA_ROOT, class_map=CLASS_MAP, split='train')
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

# Model
viewpoint_learner = ViewpointLearner(num_classes=num_classes, num_views=num_views).to(device)
optimizer = torch.optim.Adam(viewpoint_learner.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    total_loss = 0
    for verts, faces, label in train_loader:
        verts, faces, label = verts.to(device), faces.to(device), label.to(device)

        mesh = Meshes(verts=[verts], faces=[faces])
        cam_pos = viewpoint_learner(label)  # (1, V, 3)

        views = []
        for i in range(num_views):
            pts = mesh_utils.render_mesh_torch(mesh, cam_pos[0, i], num_points=num_points, device=device)
            pts = pcd_utils.normalize_pcd_tensor(pts)
            views.append(pts)

        view_tensor = torch.stack(views, dim=0)  # (V, N, 3)
        # TODO: Pass view_tensor through your encoder/classifier here
        # pred = model(view_tensor)

        # Placeholder loss
        cls_loss = torch.tensor(0.0, requires_grad=True, device=device)
        repel = viewpoint_learner.repelling_loss()
        loss = cls_loss + lambda_repel * repel

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch:02d} | Loss: {total_loss:.4f}")
