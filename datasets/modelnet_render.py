import os
from pathlib import Path
import pickle
import numpy as np
import open3d as o3d
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from pytorch3d.structures import Meshes

import utils.mesh_utils as mesh_utils
import utils.pcd_utils as pcd_utils
from configs.load_class_map import load_class_map

PROJ_ROOT = Path(__file__).resolve().parents[2]

class ModelNetRender(Dataset):
    def __init__(self, root='modelnet40_manually_aligned', class_map='modelnet11', 
                split='train', num_points=1024, 
                num_views=3, viewpoint_mode="fixed", 
                cache_dir=None, use_cache=True, device="cuda"):
        self.root_dir = PROJ_ROOT / "data" / root
        self.split = split
        self.num_points = num_points
        self.num_views = num_views
        self.device = torch.device(device) or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_map = class_map if isinstance(class_map, dict) else load_class_map(class_map)
        
        self.num_classes = len(set(self.class_map.values()))
        
        self.viewpoints = self._get_viewpoints(self.num_views, viewpoint_mode)
            
        self.cache_dir = cache_dir or os.path.join(self.root_dir, '_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_file = os.path.join(
            self.cache_dir,
            f'modelnetrender_{split}_{num_points}pts_{self.num_classes}cls_{viewpoint_mode}{self.num_views}view.pkl'
        )

        if use_cache and os.path.exists(self.cache_file):
            print(f"[ModelNetRender] Loading cached data from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                cached = pickle.load(f)
                self.data = cached['data']
                self.labels = cached['labels']
        else:
            print("[ModelNetRender] Processing raw data...")
            self.data, self.labels = self._process_raw_data()
            if use_cache:
                print(f"\n[ModelNetRender] Saving processed data to cache: {self.cache_file}")
                with open(self.cache_file, 'wb') as f:
                    pickle.dump({'data': self.data, 'labels': self.labels}, f)       
                    
        self.base_len = len(self.labels) // self.num_views
        print(
            f"[ModelNetRender] Loaded {len(self.labels)} samples: {self.base_len} objects x {self.num_views} views, "
            f"from {split} split across {self.num_classes} classes ({num_points} pts per sample).\n"
        )

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def _process_raw_data(self):
        data, labels = [], []

        for class_idx, (class_name, label) in enumerate(self.class_map.items()):
            split_dirs = ['train', 'test'] if self.split == 'all' else [self.split]

            for split in split_dirs:
                class_split_dir = os.path.join(self.root_dir, class_name, split)
                if not os.path.isdir(class_split_dir):
                    continue
                file_list = sorted([f for f in os.listdir(class_split_dir) if f.endswith('.off')])
    
                tqdm.write(f"\n[{class_idx+1}/{len(self.class_map)}] Class {label:2d} '{class_name}' ({split}): {len(file_list)} files")
                for fname in tqdm(
                    file_list,
                    desc=f"    Processing",
                    leave=False
                ):
                    mesh_path = os.path.join(class_split_dir, fname)
                    mesh = o3d.io.read_triangle_mesh(mesh_path)
                    mesh = mesh_utils.align_mesh(mesh, class_name)
                    mesh = mesh_utils.normalize_mesh(mesh)
                    
                    verts_np, faces_np = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
                    verts_np[:, [0, 2]] *= -1  # Open3D to PyTorch3D coord flip
                    verts = torch.tensor(verts_np, dtype=torch.float32, device=self.device)
                    faces = torch.tensor(faces_np, dtype=torch.int64, device=self.device)
                    mesh_torch = Meshes(verts=[verts], faces=[faces])

                    for camera_pos in torch.tensor(self.viewpoints, dtype=torch.float32, device=self.device):
                        points = mesh_utils.render_mesh_torch(mesh_torch, camera_pos, num_points=self.num_points, device=self.device)
                        
                        points[:, [0, 2]] *= -1  # PyTorch3D to Open3D coord flip
                        points = pcd_utils.normalize(points)
                        
                        data.append(points)
                        labels.append(label)

        return torch.stack(data).float().cpu().numpy(), \
               torch.tensor(labels).long().cpu().numpy()

    def _get_viewpoints(self, num_views, viewpoint_mode, distance=2.5):
        """
        Generate viewpoints according to the specified mode.
        Modes:
            - "fixed": hard-coded canonical views (only supports num_views=3)
            - "fibonacci/auto": evenly distributed quarter-sphere views (y >= 0, z >= 0),
                                balanced between +x and -x
        """
        if viewpoint_mode == "fixed":
            if num_views != 3:
                raise ValueError(f"Fixed mode only supports 3 views, got {num_views}")

            bases = np.array([
                [ 0,  1,  1],   # front-up
                [-1,  1,  1],   # left-front-up
                [ 1,  0,  1],   # right-front
            ], dtype=np.float32)

            flip = np.array([-1, 1, -1], dtype=np.float32)  # Open3D to PyTorch3D coord flip
            return (bases * distance * flip).astype(np.float32)
        
        elif viewpoint_mode == "auto":            
            # oversample using Fibonacci sphere
            n_candidates = num_views * 20  # oversample factor
            indices = np.arange(n_candidates)
            phi = np.arccos(1 - 2 * (indices + 0.5) / n_candidates)  # polar angle
            theta = np.pi * (1 + 5**0.5) * (indices + 0.5)           # azimuth

            x = np.cos(theta) * np.sin(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(phi)
            pts = np.stack([x, y, z], axis=1)  # unit sphere

            # restrict to quarter sphere (y >= 0, z >= 0)
            mask = (pts[:, 1] >= 0) & (pts[:, 2] >= 0)
            pts = pts[mask]

            # split into +x and -x
            pos_x = pts[pts[:, 0] >= 0]
            neg_x = pts[pts[:, 0] < 0]

            n_pos = num_views // 2
            n_neg = num_views - n_pos

            # ensure no similar views--angular fps (pytorch3d fps receive batched tensors)            
            # pos_x_t = torch.from_numpy(pos_x).float().unsqueeze(0).to(self.device)  # (1, N, 3)
            # neg_x_t = torch.from_numpy(neg_x).float().unsqueeze(0).to(self.device)  # (1, N, 3)
            
            # chosen_pos = pcd_utils.fps(pos_x_t, n_pos).squeeze(0).cpu().numpy()  # (n_pos, 3)
            # chosen_neg = pcd_utils.fps(neg_x_t, n_neg).squeeze(0).cpu().numpy()  # (n_neg, 3)

            # random sampling
            chosen_pos = pos_x[np.random.choice(len(pos_x), size=n_pos, replace=False)]
            chosen_neg = neg_x[np.random.choice(len(neg_x), size=n_neg, replace=False)]

            chosen = np.vstack([chosen_pos, chosen_neg])

            # scale to given distance (ensure viewpoints fall outside mesh)
            chosen *= distance

            # Open3D to PyTorch3D coord flip
            flip = np.array([-1, 1, -1], dtype=np.float32)
            return (chosen * flip).astype(np.float32)
        
        else:
            raise ValueError(f"Unsupported viewpoint_mode={viewpoint_mode}")


if __name__ == "__main__":
    # Set seeds
    from utils.train_utils import set_seed
    set_seed(42)

    dataset = ModelNetRender(
        root="modelnet40_manually_aligned", 
        class_map="modelnet11",
        # num_views=8,
        # viewpoint_mode="auto",
        # split='test'
    )

    print(f"Viewpoints: {dataset.viewpoints}\n")

    from collections import defaultdict

    label_to_classnames = defaultdict(list)
    for name, label in dataset.class_map.items():
        label_to_classnames[label].append(name)
    
    viz = True
    if viz:
        # --- Choose which classes to visualize ---
        # Option A: define manually
        chosen_labels = set([0, 1, 2])  # replace with any label IDs
        num_chosen = len(chosen_labels)

        # Option B: pick labels randomly (comment out above to use this)
        # num_chosen = 3
        # chosen_labels = set(np.random.choice(np.arange(dataset.num_classes), size=num_chosen, replace=False))

        print("[ModelNetRender] Collecting all views of a sample object from {len(chosen_labels)} classes for visualization...")
        
        num_views = dataset.num_views
        sample_points = []       

        for idx in range(dataset.base_len):    
            start = idx * num_views
            end = (idx + 1) * num_views
            label = dataset.labels[start]

            if label in chosen_labels:
                views = dataset.data[start:end]
                sample_points.extend(views)
                chosen_labels.remove(label)

                print(f"  - Class {label:2d} ({', '.join(label_to_classnames[label]):<20})")

        # Visualize all views together
        print(f"\nVisualizing {num_chosen} samples ({num_views} views per sample class)...")    
        pcd_utils.viz_pcd(sample_points, rows=num_chosen)
