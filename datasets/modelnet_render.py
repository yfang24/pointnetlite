import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import open3d as o3d
from pytorch3d.structures import Meshes

import utils.mesh_utils as mesh_utils
import utils.pcd_utils as pcd_utils
from configs.load_class_map import load_class_map

class ModelNetRender(Dataset):
    def __init__(self, root_dir, class_map, split='train', num_points=1024, num_views=3, single_view=False, cache_dir=None, use_cache=True, device=None):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.num_views = num_views
        self.single_view = single_view
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_map = class_map if isinstance(class_map, dict) else load_class_map(class_map)
        
        self.viewpoints = self._get_viewpoints(self.num_views)
            
        self.cache_dir = cache_dir or os.path.join(root_dir, '_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(
            self.cache_dir,
            f'modelnetrender_{split}_{num_points}pts_{len(set(self.class_map.values()))}cls_{self.num_views}view.pkl'
        )

        if use_cache and os.path.exists(self.cache_file):
            print(f"[ModelNetRender] Loading cached data from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                cached = pickle.load(f)
                self.data = cached['data']
                self.labels = cached['labels']
        else:
            print("[ModelNetRender] Processing and rendering data...")
            self.data, self.labels = self._process_and_render()
            if use_cache:
                print(f"[ModelNetRender] Saving processed data to cache: {self.cache_file}")
                with open(self.cache_file, 'wb') as f:
                    pickle.dump({'data': self.data, 'labels': self.labels}, f)       
                    
        self.base_len = len(self.data) // self.num_views
        print(f"\n[ModelNetRender] Loaded {len(self.labels)} samples: {self.base_len} objects x {self.num_views} views, across {len(set(self.labels.tolist()))} classes.")

    def __len__(self):
        return self.base_len if self.single_view else len(self.labels)
        
    def __getitem__(self, idx):
        if self.single_view:
            start = idx * self.num_views
            end = (idx + 1) * self.num_views
            views = self.data[start:end]
            view_idx = np.random.randint(self.num_views)
            return views[view_idx], self.labels[start]
        else:
            return self.data[idx], self.labels[idx]

    def _process_and_render(self):
        data, labels = [], []        

        for class_idx, (class_name, label) in enumerate(self.class_map.items()):
            split_dirs = ['train', 'test'] if self.split == 'all' else [self.split]

            for split in split_dirs:
                class_dir = os.path.join(self.root_dir, class_name, split)
                if not os.path.isdir(class_dir):
                    continue
                file_list = sorted([f for f in os.listdir(class_dir) if f.endswith('.off')])

                print(f"[{class_idx+1}/{len(self.class_map)}] Class '{class_name}' ({split}): {len(file_list)} files")
                for fname in tqdm(file_list, desc=f"    Rendering {class_name}", leave=False):
                    mesh_path = os.path.join(class_dir, fname)
                    mesh = o3d.io.read_triangle_mesh(mesh_path)
                    mesh = mesh_utils.align_mesh(mesh, class_name)
                    mesh = mesh_utils.normalize_mesh(mesh)
                    verts_np, faces_np = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
                    verts_np[:, [0, 2]] *= -1
                    verts = torch.tensor(verts_np, dtype=torch.float32, device=self.device)
                    faces = torch.tensor(faces_np, dtype=torch.int64, device=self.device)
                    mesh_torch = Meshes(verts=[verts], faces=[faces])

                    for camera_pos in self.viewpoints.to(self.device):
                        points = mesh_utils.render_mesh_torch(mesh_torch, camera_pos, num_points=self.num_points, device=self.device)
                        
                        points[:, [0, 2]] *= -1
                        points = pcd_utils.normalize_pcd_tensor(points)
                        
                        data.append(points)
                        labels.append(label)

        return torch.stack(data).float().cpu(), \
               torch.tensor(labels).long().cpu()

    def _get_viewpoints(self, num_views):
        if num_views == 3:
            bases = torch.tensor([
                [ 0,  1,  1],  # front-up
                [ -1,  1,  1],  # left-front-up
                [ 1,  0,  1],  # right-front
            ], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported number of views: {num_views}")
        
        distance = torch.tensor([2.5, 2.5, 2.5], dtype=torch.float32)  # (x, y, z) scale
        flip = torch.tensor([-1, 1, -1], dtype=torch.float32)  # Open3D to PyTorch3D coord flip
        
        return bases * distance * flip  # shape: (num_views, 3)


if __name__ == "__main__":
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJ_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
    
    DATA_DIR = os.path.join(PROJ_ROOT, "data/modelnet40_manually_aligned")
    CLASS_MAP_PATH = os.path.join(PROJ_ROOT, "code/configs/class_map_modelnet11.json")

    split = "train"
    
    class_map = load_class_map(CLASS_MAP_PATH)
    inv_class_map = {v: k for k, v in class_map.items()}

    dataset = ModelNetRender(
        root_dir=DATA_DIR,
        class_map=CLASS_MAP_PATH,
        split=split,
        single_view=True
    )
            
    # Pick a sample object
    print("[ModelNetRender] Collecting all vuews of a sample object for visualization...")
    idx = 0
    num_views = dataset.num_views
    start = idx * num_views
    end = (idx + 1) * num_views
    views = dataset.data[start:end].cpu().numpy()
    label = dataset.labels[start].cpu().numpy()
    class_name = inv_class_map[int(label)]
    
    # Visualize all views together
    print(f"\n[Sample {idx}] Class: {class_name}, Views: {num_views}, Points per view: {views[0].shape[0]}")    
    pcd_utils.viz_pcd([v for v in views])
