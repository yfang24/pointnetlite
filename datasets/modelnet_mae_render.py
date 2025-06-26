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


def reflect_across_plane(points, normal):
    """
    Reflect points across a plane through the origin with a given normal.
    Args:
        points: (N, 3)
        normal: (3,)
    Returns:
        reflected points: (N, 3)
    """
    normal = normal / (normal.norm() + 1e-8)  # normalize
    dot = torch.sum(points * normal, dim=1, keepdim=True)  # (N, 1)
    reflected = points - 2 * dot * normal  # (N, 3)
    return reflected


class ModelNetMAERender(Dataset):
    def __init__(self, root_dir, class_map, split='train', num_points=1024,
                 num_views=3, single_view=False, cache_dir=None, use_cache=True, device=None):
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
            f'modelnetmaerender_{split}_{num_points}pts_{len(set(self.class_map.values()))}cls_{self.num_views}view.pkl'
        )

        if use_cache and os.path.exists(self.cache_file):
            print(f"[ModelNetMAERender] Loading cached data from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                cached = pickle.load(f)
                self.data = cached['data']
                self.labels = cached['labels']
        else:
            print("[ModelNetMAERender] Processing and rendering data...")
            self.data, self.labels = self._process_and_render()
            if use_cache:
                print(f"[ModelNetMAERender] Saving processed data to cache: {self.cache_file}")
                with open(self.cache_file, 'wb') as f:
                    pickle.dump({'data': self.data, 'labels': self.labels}, f)

        self.base_len = len(self.data) // self.num_views
        print(f"\n[ModelNetMAERender] Loaded {len(self.labels)} samples: "
              f"{self.base_len} objects x {self.num_views} views, "
              f"across {len(set(self.labels.tolist()))} classes.")

    def __len__(self):
        return self.base_len if self.single_view else len(self.labels)

    def __getitem__(self, idx):
        if self.single_view:
            obj_idx = idx
            view_idx = np.random.randint(self.num_views)
            idx = obj_idx * self.num_views + view_idx

        return self.data[idx], self.labels[idx]

    def _process_and_render(self):
        data, labels = [], []

        for class_name, label in self.class_map.items():
            class_dir = os.path.join(self.root_dir, class_name, self.split)
            if not os.path.isdir(class_dir):
                continue
            file_list = sorted([f for f in os.listdir(class_dir) if f.endswith('.off')])

            print(f"[{class_idx+1}/{len(self.class_map)}] Class '{class_name}' ({split}): {len(file_list)} files")
            for fname in tqdm(file_list, desc=f"Rendering {class_name}", leave=False):
                mesh_path = os.path.join(class_dir, fname)
                mesh = o3d.io.read_triangle_mesh(mesh_path)
                mesh = mesh_utils.align_mesh(mesh, class_name)
                mesh = mesh_utils.normalize_mesh(mesh)

                verts_np = np.asarray(mesh.vertices)
                faces_np = np.asarray(mesh.triangles)
                verts_np[:, [0, 2]] *= -1  # Open3D to PyTorch3D coords

                verts = torch.tensor(verts_np, dtype=torch.float32, device=self.device)
                faces = torch.tensor(faces_np, dtype=torch.int64, device=self.device)
                mesh_torch = Meshes(verts=[verts], faces=[faces])

                for camera_pos in self.viewpoints.to(self.device):
                    vis_pts = mesh_utils.render_mesh_torch(mesh_torch, camera_pos,
                                                           num_points=self.num_points, device=self.device)
                    mask_pts = mesh_utils.render_mesh_torch(mesh_torch, -camera_pos,
                                                            num_points=self.num_points, device=self.device)

                    vis_pts[:, [0, 2]] *= -1  # Back to Open3D coord
                    mask_pts[:, [0, 2]] *= -1

                    vis_pts = pcd_utils.normalize_pcd_tensor(vis_pts)
                    mask_pts = pcd_utils.normalize_pcd_tensor(mask_pts)

                    reflected_pts = reflect_across_plane(vis_pts, camera_pos)

                    data.append((vis_pts.cpu(), mask_pts.cpu(), reflected_pts.cpu()))
                    labels.append(label)

        return data, torch.tensor(labels, dtype=torch.long)

    def _get_viewpoints(self, num_views):
        if num_views == 3:
            bases = torch.tensor([
                [0, 1, 1],     # front-up
                [-1, 1, 1],    # left-front-up
                [1, 0, 1],     # right-front
            ], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported number of views: {num_views}")

        distance = torch.tensor([2.5, 2.5, 2.5], dtype=torch.float32)
        flip = torch.tensor([-1, 1, -1], dtype=torch.float32)  # Open3D to PyTorch3D flip

        return bases * distance * flip  # (V, 3)


if __name__ == "__main__":
    from configs.load_class_map import load_class_map

    DATA_DIR = "../../data/modelnet40_manually_aligned"
    CLASS_MAP_PATH = "../../code/configs/class_map_modelnet11.json"

    dataset = ModelNetMAERender(
        root_dir=DATA_DIR,
        class_map=CLASS_MAP_PATH,
        split="train",
        single_view=True,
    )

    vis, mask, refl, label = dataset[0]
    print(f"[Sample] Class: {label}, Points shape: {vis.shape}, {mask.shape}, {refl.shape}")
