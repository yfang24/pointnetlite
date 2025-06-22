import os
import pickle
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset

import utils.mesh_utils as mesh_utils
from configs.load_class_map import load_class_map

class ModelNetMesh(Dataset):
    def __init__(self, root_dir, class_map, split='train', cache_dir=None, use_cache=True):
        self.root_dir = root_dir
        self.split = split
        self.class_map = class_map if isinstance(class_map, dict) else load_class_map(class_map)

        self.cache_dir = cache_dir or os.path.join(root_dir, "_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_file = os.path.join(
            self.cache_dir,
            f"modelnetmesh_{split}_{len(set(self.class_map.values()))}cls.pkl"
        )

        if use_cache and os.path.exists(self.cache_file):
            print(f"[ModelNetMesh] Loading cached data from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                cached = pickle.load(f)
                self.verts_list = cached['verts']
                self.faces_list = cached['faces']
                self.labels = cached['labels']
        else:
            print(f"[ModelNetMesh] Processing mesh files for split '{split}'...")
            self.verts_list, self.faces_list, self.labels = self._process_and_cache()
            if use_cache:
                print(f"[ModelNetMesh] Saving mesh cache to {self.cache_file}")
                with open(self.cache_file, 'wb') as f:
                    pickle.dump({
                        'verts': self.verts_list,
                        'faces': self.faces_list,
                        'labels': self.labels
                    }, f)

        print(f"[ModelNetMesh] Loaded {len(self.labels)} samples from {split} split.\n")

    def _process_and_cache(self):
        verts_list, faces_list, labels = [], [], []
        for class_name, label in self.class_map.items():
            class_dir = os.path.join(self.root_dir, class_name, self.split)
            if not os.path.isdir(class_dir):
                continue
            files = [f for f in os.listdir(class_dir) if f.endswith('.off')]
            for fname in tqdm(files, total=len(files), desc=f"Processing '{class_name}'", leave=False):
                mesh_path = os.path.join(class_dir, fname)
                mesh = o3d.io.read_triangle_mesh(mesh_path)
                mesh = mesh_utils.align_mesh(mesh, class_name)
                mesh = mesh_utils.normalize_mesh(mesh)

                verts = np.asarray(mesh.vertices, dtype=np.float32)
                faces = np.asarray(mesh.triangles, dtype=np.int64)

                verts_list.append(torch.tensor(verts))
                faces_list.append(torch.tensor(faces))
                labels.append(label)
        return verts_list, faces_list, torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.labels[idx]
