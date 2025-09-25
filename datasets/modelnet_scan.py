import os
from pathlib import Path
import pickle
import numpy as np
import open3d as o3d
from tqdm import tqdm
from torch.utils.data import Dataset

import utils.mesh_utils as mesh_utils
import utils.pcd_utils as pcd_utils
from configs.load_class_map import load_class_map

PROJ_ROOT = Path(__file__).resolve().parents[2]

class ModelNetScan(Dataset):
    def __init__(self, root='modelnet40_manually_aligned', class_map='modelnet11', 
                split='train', num_points=1024, cache_dir=None, use_cache=True):
        self.root_dir = PROJ_ROOT / "data" / root
        self.split = split
        self.num_points = num_points
        self.class_map = class_map if isinstance(class_map, dict) else load_class_map(class_map)

        self.num_classes = len(set(self.class_map.values()))

        self.viewpoints = [[0, 3, 2], [-2, 3, 2], [2, 0, 2]]
        self.num_views = len(self.viewpoints)
        
        self.cache_dir = cache_dir or os.path.join(self.root_dir, '_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_file = os.path.join(
            self.cache_dir,
            f'modelnetscan_{split}_{num_points}pts_{self.num_classes}cls_{self.num_views}view.pkl'
        )

        if use_cache and os.path.exists(self.cache_file):
            print(f"[ModelNetScan] Loading cached data from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                cached = pickle.load(f)
                self.data = cached['data']
                self.labels = cached['labels']
        else:
            print("[ModelNetScan] Processing raw data...")
            self.data, self.labels = self._process_raw_data()
            if use_cache:
                print(f"\n[ModelNetScan] Saving processed data to cache: {self.cache_file}")
                with open(self.cache_file, 'wb') as f:
                    pickle.dump({'data': self.data, 'labels': self.labels}, f)
            
        self.base_len = len(self.labels) // self.num_views
        print(
            f"[ModelNetScan] Loaded {len(self.labels)} samples: {self.base_len} objects x {self.num_views} views, "
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

                    for scanner_pose in self.viewpoints:
                        scan = mesh_utils.scan(mesh, scanner_pose, num_points=self.num_points)
                        scan = pcd_utils.normalize_pcd(scan)
                        points = np.asarray(scan)
                        data.append(points)
                        labels.append(label)

        return np.array(data), np.array(labels)


if __name__ == "__main__":
    # Set seeds
    from utils.train_utils import set_seed
    set_seed(42)

    dataset = ModelNetScan(
        root="modelnet40_manually_aligned", 
        class_map="modelnet11",
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

        print("[ModelNetScan] Collecting all views of a sample object from {len(chosen_labels)} classes for visualization...")
        
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
