import os
import pickle
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
from tqdm import tqdm

import utils.mesh_utils as mesh_utils
import utils.pcd_utils as pcd_utils
from configs.load_class_map import load_class_map

class ModelNet(Dataset):
    def __init__(self, root_dir, class_map, split='train', num_points=1024, cache_dir=None, use_cache=True):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.class_map = class_map if isinstance(class_map, dict) else load_class_map(class_map)

        self.cache_dir = cache_dir or os.path.join(root_dir, '_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_file = os.path.join(
            self.cache_dir,
            f'modelnet_{split}_{num_points}pts_{len(set(self.class_map.values()))}cls.pkl'
        )

        if use_cache and os.path.exists(self.cache_file):
            print(f"[ModelNet] Loading cached data from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                cached = pickle.load(f)
                self.data = cached['data']
                self.labels = cached['labels']
        else:
            print("[ModelNet] Processing raw data...")
            self.data, self.labels = self._process_raw_data()
            if use_cache:
                print(f"[ModelNet] Saving processed data to cache: {self.cache_file}")
                with open(self.cache_file, 'wb') as f:
                    pickle.dump({'data': self.data, 'labels': self.labels}, f)
                    
        print(f"[ModelNet] Loaded {len(self.labels)} samples across {len(set(self.labels))} classes.\n")
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def _process_raw_data(self):
        data, labels = [], []
        total_classes = len(self.class_map)
    
        for class_idx, (class_name, label) in enumerate(self.class_map.items()):
            split_dirs = ['train', 'test'] if self.split == 'all' else [self.split]
    
            for split in split_dirs:
                class_split_dir = os.path.join(self.root_dir, class_name, split)
                if not os.path.isdir(class_split_dir):
                    continue
                file_list = sorted([f for f in os.listdir(class_split_dir) if f.endswith('.off')])
    
                print(f"[{class_idx+1}/{total_classes}] Class '{class_name}' ({split}): {len(file_list)} files")
                for fname in tqdm(file_list, desc=f"    Processing {class_name}", leave=False):
                    mesh_path = os.path.join(class_split_dir, fname)
                    mesh = o3d.io.read_triangle_mesh(mesh_path)
                    mesh = mesh_utils.align_mesh(mesh, class_name)
                    mesh = mesh_utils.normalize_mesh(mesh)
    
                    pcd = mesh.sample_points_uniformly(self.num_points)
                    pcd = pcd_utils.normalize_pcd(pcd)
                    points = np.asarray(pcd.points)
    
                    data.append(points)
                    labels.append(label)
    
        return np.array(data), np.array(labels)


if __name__ == "__main__":

    # Resolve base directory and paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJ_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
    
    DATA_DIR = os.path.join(PROJ_ROOT, "data/modelnet40_manually_aligned")
    CLASS_MAP_PATH = os.path.join(PROJ_ROOT, "code/configs/class_map_modelnet11.json")
    
    split = "train"

    dataset = ModelNet(
        root_dir=DATA_DIR, 
        class_map=CLASS_MAP_PATH, 
        split=split
    )

    class_names = list(dataset.class_map.keys())

    # Collect one example per class for visualization
    viz = True
    if viz: 
      seen_labels = set()
      sample_points = []
  
      print("[ModelNet] Collecting 1 sample per class for visualization...")
      for i in range(len(dataset)):
          points, label = dataset[i]
          if label not in seen_labels:
              sample_points.append(points)
              seen_labels.add(label)
              print(f"  - Class {label:2d} ({class_names[label]:>15}): points shape = {points.shape}")
          if len(seen_labels) == len(class_names):
              break
  
      print(f"\nVisualizing {len(sample_points)} point clouds (1 per class)...")
      pcd_utils.viz_pcd(sample_points)    