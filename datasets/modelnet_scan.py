import os
import pickle
import numpy as np
import open3d as o3d
from tqdm import tqdm
from torch.utils.data import Dataset

import utils.mesh_utils as mesh_utils
import utils.pcd_utils as pcd_utils
from configs.load_class_map import load_class_map

class ModelNetScan(Dataset):
    def __init__(self, root_dir, class_map, split='train', num_points=1024,
                 cache_dir=None, use_cache=True, single_view=True):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.single_view = single_view
        self.class_map = class_map if isinstance(class_map, dict) else load_class_map(class_map)

        self.scanner_poses = [[0, 3, 2], [-2, 3, 2], [2, 0, 2]]
        self.num_views = len(self.scanner_poses)
        
        self.cache_dir = cache_dir or os.path.join(root_dir, '_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_file = os.path.join(
            self.cache_dir,
            f'modelnetscan_{split}_{num_points}pts_{len(set(self.class_map.values()))}cls_{self.num_views}view.pkl'
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
                print(f"[ModelNetScan] Saving processed data to cache: {self.cache_file}")
                with open(self.cache_file, 'wb') as f:
                    pickle.dump({'data': self.data, 'labels': self.labels}, f)
            
        self.base_len = len(self.data) // self.num_views
        print(f"\n[ModelNetScan] Loaded {len(self.labels)} samples: {self.base_len} objects x {self.num_views} views, across {len(set(self.labels.tolist()))} classes.")

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

                tqdm.write(f"[{class_idx + 1}/{total_classes}] Class '{class_name}' ({split}): {len(file_list)} files")
                for fname in tqdm(file_list, desc=f"Scanning {class_name}", leave=False):
                    mesh_path = os.path.join(class_split_dir, fname)
                    mesh = o3d.io.read_triangle_mesh(mesh_path)
                    mesh = mesh_utils.align_mesh(mesh, class_name)
                    mesh = mesh_utils.normalize_mesh(mesh)

                    for pose in self.scanner_poses:
                        scan = mesh_utils.scan(mesh, pose, num_points=self.num_points)
                        scan = pcd_utils.normalize_pcd(scan)
                        points = np.asarray(scan)
                        data.append(points)
                        labels.append(label)

        return np.array(data), np.array(labels)


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJ_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
    
    DATA_DIR = os.path.join(PROJ_ROOT, "data/modelnet40_manually_aligned")
    CLASS_MAP_PATH = os.path.join(PROJ_ROOT, "code/configs/class_map_modelnet11.json")
    
    split = "train"
    
    class_map = load_class_map(CLASS_MAP_PATH)
    inv_class_map = {v: k for k, v in class_map.items()}

    dataset = ModelNetScan(
        root_dir=DATA_DIR,
        class_map=CLASS_MAP_PATH,
        split=split,
        single_view=False  # or True
    )
    
    # Pick a sample object
    print("[ModelNetScan] Collecting all vuews of a sample object for visualization...")
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
