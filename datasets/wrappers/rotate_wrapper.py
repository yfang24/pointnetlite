import os
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

import utils.pcd_utils as pcd_utils

# y-axis rotation
class RotateWrapper(Dataset):
    def __init__(self, base_dataset, angle_deg, cache_dir=None, use_cache=True):
        self.base_dataset = base_dataset
        self.base_len = len(base_dataset)

        self.angle_deg = angle_deg
        self.rotation_angles = np.deg2rad(np.arange(0.0, 360.0, angle_deg))
        self.num_angles = len(self.rotation_angles)

        self.cache_dir = cache_dir or base_dataset.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.class_map = base_dataset.class_map
        self.num_classes = len(set(self.class_map.values()))
        
        base_cache_file = base_dataset.cache_file
        base_filename = os.path.basename(base_cache_file)
        base_name, ext = os.path.splitext(base_filename)
        cache_filename = f"{base_name}_{self.angle_deg}rot.pkl"
        self.cache_file = os.path.join(self.cache_dir, cache_filename)
        
        if use_cache and os.path.exists(self.cache_file):
            print(f"[RotateWrapper] Loading cached data from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                cached = pickle.load(f)
                self.data = cached['data']
                self.labels = cached['labels']
        else:
            print(f"[RotateWrapper] Creating rotated dataset...")
            self.data, self.labels = self._generate_rotated_data()
            if use_cache:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump({'data': self.data, 'labels': self.labels}, f)
                print(f"[RotateWrapper] Saving rotated dataset to cache: {self.cache_file}")

        self.numpoints = base_dataset.num_points
        print(
            f"[RotateWrapper] Loaded {len(self.labels)} samples: {self.base_len} objects x {self.num_angles} rotated copies, "
            f"from {split} split across {self.num_classes} classes ({self.num_points} pts per sample).\n"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def _generate_rotated_data(self):
        data, labels = [], []
        for i in tqdm(range(self.base_len), desc="Rotating samples", leave=False):
            base_points, label = self.base_dataset[i]
            for angle in self.rotation_angles:
                rotated = pcd_utils.rotate_xyz(base_points, [0, angle, 0])
                data.append(rotated)
                labels.append(label)
        return np.array(data), np.array(labels)


if __name__ == "__main__":
    # Resolve base directory and paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJ_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
    
    DATA_DIR = os.path.join(PROJ_ROOT, "data/modelnet40_manually_aligned")
    CLASS_MAP_PATH = os.path.join(PROJ_ROOT, "code/configs/class_map_modelnet11.json")

    dataset = RotateWrapper(
        ModelNet(
            root_dir=DATA_DIR, 
            class_map=CLASS_MAP_PATH
        )
    )

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

        print("[RotateWrapper] Collecting all rotated copies of a sample object from {len(chosen_labels)} classes for visualization...")
        
        num_angles = dataset.num_angles
        sample_points = []       

        for idx in range(dataset.base_len):    
            start = idx * num_angles
            end = (idx + 1) * num_angles
            label = dataset.labels[start]

            if label in chosen_labels:
                copies = dataset.data[start:end]
                sample_points.extend(copies)
                chosen_labels.remove(label)

                print(f"  - Class {label:2d} ({', '.join(label_to_classnames[label]):<20})")

        # Visualize all views together
        print(f"\nVisualizing {num_chosen} samples ({num_angles} rotated copies per sample class)...")    
        pcd_utils.viz_pcd(sample_points, rows=num_chosen)