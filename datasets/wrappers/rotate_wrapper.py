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
        self.angle_deg = angle_deg
        self.rotation_angles = np.deg2rad(np.arange(0.0, 360.0, angle_deg))
        
        self.cache_dir = (
            cache_dir or getattr(base_dataset, 'cache_dir', None)
        )
        if self.cache_dir is None:
            raise ValueError("[RotateWrapper] cache_dir must be provided or defined in base_dataset.")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.class_map = getattr(base_dataset, "class_map", None)
        
        base_cache_file = getattr(base_dataset, 'cache_file', None)
        if base_cache_file:
            base_filename = os.path.basename(base_cache_file)
            base_name, ext = os.path.splitext(base_filename)
            cache_filename = f"{base_name}_{self.angle_deg}rot.pkl"
        else:
            dataset_name = type(base_dataset).__name__.lower()
            base_len = len(base_dataset)
            cache_filename = f'rotated_{dataset_name}_{base_len}base_{self.angle_deg}rot.pkl'

        self.cache_file = os.path.join(self.cache_dir, cache_filename)
        
        if use_cache and os.path.exists(self.cache_file):
            print(f"[RotateWrapper] Loading cached rotated dataset from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                cached = pickle.load(f)
                self.data = cached['data']
                self.labels = cached['labels']
        else:
            print(f"[RotateWrapper] Creating rotated dataset")
            self.data, self.labels = self._generate_rotated_data()
            if use_cache:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump({'data': self.data, 'labels': self.labels}, f)
                print(f"[RotateWrapper] Saved to {self.cache_file}")

        print(f"[RotateWrapper] Loaded {len(self.labels)} samples: {base_len} base x {len(self.rotation_angles)} rotations, across {len(set(self.labels))} classes.\n")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def _generate_rotated_data(self):
        data, labels = [], []
        for i in tqdm(range(len(self.base_dataset)), desc="Rotating samples"):
            base_points, label = self.base_dataset[i]
            for angle in self.rotation_angles:
                rotated = pcd_utils.rotate_xyz(base_points, [0, angle, 0])
                data.append(rotated)
                labels.append(label)
        return np.array(data), np.array(labels)
