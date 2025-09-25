import os
from pathlib import Path
import pickle
import numpy as np
import h5py
from tqdm import tqdm
from torch.utils.data import Dataset

import utils.pcd_utils as pcd_utils
from configs.load_class_map import load_class_map

# Raw label mapping from ScanObjectNN (15 classes)
SCANOBJECTNN_RAW_CLASS_NAMES = [
    "bag", "bin", "box", "cabinet", "chair",
    "desk", "display", "door", "shelf", "table",
    "bed", "pillow", "sink", "sofa", "toilet"
]
RAW_LABEL_MAP = {name: idx for idx, name in enumerate(SCANOBJECTNN_RAW_CLASS_NAMES)}

PROJ_ROOT = Path(__file__).resolve().parents[2]

class ScanObjectNN(Dataset):
    def __init__(self, root='main_split_nobg', class_map='scanobjectnn11', 
                split='train', num_points=1024, normalize=True, cache_dir=None, use_cache=True):
        self.root_dir = PROJ_ROOT / "data" / "scanobjectnn" / root
        self.split = split
        self.num_points = num_points
        self.normalize = normalize
        self.class_map = class_map if isinstance(class_map, dict) else load_class_map(class_map)

        self.num_classes = len(set(self.class_map.values()))

        self.cache_dir = cache_dir or os.path.join(self.root_dir, "_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        norm_tag = "norm" if self.normalize else "unnorm"
        self.cache_file = os.path.join(
            self.cache_dir,
            f"scanobjectnn_{split}_{num_points}pts_{self.num_classes}cls_{norm_tag}.pkl"
        )

        if use_cache and os.path.exists(self.cache_file):
            print(f"[ScanObjectNN] Loading cached data from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                cached = pickle.load(f)
                self.data = cached["data"]
                self.labels = cached["labels"]
        else:
            print("[ScanObjectNN] Processing raw data...")
            self.data, self.labels = self._process_raw_data()
            if use_cache:
                print(f"\n[ScanObjectNN] Saving processed data to cache: {self.cache_file}")
                with open(self.cache_file, "wb") as f:
                    pickle.dump({"data": self.data, "labels": self.labels}, f)

        print(
            f"[ScanObjectNN] Loaded {len(self.labels)} samples from {split} split "
            f"across {self.num_classes} classes ({num_points} pts per sample).\n"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def _process_raw_data(self):
        with h5py.File(os.path.join(self.root_dir, "training_objectdataset.h5"), "r") as f:
            train_data = f["data"][:]                         # (N, 2048, 3)
            train_labels = f["label"][:].flatten()            # (N,)

        with h5py.File(os.path.join(self.root_dir, "test_objectdataset.h5"), "r") as f:
            test_data = f["data"][:]
            test_labels = f["label"][:].flatten()

        if self.split == "train":
            data, labels = train_data, train_labels
        elif self.split == "test":
            data, labels = test_data, test_labels
        else: # self.split == "all"
            data = np.concatenate((train_data, test_data), axis=0)
            labels = np.concatenate((train_labels, test_labels), axis=0)

        processed_data = []
        processed_labels = []

        for class_idx, (class_name, new_label) in enumerate(self.class_map.items()):
            raw_label = RAW_LABEL_MAP[class_name]    
            indices = np.where(labels == raw_label)[0]
            class_data = data[indices]
            
            tqdm.write(f"\n[{class_idx+1}/{len(self.class_map)}] Class {new_label:2d} '{class_name}' ({self.split}): {len(class_data)} files")
            for i in tqdm(
                range(len(class_data)),
                desc=f"    Processing",
                leave=False
            ):
                pc = class_data[i]
                pc = pcd_utils.uniformly_sample(pc, self.num_points)
                if self.normalize:
                    pc = pcd_utils.normalize(pc)

                processed_data.append(pc)
                processed_labels.append(new_label)

        return np.array(processed_data), np.array(processed_labels)


if __name__ == "__main__":
    # Set seeds
    from utils.train_utils import set_seed
    set_seed(42)

    dataset = ScanObjectNN(
        root="main_split_nobg",
        class_map="scanobjectnn11",
        # split='test'
    )

    from collections import defaultdict

    label_to_classnames = defaultdict(list)
    for name, label in dataset.class_map.items():
        label_to_classnames[label].append(name)

    # Collect one example per class for visualization
    viz = True
    if viz: 
        seen_labels = set()
        sample_points = []
    
        print("[ScanObjectNN] Collecting 1 sample per class for visualization...")
        for i in range(len(dataset)):
            points, label = dataset[i]

            if label not in seen_labels:
                sample_points.append(points)
                seen_labels.add(label)

                print(f"  - Class {label:2d} ({', '.join(label_to_classnames[label]):<20})")

            if len(seen_labels) == dataset.num_classes:
                break
  
        print(f"\nVisualizing {dataset.num_classes} point clouds (1 per class)...")
        pcd_utils.viz_pcd(sample_points)
