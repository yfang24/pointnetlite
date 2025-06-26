import math
import random
import torch
from torch.utils.data import Dataset

import utils.pcd_utils as pcd_utils

class AugWrapper(Dataset):
    """
    A wrapper that applies runtime transformations to point cloud datasets.
    """
    def __init__(self, base_dataset, apply_scale=True, apply_rotation=True):
        self.base_dataset = base_dataset
        self.apply_scale = apply_scale
        self.apply_rotation = apply_rotation
        self.class_map = getattr(base_dataset, "class_map", None)
        self.data = getattr(base_dataset, "data", None)
        self.labels = getattr(base_dataset, "labels", None)
        
    def __len__(self):
        return len(self.base_dataset)

    def _augment(self, points):
        points = pcd_utils.random_drop_out(points)
        if self.apply_scale:
            points = pcd_utils.random_scale(points)
        if self.apply_rotation:
            points = pcd_utils.rotate_xyz(points, [0, random.uniform(0, 2 * math.pi), 0])
        points = pcd_utils.rotate_perturb(points)
        points = pcd_utils.random_shift(points)
        points = pcd_utils.jitter(points)
        return points
        
    # def __getitem__(self, idx):
    #     points, label = self.base_dataset[idx]
    #     points = self._augment(points)
    #     return points, label

    def __getitem__(self, idx):
        data, label = self.base_dataset[idx]

        # for modelnet_mae_render
        if isinstance(data, tuple):
            # Record original lengths
            lengths = [p.shape[0] for p in data]
            total = sum(lengths)
    
            # Stack all point clouds
            stacked = torch.cat(data, dim=0)  # (sum(N), 3)
    
            # Augment the stacked point cloud
            stacked_aug = self._augment(stacked.clone())
    
            # Unstack into original shapes
            splits = torch.split(stacked_aug, lengths)
            aug_tuple = tuple(splits)
    
            return aug_tuple, label
        
        if isinstance(data, tuple):
            aug_tuple = tuple(self._augment(p) for p in data)
            return aug_tuple, label

        # fallback: (points, label)
        else:
            data = self._augment(data)
            return data, label
