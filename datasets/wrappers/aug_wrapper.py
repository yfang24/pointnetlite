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
        out = self.base_dataset[idx]
        *data, label = out        
        
        if len(data) == 1:
            return self._augment(data[0]), label
        else:
            aug_data = tuple(self._augment(d) for d in data)
            return (*aug_data, label)   # unpack so it matches base_dataset format
