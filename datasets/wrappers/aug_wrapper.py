import math
import random
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
    #     return points, label

    def __getitem__(self, idx):
        data = self.base_dataset[idx]

        # case: ((vis_pts, reflected_pts), mask_pts)
        if isinstance(data, tuple) and isinstance(data[0], tuple):
            (vis_pts, reflected_pts), mask_pts = data
            vis_pts = self._augment(vis_pts)
            reflected_pts = self._augment(reflected_pts)
            return (vis_pts, reflected_pts), mask_pts

        # fallback: (points, label)
        else:
            points, label = data
            points = self._augment(points)
            return points, label
