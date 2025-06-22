import torch
import torch.nn as nn
import torch.nn.functional as F

class ViewpointLearner(nn.Module):
    def __init__(self, num_classes, num_views, init_scale=2.5):
        super().__init__()
        self.num_classes = num_classes
        self.num_views = num_views

        # Learnable camera positions: (num_classes, num_views, 3)
        directions = F.normalize(torch.randn(num_classes, num_views, 3), dim=-1)
        self.camera_pos = nn.Parameter(directions * init_scale)

    def forward(self, class_indices):
        """
        Args:
            class_indices: (B,) or (B, 1) tensor of class labels
        Returns:
            viewpoints: (B, num_views, 3) camera positions
        """
        return self.camera_pos[class_indices]

    def repelling_loss(self):
        """
        Penalize camera positions that are too close to each other (per class).
        Encourages spatial diversity of viewpoints per class.
        """
        loss = 0.0
        for c in range(self.num_classes):
            cams = self.camera_pos[c]  # (num_views, 3)
            diffs = cams[:, None, :] - cams[None, :, :]  # (V, V, 3)
            dists = torch.norm(diffs, dim=-1)  # (V, V)
            mask = ~torch.eye(self.num_views, dtype=torch.bool, device=cams.device)
            loss += (1.0 / (dists[mask] + 1e-6)).mean()
        return loss / self.num_classes
