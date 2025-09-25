import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetLoss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001, weight=None):
        '''
        weight: a list of per class weight for cross entropy loss
        '''
        super().__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.weight = weight

    def forward(self, outputs, target):
        pred, trans_feat = outputs   # PointNetClsHead returns (logits, trans_feat)

        if self.weight is not None:
            weight = torch.as_tensor(self.weight, dtype=torch.float32, device=pred.device)
        else:
            weight = None
        loss = F.cross_entropy(pred, target, weight=weight)

        if trans_feat is not None:
            mat_diff_loss = self._feature_transform_regularizer(trans_feat)
            loss += mat_diff_loss * self.mat_diff_loss_scale

        return loss

    def _feature_transform_regularizer(self, trans):
        '''
        to prevent the STN matrix from drifting too far away from orthogonal
        '''
        d = trans.size()[1]
        I = torch.eye(d, device=trans.device).unsqueeze(0)
        return torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
