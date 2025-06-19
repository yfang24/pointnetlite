import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses.pointnet_loss import PointNetLoss
from models.losses.chamfer_distance_loss import ChamferDistanceL2, ChamferDistanceL1

# Registry of named loss classes or functions
LOSS_REGISTRY = {
    "cross_entropy": nn.CrossEntropyLoss,
    "pointnet_loss": PointNetLoss,
    "chamfer_distance_l2": ChamferDistanceL2,
    "chamfer_distance_l1": ChamferDistanceL1
}

def get_loss(config):
    """
    Given full config dict, return a callable loss_fn(pred, target, **kwargs)
    """
    loss_config = config.get("loss", None)
    if loss_config is None:
        raise ValueError("Missing 'loss' in config")

    # If just a string is given
    if isinstance(loss_config, str):
        name = loss_config
        args = {}
    elif isinstance(loss_config, dict):
        name = loss_config.get("name")
        args = loss_config.get("args", {})
        if name is None:
            raise ValueError("Missing 'name' field in loss config dictionary")
    else:
        raise ValueError(f"Invalid format for config['loss']: {type(loss_config)}")

    if name not in LOSS_REGISTRY:
        raise ValueError(f"[get_loss] Unknown loss: '{name}'. Available: {list(LOSS_REGISTRY.keys())}")

    loss_entry = LOSS_REGISTRY[name]

    # Instantiate class-based loss
    base_loss = loss_entry(**args)

    # If it's PointNetLoss, wrap to accept trans_feat
    if isinstance(base_loss, PointNetLoss):
        def loss_fn(pred, target, trans_feat=None, **kwargs):
            return base_loss(pred, target, trans_feat=trans_feat)
        return loss_fn

    # Generic loss wrapper
    def loss_fn(pred, target, **kwargs):
        return base_loss(pred, target)
    
    return loss_fn