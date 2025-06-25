import torch.nn as nn

def build_shared_mlp(in_dim, mlp_channels, dim=2): # set dim=2 if using grouped points (B, G, S, in_dim)
    layers = []
    for out_dim in mlp_channels:
        if dim == 2:
            layers += [nn.Conv2d(in_dim, out_dim, 1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)]
        elif dim == 1:
            layers += [nn.Conv1d(in_dim, out_dim, 1), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True)]
        else:
            raise ValueError("Only dim=1 or dim=2 supported")
        in_dim = out_dim
    return nn.Sequential(*layers)
