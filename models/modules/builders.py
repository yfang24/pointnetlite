import torch.nn as nn

# shared mlp = conv + bn + act
def build_shared_mlp(dims, dim=1, act=nn.ReLU(inplace=True)): # set dim=2 if using grouped points (B, G, S, in_dim)
    assert dim in [1, 2]
    conv = nn.Conv1d if dim == 1 else nn.Conv2d
    bn = nn.BatchNorm1d if dim == 1 else nn.BatchNorm2d
    
    layers = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers += [conv(in_dim, out_dim, 1), bn(out_dim), get_activation(act)]
    return nn.Sequential(*layers)

# fc = linear + bn + act
def build_fc_layers(dims, act=nn.ReLU(inplace=True), final_act=False, dropout=0.0)
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.BatchNorm1d(dims[i + 1]))
        if i < len(dims) - 2 or final_act:
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)
