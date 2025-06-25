import torch.nn as nn

# shared mlp = conv + bn + act
def build_shared_mlp(dims, conv_dim=1, act=nn.ReLU(inplace=True)):
    '''
    conv_dim: default=1; set conv_dim=2 if using grouped points (B, G, S, in_dim)
    '''
    assert conv_dim in [1, 2]
    conv = nn.Conv1d if conv_dim == 1 else nn.Conv2d
    bn = nn.BatchNorm1d if conv_dim == 1 else nn.BatchNorm2d
    
    layers = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers += [conv(in_dim, out_dim, 1), bn(out_dim), get_activation(act)]
    return nn.Sequential(*layers)

# fc = linear + bn + act + dropout
def build_fc_layers(dims, act=nn.ReLU(inplace=True), final_act=False, dropout=0.0):
    '''
    dropout: float or list of float(s), dropout rate(s) per layer
    '''
    # Normalize dropout to list
    if isinstance(dropout, (float, int)):
        dropout = [dropout] * n
    assert len(dropout) == n, "Dropout length must match number of layers"

    layers = []
    n = len(dims) - 1
    for i in range(n):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(nn.BatchNorm1d(dims[i+1]))
        if final_act or i < n - 1:
            layers.append(act)
        if dropout[i] > 0:
            layers.append(nn.Dropout(dropout[i]))
    
    return nn.Sequential(*layers)
