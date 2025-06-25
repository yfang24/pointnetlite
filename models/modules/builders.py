import torch.nn as nn

# shared mlp = conv + bn + act
# last mlp = conv + bn  (typically)
def build_shared_mlp(dims, conv_dim=1, act=nn.ReLU(inplace=True), final_act=False):
    '''
    conv_dim: default=1; set conv_dim=2 if using grouped points (B, G, S, in_dim)
    '''
    assert conv_dim in [1, 2], "conv_dim must be 1 or 2"
    conv = nn.Conv1d if conv_dim == 1 else nn.Conv2d
    bn = nn.BatchNorm1d if conv_dim == 1 else nn.BatchNorm2d

    layers = []
    n_layers = len(dims) - 1

    for i in range(n_layers):
        in_dim, out_dim = dims[i], dims[i + 1]
        is_last = i == n_layers - 1

        layers.append(conv(in_dim, out_dim, 1))
        layers.append(bn(out_dim))
        if not is_last or final_act:
            layers.append(act)

    return nn.Sequential(*layers)

# fc_layer = linear + bn + act + dropout
# final fc = linear (typically)
def build_fc_layers(dims, act=nn.ReLU(inplace=True), dropout=0.0, final_act=False, final_bn=False):
    '''
    dropout: float or list of float(s), dropout rate(s) per layer
    '''
    layers = []
    n_layers = len(dims) - 1
    
    # Normalize dropout to list
    if isinstance(dropout, (float, int)):
        dropout = [dropout] * n_layers
    assert len(dropout) == n_layers, f"Dropout must match number of layers ({n_layers})"

    for i in range(n_layers):
        in_dim, out_dim = dims[i], dims[i + 1]
        is_last = i == n_layers - 1

        layers.append(nn.Linear(in_dim, out_dim))

        if not is_last or final_bn:
            layers.append(nn.BatchNorm1d(out_dim))
        if not is_last or final_act:
            layers.append(act)
        if dropout[i] > 0:
            layers.append(nn.Dropout(dropout[i]))

    return nn.Sequential(*layers)
