import torch
import torch.nn as nn

from models.modules.builders import build_fc_layers

class PointNetClsHead(nn.Module):
    def __init__(self, embed_dim=1024, hidden_dims=[512, 256], out_dim=40, dropout=[0, 0.4]):
        """
        dropout: 
            [0, 0.4] for pointnetlite/pointnet, 
            0.4 for pointnet2 ssg, 
            [0.4, 0.5] for pointnet2 msg,
            0.5 for pointpn
        """
        super().__init__()
        self.fc = build_fc_layers([embed_dim] + hidden_dims + [out_dim], dropout=dropout)

    def forward(self, x):  
        if isinstance(x, tuple):  # PointNetEncoder returns (global_feat, trans_feat, (local_feat if return_all))
            return self.fc(x[0]), x[1]   # logits, trans_feat
        
        return self.fc(x)
       
        
