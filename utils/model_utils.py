import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count

def get_model_profile(model, input_tensor):
    model.eval()
    
    if isinstance(input_tensor, tuple):  # for modelnet_mae_render
        model.to(input_tensor[0].device)
    else:
        model.to(input_tensor.device)

    with torch.no_grad():
        flops = FlopCountAnalysis(model, (input_tensor,)).total()
        params = sum(p.numel() for p in model.parameters())

    return flops, params

def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
