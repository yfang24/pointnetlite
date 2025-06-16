import torch
from thop import profile

def get_model_profile(model, input_tensor):
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops = macs * 2
    return flops, params

def freeze_module(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False