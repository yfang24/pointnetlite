import torch
from thop import profile

def get_model_profile(model, input_tensor):
    model.eval()
    
    if isinstance(input_tensor, tuple):  # for modelnet_mae_render
        model.to(input_tensor[0].device)
    else:
        model.to(input_tensor.device)
    
    with torch.no_grad():
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops = macs * 2
    
    # Safely remove THOP-injected buffers from all submodules
    for module in model.modules():
        for key in list(module._buffers.keys()):
            if "total_ops" in key or "total_params" in key:
                del module._buffers[key]
            
    return flops, params

def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
