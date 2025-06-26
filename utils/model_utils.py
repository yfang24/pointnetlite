import torch
from thop import profile

def get_model_profile(model, input_tensor):
    model.eval()
    
    if isinstance(input_tensor, tuple):  # for modelnet_mae_render
        model.to(input_tensor[0].device)
    else:
        model.to(input_tensor.device)

    # ---- Patch inplace=True ReLU to inplace=False ----
    relu_inplace_flags = []
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            relu_inplace_flags.append((module, module.inplace))
            module.inplace = False  # patch for THOP compatibility

    # ---- Profile ----
    with torch.no_grad():
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops = macs * 2

    # ---- Revert ReLU to original inplace setting ----
    for module, was_inplace in relu_inplace_flags:
        module.inplace = was_inplace
        
    # ---- Clean up THOP buffers ----
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
