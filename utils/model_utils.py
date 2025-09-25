import torch
import torch.nn as nn
from thop import profile

def get_model_profile(model, input_tensor):
    model.eval()
    model.to(input_tensor.device)

    # ---- Temporarily replace non-parametric modules with Identity ----
    replacements = []
    for module in model.modules():
        if isinstance(module, (
            # Normalizations
            nn.modules.batchnorm._BatchNorm,   
            nn.LayerNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            # Pooling
            nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
            nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
            nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
            nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
        )) or module.__module__.startswith("torch.nn.modules.activation"):
            replacements.append((module, type(module)))
            module.__class__ = nn.Identity

    # ---- Run THOP Profile ----        
    with torch.no_grad():
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops = macs * 2

    # model params
    # params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # for trainable params
    # params = sum(p.numel() for p in model.parameters())  # for trainable + frozen params--actual model size in memory


    # ---- Restore original patched modules ----
    for module, original_class in replacements:
        module.__class__ = original_class
        
    # ---- Clean up THOP-injected buffers ----
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
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
