import torch
from torch.optim import Optimizer
from timm.scheduler import CosineLRScheduler
from typing import List, Tuple, Dict, Any


def add_weight_decay(
    named_params: List[Tuple[str, torch.nn.Parameter]],
    weight_decay: float = 1e-5,
    skip_list: Tuple[str, ...] = ()
) -> List[Dict[str, Any]]:
    """
    Splits parameters into two groups: with weight decay and without.

    Args:
        named_params: list of (name, parameter) tuples, e.g., list(model.named_parameters()).
        weight_decay: global weight decay factor to apply.
        skip_list: parameter names to exclude from decay.

    Returns:
        A list of two parameter groups suitable for passing to an optimizer.
    """
    decay, no_decay = [], []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1  # e.g. biases, LayerNorm
            or name.endswith(".bias")
            or "token" in name
            or name in skip_list
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def get_optimizer(config: dict, named_params: List[Tuple[str, torch.nn.Parameter]]) -> Optimizer:
    """
    Create an optimizer based on config.

    Args:
        config: experiment config dict containing an "optimizer" field.
        named_params: list of (name, parameter) tuples, e.g., list(model.named_parameters()).

    Returns:
        A torch Optimizer instance.
    """
    opt_cfg = config.get("optimizer", None)

    if opt_cfg is None:
        return torch.optim.Adam(
            [p for _, p in named_params if p.requires_grad],
            lr=0.001,
            weight_decay=0.0001,
        )

    opt_type = opt_cfg.get("name", "adam").lower()
    opt_args = opt_cfg.get("args", {})

    if opt_type == "adamw":
        param_groups = add_weight_decay(named_params, opt_args.get("weight_decay", 1e-5))
        return torch.optim.AdamW(param_groups, **opt_args)
    elif opt_type == "adam":
        params = [p for _, p in named_params if p.requires_grad]
        return torch.optim.Adam(params, **opt_args)
    elif opt_type == "sgd":
        params = [p for _, p in named_params if p.requires_grad]
        return torch.optim.SGD(params, nesterov=True, momentum=0.9, **opt_args)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")


def get_scheduler(config: dict, optimizer: Optimizer):
    """
    Create a learning rate scheduler based on config.

    Args:
        config: experiment config dict containing a "scheduler" field.
        optimizer: torch Optimizer instance.

    Returns:
        A torch scheduler instance (PyTorch or timm).
    """
    sched_cfg = config.get("scheduler", None)

    if sched_cfg is None:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    sched_type = sched_cfg.get("name", "steplr").lower()
    sched_args = sched_cfg.get("args", {})

    if sched_type == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, **sched_args)
    elif sched_type == "coslr":
        return CosineLRScheduler(
            optimizer,
            t_initial=sched_args.get("epochs", config.get("epochs", 200)),
            lr_min=1e-6,
            cycle_mul=1,
            warmup_lr_init=1e-6,
            warmup_t=sched_args.get("initial_epochs", 10),
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")
