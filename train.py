import os
import torch
import argparse
from pathlib import Path
import torch.multiprocessing as mp
import torch.distributed as dist

from utils.train_utils import run_training, run_pretraining
from configs.load_config import load_config

def main():
    parser = argparse.ArgumentParser(description="Train/pretrain/finetune model from config and/or experiment.")

    # train/pretrain: -cfg
    # resumed train/pretrain/fintune: -exp [-ckpt]
    # finetune: -cfg -exp [-ckpt]
    
    parser.add_argument("-cfg", "--config_name", type=str, help="Name of config file in /code/configs/")
    parser.add_argument("-exp", "--exp_name", type=str, help="Name of experiment folder under /code/experiments/")
    parser.add_argument("-ckpt", "--ckpt_type", type=str, choices=["best", "last"],
                        default="best", help="Which checkpoint to use when resuming: 'best' (default) or 'last'")
    
    args = parser.parse_args()

    # Resolve config path
    root_dir = Path(__file__).resolve().parent    
    if args.config_name:
        config_path = root_dir / "configs" / (args.config_name if args.config_name.endswith(".yaml") else args.config_name + ".yaml")
    elif args.exp_name:  # resumed training (if only exp name is provided--no config; it is finetune if config is also provided)
        config_path = root_dir / "experiments" / args.exp_name / "config.yaml"
    else:
        raise ValueError("At least one of --config_name or --exp_name must be provided.")        
    
    config = load_config(config_path)

    # Determine mode
    use_ddp = config.get("ddp", False)
    is_pretrain = config.get("pretrain", False)

    if is_pretrain:
        run_func = run_pretraining
    elif args.config_name and args.exp_name:
        run_func = run_finetuning
    else:
        run_func = run_training
        
    if use_ddp:
        # Setup environment from SLURM if not already set
        if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
        if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" in os.environ:
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    
        dist.init_process_group(backend="nccl")

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        try:
            run_func(rank, world_size, local_rank, config, config_path, 
                     device, use_ddp, exp_name=args.exp_name, ckpt_type=args.ckpt_type)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        run_func(rank=0, world_size=1, local_rank=0, config=config, config_path=config_path, 
                 device=device, use_ddp=use_ddp, exp_name=args.exp_name, ckpt_type=args.ckpt_type)

if __name__ == "__main__":
    main()
