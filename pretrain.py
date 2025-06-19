import os
import torch
import argparse
from pathlib import Path
import torch.multiprocessing as mp
import torch.distributed as dist

from utils.train_utils import run_pretraining
from configs.load_config import load_config

def main():
    parser = argparse.ArgumentParser(description="Train model from config or experiment.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-cfg", "--config_name", type=str, help="Name of config in /code/configs/")
    group.add_argument("-exp", "--exp_name", type=str, help="Name of experiment under /code/experiments/")
    
    args = parser.parse_args()
    
    root_dir = Path(__file__).resolve().parent    
    if args.config_name:
        config_path = root_dir / "configs" / (args.config_name if args.config_name.endswith(".yaml") else args.config_name + ".yaml")
    else:
        config_path = root_dir / "experiments" / args.exp_name / "config.yaml"
    
    config = load_config(config_path)
    use_ddp = config.get("ddp", False)

    if use_ddp:
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
            run_pretraining(rank, world_size, local_rank, config, config_path, device, use_ddp)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        run_pretraining(rank=0, world_size=1, local_rank=0, config=config, config_path=config_path, device=device, use_ddp=use_ddp)

if __name__ == "__main__":
    main()
