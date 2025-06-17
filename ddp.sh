#!/bin/bash 
#SBATCH --job-name=full_ddp_train
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2          # =GPUs per node
#SBATCH --gres=gpu:2                 # GPUs per node
#SBATCH --cpus-per-task=16          # 64 CPU cores per node / ntasks-per-node
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
##SBATCH --mail-user=yfang24@stevens.edu
##SBATCH --mail-type=BEGIN,END,FAIL

module load cuda12.4
source ~/myenv/bin/activate

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12355
export PYTHONPATH=.

srun python train.py -cfg exp_mn2nn_11cls