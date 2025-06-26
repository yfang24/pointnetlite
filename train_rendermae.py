import os
import shutil
import torch
import torch.nn as nn
import time
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from sklearn.metrics import confusion_matrix

from utils.train_utils import set_seed, get_optimizer, get_scheduler, smart_collate_fn
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.model_utils import get_model_profile
from utils.log_utils import setup_logger
from datasets.get_dataset import get_dataset
from models.encoders.get_encoder import get_encoder
from models.heads.get_head import get_head
from models.losses.get_loss import get_loss

vis_token, vis_centers = encoder(vis_pts)
outputs = head(vis_token, vis_centers, mask_pts)


class RenderMAE(nn.Module):
    def __init__(noaug=False):
        super().__init__()

        self.encoder = get_encoder(
    
