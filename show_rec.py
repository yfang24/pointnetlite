import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from configs.load_config import load_config
from datasets.get_dataset import get_dataset
from models.encoders.get_encoder import get_encoder
from models.heads.get_head import get_head
from models.losses.get_loss import get_loss
from utils.train_utils import evaluate, pretrain_evaluate
from utils.checkpoint_utils import load_checkpoint
from utils.pcd_utils import init_pcd, viz_pcd
from utils.pcd_utils import fps, knn_group, group_points

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp", "--exp_name", type=str, required=True,
                    help="Experiment folder name (e.g., pointnet_test_20250612_103045)")
    parser.add_argument(
        "-ckpt", "--ckpt_type", type=str, choices=["best", "last"], default="best",
        help="Which checkpoint to use: 'best' (default) or 'last'"
    )
    parser.add_argument(
        "-vset", "--viz_dataset", type=str, choices=["train", "test"], default="test",
        help="Which dataset to viz"
    )
    parser.add_argument(
    "-viz", "--viz_reconstruction",
    nargs="?", const=5, default=None, type=int,
    help="Visualize reconstructions in pretraining; optionally pass number of examples (default=5)"
    )

    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    exp_dir = root_dir / "experiments" / args.exp_name
    config_path = exp_dir / "config.yaml"
    ckpt_path = exp_dir / f"checkpoint_{args.ckpt_type}.pth"

    assert config_path.exists(), f"Config not found at {config_path}"
    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    if args.viz_dataset == "train":
        dataset = get_dataset(config, "train_dataset")
    else:
        dataset, _, _ = get_dataset(config, "test_dataset")

    # Load model
    encoder = get_encoder(config).to(device)
    head = get_head(config).to(device)
    
    load_checkpoint(ckpt_path, device, encoder=encoder, head=head)
    
    # Show visible, reconstructed, ground truth pcs
    
    if args.viz_reconstruction:
        inv_class_map = {v: k for k, v in dataset.class_map.items()}
        shown = set()            
        num_viz = 5
        viz_pcds = []
        viz_class_names = []

        encoder.eval()
        head.eval()
        
        with torch.no_grad():
            for pc, label in dataset:
                if isinstance(pc, tuple):                    
                    vis_pts, mask_pts, _ = tuple(t.unsqueeze(0).float().to(device) for t in pc)  # (1, N, 3)
                    vis = vis_pts.squeeze(0).cpu().numpy()
                    mask = mask_pts.squeeze(0).cpu().numpy()
                else:
                    pc = pc.unsqueeze(0).float().to(device)  # (1, N, 3) 
                    
                label = label.item()
                if label in shown:
                    continue

                vis_token, vis_centers = encoder(vis_pts)
                rec_group, mask_group = head(vis_token, vis_centers, mask_pts)  # (G, S, 3)
                G, S, _ = rec_group.shape
                
                vis_centers = fps(vis_pts, G) 
                vis_pts = group_points(vis_pts.clone(), idx=knn_group(vis_pts.clone(), vis_centers, S))
                vis_pts = vis_pts.reshape(-1, 3).cpu().numpy() # (G * S, 3)
                
                mask_centers = fps(mask_pts, G)  # (1, G, 3)
                rec_pts = rec_group.unsqueeze(0) + mask_centers.unsqueeze(2) # (1, G, S, 3)
                rec_pts = rec_pts.reshape(-1, 3).cpu().numpy() # (G * S, 3)   

                gt_pts = mask_group.unsqueeze(0) + mask_centers.unsqueeze(2)
                gt_pts = gt_pts.reshape(-1, 3).cpu().numpy()

                viz_pcds.extend([vis, mask, vis_pts, rec_pts, gt_pts])
                
                class_name = inv_class_map[label]
                viz_class_names.append(class_name)
                
                shown.add(label)
                if len(shown) >= num_viz:
                    break

        # all_vis, all_mask, vis_all, rec_all, gt_all = zip(*viz_pcds)
        # viz_pcds = list(all_vis + all_mask + vis_all + rec_all + gt_all)
        
        print(f"\n[Visualization Summary]")
        print(f"- Each column is a single object from classes: {viz_class_names}")
        print(f"- Row 1: Visible")
        print(f"- Row 2: Reconstructed")
        print(f"- Row 3: Ground Truth")
        viz_pcd(viz_pcds, spacing=2, rows=args.viz_reconstruction)
        
if __name__ == "__main__":
    main()
