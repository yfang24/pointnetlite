import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from configs.load_config import load_config
from datasets.get_dataset import get_dataset
from models.encoders.get_encoder import get_encoder
from models.heads.get_head import get_head
from models.losses.get_loss import get_loss
from utils.train_utils import evaluate, pretrain_evaluate
from utils.checkpoint_utils import load_checkpoint
import utils.pcd_utils as pcd_utils

def main(exp_name):    
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp", "--exp_name", type=str, required=True,
                    help="Experiment folder name (e.g., pointnet_test_20250612_103045)")
    parser.add_argument(
        "-ckpt", "--ckpt_type", type=str, choices=["best", "last"], default="best",
        help="Which checkpoint to use: 'best' (default) or 'last'"
    )
    parser.add_argument(
        "-cm", "--confusion_matrix", action="store_true",
        help="Whether to show the confusion matrix"
    )
    parser.add_argument(
        "-viz", "--viz_reconstruction", action="store_true", 
        help="Visualize reconstructions in pretraining")

    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    exp_dir = root_dir / "experiments" / args.exp_name
    config_path = exp_dir / "config.yaml"
    checkpoint_path = exp_dir / f"checkpoint_{args.ckpt_type}.pth"

    assert config_path.exists(), f"Config not found at {config_path}"
    assert checkpoint_path.exists(), f"Checkpoint not found at {checkpoint_path}"

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    test_set, rotation_vote, num_votes = get_dataset(config, "test_dataset")
    test_loader = DataLoader(
        test_set,
        batch_size=config.get("batch_size", 32),
        shuffle=False,
        num_workers=config.get("num_workers", 4)
    )

    # Load model
    encoder = get_encoder(config).to(device)
    head = get_head(config).to(device)
    model = torch.nn.Sequential(encoder, head).to(device)
    loss_fn = get_loss(config)

    load_checkpoint(encoder, head, checkpoint_path, device)

    # Evaluate
    is_pretrain = config.get("pretrain", False)
    if is_pretrain:
        _ = pretrain_evaluate(
            model, test_loader, loss_fn, device, None, rotation_vote, num_votes
        )
    else:
        _, _, _, _, _, _, test_cm = evaluate(
            model, test_loader, loss_fn, device, None, rotation_vote, num_votes
        )

    # Plot confusion matrix
    if args.confusion_matrix:
        class_names = getattr(test_set, "class_names", [str(i) for i in range(test_cm.shape[0])])
        disp = ConfusionMatrixDisplay(confusion_matrix=test_cm, display_labels=class_names)
        disp.plot(xticks_rotation=45, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(exp_dir / "confusion_matrix.png")
        plt.show()

    # Show visible, reconstructed, ground truth pcs
    if args.viz_reconstruction:
        inv_class_map = {v: k for k, v in test_set.class_map.items()}
        shown = set()            
        num_viz = 5
        viz_pcds = []
        viz_class_names = []

        encoder.eval()
        head.eval()
        with torch.no_grad():
            for pc, label in dataloader:
                pc = pc.float().to(device)
                pc = pc.unsqueeze(0).float().to(device)  # (1, N, 3)
                full_pts = pc[0].cpu().numpy()           # (N, 3)
                
                label = label.item()
                if label in shown:
                    continue

                x_vis, mask, neighborhood, center = encoder(pc, noaug=False)
                rec, gt = head(x_vis, mask, neighborhood, center)

                vis_pts = neighborhood[0][~mask[0]].reshape(-1, 3).cpu().numpy() # (G_visible * S, 3)

                visible_groups = neighborhood[0][~mask[0]].cpu()                  # (G_visible, S, 3)
                reconstructed_groups = rec.reshape(-1, neighborhood.shape[2], 3).cpu()  # (G_masked, S, 3)
        
                full_reconstructed = torch.cat([visible_groups, reconstructed_groups], dim=0)  # (G, S, 3)
                rec_pts = full_reconstructed.reshape(-1, 3).numpy()                            # (G * S, 3)
                
                rec_pts = rec[0].reshape(-1, 3).cpu().numpy()
                gt_pts = gt[0].reshape(-1, 3).cpu().numpy()
                viz_pcds.extend([vis_pts, rec_pts, full_pts])
                
                class_name = inv_class_map[label]
                viz_class_names.append(class_name)
                
                shown.add(label)
                if len(shown) >= num_viz:
                    break
                    
        print(visualizing one sample of viz_class_names(list of classes), in order of Visible", "Reconstructed", "Ground Truth of one sample input per row)
                    
if __name__ == "__main__":
    main()
