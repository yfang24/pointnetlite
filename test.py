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
from utils.train_utils import evaluate
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
    loss_fn = get_loss(config)

    load_checkpoint(encoder, head, checkpoint_path, device)

    # Evaluate
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

if __name__ == "__main__":
    main()
