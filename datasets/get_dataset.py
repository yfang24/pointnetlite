from pathlib import Path

from datasets.modelnet import ModelNet
from datasets.modelnet_render import ModelNetRender
from datasets.modelnet_mae_render import ModelNetMAERender
from datasets.modelnet_scan import ModelNetScan
from datasets.scanobjectnn import ScanObjectNN
from datasets.wrappers.rotate_wrapper import RotateWrapper
from datasets.wrappers.aug_wrapper import AugWrapper

PROJ_ROOT = Path(__file__).resolve().parents[2]

ROOTS = {
    "modelnet40_manually_aligned": PROJ_ROOT / "data/modelnet40_manually_aligned",
    "main_split_nobg": PROJ_ROOT / "data/scanobjectnn/main_split_nobg"
}

CLASS_MAPS = {
    "modelnet11": PROJ_ROOT / "code/configs/class_map_modelnet11.json",
    "scanobjectnn11": PROJ_ROOT / "code/configs/class_map_scanobjectnn11.json"
}

# Base dataset registry
DATASET_REGISTRY = {
    "modelnet": ModelNet,
    "modelnet_render": ModelNetRender,
    "modelnet_mae_render": ModelNetMAERender,
    "modelnet_scan": ModelNetScan,
    "scanobjectnn": ScanObjectNN
}

# Optional transformation wrappers
TRANSFORM_REGISTRY = {
    "rotate": RotateWrapper,
    "aug": AugWrapper
}

def get_dataset(config, key: str):
    ds_cfg = config[key]  # key = "train_dataset" or "test_dataset"
    ds_name = ds_cfg["name"]
    ds_class = DATASET_REGISTRY.get(ds_name)
    if ds_class is None:
        raise ValueError(f"Unknown dataset '{ds_name}'")

    ds_args = ds_cfg.get("args", {})
    
    if "root_dir" in ds_args:
        root_key = ds_args["root_dir"]
        if root_key not in ROOTS:
            raise ValueError(f"[get_dataset] root_dir key '{root_key}' not found in ROOTS.")
        ds_args["root_dir"] = ROOTS[root_key]

    if "class_map" in ds_args:
        map_key = ds_args["class_map"]
        if map_key not in CLASS_MAPS:
            raise ValueError(f"[get_dataset] class_map key '{map_key}' not found in CLASS_MAPS.")
        ds_args["class_map"] = CLASS_MAPS[map_key]
        
    dataset = ds_class(**ds_args)

    # Apply transforms
    transforms = ds_cfg.get("transform", [])
    if isinstance(transforms, dict):
        transforms = [transforms]
    elif isinstance(transforms, str):
        transforms = [{"name": transforms, "args": {}}]
    
    # Rotation vote flags (for test_dataset only)
    rotation_vote = False
    num_votes = None
    
    for tf in transforms:
        name = tf["name"]
        args = tf.get("args", {})
        wrapper = TRANSFORM_REGISTRY.get(name)
        if wrapper is None:
            raise ValueError(f"Unknown transform wrapper: {name}")
        dataset = wrapper(dataset, **args)
        
        if key == "test_dataset" and name == "rotate":
            angle_deg = args.get("angle_deg", None)
            if angle_deg is not None and isinstance(angle_deg, (int, float)):
                rotation_vote = True
                num_votes = int(360 / angle_deg)
                
    if key == "test_dataset":
        return dataset, rotation_vote, num_votes
    else:
        return dataset
