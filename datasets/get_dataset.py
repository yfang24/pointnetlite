from datasets.modelnet import ModelNet
from datasets.modelnet_mesh import ModelNetMesh
from datasets.modelnet_render import ModelNetRender
from datasets.modelnet_scan import ModelNetScan
from datasets.scanobjectnn import ScanObjectNN
from datasets.wrappers.rotate_wrapper import RotateWrapper
from datasets.wrappers.aug_wrapper import AugWrapper

# Base dataset registry
DATASET_REGISTRY = {
    "modelnet": ModelNet,
    "modelnet_mesh": ModelNetMesh,
    "modelnet_render": ModelNetRender,
    "modelnet_scan": ModelNetScan,
    "scanobjectnn": ScanObjectNN
}

# Optional transformation wrappers
TRANSFORM_REGISTRY = {
    "rotate": RotateWrapper,
    "aug": AugWrapper
}

def get_dataset(config, key: str):
    ds_cfg = config["dataset"][key]  # key = "train", "val", or "test"
    ds_name = ds_cfg["name"]
    ds_class = DATASET_REGISTRY.get(ds_name)
    if ds_class is None:
        raise ValueError(f"Unknown dataset '{ds_name}'")

    ds_args = ds_cfg.get("args", {})
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
        
        if key == "test" and name == "rotate":
            angle_deg = args.get("angle_deg", None)
            if angle_deg and isinstance(angle_deg, (int, float)):
                rotation_vote = True
                num_votes = int(360 / angle_deg)
                
    if key == "test":
        return dataset, rotation_vote, num_votes
    else:
        return dataset
