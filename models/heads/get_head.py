from models.heads.pointnet_cls_head import PointNetClsHead
from models.heads.pointnet_semseg_head import PointNetSemSegHead

HEAD_REGISTRY = {
    "pointnet_cls_head": PointNetClsHead,
    "pointnet_semseg_head": PointNetSemSegHead
}

def get_head(config):
    head_name = config["head"]["name"]
    head_args = config["head"].get("args", {})
    
    head_class = HEAD_REGISTRY.get(head_name)    
    if head_class is None:
        raise ValueError(f"Unknown encoder head: {head_name}")
        
    head = head_class(**head_args)
    return head