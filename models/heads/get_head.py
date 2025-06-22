from models.heads.pointnet_cls_head import PointNetClsHead
from models.heads.pointnet_semseg_head import PointNetSemSegHead
from models.heads.pointmae_decoder import PointMAEDecoder
from models.heads.pointmae_cls_head import PointMAEClsHead
from models.heads.pointnet2_cls_head import PointNet2ClsHead
from models.heads.dgcnn_cls_head import DGCNNClsHead

HEAD_REGISTRY = {
    "pointnet_cls_head": PointNetClsHead,
    "pointnet_semseg_head": PointNetSemSegHead,
    "pointmae_decoder": PointMAEDecoder,
    "pointmae_cls_head": PointMAEClsHead,
    "pointnet2_cls_head": PointNet2ClsHead,
    "dgcnn_cls_head": DGCNNClsHead
}

def get_head(config):
    head_name = config["head"]["name"]
    head_args = config["head"].get("args", {})
    
    head_class = HEAD_REGISTRY.get(head_name)    
    if head_class is None:
        raise ValueError(f"Unknown encoder head: {head_name}")
        
    head = head_class(**head_args)
    return head
