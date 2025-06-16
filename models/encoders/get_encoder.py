from models.encoders.pointnet_encoder import PointNetEncoder
from models.encoders.pointnetlite_encoder import PointNetLiteEncoder

ENCODER_REGISTRY = {
    "pointnet_encoder": PointNetEncoder,
    "pointnetlite_encoder": PointNetLiteEncoder
}

def get_encoder(config):
    encoder_name = config["encoder"]["name"]
    encoder_args = config["encoder"].get("args", {})
    
    encoder_class = ENCODER_REGISTRY.get(encoder_name)
    if encoder_class is None:
        raise ValueError(f"Unknown encoder: {encoder_name}")
        
    encoder = encoder_class(**encoder_args)
    return encoder