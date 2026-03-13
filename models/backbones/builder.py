"""Backbone factory for few-shot segmentation."""

from .dinov3_vit import DINOv3ViTBackbone
from .vgg import VGGEncoder


def build_backbone(cfg, pretrained_path=None, in_channels=3):
    """Build the requested backbone and return a dense feature extractor."""
    backbone_name = (cfg or {}).get("backbone", "vgg")

    if backbone_name == "vgg":
        weights_path = pretrained_path
        if isinstance(pretrained_path, dict):
            weights_path = pretrained_path.get("init_path")
        return VGGEncoder(in_channels=in_channels, pretrained_path=weights_path)

    if backbone_name == "dinov3_vitb16":
        weights_path = None
        if isinstance(pretrained_path, dict):
            weights_path = pretrained_path.get("dinov3_init_path")
        elif isinstance(pretrained_path, str):
            weights_path = pretrained_path
        return DINOv3ViTBackbone(
            pretrained_path=weights_path,
            freeze=(cfg or {}).get("freeze_backbone", True),
            out_index=((cfg or {}).get("dinov3") or {}).get("out_index", -1),
        )

    raise ValueError(f"Unsupported backbone '{backbone_name}'.")
