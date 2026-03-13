"""DINOv3 ViT backbone wrapper for PANet."""

import torch
import torch.nn as nn

from dinov3.models.vision_transformer import vit_base


class DINOv3ViTBackbone(nn.Module):
    """Expose DINOv3 ViT-B/16 as a dense feature extractor."""

    def __init__(self, pretrained_path, freeze=True, out_index=-1):
        super().__init__()
        if not pretrained_path:
            raise ValueError("DINOv3 backbone requires a local pretrained weight path.")

        self.backbone = vit_base(
            patch_size=16,
            img_size=224,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            qkv_bias=True,
            layerscale_init=1e-5,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        )
        state_dict = torch.load(pretrained_path, map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=True)
        self.out_index = out_index
        self.freeze = freeze

        if freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                outputs = self.backbone.get_intermediate_layers(x, n=1, reshape=True)
        else:
            outputs = self.backbone.get_intermediate_layers(x, n=1, reshape=True)
        feature_map = outputs[self.out_index]
        return feature_map.float()
