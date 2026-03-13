"""Backward-compatible VGG backbone import."""

from .backbones.vgg import VGGEncoder as Encoder

__all__ = ["Encoder"]
