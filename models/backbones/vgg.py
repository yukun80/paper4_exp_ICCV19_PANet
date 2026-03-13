"""VGG backbone for PANet."""

import torch
import torch.nn as nn


class VGGEncoder(nn.Module):
    """VGG16-style encoder used by the original PANet implementation."""

    def __init__(self, in_channels=3, pretrained_path=None):
        super().__init__()
        self.pretrained_path = pretrained_path

        self.features = nn.Sequential(
            self._make_layer(2, in_channels, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(2, 64, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(3, 128, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(3, 256, 512),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            self._make_layer(3, 512, 512, dilation=2, last_relu=False),
        )

        self._init_weights()

    def forward(self, x):
        return self.features(x)

    def _make_layer(self, n_convs, in_channels, out_channels, dilation=1, last_relu=True):
        layer = []
        for i in range(n_convs):
            layer.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    dilation=dilation,
                    padding=dilation,
                )
            )
            if i != n_convs - 1 or last_relu:
                layer.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layer)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

        if self.pretrained_path is None:
            return

        state_dict = torch.load(self.pretrained_path, map_location="cpu")
        current_state = self.state_dict()
        current_keys = list(current_state.keys())
        pretrained_keys = list(state_dict.keys())

        for index in range(26):
            current_state[current_keys[index]] = state_dict[pretrained_keys[index]]

        self.load_state_dict(current_state)
