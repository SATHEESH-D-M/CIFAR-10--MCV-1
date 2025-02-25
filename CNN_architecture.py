import torch
import torch.nn as nn


class VGG11(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, dropout=0.0):
        super(VGG11, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),  # 32x32 -> 32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 16x16 -> 16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 8x8 -> 8x8
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 8x8 -> 8x8
            nn.ReLU(),
            # No final pooling layer — keep 8x8 feature map
        )

        # Fully connected layers
        self.linear_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)  # (batch_size, 256, 8, 8) -> (batch_size, 256*8*8)
        x = self.linear_layers(x)
        return x
