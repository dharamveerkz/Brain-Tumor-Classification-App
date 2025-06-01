import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attention_map = self.sigmoid(self.conv(combined))  # (B, 1, H, W)
        return x * attention_map  # Broadcasts across channels

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, channel_first=True):
        super(CBAM, self).__init__()
        self.channel_first = channel_first
        self.ca = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        if self.channel_first:
            x = self.ca(x)
            x = self.sa(x)
        else:
            x = self.sa(x)
            x = self.ca(x)
        return x