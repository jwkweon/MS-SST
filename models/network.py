import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import ConvBlock, upBlock


class MSSST(nn.Module):
    """MS-SST: Multi-domain Single image reconstruction-based Stain-Style Transfer.

    The network consists of two paths:
      1) Style path: transforms domain label vectors into spatial style features
         via linear layers and upsampling blocks (8x8 -> 256x256).
      2) Content path: preserves local information of the source image through
         convolution blocks, while receiving style features via element-wise addition.

    Args:
        img_ch (int): Number of image channels (default: 3).
        net_ch (int): Number of feature channels (default: 128).
        n_class (int): Number of stain domains (default: 4).
        n_body (int): Number of body convolution blocks (default: 4).
        pool_size (int): Style smoothing resolution. Lower values produce
            smoother style maps that reduce spatial artifacts at the cost of
            fine-grained style variation (default: 8).
    """

    def __init__(self, img_ch=3, net_ch=128, n_class=4, n_body=4, pool_size=8):
        super().__init__()
        self.n_class = n_class
        self.n_body = n_body
        self.pool_size = pool_size

        # Content path: extract features from source image
        self.from_rgb = nn.Sequential(
            ConvBlock(img_ch, net_ch // 2, norm='in', act='leakyrelu'),
            ConvBlock(net_ch // 2, net_ch, norm='in', act='leakyrelu'),
        )
        self.content_extract = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(net_ch, net_ch, 3, 1, 0),
            nn.InstanceNorm2d(net_ch),
            nn.ReLU(),
            nn.Conv2d(net_ch, net_ch, 1, 1, 0),
        )

        # Style path: generate style features from domain label
        self.from_code = nn.Sequential(
            nn.Linear(self.n_class, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128 * 8 * 8),
        )
        self.code_norm = nn.InstanceNorm2d(128)
        self.code_upsample = nn.Sequential(
            upBlock(net_ch, net_ch),  # 8 -> 16
            upBlock(net_ch, net_ch),  # 16 -> 32
            upBlock(net_ch, net_ch),  # 32 -> 64
            upBlock(net_ch, net_ch),  # 64 -> 128
            upBlock(net_ch, net_ch),  # 128 -> 256
        )

        # Per-layer style projection (each body layer gets its own style transform)
        self.style_proj = nn.ModuleList([
            nn.Conv2d(net_ch, net_ch, 1, 1, 0)
            for _ in range(n_body)
        ])

        # Translation body: convolution blocks with style injection
        self.body = nn.ModuleList([
            ConvBlock(net_ch, net_ch, ker_s=3, norm='in', act='leakyrelu')
            for _ in range(n_body)
        ])

        # Output: convert features back to RGB
        self.to_rgb = nn.Sequential(
            nn.Conv2d(net_ch, net_ch // 2, 1, 1, 0),
            nn.Conv2d(net_ch // 2, img_ch, 1, 1, 0),
            nn.Tanh(),
        )

    def forward(self, x, code):
        """
        Args:
            x (Tensor): Source image, shape (B, 3, 256, 256), range [-1, 1].
            code (Tensor): One-hot domain label, shape (B, n_class).

        Returns:
            Tensor: Translated/reconstructed image, shape (B, 3, 256, 256), range [-1, 1].
        """
        # Style path
        style = self.from_code(code)
        style = style.view(-1, 128, 8, 8)
        style = self.code_norm(style)
        style = self.code_upsample(style)

        # Content path
        x = self.from_rgb(x)
        content = self.content_extract(x)  # noqa: F841

        # Smooth style: reduce spatial resolution to prevent hotspots
        if self.pool_size < style.shape[2]:
            style = F.adaptive_avg_pool2d(style, self.pool_size)
            style = F.interpolate(style, size=x.shape[2:],
                                  mode='bilinear', align_corners=False)

        # Translation with per-layer style injection
        for proj, layer in zip(self.style_proj, self.body):
            x = x + proj(style)
            x = layer(x)

        x = self.to_rgb(x)
        return x


class MSSSTAdaIN(nn.Module):
    """MS-SST variant using Adaptive Instance Normalization for style injection.

    Instead of element-wise addition, style is injected through AdaIN:
    the style path produces per-layer affine parameters (gamma, beta)
    that modulate the content features after instance normalization.

    Args:
        img_ch (int): Number of image channels (default: 3).
        net_ch (int): Number of feature channels (default: 128).
        n_class (int): Number of stain domains (default: 4).
        n_body (int): Number of body convolution blocks (default: 4).
    """

    def __init__(self, img_ch=3, net_ch=128, n_class=4, n_body=4):
        super().__init__()
        self.n_class = n_class
        self.n_body = n_body

        # Content path
        self.from_rgb = nn.Sequential(
            ConvBlock(img_ch, net_ch // 2, norm='in', act='leakyrelu'),
            ConvBlock(net_ch // 2, net_ch, norm='in', act='leakyrelu'),
        )
        self.content_extract = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(net_ch, net_ch, 3, 1, 0),
            nn.InstanceNorm2d(net_ch),
            nn.ReLU(),
            nn.Conv2d(net_ch, net_ch, 1, 1, 0),
        )

        # Style path (same as MSSST)
        self.from_code = nn.Sequential(
            nn.Linear(self.n_class, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128 * 8 * 8),
        )
        self.code_norm = nn.InstanceNorm2d(128)
        self.code_upsample = nn.Sequential(
            upBlock(net_ch, net_ch),
            upBlock(net_ch, net_ch),
            upBlock(net_ch, net_ch),
            upBlock(net_ch, net_ch),
            upBlock(net_ch, net_ch),
        )

        # AdaIN: per-layer gamma and beta from style
        self.adain_gamma = nn.ModuleList([
            nn.Conv2d(net_ch, net_ch, 1) for _ in range(n_body)
        ])
        self.adain_beta = nn.ModuleList([
            nn.Conv2d(net_ch, net_ch, 1) for _ in range(n_body)
        ])

        # Translation body: conv + instance norm (no affine) + AdaIN + activation
        self.body_conv = nn.ModuleList([
            nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(net_ch, net_ch, 3, 1, 0))
            for _ in range(n_body)
        ])
        self.body_norm = nn.ModuleList([
            nn.InstanceNorm2d(net_ch, affine=False) for _ in range(n_body)
        ])
        self.body_act = nn.ModuleList([
            nn.LeakyReLU(0.2, inplace=True) for _ in range(n_body)
        ])

        # Output
        self.to_rgb = nn.Sequential(
            nn.Conv2d(net_ch, net_ch // 2, 1, 1, 0),
            nn.Conv2d(net_ch // 2, img_ch, 1, 1, 0),
            nn.Tanh(),
        )

    def forward(self, x, code):
        """
        Args:
            x (Tensor): Source image, shape (B, 3, 256, 256), range [-1, 1].
            code (Tensor): One-hot domain label, shape (B, n_class).

        Returns:
            Tensor: Translated/reconstructed image, shape (B, 3, 256, 256), range [-1, 1].
        """
        # Style path → global pool for channel-only modulation
        style = self.from_code(code)
        style = style.view(-1, 128, 8, 8)
        style = self.code_norm(style)
        style = self.code_upsample(style)
        style = F.adaptive_avg_pool2d(style, 1)  # (B, C, 1, 1)

        # Content path
        x = self.from_rgb(x)
        _ = self.content_extract(x)

        # Translation with AdaIN style injection
        for i in range(self.n_body):
            x = self.body_conv[i](x)
            x = self.body_norm[i](x)
            gamma = 1.0 + self.adain_gamma[i](style)  # (B, C, 1, 1)
            beta = self.adain_beta[i](style)           # (B, C, 1, 1)
            x = gamma * x + beta
            x = self.body_act[i](x)

        x = self.to_rgb(x)
        return x


def build_model(model_type='default', img_ch=3, net_ch=128, n_class=4,
                n_body=4, pool_size=8):
    """Factory function to create the appropriate model variant.

    Args:
        model_type (str): 'default' for addition-based, 'adain' for AdaIN-based.
        pool_size (int): Style smoothing resolution (only used for 'default').
    """
    if model_type == 'adain':
        return MSSSTAdaIN(img_ch, net_ch, n_class, n_body)
    else:
        return MSSST(img_ch, net_ch, n_class, n_body, pool_size)
