import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, ker_s=3, stride=1, padding=0,
                 norm='in', act='leakyrelu'):
        super().__init__()

        if ker_s == 3:
            self.conv2d = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_c, out_c, ker_s, stride, padding),
            )
        elif ker_s == 1:
            self.conv2d = nn.Sequential(
                nn.Conv2d(in_c, out_c, ker_s, stride, padding),
            )

        norm_layer = {
            'in': nn.InstanceNorm2d(out_c),
            'bn': nn.BatchNorm2d(out_c),
            'identity': nn.Identity(),
        }
        act_layer = {
            'relu': nn.ReLU(True),
            'leakyrelu': nn.LeakyReLU(inplace=True, negative_slope=0.2),
            'sigmoid': nn.Sigmoid(),
            'identity': nn.Identity(),
        }

        self.layer = nn.Sequential(
            self.conv2d,
            norm_layer[norm],
            act_layer[act],
        )

    def forward(self, x):
        return self.layer(x)


def upBlock(in_c, out_c):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(out_c),
        nn.LeakyReLU(inplace=True, negative_slope=0.2),
    )
