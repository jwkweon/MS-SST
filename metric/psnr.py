# Codes are obtained from https://github.com/bonlime/pytorch-tools

import torch


class PSNR:
    """Peak Signal to Noise Ratio.

    Usage:
        psnr = PSNR(data_range=1.0)
        score = psnr(x, y)
    """

    def __init__(self, data_range=255):
        self.name = "PSNR"
        self.data_range = data_range

    def __call__(self, X, Y):
        mse = torch.mean((X - Y) ** 2)
        return 20 * torch.log10(self.data_range / torch.sqrt(mse))
