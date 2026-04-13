import torch
import torch.nn as nn
import lpips

from utils.functions import denorm
from utils.ssim import SSIM


class Loss(nn.Module):
    """Composable loss module supporting MAE, MSE, LPIPS, and SSIM losses."""

    def __init__(self, losses, device=None):
        super().__init__()
        self.loss_funcs = []
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for loss in losses:
            if 'ssim' in loss:
                win_size = int(loss[loss.rfind('m') + 1:])
                self.ssim = SSIM(data_range=1.0, win_size=win_size,
                                 nonnegative_ssim=True)
                self.loss_funcs.append(self._ssim)
            elif loss == 'mse':
                self.mse = nn.MSELoss()
                self.loss_funcs.append(self._mse)
            elif loss == 'mae':
                self.mae = nn.L1Loss()
                self.loss_funcs.append(self._mae)
            elif loss == 'lpips':
                self.lpips = lpips.LPIPS(net='vgg').to(device)
                self.loss_funcs.append(self._lpips)

    def _ssim(self, x, y):
        x, y = denorm(x), denorm(y)
        return 1 - self.ssim(x, y)

    def _mse(self, x, y):
        return self.mse(x, y)

    def _mae(self, x, y):
        return self.mae(x, y)

    def _lpips(self, x, y):
        return self.lpips(x, y)

    def forward(self, x, y):
        loss = 0
        for loss_func in self.loss_funcs:
            loss = loss + loss_func(x, y)
        return loss
