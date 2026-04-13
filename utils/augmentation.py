import torch
import torch.nn.functional as F


class JitAugment:
    """Differentiable augmentation for reconstruction-based training.

    Based on: Differentiable Augmentation for Data-Efficient GAN Training
    (Zhao et al., https://arxiv.org/pdf/2006.10738)
    """

    def __init__(self, policy=''):
        self.augment_fns = {
            'color': [self.rand_brightness, self.rand_saturation,
                      self.rand_contrast],
            'translation': [self.rand_translation],
            'cutout': [self.rand_cutout],
            'shuffle': [self.shuffle_pixel],
        }
        self.policy = policy

    def rand_brightness(self, x):
        x = x + (torch.rand(x.size(0), 1, 1, 1,
                             dtype=x.dtype, device=x.device) - 0.5)
        return x

    def rand_saturation(self, x):
        x_mean = x.mean(dim=1, keepdim=True)
        x = ((x - x_mean) *
             (torch.rand(x.size(0), 1, 1, 1,
                         dtype=x.dtype, device=x.device) * 2) + x_mean)
        return x

    def rand_contrast(self, x):
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        x = ((x - x_mean) *
             (torch.rand(x.size(0), 1, 1, 1,
                         dtype=x.dtype, device=x.device) + 0.5) + x_mean)
        return x

    def rand_translation(self, x, ratio=0.125):
        shift_x = int(x.size(2) * ratio + 0.5)
        shift_y = int(x.size(3) * ratio + 0.5)
        translation_x = torch.randint(
            -shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
        translation_y = torch.randint(
            -shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
            indexing='ij',
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        x = (x_pad.permute(0, 2, 3, 1).contiguous()
             [grid_batch, grid_x, grid_y]
             .permute(0, 3, 1, 2).contiguous())
        return x

    def rand_cutout(self, x, ratio=0.4):
        cutout_size = (int(x.size(2) * ratio + 0.5),
                       int(x.size(3) * ratio + 0.5))
        offset_x = torch.randint(
            0, x.size(2) + (1 - cutout_size[0] % 2),
            size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(
            0, x.size(3) + (1 - cutout_size[1] % 2),
            size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
            indexing='ij',
        )
        grid_x = torch.clamp(
            grid_x + offset_x - cutout_size[0] // 2,
            min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(
            grid_y + offset_y - cutout_size[1] // 2,
            min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3),
                          dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0

        alpha = torch.rand(1).to(x.device)
        x = x * mask.unsqueeze(1) + (1 - mask.unsqueeze(1)) * alpha
        return x

    def shuffle_pixel(self, x, p=0.005):
        if p == 0:
            return x.clone()
        b, c, h, w = x.shape
        out = x.clone()
        original_idx = torch.arange(h * w)
        shuffle_idx = torch.randperm(h * w)
        shuffle_idx = torch.where(
            torch.rand(*original_idx.shape) > p, original_idx, shuffle_idx)
        out = out.view(b, 3, -1)[:, :, shuffle_idx].view(*out.shape)
        return out

    def transform(self, x):
        if self.policy:
            for p in self.policy.split(','):
                for f in self.augment_fns[p]:
                    x = f(x)
            x = x.contiguous()
        return x
