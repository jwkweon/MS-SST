import os
import random
import datetime
import dateutil.tz

import numpy as np
import torch
import torch.nn.functional as F


def denorm(x):
    """Convert from [-1, 1] to [0, 1]."""
    return ((x + 1) / 2).clamp(0, 1)


def norm(x):
    """Convert from [0, 1] to [-1, 1]."""
    return ((x - 0.5) * 2).clamp(-1, 1)


def interp(x, img_shape):
    """Bicubic interpolation."""
    if isinstance(img_shape, (tuple, list)):
        return F.interpolate(x, size=img_shape, mode='bicubic',
                             align_corners=True).clamp(-1, 1)
    elif isinstance(img_shape, int):
        h, w = x.shape[-2:]
        sf = img_shape / max(h, w)
        h, w = int(h * sf), int(w * sf)
        return F.interpolate(x, size=(h, w), mode='bicubic',
                             align_corners=True).clamp(-1, 1)
    else:
        raise ValueError(f"Unsupported img_shape type: {type(img_shape)}")


def generate_dir2save(opt):
    dir2save = 'TrainedModels/{}/'.format(opt.timestamp)
    return dir2save


def post_config(opt):
    if opt.not_cuda:
        opt.device = torch.device("cpu")
    elif torch.cuda.is_available():
        opt.device = torch.device("cuda:{}".format(opt.gpu))
    elif torch.backends.mps.is_available():
        opt.device = torch.device("mps")
    else:
        opt.device = torch.device("cpu")

    opt.timestamp = datetime.datetime.now(
        dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    return opt
