"""
MS-SST Training Script.

Train a multi-domain stain-style transfer model using reconstruction-based learning.

Usage:
    python train.py --dir_path <path_to_data> [--gpu 0] [--niter 300]

Data directory structure (ImageFolder format):
    data_dir/
        domain_0/
            image_0.jpg
        domain_1/
            image_1.jpg
        ...
"""

import os
import logging

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image
from tqdm import tqdm

from config.config import get_arguments
from models.network import build_model
from utils.loader import PathologyLoader
from utils.losses import Loss
from utils.augmentation import JitAugment
from utils.functions import post_config, generate_dir2save, denorm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def train(opt):
    # Data
    loader = PathologyLoader(opt)
    num_classes = loader.num_classes
    if opt.n_class is not None:
        num_classes = opt.n_class
    opt.num_classes = num_classes
    logging.info(f"Detected {num_classes} domains from {opt.dir_path}")

    train_iter = loader.dataloader_iter

    # Augmentation (color modulation)
    aug_color = JitAugment(policy='color')
    aug_cutout = JitAugment(policy='cutout')

    # Model
    model = build_model(opt.model_type, opt.nc_im, opt.nfc, num_classes,
                        pool_size=opt.pool_size).to(opt.device)
    logging.info(f"Model: {opt.model_type} (pool_size={opt.pool_size}), "
                 f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss functions
    loss_mae = Loss(['mae'], device=opt.device)
    loss_lpips = Loss(['lpips'], device=opt.device)
    loss_ssim = Loss(['ssim7'], device=opt.device)

    # Optimizer
    optimizer = Adam(model.parameters(), opt.lr, [opt.beta1, opt.beta2])
    scheduler = CosineAnnealingLR(optimizer, opt.niter)

    # Save directory
    save_dir = generate_dir2save(opt)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/samples", exist_ok=True)
    logging.info(f"Saving to {save_dir}")

    # Training loop
    model.train()
    pbar = tqdm(range(1, opt.niter + 1), desc='Training')
    for epoch in pbar:
        img, label = next(train_iter)
        img = img.to(opt.device)
        label = label.to(opt.device)

        # Data augmentation (color modulation for reconstruction objective)
        if torch.rand(1) <= 0.25:
            img_aug = aug_cutout.transform(img)
        else:
            img_aug = img
        img_aug = aug_color.transform(img_aug)

        # Forward: reconstruct source from corrupted input + source label
        code = torch.eye(num_classes)[label.cpu()].to(opt.device)
        out = model(img_aug, code)

        # Loss (Eq. 5 in paper)
        # MAE: reconstruct original color, LPIPS/SSIM: preserve input structure
        l_mae = loss_mae(out, img)
        l_lpips = loss_lpips(out, img_aug)
        l_ssim = loss_ssim(out, img_aug)
        loss = (opt.lambda_rec * l_mae +
                opt.lambda_perceptual * l_lpips.mean() +
                opt.lambda_ssim * l_ssim)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_postfix_str(
            f'Total:{loss.item():.4f} '
            f'MAE:{l_mae.item():.4f} '
            f'LPIPS:{l_lpips.mean().item():.4f} '
            f'SSIM:{l_ssim.item():.4f}')

        # Save sample images periodically
        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                save_image(denorm(out), f"{save_dir}/samples/epoch_{epoch}.png")

    # Save model
    torch.save({
        'model': model.state_dict(),
        'num_classes': num_classes,
        'model_type': opt.model_type,
        'pool_size': opt.pool_size,
        'opt': vars(opt),
    }, f"{save_dir}/model.pth")
    logging.info(f"Model saved to {save_dir}/model.pth")

    return save_dir


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--dir_path', required=True,
                        help='training data directory (ImageFolder format)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--img_shape', type=int, nargs=2, default=[256, 256],
                        help='target image size (H W)')

    opt = parser.parse_args()
    opt.img_shape = tuple(opt.img_shape)
    opt = post_config(opt)

    if not os.path.exists(opt.dir_path):
        print(f"Data directory does not exist: {opt.dir_path}")
        exit(1)

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)

    train(opt)
