"""
MS-SST Evaluation Script.

Compute PSNR, SSIM, and MS-SSIM metrics between translated and reference images.

Usage:
    python evaluate.py --pred_dir results/ --gt_dir ground_truth/
"""

import os
import argparse

import torch
from torchvision import transforms
from PIL import Image

from metric.psnr import PSNR
from utils.ssim import SSIM, MS_SSIM


def load_image(path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description='MS-SST Evaluation')
    parser.add_argument('--pred_dir', required=True,
                        help='directory of predicted/translated images')
    parser.add_argument('--gt_dir', required=True,
                        help='directory of ground-truth/reference images')
    args = parser.parse_args()

    psnr_fn = PSNR(data_range=1.0)
    ssim_fn = SSIM(data_range=1.0, win_size=11, nonnegative_ssim=True)
    ms_ssim_fn = MS_SSIM(data_range=1.0)

    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    pred_files = sorted([
        f for f in os.listdir(args.pred_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    gt_files = sorted([
        f for f in os.listdir(args.gt_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])

    assert len(pred_files) == len(gt_files), \
        f"Mismatch: {len(pred_files)} predictions vs {len(gt_files)} references"

    psnr_scores, ssim_scores, ms_ssim_scores = [], [], []

    for pred_f, gt_f in zip(pred_files, gt_files):
        pred = load_image(os.path.join(args.pred_dir, pred_f))
        gt = load_image(os.path.join(args.gt_dir, gt_f))

        psnr_scores.append(psnr_fn(pred, gt).item())
        ssim_scores.append(ssim_fn(pred, gt).item())
        ms_ssim_scores.append(ms_ssim_fn(pred, gt).item())

    n = len(psnr_scores)
    print(f"Evaluated {n} image pairs")
    print(f"  PSNR:    {sum(psnr_scores)/n:.4f}")
    print(f"  SSIM:    {sum(ssim_scores)/n:.4f}")
    print(f"  MS-SSIM: {sum(ms_ssim_scores)/n:.4f}")


if __name__ == '__main__':
    main()
