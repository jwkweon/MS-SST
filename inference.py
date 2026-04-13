"""
MS-SST Inference Script.

Translate source images to target stain domains using a trained MS-SST model.

Usage:
    # Translate a single image to target domain 2
    python inference.py --model_path TrainedModels/.../model.pth \
                        --source_path source.jpg \
                        --target_domain 2

    # Translate all images in a directory to all domains
    python inference.py --model_path TrainedModels/.../model.pth \
                        --source_dir source_images/ \
                        --all_domains

    # Translate a directory of images to a specific domain
    python inference.py --model_path TrainedModels/.../model.pth \
                        --source_dir source_images/ \
                        --target_domain 0
"""

import os
import argparse

import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from models.network import build_model
from utils.functions import denorm


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint['num_classes']
    model_type = checkpoint.get('model_type', 'default')
    pool_size = checkpoint.get('pool_size', 8)
    saved_opt = checkpoint.get('opt', {})
    nfc = saved_opt.get('nfc', 128)
    nc_im = saved_opt.get('nc_im', 3)

    model = build_model(model_type, nc_im, nfc, num_classes,
                        pool_size=pool_size).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, num_classes


def load_image(img_path, img_shape=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(img_shape),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0)


def translate(model, img, target_domain, num_classes, device):
    """Translate a source image to the target domain."""
    with torch.no_grad():
        img = img.to(device)
        code = torch.eye(num_classes)[target_domain].unsqueeze(0).to(device)
        out = model(img, code)
    return out


def main():
    parser = argparse.ArgumentParser(
        description='MS-SST Inference: Stain-Style Transfer')
    parser.add_argument('--model_path', required=True,
                        help='path to trained model checkpoint')
    parser.add_argument('--source_path', default=None,
                        help='path to a single source image')
    parser.add_argument('--source_dir', default=None,
                        help='path to directory of source images')
    parser.add_argument('--target_domain', type=int, default=None,
                        help='target domain index for translation')
    parser.add_argument('--all_domains', action='store_true',
                        help='translate to all domains')
    parser.add_argument('--output_dir', default='results',
                        help='output directory for translated images')
    parser.add_argument('--img_shape', type=int, nargs=2, default=[256, 256],
                        help='image size (H W)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')

    args = parser.parse_args()

    if args.source_path is None and args.source_dir is None:
        print("Please specify --source_path or --source_dir")
        exit(1)
    if args.target_domain is None and not args.all_domains:
        print("Please specify --target_domain or --all_domains")
        exit(1)

    device = torch.device(f'cuda:{args.gpu}'
                          if torch.cuda.is_available() else 'cpu')
    model, num_classes = load_model(args.model_path, device)
    img_shape = tuple(args.img_shape)

    # Collect source images
    if args.source_path:
        source_images = [args.source_path]
    else:
        exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        source_images = sorted([
            os.path.join(args.source_dir, f)
            for f in os.listdir(args.source_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])

    # Determine target domains
    if args.all_domains:
        target_domains = list(range(num_classes))
    else:
        target_domains = [args.target_domain]

    os.makedirs(args.output_dir, exist_ok=True)

    for src_path in source_images:
        img = load_image(src_path, img_shape)
        name = os.path.splitext(os.path.basename(src_path))[0]

        for domain in target_domains:
            out = translate(model, img, domain, num_classes, device)
            out_path = os.path.join(
                args.output_dir, f"{name}_to_domain{domain}.png")
            save_image(denorm(out), out_path)
            print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
