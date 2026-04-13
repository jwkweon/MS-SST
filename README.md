# MS-SST

Official implementation of **"MS-SST: Single Image Reconstruction-Based Stain-Style Transfer for Multi-Domain Hematoxylin & Eosin Stained Pathology Images"** (IEEE Access, 2023).

by Juwon Kweon, Mujung Kim, Gilly Yun, Soonchul Kwon, and Jisang Yoo

[[Paper]](https://doi.org/10.1109/ACCESS.2023.3274877)

## Overview

MS-SST is a multi-domain stain-style transfer model that achieves translation among multiple stain domains using a single training image per domain. Instead of GAN-based adversarial training, MS-SST uses a **reconstruction-based learning framework**, which reduces complexity and training time (~1 minute) compared to GAN objectives (~7 hours).

### Key Features
- **Reconstruction-based training**: No GAN objective needed
- **Single image per domain**: Only one training image required per stain domain
- **Multi-domain transfer**: Single model handles all domain translations
- **Fast training**: ~366x faster than MultiPathGAN

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

### Data Preparation

Organize your training data in ImageFolder format:
```
data/
    domain_0/
        image_0.jpg
    domain_1/
        image_1.jpg
    ...
```

Each domain folder should contain at least one 256x256 pathology image patch.

### Training

```bash
# Default model (addition-based style injection, pool_size=8)
python train.py --dir_path data/ --gpu 0

# AdaIN variant
python train.py --dir_path data/ --gpu 0 --model_type adain
```

**Options:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--dir_path` | (required) | Training data directory |
| `--gpu` | 0 | GPU device ID |
| `--model_type` | `default` | Model variant: `default` or `adain` |
| `--pool_size` | 8 | Style smoothing resolution (`default` model only) |
| `--niter` | 300 | Number of training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--batch_size` | 1 | Batch size |
| `--nfc` | 128 | Number of feature channels |
| `--lambda_rec` | 10.0 | Weight for MAE loss |
| `--lambda_perceptual` | 0.1 | Weight for LPIPS loss |
| `--lambda_ssim` | 0.1 | Weight for SSIM loss |

### Inference

Translate a source image to a target stain domain:

```bash
# Single image to single domain
python inference.py --model_path TrainedModels/<timestamp>/model.pth \
                    --source_path source.jpg \
                    --target_domain 2

# All images in a directory to all domains
python inference.py --model_path TrainedModels/<timestamp>/model.pth \
                    --source_dir source_images/ \
                    --all_domains \
                    --output_dir results/
```

The model type and pool size are automatically detected from the checkpoint.

### Evaluation

Compute PSNR, SSIM, and MS-SSIM metrics:

```bash
python evaluate.py --pred_dir results/ --gt_dir ground_truth/
```

## Method

MS-SST learns multi-domain stain-style transfer through image reconstruction:

1. **Training**: The model receives a color-augmented version of the source image along with its domain label, and is trained to reconstruct the original image.
2. **Inference**: Given a source image and a target domain label, the model translates the image to the target stain style.

### Loss Function

$$\mathcal{L}_{total} = \lambda_{rec}\mathcal{L}_{rec} + \lambda_{p}\mathcal{L}_{p} + \lambda_{ssim}\mathcal{L}_{ssim}$$

where $\lambda_{rec}=10$, $\lambda_{p}=0.1$, $\lambda_{ssim}=0.1$.

### Architecture

The network consists of two paths:
- **Style path**: Transforms domain label vectors into spatial style features via linear layers and upsampling blocks (8x8 -> 256x256)
- **Content path**: Preserves local information through convolution blocks with style feature injection

Two style injection methods are supported:

| Variant | Injection | Description |
|---------|-----------|-------------|
| `default` | Element-wise addition | Style features are projected and added to content features per layer, with spatial smoothing controlled by `pool_size` |
| `adain` | Adaptive Instance Normalization | Style provides per-channel affine parameters (gamma, beta) after instance normalization |

## Citation

```bibtex
@article{kweon2023mssst,
  title={MS-SST: Single Image Reconstruction-Based Stain-Style Transfer for Multi-Domain Hematoxylin \& Eosin Stained Pathology Images},
  author={Kweon, Juwon and Kim, Mujung and Yun, Gilly and Kwon, Soonchul and Yoo, Jisang},
  journal={IEEE Access},
  volume={11},
  pages={50090--50097},
  year={2023},
  doi={10.1109/ACCESS.2023.3274877}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
