import argparse


def get_arguments():
    parser = argparse.ArgumentParser(
        description='MS-SST: Multi-domain Single image reconstruction-based '
                    'Stain-Style Transfer')

    # Workspace
    parser.add_argument('--not_cuda', action='store_true', default=False,
                        help='disables cuda')
    parser.add_argument('--manualSeed', type=int, default=3473,
                        help='manual seed')

    # Network parameters
    parser.add_argument('--nc_im', type=int, default=3,
                        help='number of image channels')
    parser.add_argument('--nfc', type=int, default=128,
                        help='number of feature channels in the network')
    parser.add_argument('--n_class', type=int, default=None,
                        help='number of domains (auto-detected from data '
                             'if not specified)')
    parser.add_argument('--model_type', type=str, default='default',
                        choices=['default', 'adain'],
                        help='model variant: default (addition) or adain')
    parser.add_argument('--pool_size', type=int, default=8,
                        help='style smoothing resolution (default model only)')

    # Training parameters (matching paper: Section IV.B)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='beta2 for Adam optimizer')
    parser.add_argument('--niter', type=int, default=300,
                        help='number of training epochs')

    # Loss weights (matching paper: Eq. 5)
    parser.add_argument('--lambda_rec', type=float, default=10.0,
                        help='weight for reconstruction loss (L1/MAE)')
    parser.add_argument('--lambda_perceptual', type=float, default=0.1,
                        help='weight for perceptual loss (LPIPS)')
    parser.add_argument('--lambda_ssim', type=float, default=0.1,
                        help='weight for SSIM loss')

    return parser
