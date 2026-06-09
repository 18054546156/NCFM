import math

import torch
import torch.nn.functional as F


def _parse_grids(value):
    if value is None:
        return [1]
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        grids = [int(item) for item in items]
    elif isinstance(value, (list, tuple)):
        grids = [int(item) for item in value]
    else:
        grids = [int(value)]
    grids = [grid for grid in grids if grid >= 1]
    if not grids:
        raise ValueError("ssim_grids selected no valid grids")
    return grids


def _gaussian_1d(kernel_size, sigma, device, dtype):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel


def gaussian_window(kernel_size, channels, device, dtype, sigma=None):
    if kernel_size % 2 == 0:
        raise ValueError(f"SSIM kernel_size must be odd, got {kernel_size}")
    if sigma is None:
        sigma = 1.5 if kernel_size == 7 else max(float(kernel_size) / 6.0, 0.5)
    kernel_1d = _gaussian_1d(kernel_size, sigma, device, dtype)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    window = kernel_2d.view(1, 1, kernel_size, kernel_size)
    return window.repeat(channels, 1, 1, 1)


def differentiable_ssim(x, y, kernel_size=7, data_range=1.0, eps=1e-8):
    """Return mean SSIM for tensors shaped [B, C, H, W]."""
    if x.shape != y.shape:
        raise ValueError(f"SSIM expects matching shapes, got x={x.shape} y={y.shape}")
    if x.dim() != 4:
        raise ValueError(f"SSIM expects [B, C, H, W], got shape={x.shape}")

    _, channels, height, width = x.shape
    min_hw = min(height, width)
    kernel_size = int(kernel_size)
    if kernel_size > min_hw:
        kernel_size = min_hw if min_hw % 2 == 1 else min_hw - 1
    if kernel_size < 1:
        raise ValueError(f"Invalid SSIM kernel_size for input shape={x.shape}")
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size < 1:
        kernel_size = 1

    padding = kernel_size // 2
    window = gaussian_window(kernel_size, channels, x.device, x.dtype)
    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy

    data_range = float(data_range)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator.clamp_min(eps)
    return ssim_map.mean()


def split_grid(images, grid):
    """Split [B, C, H, W] into [B * grid * grid, C, H/grid, W/grid]."""
    if grid == 1:
        return images
    if images.dim() != 4:
        raise ValueError(f"split_grid expects [B, C, H, W], got shape={images.shape}")
    batch, channels, height, width = images.shape
    if height % grid != 0 or width % grid != 0:
        raise ValueError(f"Image size {(height, width)} is not divisible by grid={grid}")
    patch_h = height // grid
    patch_w = width // grid
    patches = images.view(batch, channels, grid, patch_h, grid, patch_w)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    return patches.view(batch * grid * grid, channels, patch_h, patch_w)


def _prototype_pair(img_real, img_syn):
    real_proto = img_real.mean(dim=0, keepdim=True)
    syn_proto = img_syn.mean(dim=0, keepdim=True)
    return real_proto, syn_proto


def multiscale_ssim_prototype_loss(img_real, img_syn, args):
    """Compute class-prototype multi-scale SSIM regularization.

    This is image-space regularization, not local patch feature NCFD:
    real/synthetic images are averaged into class prototypes, optionally split into
    spatial grids, and compared using differentiable SSIM.
    """
    pairing = getattr(args, "ssim_pairing", "prototype")
    if pairing != "prototype":
        raise ValueError("SSIM v1 only supports ssim_pairing='prototype'")

    grids = _parse_grids(getattr(args, "ssim_grids", [1]))
    kernel_size = int(getattr(args, "ssim_kernel_size", 7))
    data_range = float(getattr(args, "ssim_data_range", 1.0))

    real_proto, syn_proto = _prototype_pair(img_real, img_syn)
    losses = []
    components = {}
    for grid in grids:
        real_patches = split_grid(real_proto, grid)
        syn_patches = split_grid(syn_proto, grid)
        ssim_value = differentiable_ssim(
            syn_patches,
            real_patches.detach(),
            kernel_size=kernel_size,
            data_range=data_range,
        )
        loss = 1.0 - ssim_value
        losses.append(loss)
        components[f"grid{grid}"] = float(loss.detach().item())

    total = torch.stack(losses).mean()
    if bool(getattr(args, "ssim_log_components", True)):
        args._ssim_last_loss = float(total.detach().item())
        args._ssim_last_components = components
    return total
