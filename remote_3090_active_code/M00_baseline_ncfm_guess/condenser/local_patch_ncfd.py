import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def patchify_images(images, grid):
    """Split images into a regular grid of non-overlapping patches."""
    if grid <= 0:
        raise ValueError(f"local_patch_grid must be positive, got {grid}")
    bsz, channels, height, width = images.shape
    if height % grid != 0 or width % grid != 0:
        raise ValueError(
            f"Image size {(height, width)} must be divisible by local_patch_grid={grid}"
        )
    patch_h = height // grid
    patch_w = width // grid
    patches = images.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(bsz, grid * grid, channels, patch_h, patch_w)


class FrozenConvPatchEncoder(nn.Module):
    """Frozen shallow ConvNet encoder that maps each patch to a 128-d feature."""

    def __init__(self, convnet, num_blocks=2):
        super().__init__()
        if not hasattr(convnet, "layers"):
            raise TypeError("FrozenConvPatchEncoder expects a ConvNet with a `layers` dict")

        max_blocks = len(convnet.layers["conv"])
        if num_blocks < 1 or num_blocks > max_blocks:
            raise ValueError(
                f"local_patch_encoder_blocks must be in [1, {max_blocks}], got {num_blocks}"
            )

        blocks = []
        has_norm = len(convnet.layers["norm"]) > 0
        has_pool = len(convnet.layers["pool"]) > 0
        for idx in range(num_blocks):
            modules = [copy.deepcopy(convnet.layers["conv"][idx])]
            if has_norm:
                modules.append(copy.deepcopy(convnet.layers["norm"][idx]))
            modules.append(copy.deepcopy(convnet.layers["act"][idx]))
            if has_pool:
                modules.append(copy.deepcopy(convnet.layers["pool"][idx]))
            blocks.append(nn.Sequential(*modules))

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


def build_frozen_patch_encoder(args):
    """Build a frozen patch encoder from a trained premodel checkpoint."""
    from utils.ddp import load_state_dict
    from utils.utils import define_model

    source = getattr(args, "local_patch_encoder_source", "premodel0_trained")
    if source != "premodel0_trained":
        raise ValueError(
            "v1 only supports local_patch_encoder_source='premodel0_trained'"
        )

    model = define_model(
        args.dataset,
        args.norm_type,
        args.net_type,
        args.nch,
        args.depth,
        args.width,
        args.nclass,
        args.logger,
        args.size,
    ).to(args.device)
    checkpoint_path = os.path.join(args.pretrain_dir, "premodel0_trained.pth.tar")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Patch encoder checkpoint not found: {checkpoint_path}"
        )
    load_state_dict(checkpoint_path, model)
    model.eval()

    num_blocks = int(getattr(args, "local_patch_encoder_blocks", 2))
    patch_encoder = FrozenConvPatchEncoder(model, num_blocks=num_blocks).to(args.device)
    patch_encoder.eval()
    if getattr(args, "rank", 0) == 0:
        args.logger(
            "Local patch-feature NCFD encoder: "
            f"source={source}, blocks={num_blocks}, checkpoint={checkpoint_path}"
        )
    return patch_encoder


def _extract_patch_features(images, patch_encoder, grid):
    patches = patchify_images(images, grid)
    bsz, num_patches, channels, patch_h, patch_w = patches.shape
    patches = patches.view(bsz * num_patches, channels, patch_h, patch_w)
    feats = patch_encoder(patches)
    return feats.view(bsz, num_patches, -1)


def local_patch_feature_ncfd_loss(img_real, img_syn, patch_encoder, cf_loss_func, args):
    """Compute local NCFD over original-image patches encoded as frozen features."""
    if patch_encoder is None:
        raise ValueError("patch_encoder is required when local patch-feature NCFD is enabled")

    grid = int(getattr(args, "local_patch_grid", 4))
    local_num_freqs = int(
        getattr(args, "local_patch_num_freqs", min(int(args.num_freqs), 256))
    )

    with torch.no_grad():
        feat_real = _extract_patch_features(img_real, patch_encoder, grid)
        feat_real = F.normalize(feat_real, dim=2)
    feat_syn = _extract_patch_features(img_syn, patch_encoder, grid)
    feat_syn = F.normalize(feat_syn, dim=2)

    num_patches = feat_syn.shape[1]
    feature_dim = feat_syn.shape[2]
    if feature_dim != int(getattr(args, "local_patch_feature_dim", feature_dim)):
        raise ValueError(
            f"Expected local patch feature dim {args.local_patch_feature_dim}, got {feature_dim}"
        )

    t = torch.randn((local_num_freqs, feature_dim), device=feat_syn.device)
    loss = feat_syn.new_tensor(0.0)
    for patch_idx in range(num_patches):
        loss = loss + cf_loss_func(
            feat_real[:, patch_idx, :],
            feat_syn[:, patch_idx, :],
            t,
            args,
        )

    return 300.0 * loss / num_patches
