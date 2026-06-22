"""Patch embedding and patch reconstruction helpers."""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Split an image into non overlapping patches and project each one.

    A single convolution with kernel and stride equal to the patch size
    turns an image of shape (batch, channels, H, W) into a sequence of
    patch tokens of shape (batch, num_patches, embed_dim).
    """

    def __init__(self, img_size: int = 32, patch_size: int = 8,
                 in_chans: int = 3, embed_dim: int = 64):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                "img_size %d must be divisible by patch_size %d"
                % (img_size, patch_size)
            )
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, chans, height, width = x.shape
        if height != self.img_size or width != self.img_size:
            raise ValueError(
                "input size (%d, %d) does not match img_size %d"
                % (height, width, self.img_size)
            )
        x = self.proj(x)              # (batch, embed_dim, grid, grid)
        x = x.flatten(2)              # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)         # (batch, num_patches, embed_dim)
        return x


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Turn images into a sequence of flattened pixel patches.

    Args:
        imgs: tensor of shape (batch, channels, H, W) with H == W.
        patch_size: side length of each square patch.

    Returns:
        tensor of shape (batch, num_patches, patch_size * patch_size * channels).
    """
    batch, chans, height, width = imgs.shape
    if height != width:
        raise ValueError("patchify expects square images")
    if height % patch_size != 0:
        raise ValueError("image size must be divisible by patch_size")

    grid = height // patch_size
    x = imgs.reshape(batch, chans, grid, patch_size, grid, patch_size)
    # reorder to (batch, grid, grid, patch, patch, chans) then flatten
    x = x.permute(0, 2, 4, 3, 5, 1)
    x = x.reshape(batch, grid * grid, patch_size * patch_size * chans)
    return x
