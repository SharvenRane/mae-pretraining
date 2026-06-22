"""A compact Masked Autoencoder built on top of Transformer blocks.

The model follows the structure of the MAE paper: a patch embedding,
random masking of most patches, a Transformer encoder that only sees the
visible patches, and a lightweight Transformer decoder that fills in mask
tokens and reconstructs the raw pixels. The training objective is the mean
squared error on the masked patches only.
"""

import torch
import torch.nn as nn

from .patch_embed import PatchEmbed, patchify
from .masking import random_masking


def sincos_pos_embed(num_patches: int, dim: int) -> torch.Tensor:
    """Fixed sinusoidal position embedding of shape (1, num_patches, dim)."""
    position = torch.arange(num_patches).unsqueeze(1).float()
    div = torch.exp(
        torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim)
    )
    pe = torch.zeros(num_patches, dim)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


class TransformerBlock(nn.Module):
    """Pre norm Transformer block with multi head attention and an MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MAE(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 8, in_chans: int = 3,
                 embed_dim: int = 64, depth: int = 2, num_heads: int = 4,
                 decoder_embed_dim: int = 32, decoder_depth: int = 1,
                 decoder_num_heads: int = 4, mlp_ratio: float = 2.0,
                 mask_ratio: float = 0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.in_chans = in_chans

        # ---- encoder ----
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.register_buffer(
            "pos_embed", sincos_pos_embed(num_patches, embed_dim), persistent=False
        )
        self.encoder_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ---- decoder ----
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer(
            "decoder_pos_embed",
            sincos_pos_embed(num_patches, decoder_embed_dim),
            persistent=False,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio)
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        patch_dim = patch_size * patch_size * in_chans
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim)

    def forward_encoder(self, imgs: torch.Tensor, mask_ratio: float):
        x = self.patch_embed(imgs)
        x = x + self.pos_embed
        x_masked, mask, ids_restore = random_masking(x, mask_ratio)
        for block in self.encoder_blocks:
            x_masked = block(x_masked)
        x_masked = self.encoder_norm(x_masked)
        return x_masked, mask, ids_restore

    def forward_decoder(self, latent: torch.Tensor, ids_restore: torch.Tensor):
        x = self.decoder_embed(latent)
        batch = x.shape[0]
        num_masked = ids_restore.shape[1] - x.shape[1]
        mask_tokens = self.mask_token.expand(batch, num_masked, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        # unshuffle to restore the original patch order
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )
        x = x + self.decoder_pos_embed
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

    def forward(self, imgs: torch.Tensor, mask_ratio: float = None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = mae_reconstruction_loss(imgs, pred, mask, self.patch_size)
        return loss, pred, mask


def mae_reconstruction_loss(imgs: torch.Tensor, pred: torch.Tensor,
                            mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Mean squared error between predicted and true patches, masked patches only.

    Args:
        imgs: original images, shape (batch, channels, H, W).
        pred: predicted patches, shape (batch, num_patches, patch_dim).
        mask: binary mask, shape (batch, num_patches), 1 for masked patches.
        patch_size: side length of each square patch.

    Returns:
        scalar tensor, the average per pixel squared error over masked patches.
    """
    target = patchify(imgs, patch_size)
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)               # per patch mean over pixels
    denom = mask.sum()
    if denom.item() == 0:
        return loss.mean()
    loss = (loss * mask).sum() / denom     # mean over masked patches
    return loss
