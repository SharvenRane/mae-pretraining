"""Random patch masking for masked autoencoder pretraining."""

import torch


def random_masking(x: torch.Tensor, mask_ratio: float):
    """Randomly mask a fraction of the patches in a sequence.

    This implements the per sample shuffling strategy from the MAE paper.
    For every sample we draw a random ordering of the patches, keep the
    first ``len_keep`` of them, and drop the rest. The drop count is fixed
    per batch so the kept set has a constant length that the encoder can
    process as a dense tensor.

    Args:
        x: tensor of shape (batch, num_patches, dim).
        mask_ratio: fraction of patches to remove, in the half open
            interval [0, 1).

    Returns:
        x_masked: the kept patches, shape (batch, len_keep, dim).
        mask: binary mask of shape (batch, num_patches) where 1 marks a
            removed (masked) patch and 0 marks a kept (visible) patch.
        ids_restore: indices of shape (batch, num_patches) that undo the
            random shuffle, used by the decoder to place tokens back in
            their original order.
    """
    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in [0, 1), got %r" % (mask_ratio,))

    batch, num_patches, _ = x.shape
    len_keep = int(round(num_patches * (1.0 - mask_ratio)))
    # Always keep at least one patch so the encoder has something to read.
    len_keep = max(1, len_keep)

    noise = torch.rand(batch, num_patches, device=x.device)

    # Sort noise ascending; the small noise values are the ones we keep.
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    )

    # Build the binary mask: 0 for kept, 1 for removed.
    mask = torch.ones(batch, num_patches, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore
