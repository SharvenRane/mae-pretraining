import torch
import pytest

from src.masking import random_masking


@pytest.mark.parametrize("mask_ratio", [0.0, 0.25, 0.5, 0.75, 0.9])
def test_correct_number_of_patches_masked(mask_ratio):
    batch, num_patches, dim = 4, 16, 8
    x = torch.randn(batch, num_patches, dim)

    x_masked, mask, ids_restore = random_masking(x, mask_ratio)

    expected_keep = max(1, int(round(num_patches * (1.0 - mask_ratio))))
    expected_masked = num_patches - expected_keep

    # The visible sequence length must equal the kept count.
    assert x_masked.shape == (batch, expected_keep, dim)

    # Every sample must mask exactly the expected number of patches.
    per_sample_masked = mask.sum(dim=1)
    assert torch.all(per_sample_masked == expected_masked)

    # The mask is strictly binary.
    assert torch.all((mask == 0) | (mask == 1))


def test_kept_patches_match_originals():
    # Patches that survive masking must be exact copies of the input patches,
    # placed back in their original positions by ids_restore.
    batch, num_patches, dim = 2, 10, 5
    x = torch.randn(batch, num_patches, dim)
    mask_ratio = 0.6

    x_masked, mask, ids_restore = random_masking(x, mask_ratio)
    len_keep = x_masked.shape[1]

    # Reconstruct the full shuffled order: kept tokens then zero filled rest.
    pad = torch.zeros(batch, num_patches - len_keep, dim)
    shuffled = torch.cat([x_masked, pad], dim=1)
    restored = torch.gather(
        shuffled, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, dim)
    )

    visible = (mask == 0).unsqueeze(-1)
    assert torch.allclose(restored * visible, x * visible)


def test_ids_restore_is_a_permutation():
    batch, num_patches, dim = 3, 12, 4
    x = torch.randn(batch, num_patches, dim)
    _, _, ids_restore = random_masking(x, 0.5)

    sorted_ids, _ = torch.sort(ids_restore, dim=1)
    reference = torch.arange(num_patches).unsqueeze(0).expand(batch, -1)
    assert torch.all(sorted_ids == reference)


def test_invalid_mask_ratio_raises():
    x = torch.randn(1, 8, 4)
    with pytest.raises(ValueError):
        random_masking(x, 1.0)
    with pytest.raises(ValueError):
        random_masking(x, -0.1)
