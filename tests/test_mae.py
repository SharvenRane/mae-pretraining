import torch
import pytest

from src.mae import MAE, mae_reconstruction_loss
from src.patch_embed import PatchEmbed, patchify


def make_model():
    torch.manual_seed(0)
    return MAE(
        img_size=16,
        patch_size=4,
        in_chans=3,
        embed_dim=32,
        depth=2,
        num_heads=4,
        decoder_embed_dim=24,
        decoder_depth=1,
        decoder_num_heads=4,
        mask_ratio=0.75,
    )


def test_patch_embed_shapes():
    pe = PatchEmbed(img_size=16, patch_size=4, in_chans=3, embed_dim=32)
    x = torch.randn(2, 3, 16, 16)
    out = pe(x)
    assert out.shape == (2, pe.num_patches, 32)
    assert pe.num_patches == (16 // 4) ** 2


def test_patchify_roundtrip_dimensions():
    imgs = torch.randn(2, 3, 16, 16)
    patches = patchify(imgs, patch_size=4)
    grid = 16 // 4
    assert patches.shape == (2, grid * grid, 4 * 4 * 3)


def test_forward_pass_shapes_and_finite():
    model = make_model()
    imgs = torch.randn(2, 3, 16, 16)
    loss, pred, mask = model(imgs)
    patch_dim = 4 * 4 * 3
    assert pred.shape == (2, model.num_patches, patch_dim)
    assert mask.shape == (2, model.num_patches)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_loss_is_zero_on_perfect_reconstruction():
    imgs = torch.randn(2, 3, 16, 16)
    target = patchify(imgs, patch_size=4)
    mask = torch.ones(2, target.shape[1])
    loss = mae_reconstruction_loss(imgs, target.clone(), mask, patch_size=4)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_reconstruction_loss_decreases_on_one_batch():
    # Overfitting a single tiny batch must drive the masked reconstruction
    # loss down. We fix the masking pattern across steps so the objective is
    # stationary and the comparison is fair.
    torch.manual_seed(123)
    model = make_model()
    imgs = torch.randn(4, 3, 16, 16)

    # Freeze the random mask once so every step optimizes the same target.
    latent, mask, ids_restore = model.forward_encoder(imgs, model.mask_ratio)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def compute_loss():
        latent, mask_b, ids = model.forward_encoder(imgs, model.mask_ratio)
        pred = model.forward_decoder(latent, ids)
        return mae_reconstruction_loss(imgs, pred, mask_b, model.patch_size)

    # Use a deterministic mask by seeding right before each forward so the
    # mask draw is identical step to step.
    def step_loss(train):
        torch.manual_seed(777)
        loss = compute_loss()
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss

    with torch.no_grad():
        torch.manual_seed(777)
        initial = compute_loss().item()

    for _ in range(80):
        step_loss(train=True)

    with torch.no_grad():
        torch.manual_seed(777)
        final = compute_loss().item()

    assert final < initial * 0.7, (
        "expected loss to drop substantially, got initial=%.5f final=%.5f"
        % (initial, final)
    )


def test_masked_patches_count_in_forward():
    model = make_model()
    imgs = torch.randn(3, 3, 16, 16)
    _, _, mask = model(imgs)
    expected_keep = max(1, int(round(model.num_patches * (1.0 - model.mask_ratio))))
    expected_masked = model.num_patches - expected_keep
    assert torch.all(mask.sum(dim=1) == expected_masked)
