# mae-pretraining

A small, readable implementation of Masked Autoencoder pretraining in PyTorch. The goal here is to show the core mechanics clearly rather than to chase scale: random patch masking, an encoder that only ever sees the visible patches, a lightweight decoder that fills in the gaps, and a reconstruction loss computed on the masked patches only.

## The idea

A Masked Autoencoder is a self supervised method. You take an image, cut it into a grid of patches, hide most of them, and ask a model to reconstruct the missing pixels from the few patches that remain. Because the masking ratio is high (often around 75 percent), the model cannot solve the task by copying neighbors. It has to learn something about the structure of the data. After pretraining you usually throw the decoder away and keep the encoder as a feature extractor.

This repo implements that loop end to end on tiny tensors so the whole thing runs on a CPU in seconds.

## How it works here

The forward pass has four stages.

1. **Patch embedding.** A single strided convolution turns an image of shape `(batch, channels, H, W)` into a sequence of patch tokens of shape `(batch, num_patches, embed_dim)`. Fixed sinusoidal position embeddings are added so the model knows where each patch came from.

2. **Random masking.** For each sample we draw a random ordering of the patches, keep the first `len_keep` of them, and drop the rest. The kept count is fixed per batch, so the visible set is a dense tensor the encoder can process directly. We also return a binary mask and the indices needed to undo the shuffle later. This follows the per sample shuffle trick from the original paper, which avoids any need for sparse attention.

3. **Encoding.** The Transformer encoder runs only on the visible patches. That is the efficiency win of MAE: with a high mask ratio the encoder sees a small fraction of the sequence.

4. **Decoding and loss.** The decoder takes the encoded visible tokens, inserts a learned mask token at every dropped position, restores the original patch order, adds its own position embeddings, and predicts raw pixels per patch. The loss is the mean squared error between the prediction and the true pixels, averaged over the masked patches only. Visible patches do not contribute to the loss.

## Layout

```
src/
  masking.py      random patch masking
  patch_embed.py  patch embedding and pixel patchify
  mae.py          the MAE model and the reconstruction loss
tests/
  test_masking.py
  test_mae.py
```

## Running the tests

The tests use tiny synthetic tensors, so there are no downloads and no GPU needed.

```
python -m pytest tests/ -q
```

What the tests check, in plain terms:

* For a range of mask ratios, the number of patches actually masked matches what the ratio asks for, and the visible sequence length lines up with the kept count.
* The patches that survive masking are exact copies of the input patches, sitting in their original positions once the shuffle is undone.
* The restore indices form a real permutation of the patch positions.
* A perfect reconstruction gives a loss of zero, which pins down the masked averaging.
* Overfitting a single fixed batch drives the masked reconstruction loss down. The masking pattern is held constant across steps so the objective is stationary and the before and after comparison is honest.

On the machine where this was written, all tests pass. The overfitting test reports a substantial drop in the masked reconstruction loss over a short run rather than any fixed benchmark figure.

## Using the model

```python
import torch
from src.mae import MAE

model = MAE(img_size=32, patch_size=8, in_chans=3, mask_ratio=0.75)
imgs = torch.randn(8, 3, 32, 32)
loss, pred, mask = model(imgs)
loss.backward()
```

`loss` is the masked reconstruction MSE, `pred` holds the predicted pixel patches, and `mask` marks which patches were hidden.

## Notes and limitations

This is a teaching scale build. The defaults use a shallow encoder and a single decoder block so the tests stay fast. The position embeddings are fixed sinusoidal rather than learned, and there is no class token, since the focus is the masking and reconstruction objective itself. Swapping in a deeper encoder, learned position embeddings, and a real image dataset is straightforward from here.
