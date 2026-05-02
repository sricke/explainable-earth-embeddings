# CLIP-Surgery inference (GeoCLIP / SatCLIP)

# Surgery Vision Transformer — what changes from a traditional ViT

This note documents how `images/clip_surgery/surgery_vision_transformer.py` differs from a vanilla `timm` ViT. The file is essentially the standard `timm` `VisionTransformer`, but with **CLIP-Surgery** modifications grafted onto the attention path. Concretely, four things differ from a "vanilla" ViT.

## 1. Last 6 blocks use a different attention (`ConsistentAttention`)

In a normal ViT, every block uses the same `Attention` class. Here, while building blocks, the last `min(6, depth)` blocks are forced to `consistent_attn`:

```python
dpr = calculate_drop_path_rates(drop_path_rate, depth)  # stochastic depth decay rule
self.blocks = nn.ModuleList()
for i in range(depth):
    # apply architecture surgery on the last 6 blocks
    start_index = max(0, depth - 6)
    apply_surgery = i >= start_index
    attn_layer_i = "consistent_attn" if apply_surgery else attn_layer
    self.blocks.append(
        block_fn(
            ...
            attn_layer=attn_layer_i,
            depth=i,
            ...
        )
    )
```

`ConsistentAttention` is the CLIP-Surgery "v-v" (a.k.a. "consistent self-attention") variant from `images/clip_surgery/modified_attention.py` and is wired in via the `ATTN_LAYERS` registry:

```python
ATTN_LAYERS = {
    "": Attention,
    "attn": Attention,
    "consistent_attn": ConsistentAttention,
    # "diff": DiffAttention,
}
```

A vanilla ViT has no such registry and no per-depth attention swap.

## 2. The block forward becomes dual-path

A normal ViT block is:

```text
x = x + drop_path(ls(attn(norm(x))))
x = x + drop_path(ls(mlp(norm(x))))
```

Here, when the block uses `ConsistentAttention`, attention returns **two outputs** `(x_attn, x_ori_attn)` and the block routes them as two parallel streams:

```python
def forward(self, x, attn_mask=None):
    if isinstance(self.attn, ConsistentAttention):
        if isinstance(x, list):
            x, x_ori = x
            x_attn, x_ori_attn = self.drop_path1(self.ls1(self.attn(self.norm1(x_ori), attn_mask=attn_mask)))
            x_ori += x_ori_attn
            x_ori_ffn = self.drop_path2(self.ls2(self.mlp(self.norm2(x_ori))))
            x_ori += x_ori_ffn

            x += x_attn  # skip ffn for the new path
            return [x, x_ori]
        else:
            x_attn, x_ori_attn = self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
            x_ori = x + x_ori_attn
            x_ori_ffn = self.drop_path2(self.ls2(self.mlp(self.norm2(x_ori))))
            x_ori += x_ori_ffn
            x += x_attn  # skip ffn for the new path
            return [x, x_ori]
    else:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x
```

Two things to note:

- `x_ori` is the **standard** ViT path (residual attn + residual MLP).
- `x_new` (the one named `x` in the code) gets only the new attention residual and **skips the MLP**. That's the key surgery trick: the new path produces cleaner localization features without mixing them through MLPs again.
- Pre-norms (`norm1`, `norm2`) are always applied to `x_ori`, not the new path.

A traditional ViT block has no list-input handling, no dual residual, and never skips the MLP.

## 3. End-of-backbone merge (CLS from original path, tokens from new path)

After all blocks run, the dual streams are merged so that the final token tensor has the **CLS token from the original stream** but the **patch tokens from the new stream**:

```python
if attn_mask is not None:
    for blk in self.blocks:
        x = blk(x, attn_mask=attn_mask)
elif self.grad_checkpointing and not torch.jit.is_scripting():
    x = checkpoint_seq(self.blocks, x)
else:
    x = self.blocks(x)

# x is a list [x_new, x_ori] because last blocks use ConsistentAttention
if isinstance(x, list) and len(x) == 2:
    x_new, x_ori = x
    x_new[:, 0, :] = x_ori[:, 0, :]  # cls from original path, img tokens from new path
    x = x_new

x = self.norm(x)
```

The same merge is reused inside `forward_intermediates` via a small helper:

```python
def _merge_satclip_dual_path_nlc(x):
    if isinstance(x, list) and len(x) == 2:
        x_new, x_ori = x
        out = x_new.clone()
        out[:, 0, :] = x_ori[:, 0, :]
        return out
    return x
```

A vanilla ViT just runs `x = self.norm(self.blocks(x))` with no list merging.

## 4. `forward_intermediates` is dual-path-aware

The intermediates API exists in upstream `timm` too, but here it’s extended to:

- merge the dual path before saving each layer's output, and
- **clone** detached tensors, because later `ConsistentAttention` blocks mutate the path tensors in-place — without cloning, all "saved" layers would converge to the final activations:

```python
if i in take_indices:
    # Dual-path blocks return [x_new, x_ori]; merge before norm / storage.
    xm = _merge_satclip_dual_path_nlc(x)
    # Must clone: later ConsistentAttention blocks mutate `x` / list paths in-place;
    # shared refs would make every saved "layer" converge to the final activations.
    to_store = self.norm(xm) if norm else xm
    intermediates.append(to_store.detach().clone())
```

This is precisely what `inference_satclip.py` relies on to draw `layers_*.png` (per-layer similarity heatmaps).

## What stays the same as a vanilla ViT

Everything else is bog-standard `timm` ViT and unchanged in spirit:

- `PatchEmbed` patchification (`patch_embed`).
- Optional CLS token, optional register tokens, `pos_embed` (learned), `pos_drop`, `patch_drop`, optional `norm_pre`.
- Pre-norm blocks with `ls1`/`drop_path1` and `ls2`/`drop_path2`.
- Final `norm`, optional `attn_pool` (MAP head), `fc_norm`, `head_drop`, `head`.
- Multiple block flavors are still available (`Block`, `ResPostBlock`, `ParallelScalingBlock`, `DiffParallelScalingBlock`, `ParallelThingsBlock`).
- Init schemes (`jax`, `moco`, `timm`), checkpoint adapters for OpenAI-CLIP / DINOv2 / AIMv2 / BEiT3, and `_load_weights` for Flax `.npz` checkpoints — these are upstream `timm` features carried over so SatCLIP weights still load cleanly.

## TL;DR

The only real architectural change vs a traditional ViT is the **CLIP-Surgery dual-path attention** in the last 6 blocks: `ConsistentAttention` produces two outputs, the block runs them as two streams (`x_new` skipping the MLP, `x_ori` being the normal ViT residual), and the backbone merges them at the end by taking the CLS token from `x_ori` and patch tokens from `x_new`. Everything else (patch embed, pos embed, MLP, pre-norm, head, init) is untouched.
