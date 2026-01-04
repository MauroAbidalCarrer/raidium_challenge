from __future__ import annotations
from itertools import repeat
from typing import Tuple, Optional, Any

import torch
import numpy as np
from torch import Tensor, nn
from einops import rearrange
from torch.nn import functional as F
from monai.networks.nets import UNet
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from transformers import (
    CONFIG_MAPPING,
    IMAGE_PROCESSOR_MAPPING,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForMaskedImageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

import src.configs as cfg


def gather_batch(x: Tensor, idx: Tensor) -> Tensor:
    """
    Gather tokens from x (B, T, C) according to idx (B, T_idx) per-batch.
    Returns out (B, T_idx, C) where out[b, t, :] = x[b, idx[b, t], :].
    """
    if x.ndim != 3 or idx.ndim != 2:
        raise ValueError("x must be (B,T,C) and idx must be (B,T_idx)")
    B, T, C = x.shape
    _, T_idx = idx.shape
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, C)  # (B, T_idx, C)
    return torch.gather(x, dim=1, index=idx_exp)


class PatchShuffleBF(nn.Module):
    """
    Batch-first patch shuffle + mask:
      - Input: patches (B, T, C)
      - Output: visible_patches (B, T_vis, C), forward_idx_bt (B, T), backward_idx_bt (B, T)
    """
    def __init__(self, mask_ratio: float) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, patches_bt: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        patches_bt: (B, T, C)
        returns:
          - visible_patches: (B, T_vis, C)
          - forward_idx_bt: (B, T) permutation indexes
          - backward_idx_bt: (B, T) inverse permutation
        """
        if patches_bt.ndim != 3:
            raise ValueError("patches_bt must be (B, T, C)")

        B, T, C = patches_bt.shape
        device = patches_bt.device

        mask_ratio = 0 if torch.is_inference_mode_enabled() else self.mask_ratio
        T_vis = int(T * (1.0 - mask_ratio))
        if T_vis == T:
            # trivial: identity permutation
            forward_idx_bt = torch.arange(T, device=device).unsqueeze(0).expand(B, -1).contiguous()
            backward_idx_bt = forward_idx_bt.argsort(dim=1)
            return patches_bt, forward_idx_bt, backward_idx_bt

        # Generate random permutations on GPU
        # rand: (B, T) -> argsort -> forward_idx_bt (B, T)
        rand = torch.rand(B, T, device=device)
        forward_idx_bt = rand.argsort(dim=1)  # (B, T)

        # Permute tokens and keep first T_vis tokens
        permuted = gather_batch(patches_bt, forward_idx_bt)  # (B, T, C)
        visible = permuted[:, :T_vis, :].contiguous()        # (B, T_vis, C)

        # inverse permutation
        backward_idx_bt = forward_idx_bt.argsort(dim=1)      # (B, T)

        return visible, forward_idx_bt, backward_idx_bt

class MAE_EncoderBF(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 2,
        emb_dim: int = 192,
        num_layer: int = 12,
        num_head: int = 3,
        mask_ratio: float = 0.75,
        in_channels: int = 1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.in_channels = in_channels

        self.num_patches = (image_size // patch_size) ** 2

        # patchify: conv maps image -> patch embeddings (B, emb, h, w)
        self.patchify = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

        # positional embedding: (1, T, C) for batch-first addition
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))

        # class token: (1, 1, C)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        # shuffle
        self.shuffle = PatchShuffleBF(mask_ratio)

        # transformer: expects batch-first (B, T, C)
        self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, batch_dict: dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        img: (B, in_channels, H, W)
        returns:
          - features_bt: (B, T_vis+1, C)  (batch-first: cls + visible tokens)
          - forward_idx_bt: (B, T)
          - backward_idx_bt: (B, T)
        """
        x = batch_dict["x"]
        B = x.shape[0]

        # patchify -> (B, C, h, w)
        patches = self.patchify(x)  # (B, emb, h, w)
        patches = rearrange(patches, 'b c h w -> b (h w) c').contiguous()  # (B, T, C)

        # add pos embedding (broadcast over batch)
        patches = patches + self.pos_embedding  # (B, T, C)

        # shuffle and mask (batch-first)
        visible, forward_idx_bt, backward_idx_bt = self.shuffle(patches)  # visible: (B, T_vis, C)

        # prepend cls token and run transformer
        cls = self.cls_token.expand(B, -1, -1).contiguous()  # (B,1,C)
        seq_bt = torch.cat([cls, visible], dim=1)            # (B, T_vis+1, C)

        features_bt = self.transformer(seq_bt)               # (B, T_vis+1, C)
        features_bt = self.layer_norm(features_bt)

        return features_bt, forward_idx_bt, backward_idx_bt

class MAE_DecoderBF(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 2,
        emb_dim: int = 192,
        num_layer: int = 4,
        num_head: int = 3,
        out_channels: int = 1,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.num_patches = (image_size // patch_size) ** 2

        # mask token (1,1,C) to be expanded
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        # full pos embedding for cls + all patches
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_dim))

        # transformer (batch-first)
        self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        # projection to patch pixels
        self.head = nn.Linear(emb_dim, out_channels * patch_size * patch_size)

        # reconstruct patches -> image (expects shape (B, T, out_ch * p1 * p2))
        self.patch2img = Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                                   p1=patch_size, p2=patch_size, h=image_size // patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, features_bt: Tensor, backward_idx_bt: Tensor) -> Tuple[Tensor, Tensor]:
        """
        features_bt: (B, T_vis+1, C)  (batch-first)
        backward_idx_bt: (B, T) inverse permutation giving original positions
        returns:
          - img: (B, out_ch, H, W)
          - mask_img: (B, out_ch, H, W) where predicted pixels are 1, others 0
        """
        B, L, C = features_bt.shape
        device = features_bt.device
        T = backward_idx_bt.shape[1]  # full number of patches

        # total sequence length including cls
        total_len = T + 1

        # number of mask tokens to append so feats_padded length == total_len
        n_mask = total_len - L
        if n_mask > 0:
            mask_tokens = self.mask_token.expand(B, n_mask, C).contiguous()  # (B, n_mask, C)
            feats_padded = torch.cat([features_bt, mask_tokens], dim=1)      # (B, total_len, C)
        else:
            feats_padded = features_bt  # already full

        # Build full_back_idx_bt in batch-first form: shape (B, total_len)
        # We want full_back_idx_bt[b, 0] = 0 (cls), and full_back_idx_bt[b, 1:] = backward_idx_bt[b] + 1
        cls_col = torch.zeros(B, 1, dtype=backward_idx_bt.dtype, device=device)
        body_idx = backward_idx_bt + 1  # shift by 1 to account for cls
        full_back_idx_bt = torch.cat([cls_col, body_idx], dim=1)  # (B, total_len)

        # Unshuffle: gather from feats_padded according to full_back_idx_bt -> (B, total_len, C)
        feats_unshuffled = gather_batch(feats_padded, full_back_idx_bt)

        # Add positional embedding (batch-first)
        feats_unshuffled = feats_unshuffled + self.pos_embedding  # (B, total_len, C)

        # Run decoder transformer
        out_bt = self.transformer(feats_unshuffled)  # (B, total_len, C)

        # Drop cls token and keep patch tokens
        patch_feats = out_bt[:, 1:, :]  # (B, T, C)

        # Project to patch pixels
        patches = self.head(patch_feats)  # (B, T, out_ch * p1 * p2)

        # Convert to image
        img = self.patch2img(patches)  # (B, out_ch, H, W)

        # Build mask image indicating which patches were reconstructed (i.e., originally masked)
        # The reconstructed patches correspond to indices that came from appended mask tokens.
        # Those appended mask tokens are at indices >= L in feats_padded, so any position t where full_back_idx_bt[:, t+1] >= L
        # corresponds to a predicted patch.
        idx_body = full_back_idx_bt[:, 1:]  # (B, T)
        predicted_pos = (idx_body >= L)     # (B, T) boolean

        # Build mask patches of shape (B, T, out_ch * p1 * p2)
        mask_patches = predicted_pos.unsqueeze(-1).to(dtype=patches.dtype) * torch.ones_like(patches)
        # Convert mask patches to image
        mask_img = self.patch2img(mask_patches)  # (B, out_ch, H, W)

        return img, mask_img

class MAE_ViT(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        mask_ratio: int = 0.75,
        emb_dim: int = 256,
        encoder_layer: int = 6,
        encoder_head: int = 8,
        decoder_layer: int = 3,
        decoder_head: int = 8,
        in_channels: int = 1,
        out_channels: int = cfg.N_CLASSES + 1,
    ):
        super().__init__()
        self.encoder = MAE_EncoderBF(
            image_size=image_size,
            patch_size=patch_size,
            emb_dim=emb_dim,
            num_layer=encoder_layer,
            num_head=encoder_head,
            mask_ratio=mask_ratio,
            in_channels=in_channels,
        )
        self.decoder = MAE_DecoderBF(
            image_size=image_size,
            patch_size=patch_size,
            emb_dim=emb_dim,
            num_layer=decoder_layer,
            num_head=decoder_head,
            out_channels=out_channels,
        )

    def forward(self, batch_dict: dict[str, Any]) -> dict[str, Tensor]:
        """
        img: (B, in_channels, H, W)
        returns:
          - predicted image: (B, out_ch, H, W)
          - mask image: (B, out_ch, H, W)
        """
        features_bt, forward_idx_bt, backward_idx_bt = self.encoder(batch_dict)
        preds, mask = self.decoder(features_bt, backward_idx_bt)
        output_dict = {
            "x_hat": preds[:, :1],
            "y_pred": preds[:, 1:],
            "mask": mask,
        }
        return output_dict

class DownScalingWrapper(nn.Module):

    def __init__(self, model: nn.Module, downscaling: int = 2):
        super().__init__()
        self.downscaling = downscaling
        self.model = model

    def forward(self, batch_dict: dict[str, Any]) -> dict[str, Tensor]:
        batch_dict = {
            "x": F.max_pool2d(batch_dict["x"], self.downscaling, self.downscaling),
        }
        output_dict = self.model(batch_dict)
        for output_k in ("x_hat", "mask", "y_pred"):
            if output_k in output_dict:
                output_dict[output_k] = F.interpolate(
                    output_dict[output_k],
                    (256, 256),
                )
        return output_dict

def mk_model_from_cfg(model_cfg: cfg.ModelConfig) -> nn.Module:
    kwargs = model_cfg.constructor_kwargs
    downscaling_remainder = 256 % (model_cfg.downscaling or 1)
    assert downscaling_remainder == 0, f"Downscaling remainder must be 0, got {downscaling_remainder}."
    if model_cfg.architecture == "unet":
        channels = kwargs["channels"]
        model = (
            UnetWrapper(
                spatial_dims=2,
                in_channels=1,
                out_channels=cfg.N_CLASSES,
                kernel_size=3,
                strides=tuple(repeat(2, len(channels) - 1)),
                bias=True,
                **kwargs,
            )
            .to(cfg.DEVICE)
        )
    elif model_cfg.architecture == "mae_vit":
        model = MAE_ViT(
            image_size=256 // model_cfg.downscaling or 1,
            **kwargs
        )
    elif model_cfg.architecture == "hf_swin_vit":
         model = AutoModelForMaskedImageModeling.from_config(**model_cfg.constructor_kwargs)
    if model_cfg.downscaling is not None:
        model = DownScalingWrapper(model, model_cfg.downscaling)
    if model_cfg.compile:
        model = torch.compile(model)
    model.cfg = model_cfg
    model = model.to(cfg.DEVICE)
    return model

class UnetWrapper(UNet):
    def forward(self, batch_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return {"y_pred": super().forward(batch_dict["x"])}

def print_params_count(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("number of parameters:", str(params // 1e6) + "M")
