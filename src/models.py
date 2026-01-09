from __future__ import annotations
from itertools import repeat
from typing import Tuple, Optional, Any

import torch
import numpy as np
from torch import Tensor, nn
from torch.nn import functional as F
from monai.networks.nets import UNet
from transformers import SwinConfig
from transformers import SwinForMaskedImageModeling as MiMModel

import src.configs as cfg


class MaskGenerator:
    def __init__(
            self,
            img_size: int,
            model_patch_size: int,
            mask_ratio: float,
            mask_patch_size: int,
        ):
        self.input_size = img_size
        self.model_patch_size = model_patch_size
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise ValueError("Input size must be divisible by mask patch size")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError("Mask patch size must be divisible by model patch size")

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return torch.tensor(mask.flatten())

class MIMWrapper(MiMModel):
    def __init__(
            self,
            n_layers: int,
            per_layer_depth: int,
            patch_size: int,
            embed_dim: int,
            mask_ratio: int,
            mask_patch_size: int,
            down_scaling_factor: Optional[int] = None,
        ):
        down_scaling_factor = down_scaling_factor or 1
        img_size=256 // down_scaling_factor
        swin_cfg = SwinConfig(
            image_size=img_size,
            window_size=6 // down_scaling_factor,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=list(repeat(per_layer_depth, n_layers)),
            num_heads=[2**(2 + i) for i in range(n_layers)],
            num_channels=1,
        )
        super().__init__(swin_cfg)
        self.mask_generator = MaskGenerator(
            model_patch_size=patch_size,
            img_size=img_size,
            mask_ratio=mask_ratio,
            mask_patch_size=mask_patch_size,
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        pixel_values = batch["x"]
        bool_masked_pos = torch.stack(
            [self.mask_generator() for _ in range(pixel_values.shape[0])],
            dim=0,
        ).to(pixel_values.device)

        outputs = super().forward(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
        )
        outputs = vars(outputs)
        if "reconstruction" in outputs:
            outputs["x_hat"] = outputs["reconstruction"]
        return outputs


class DownScalingWrapper(nn.Module):
    def __init__(self, model: nn.Module, downscaling: int = 2, up_scale_output: bool=True):
        super().__init__()
        self.downscaling = downscaling
        self.model = model
        self.up_scale_output = up_scale_output

    def forward(self, batch_dict: dict[str, Any]) -> dict[str, Tensor]:
        batch_dict = {
            "x": F.max_pool2d(batch_dict["x"], self.downscaling, self.downscaling),
        }
        output_dict = self.model(batch_dict)
        if not self.up_scale_output:
            return output_dict
        for output_k in ("x_hat", "mask", "y_pred"):
            if output_k in output_dict:
                output_dict[output_k] = F.interpolate(
                    output_dict[output_k],
                    (256, 256),
                )
        return output_dict

class UnetWrapper(UNet):
    def forward(self, batch_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return {"y_pred": super().forward(batch_dict["x"])}

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
    elif model_cfg.architecture == "hf_swin_vit":
        model = MIMWrapper(down_scaling_factor=model_cfg.downscaling, **model_cfg.constructor_kwargs)
    if model_cfg.downscaling is not None:
        model = DownScalingWrapper(model, model_cfg.downscaling, model_cfg.up_scale_output)
    if model_cfg.compile:
        model = torch.compile(model)
    model.cfg = model_cfg
    model = model.to(cfg.DEVICE)
    return model

def print_params_count(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("number of parameters:", str(params // 1e6) + "M")
