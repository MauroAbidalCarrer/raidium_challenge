import torch
from torch import nn
from monai.networks.nets import UNet

from src.configs import ModelConfig, N_CLASSES, DEVICE, N_MODEL_OUT_CHANNELS


def mk_model(model_cfg: ModelConfig, compile: bool=True) -> nn.Module:
    model = (
        UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=N_MODEL_OUT_CHANNELS,
            channels=model_cfg.channels,
            strides=model_cfg.strides,
            kernel_size=model_cfg.kernel_size,
            num_res_units=model_cfg.num_res_units,
            act=model_cfg.act,
            norm=model_cfg.norm,
            dropout=model_cfg.dropout,
            bias=model_cfg.bias,
        )
        .to(DEVICE)
    )
    model.cfg = model_cfg
    if compile:
        model = torch.compile(model)
    return model
