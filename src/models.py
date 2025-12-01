import torch
from torch import nn
from monai.networks.nets import UNet

from src.configs import TrainingConfig, ModelConfig, N_CLASSES


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mk_model(train_cfg: TrainingConfig, model_cfg: ModelConfig) -> nn.Module:
    model = (
        UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=N_CLASSES,
            channels=model_cfg.channels,
            strides=model_cfg.strides,
            kernel_size=model_cfg.kernel_size,
            num_res_units=model_cfg.num_res_units,
            act=model_cfg.act,
            norm=model_cfg.norm,
            dropout=model_cfg.dropout,
            bias=model_cfg.bias,
        )
        .to(device)
    )
    model.cfg = model_cfg
    return model
