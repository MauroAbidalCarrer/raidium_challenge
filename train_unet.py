from itertools import repeat

import torch
from torch import nn
from monai.networks.nets import UNet

from src import configs as cfg
from src import dataset, training, models, metrics


def main():
    dataset.mk_dataset(verbose=False)
    train_cfg = cfg.TRAIN_CONFIGS["unet_training"]
    wandb_cfg = cfg.WandbConfig(["unet"])
    model_cfg = cfg.DFLT_MODELS_CFGS["unet"]
    model = mk_model(model_cfg)
    optimizer = training.mk_optimizer(model, train_cfg)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 1)
    criterion = metrics.SegmentationLoss(train_cfg)
    data_loaders = dataset.mk_data_loaders_for_segmentation(train_cfg)
    trainer = training.Trainer(
        model,
        train_cfg, 
        optimizer,
        lr_scheduler,
        wandb_cfg
    )
    trainer.train_model(
        data_loaders,
        criterion,
        "checkpoints/unet/unet_epoch_{epoch}.pt",
    )

def mk_model(model_cfg: cfg.ModelConfig) -> nn.Module:
    kwargs = model_cfg.constructor_kwargs
    channels = kwargs["channels"]
    model = (
        UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=cfg.N_CLASSES,
            kernel_size=3,
            strides=tuple(repeat(2, len(channels) - 1)),
            bias=True,
            **kwargs,
            # channels=channels,
            # num_res_units=model_cfg.num_res_units,
            # act=model_cfg.act,
            # norm=model_cfg.norm,
            # dropout=model_cfg.dropout,
        )
        .to(cfg.DEVICE)
    )
    model.cfg = model_cfg
    return model


if __name__ == "__main__":
    main()