import sys

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.transforms import v2
from transformers import (
    SwinModel,
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
)

import src.configs as cfg
from src import models, training, dataset, metrics


class Mask2FormerWrapper(Mask2FormerForUniversalSegmentation):
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        output = super().forward(pixel_values=batch["x"])

def main():
    if len(sys.argv) < 2:
        print("Error: An hugging face swin checkpoint argument is required.")
        exit(1)
    chkpt_pth = sys.argv[1]
    # Set configs
    train_cfg = cfg.TrainingConfig(
        n_epochs=100,
        batch_size=32,
        transform=v2.RandomApply(torch.nn.ModuleList([v2.RandomResizedCrop(256, (0.3, 0.5)),])),
        checkpointing_interval=50,
        eval_interval=10,
    )
    optim_cfg = cfg.OPTIM_CFGS["unet"]
    wandb_tags = cfg.WANDB_RUN_TAGS["donwscaled_swin_finetuning"]
    # Instantiate objects
    backbone = SwinModel.from_pretrained(chkpt_pth, add_pooling_layer=False)
    backbone_cfg = backbone.config
    mask2former_cfg = Mask2FormerConfig.from_backbone_config(backbone_cfg, )
    model = Mask2FormerForUniversalSegmentation(mask2former_cfg)
    model_state = model.state_dict()
    backbone_state = backbone.state_dict()
    copied, skipped = 0, 0
    skipped_keys = []
    for k, v in backbone_state.items():
        target_key = f"model.pixel_level_module.encoder.{k}"
        if target_key in model_state and model_state[target_key].shape == v.shape:
            model_state[target_key] = v
            copied += 1
        else:
            skipped += 1
            skipped_keys.append(k)

    model.load_state_dict(model_state, strict=False)
    print(model.model.enco)
    print(f"Copied {copied} Swin weights, skipped {skipped}")
    print(skipped_keys)
    model = models.DownScalingWrapper(model).to(cfg.DEVICE)
    model.cfg = cfg.ModelConfig(
        "swin_fpn_segmentation",
        constructor_kwargs={"decoder_dim": 256},
        downscaling=2,
        up_scale_output=True,
    )
    optim = training.mk_optimizer(model, optim_cfg)
    lr_scheduler = training.mk_lr_scheduler(train_cfg, optim)
    wandb_run = training.wandb_init(optim_cfg, train_cfg, tags=wandb_tags)
    trainer = training.Trainer(model, train_cfg, optim, lr_scheduler, wandb_run)
    data_loaders = dataset.mk_segmentation_data_loaders(train_cfg)
    criterion = metrics.SegmentationLoss(train_cfg)
    # Start training
    trainer.train_model(
        data_loaders,
        criterion,
        "checkpoints/finetuned_swin/{wandb_run_name}/unet_epoch_{epoch}.pt",
    )

if __name__ == "__main__":
    main()