import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.transforms import v2
from transformers import SwinModel

import src.configs as cfg
from src import models, training, dataset, metrics


class SwinSegmentationModel(nn.Module):
    def __init__(self, backbone: SwinModel, decoder_dim=256, mask_ratio: float=0):
        super().__init__()
        self.backbone = backbone
        embed_dim = backbone.config.embed_dim * 2
        self.stage_dims = [
            embed_dim,
            embed_dim * 2,
            embed_dim * 4,
            embed_dim * 4,
        ]

        self.mask_ratio = mask_ratio
        if self.mask_ratio:
            self.mask_generator = models.MaskGenerator(128, backbone.config.patch_size, mask_ratio, 8)

        self.proj = nn.ModuleList([
            nn.Conv2d(dim, decoder_dim, kernel_size=1)
            for dim in self.stage_dims
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(decoder_dim, cfg.N_CLASSES, kernel_size=1)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        pixel_values = batch["x"]
        if self.mask_generator:
            bool_masked_pos = torch.stack(
                [self.mask_generator() for _ in range(pixel_values.shape[0])],
                dim=0,
            ).to(pixel_values.device)
        else:
            bool_masked_pos = None
        print("pixe values", pixel_values.shape, "bool mask", bool_masked_pos.shape)
        outputs = self.backbone(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
            bool_masked_pos=bool_masked_pos,
        )

        hidden_states = outputs.hidden_states[1:]  # 4 stages
        feats = []
        for i, h in enumerate(hidden_states):
            B, L, C = h.shape
            H = W = int(L ** 0.5)

            x = h.transpose(1, 2).reshape(B, C, H, W)
            x = self.proj[i](x)

            x_shape = x.shape
            if i > 0:
                x = F.interpolate(
                    x,
                    (16, 16),
                    mode="bilinear",
                    align_corners=False,
                )

            feats.append(x)
        
        x = torch.cat(feats, dim=1)
        x = self.fuse(x)
        x = self.head(x)

        return {"y_pred": x}


def main():
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
    backbone = SwinModel.from_pretrained(
        "hf_swin_pretrained",
        add_pooling_layer=False,
    ).to(cfg.DEVICE)
    backbone = backbone.train()
    for param in backbone.parameters():
        param.requires_grad_(False)
    model = SwinSegmentationModel(backbone, mask_ratio=0.3).to(cfg.DEVICE)
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