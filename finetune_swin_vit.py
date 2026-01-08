import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.transforms import v2
from transformers import SwinModel

import src.configs as cfg
from src import models, training, dataset, metrics


class SwinSegmentationModel(nn.Module):
    def __init__(self, backbone: SwinModel, decoder_dim=256):
        super().__init__()
        self.backbone = backbone

        embed_dim = backbone.config.embed_dim
        self.stage_dims = [
            embed_dim,
            embed_dim * 2,
            embed_dim * 4,
            embed_dim * 8,
        ]

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
        outputs = self.backbone(
            batch["x"],
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[1:]  # 4 stages

        feats = []
        for i, h in enumerate(hidden_states):
            B, L, C = h.shape
            H = W = int(L ** 0.5)

            x = h.transpose(1, 2).reshape(B, C, H, W)
            x = self.proj[i](x)

            if i > 0:
                x = F.interpolate(
                    x,
                    scale_factor=2 ** i,
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
    model = SwinSegmentationModel(backbone).to(cfg.DEVICE)
    model = models.DownScalingWrapper(model).to(cfg.DEVICE)
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