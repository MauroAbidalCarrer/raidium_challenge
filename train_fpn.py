import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.transforms import v2
from transformers import SwinModel

import src.configs as cfg
from src import models, training, dataset, metrics


class SwinSegmentationModel(nn.Module):
    def __init__(self, backbone: SwinModel, backbone_cfg: cfg.ModelConfig, decoder_dim=256):
        super().__init__()
        self.backbone = backbone
        embed_dim = backbone.config.embed_dim * 2
        self.stage_dims = [
            embed_dim,
            embed_dim * 2,
            embed_dim * 4,
            embed_dim * 4,
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

            x_shape = x.shape
            # if i > 0:
            x = F.interpolate(
                x,
                (128, 128),
                # scale_factor=2 ** min(i, 2),
                mode="bilinear",
                align_corners=False,
            )

            feats.append(x)
        #     print("og shape", x_shape, "interpolated", x.shape)
        # print("=========")
        
        x = torch.cat(feats, dim=1)
        x = self.fuse(x)
        x = self.head(x)

        return {"y_pred": x}


def main():
    # Set configs
    train_cfg = cfg.TrainingConfig(
        n_epochs=2000,
        batch_size=32,
        transform=v2.Compose([
            v2.RandomApply(torch.nn.ModuleList([v2.RandomResizedCrop(256, (0.3, 0.5)),])),
            v2.RandomApply(torch.nn.ModuleList([v2.GaussianBlur(9, sigma=5)])),
            v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
            v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
            v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
            v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
            v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
            v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
            v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
            v2.RandomErasing(p=0.2, scale=(0.05, 0.1)),
            v2.RandomAffine(degrees=(0, 0), translate=(0.1, 0.3), scale=(0.75, 1)),
        ]),
        checkpointing_interval=50,
        eval_interval=10,
    )
    optim_cfg = cfg.OPTIM_CFGS["unet"]
    wandb_tags = cfg.WANDB_RUN_TAGS["fpn"]
    # Instantiate objects
    backbone = SwinModel.from_pretrained(
        "hf_swin_pretrained",
        add_pooling_layer=False,
    ).to(cfg.DEVICE)
    backbone = backbone.train()
    for param in backbone.parameters():
        param.requires_grad_(False)
    model = SwinSegmentationModel(backbone, cfg.MODELS_CFGS["downscaled_swin_vit"]).to(cfg.DEVICE)
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
