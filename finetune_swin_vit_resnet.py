import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.transforms import v2
from transformers import SwinModel, Mask2FormerForUniversalSegmentation

import src.configs as cfg
from src import models, training, dataset, metrics

class SqueezeExcitationBlock(nn.Module):
    # Copy/paste of https://www.kaggle.com/code/wasupandceacar/lb-0-82-5fold-single-bert-model#Model implementation
    def __init__(self, channels:int, reduction:int=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        se = F.adaptive_avg_pool2d(x, 1)[:, :, 0, 0]      # -> (B, C)
        # print(se.shape)
        se = F.relu(self.fc1(se), inplace=True)          # -> (B, C//r)
        se = self.sigmoid(self.fc2(se)).unsqueeze(-1).unsqueeze(-1)    # -> (B, C, 1, 1)
        return x * se

class ResidualBlock(nn.Module):
    def __init__(self, in_chns:int, out_chns:int, dropout_ratio:float=0.3, se_reduction:int=8, kernel_size:int=3):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_chns, out_chns, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_chns),
            nn.ReLU(),
            nn.Conv2d(out_chns, out_chns, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_chns),
            SqueezeExcitationBlock(out_chns, se_reduction),
        )
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout_ratio))
        if in_chns == out_chns:
            self.skip_connection = nn.Identity() 
        else:
            # TODO: set bias to False ?
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_chns, out_chns, 1, bias=False),
                nn.BatchNorm2d(out_chns)
            )
            self.head.insert(1, nn.MaxPool2d(2))

    def forward(self, x:Tensor) -> Tensor:
        activaition_maps = self.skip_connection(x) + self.blocks(x)
        return self.head(activaition_maps)

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


        # self.proj = nn.ModuleList([
        #     nn.Conv2d(dim, decoder_dim, kernel_size=1)
        #     for dim in self.stage_dims
        # ])

        # self.fuse = nn.Sequential(
        #     nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(decoder_dim),
        #     nn.ReLU(inplace=True),
        # )

        
        # self.head = nn.Conv2d(sum(self.stage_dims), cfg.N_CLASSES, kernel_size=3, padding=1)
        # print("caleddddddddddddddddddddd")
        # print("sun stage dims", sum(self.stage_dims))
        self.head = nn.Sequential(*[
            ResidualBlock(sum(self.stage_dims), decoder_dim),
            ResidualBlock(decoder_dim, decoder_dim),
            ResidualBlock(decoder_dim, cfg.N_CLASSES),
        ])

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        outputs = self.backbone(
            batch["x"],
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[1:]
        feats = []
        for i, h in enumerate(hidden_states):
            B, L, C = h.shape
            H = W = int(L ** 0.5)

            x = h.transpose(1, 2).reshape(B, C, H, W)
            # x = self.proj[i](x)
            

            if i > 0:
                x = F.interpolate(
                    x,
                    (16, 16),
                    # scale_factor=2 ** min(i, 2),
                    # mode="bilinear",
                    # align_corners=False,
                )
            
            feats.append(x)
            # print("x shape", x.shape)
        x = torch.cat(feats, dim=1)
        # print("x shape", x.shape)
        # x = self.fuse(x)
        # exit(0)
        x = self.head(x)

        return {"y_pred": x}


class Mask2FormerWrapper(Mask2FormerForUniversalSegmentation):
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        outputs = super().forward(pixel_values=batch["x"])
        print(outputs)
        exit(0)

def main():
    # Set configs
    train_cfg = cfg.TrainingConfig(
        n_epochs=2000,
        batch_size=64,
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
    # for param in backbone.parameters():
    #     param.requires_grad_(False)
    model = SwinSegmentationModel(backbone, cfg.MODELS_CFGS["downscaled_swin_vit"]).to(cfg.DEVICE)
    # model = Mask2FormerWrapper.from_pretrained("hf_swin_pretrained")
    model = models.DownScalingWrapper(model).to(cfg.DEVICE)
    model.cfg = cfg.ModelConfig(
        "swin_fpn_segmentation",
        constructor_kwargs={"decoder_dim": 256},
        downscaling=2,
        up_scale_output=True,
    )
    # optim = training.mk_optimizer(model, optim_cfg)
    optim = torch.optim.AdamW(
        model.parameters(),
        # params=[
        #     {"params": model.model.backbone.parameters(), "lr": 1e-5},
        #     {"params": model.model.head.parameters(), "lr": 1e-3},
        # ],
        lr=optim_cfg.starting_lr,
        betas=(optim_cfg.beta0, optim_cfg.beta1),
    )
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