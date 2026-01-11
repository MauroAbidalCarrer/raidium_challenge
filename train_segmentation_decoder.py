import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.transforms import v2
from transformers import SwinModel

import src.configs as cfg
from src import models, training, dataset, metrics


class SwinSegmentation(nn.Module):
    def __init__(self, backbone: SwinModel):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.decoder = nn.LazyLinear(
            cfg.N_CLASSES * 32 ** 2,
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        b_size = batch["x"].shape[0]
        outputs = self.backbone(
            batch["x"],
            output_hidden_states=True,
            return_dict=True,
        )
        # for h in outputs.hidden_states:
        #     print(h.shape)
        last_hidden_state = torch.cat(outputs.hidden_states[-2:], dim=-1)
        y_pred = (
            self.decoder(last_hidden_state) # B, 16, n_classes * 32 ** 2
            .reshape(b_size, 128, 128, cfg.N_CLASSES) # B, 128, 128, n classes
        )
        y_pred = torch.permute(y_pred, (0, 3, 1, 2))
        return {"y_pred": y_pred}


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
    model = SwinSegmentation(backbone).to(cfg.DEVICE)
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
