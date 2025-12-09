import os
import sys
import math
from tqdm import tqdm
from pathlib import Path
from typing import Any, overload
from collections import defaultdict

import torch
import wandb
import numpy as np
from torch import nn, Tensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import Optimizer, AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler, LambdaLR

from mae.model import MAE_ViT
from src import configs as cfg
from mae.utils import setup_seed
from src.plotting import plt_sample
from src import dataset, metrics, timing


# TODO:
# - Add checkpoints save/loading
# - Optimize code (again) ... or just spend a lot of money on n H100
# - Add submission creation and submit
# - Switch to one cycle lr


def main():
    if len(sys.argv) > 1:
        trainer = Trainer.from_checkpoint(sys.argv[1])
    else:
        trainer = mk_trainer_from_scratch()
    data_loaders = mk_semi_supervised_data_loaders(trainer.train_cfg)
    criterion = SemiSupervisedLoss(trainer.train_cfg)
    trainer.train_model(data_loaders, criterion)

def mk_trainer_from_scratch() -> "Trainer":
    # setup configs
    train_cfg = cfg.TrainingConfig(
        max_lr=1e-3,
        n_epochs=5000,
        batch_size=64,
    )
    model_cfg = cfg.ModelConfig(
        n_encoder_heads=8,
        n_encoder_layers=4,
        compile=False,
    )
    # setup objects (no I didn't count the configs as objects...)
    model = mk_model(model_cfg, train_cfg)
    optimizer = mk_optimizer(model, train_cfg)
    lr_scheduler = mk_lr_scheduler(train_cfg, optimizer)
    # start training
    return Trainer(
        model,
        train_cfg,
        optimizer,
        lr_scheduler,
    )

def mk_model(model_cfg: cfg.ModelConfig, training_cfg: cfg.TrainingConfig, print_params_count=False) -> nn.Module:
    model = MAE_ViT(
        image_size=256,
        mask_ratio=training_cfg.mask_ratio,
        patch_size=16,
        emb_dim=256,
        out_channels=cfg.N_CLASSES + 1,
        decoder_head=model_cfg.n_decoder_heads,
        encoder_head=model_cfg.n_encoder_heads,
        decoder_layer=model_cfg.n_encoder_layers,
    ).to(cfg.DEVICE)
    model.cfg = model_cfg
    if print_params_count:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("number of parameters:", str(params // 1e6) + "M")
    return model

def mk_lr_scheduler(train_cfg: cfg.TrainingConfig, optimizer: Optimizer) -> LRScheduler:
    def lr_func(epoch: int) -> float:
        return min(
            (epoch + 1) / (train_cfg.n_epochs // 10 + 1e-8),
            0.5 * (math.cos(epoch / train_cfg.n_epochs * math.pi) + 1)
        )
    return LambdaLR(optimizer, lr_lambda=lr_func)

def mk_semi_supervised_data_loaders(train_cfg: cfg.TrainingConfig) -> dict[str, DataLoader]:
    x_train, y_train, x_test = dataset.load_raw_dataset(cfg.DEVICE)
    if not all(map(lambda t: t.dtype == torch.uint8, (x_train, x_test, y_train))):
        raise ValueError("Not all raw tensors are of dtype uint8.")
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=train_cfg.test_size,
    )
    x_train = torch.cat((x_train, x_test))
    y_test_zeros = torch.zeros(
        x_test.shape[0], 256, 256,
        dtype=torch.uint8,
        device=y_train.device,
    )

    y_train = torch.cat((y_train, y_test_zeros))
    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_valid, y_valid)
    test_dataset  = TensorDataset(x_test, y_test_zeros)
    train_loader = DataLoader(train_dataset, train_cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, train_cfg.batch_size)
    test_loader = DataLoader(test_dataset, train_cfg.batch_size)
    return {
        "train": train_loader,
        "valid": valid_loader,
        "test":  test_loader,
    }

def mk_optimizer(model: nn.Module, train_cfg: cfg.TrainingConfig) -> Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        # TODO: Understand the scaling of the max_lr
        lr=train_cfg.max_lr,
        betas=(0.9, 0.95),
    )


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_cfg: cfg.TrainingConfig,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        ):
        if isinstance(model, nn.Module):
            self.model = model
            self.train_cfg = train_cfg
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
            self.epoch = 0
            self.training_samples_seen = 0
        
        dataset.mk_dataset(verbose=False)
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        setup_seed(train_cfg.random_state)
        # Initilaze weights and biases run
        wandb_init(
            self.model,
            cfg.WandbConfig(["pretraining", "MAE"], "pretraining"),
            self.model.cfg,
            train_cfg,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path):
        print("Starting training from checkpoint:", path)
        chkpt = torch.load(path, weights_only=False)
        train_cfg = cfg.TrainingConfig(**chkpt["train_cfg"])
        model_cfg = cfg.ModelConfig(**chkpt["model_cfg"])
        model = mk_model(model_cfg, train_cfg, True)
        optimizer = mk_optimizer(model, train_cfg)
        lr_sched = mk_lr_scheduler(train_cfg, optimizer)
        trainer = cls(model, train_cfg, optimizer, lr_sched)
        trainer.epoch = chkpt["epoch"]
        trainer.training_samples_seen = chkpt["training_samples_seen"]
        return trainer

    def train_model(
            self,
            data_loaders: dict[str, DataLoader],
            criterion: cfg.criterion_t,
        ) -> dict[str, Tensor]:
        self.training_samples_seen = 0
        for _ in tqdm(range(self.epoch, self.train_cfg.n_epochs)):
            is_last_epoch = self.epoch == self.train_cfg.n_epochs - 1
            if self.epoch % 100 == 0 or is_last_epoch:
                with timing.time_to_run("evaluation/total"):
                    self.evaluate_model(data_loaders, criterion=criterion)
            with timing.time_to_run("training/total"):
                training_dict = self.train_model_for_single_epoch(data_loaders["train"], criterion)
            wandb_log_dict_with_prefix(training_dict, "training", self.epoch)
            if self.epoch % 10 == 0 or is_last_epoch:
                timing.print_time_dict()
            if (self.epoch % 100 == 0 and self.epoch != 0) or is_last_epoch:
                self.save_checkpoint()
            self.epoch += 1

    def save_checkpoint(self):
        """Saves checkpoint on wandb and on the machine."""
        chkpt_dict = {
            "model": self.model.state_dict(),
            "model_cfg": vars(self.model.cfg),
            "train_cfg": vars(self.train_cfg),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epoch": self.epoch,
            "training_samples_seen": self.training_samples_seen
        }
        os.makedirs("checkpoints/", exist_ok=True)
        pth = f"checkpoints/mae_vit_{self.epoch}_chkpt.pt"
        torch.save(chkpt_dict, pth)
        print("Saved checpoint at", pth)

    def train_model_for_single_epoch(
            self,
            train_loader: torch.utils.data.DataLoader,
            criterion: cfg.criterion_t,
        ) -> dict[str, float]:
        self.model.train()
        n_batches = len(train_loader)
        mk_epoch_buff = lambda : torch.empty(n_batches, device=cfg.DEVICE)
        epochs_steps_dicts: dict[str, Tensor] = defaultdict(mk_epoch_buff)
        batch_it = iter(train_loader)
        for batch_i in range(len(train_loader)):
            with timing.time_to_run("training/get_batch and preprocess"):
                x, y_true = next(batch_it)
                x = dataset.preprocess_imgs(x)
            with timing.time_to_run("training/step"):
                step_dict = self.perform_training_step(x, y_true, criterion)
            with timing.time_to_run("training/epoch_dict_store_values"):
                for k, v in step_dict.items():
                    epochs_steps_dicts[k][batch_i] = v.detach()
        with timing.time_to_run("training/mk_ epoch_dict"):
            epoch_dict = {k: v.mean().item() for k, v in epochs_steps_dicts.items()}
            epoch_dict["training_samples_seen"] = self.training_samples_seen
            epoch_dict["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        # TODO: Check if this shouldn't get called per step instead of per epoch.
        self.lr_scheduler.step()
        return epoch_dict

    def perform_training_step(self, x: Tensor, y_true: Tensor, criterion: cfg.criterion_t) -> dict[str, Tensor]:
        self.optimizer.zero_grad()
        with torch.autocast(cfg.DEVICE.type, torch.bfloat16):
            with timing.time_to_run("training/forward"):
                predicted_img, mask = self.model(x)
            with timing.time_to_run("training/loss"):
                loss_dict = criterion(x, predicted_img, mask, y_true)
        with timing.time_to_run("training/backprop"):
            loss_dict["loss"].backward()
        with timing.time_to_run("training/clip_grad_norm_"):
            loss_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                1.0, # TODO: Hypertune this
            )
        with timing.time_to_run("training/optimizer step"):
            self.optimizer.step()
        self.training_samples_seen += len(x)
        return {**loss_dict, "loss_norm": loss_norm}

    @torch.no_grad
    def evaluate_model(self, data_loaders: dict[str, DataLoader], criterion: cfg.criterion_t):
        """
        Plots the reconstruction and segmentation of masked batches from all splits.
        Evaluates the reconstruction and seg losses of validation split in eval mode.
        Evaluates the seg losses of validation split in inference mode.
        Evaluates the recon of test split in eval mode.
        """
        self.model = self.model.eval()
        if self.epoch != 0:
            for split_name, data_loader in data_loaders.items():
                print(f"Training visualization on {split_name} batch:")
            # plt_model_preds(model, data_loader)
        valid_eval_dict = self.evaluate_model_on_single_split(data_loaders["valid"], criterion)
        wandb_log_dict_with_prefix(valid_eval_dict, "validation", self.epoch)
        test_infer_dict  = self.evaluate_model_on_single_split(data_loaders["test"],  criterion)
        wandb_log_dict_with_prefix(test_infer_dict,  "inference_on_test",  self.epoch)
        with torch.autograd.grad_mode.inference_mode(True):
            valid_infer_dict = self.evaluate_model_on_single_split(data_loaders["valid"], criterion)
            wandb_log_dict_with_prefix(valid_infer_dict, "inference_on_valid", self.epoch)
            # if self.epoch != 0:
            #     for split_name, data_loader in data_loaders.items():
            #         print(f"Inference visualization on {split_name} batch:")
            #         plt_model_preds(model, data_loader, plt_recon=False)

    @torch.no_grad
    def evaluate_model_on_single_split(
            self,
            data_loader: DataLoader,
            criterion: cfg.criterion_t,
        ) -> dict[str, Any]:
        self.model = self.model.eval()
        n_batches = len(data_loader)
        mk_epoch_buff = lambda : torch.empty(n_batches, device=cfg.DEVICE)
        eval_dict: dict[str, Tensor] = defaultdict(mk_epoch_buff)
        seg_preds = []
        seg_y_true = []
        for batch_i, (x, y_true) in enumerate(data_loader):
            x = dataset.preprocess_imgs(x)
            with torch.autocast(cfg.DEVICE.type, torch.bfloat16):
                predicted_img, mask = self.model(x)
                loss_dict = criterion(x, predicted_img, mask, y_true)
            loss_dict["loss_norm"] = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                1.0, # TODO: Hypertune this
            )
            for k, v in loss_dict.items():
                eval_dict[k][batch_i] = v.detach()
            y_pred_logits = predicted_img[:, 1:]
            pred = torch.argmax(y_pred_logits, dim=1)
            seg_y_true.append(y_true.squeeze().cpu().numpy())
            seg_preds.append(pred.squeeze().cpu().numpy())
        with timing.time_to_run("evaluation/mk_eval_dict"):
            eval_dict =  {k: v.mean().item() for k, v in eval_dict.items()}
            eval_dict["training_samples_seen"] = self.training_samples_seen
        with timing.time_to_run("evaluation/dice_score"):
            seg_preds = np.concat(seg_preds).reshape(-1 , 256 * 256)
            seg_y_true = np.concat(seg_y_true).reshape(-1, 256 * 256)
            eval_dict["dice_score"] = metrics.dice_pandas(seg_y_true, seg_preds)
        return eval_dict

def wandb_init(
        model: nn.Module,
        wandb_cfg: cfg.WandbConfig,
        *configs: list[Any],
    ):
    cfg_vars = {}
    for cfg in configs:
        cfg_vars |= vars(cfg)
    wandb.init(
        project="raidium-challenge",
        config={
            **cfg_vars,
            "model_class": type(model).__class__,
        },
        **vars(wandb_cfg),
    )

def wandb_log_dict_with_prefix(data: dict[str, Any], prefix: str, step: int):
    wandb.log(
        data={prefix + "/" + k: v for k, v in data.items()},
        step=step,
    )

class SemiSupervisedLoss:
    """
    Computes a weighted average of mse loss of the images reconstruction, dice score and cross entropy.
    """
    def __init__(self, train_cfg: cfg.TrainingConfig):
        weight = metrics.get_class_weights().to(cfg.DEVICE)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=weight)
        self.train_cfg = train_cfg

    def __call__(
            self,
            x: Tensor,
            x_hat: Tensor,
            mask: Tensor,
            y_true: Tensor,
        ) -> dict[str, Tensor]:
        # reconstruction loss
        pix_wise_rec_loss = F.mse_loss(x_hat[:, 0], x[:, 0], reduction="none")
        pix_wise_rec_loss = pix_wise_rec_loss * mask[:, 0] / self.train_cfg.mask_ratio
        rec_loss = pix_wise_rec_loss.mean()
        # # Create labeled samples mask to compute seg loss only on them
        # seg_loss_mask = (y_true.flatten(1) != 0 ).any(dim=1)
        # # Cross entropy loss
        # y_pred = x_hat[:, 1:]
        # y_true = y_true.long()
        # ce_loss = self.cross_entropy_loss(
        #     y_pred[seg_loss_mask],
        #     y_true[seg_loss_mask],
        # )
        # # Dice loss
        # base_d_loss = metrics.torch_dice_loss(
        #     y_pred[seg_loss_mask],
        #     y_true[seg_loss_mask],
        # )
        # Weighted average
        # loss = base_d_loss * self.train_cfg.dice_loss_weight          \
        #      + ce_loss     * self.train_cfg.cross_entropy_loss_weight 
            #  + rec_loss    * self.train_cfg.rec_loss_weight
        return {
            "loss": rec_loss,
            # "cross_entropy_loss": ce_loss,
            # "dice_loss": base_d_loss,
            "rec_loss": rec_loss,
        }

N_IMGS_TO_PLT = 5
@torch.no_grad
def plt_model_preds(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        plt_recon: bool=True,
    ):
    """Plots the model's reconstruction and segmentation."""
    model.eval()
    x, y_true = next(iter(data_loader))
    sample_has_seg = y_true.flatten(1).__ne__(0).any(dim=1)
    print("n samples with seg:", sample_has_seg.sum(), sample_has_seg.shape)
    if sample_has_seg.any():
        x = x[sample_has_seg]
        y_true = y_true[sample_has_seg]
    x = dataset.preprocess_imgs(x)
    x = x[:N_IMGS_TO_PLT]
    x_hat, mask = model(x)
    x_hat_img = x_hat[:, :1]
    mask = mask[:, :1]
    x_hat_img = x_hat_img * mask + x * (1 - mask)
    plt_seg_imgs(x, y_true, x_hat[:, 1:], mask)
    if plt_recon:
        plt_recon_imgs(x, x_hat_img, mask)

def plt_recon_imgs(
    x: Tensor,
    x_hat_img: Tensor,
    mask: Tensor,
    n_cols: int = 8,
):
    """
    Matplotlib version of the reconstruction visualization.
    Shows reconstructed images and originals in one grid.
    """
    # Combine images along batch dimension
    img = torch.cat(
        [
            # x * (1 - mask),   # optional: masked input
            x_hat_img,
            x,
        ],
        dim=0,
    )
    # unnormalize
    img = img * dataset.STD + dataset.MEAN  
    # B, C, H, W → B, H, W
    np_imgs = img.detach().cpu().numpy().squeeze()
    n_imgs = np_imgs.shape[0]
    n_rows = (n_imgs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        2, N_IMGS_TO_PLT,
        figsize=(n_cols * 2, n_rows * 2),
        squeeze=False
    )
    for i, ax in enumerate(axes.flat):
        if i < n_imgs:
            ax.imshow(np_imgs[i], cmap="rainbow")
            ax.set_title(f"{i}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def plt_seg_imgs(x: Tensor, y_true: Tensor, y_pred: Tensor, mask: Tensor):
    x = x * (1 - mask)
    for i in range(min(N_IMGS_TO_PLT, len(x))):

        plt_sample(
            x[i].squeeze().detach().cpu().numpy(),
            y_pred[i].squeeze().detach().cpu().argmax(dim=0).numpy(),
            y_true[i].squeeze().detach().cpu().numpy(),
        )

if __name__ == "__main__":
    main()