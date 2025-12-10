import os
import math
from tqdm import tqdm
from typing import Any
from pathlib import Path
from collections import defaultdict

import torch
import wandb
import numpy as np
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler, LambdaLR

from src import (
    dataset,
    metrics,
    timing,
    utils,
    models,
)
from src import configs as cfg


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_cfg: cfg.TrainingConfig,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
            wandb_cfg: cfg.WandbConfig,
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
        utils.setup_seed(train_cfg.random_state)
        # Initilaze weights and biases run
        wandb_init(
            self.model,
            wandb_cfg,
            self.model.cfg,
            train_cfg,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path):
        print("Starting training from checkpoint:", path)
        chkpt = torch.load(path, weights_only=False)
        train_cfg = cfg.TrainingConfig(**chkpt["train_cfg"])
        model_cfg = cfg.ModelConfig(**chkpt["model_cfg"])
        model = models.MAE_ViT.from_config(model_cfg, print_params_count=True)
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
            chkpt_pth_format: str,
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
                self.save_checkpoint(chkpt_pth_format)
            self.epoch += 1

    def save_checkpoint(self, chkpt_pth_format: str):
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
        pth = chkpt_pth_format.format(epoch=self.epoch)
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
                if self.train_cfg.transform:
                    x = self.train_cfg.transform(x)
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
                predicted_img, mask = self.model(x, self.train_cfg.mask_ratio)
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
        valid_eval_dict = self.evaluate_model_on_single_split(data_loaders["valid"], criterion)
        test_infer_dict  = self.evaluate_model_on_single_split(data_loaders["test"],  criterion)
        valid_infer_dict = self.evaluate_model_on_single_split(data_loaders["valid"], criterion)
        wandb_log_dict_with_prefix(valid_eval_dict, "validation", self.epoch)
        wandb_log_dict_with_prefix(test_infer_dict,  "inference_on_test",  self.epoch)
        wandb_log_dict_with_prefix(valid_infer_dict, "inference_on_valid", self.epoch)

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
                predicted_img, mask = self.model(x, self.train_cfg.mask_ratio)
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
    
def mk_lr_scheduler(train_cfg: cfg.TrainingConfig, optimizer: Optimizer) -> LRScheduler:
    def lr_func(epoch: int) -> float:
        return min(
            (epoch + 1) / (train_cfg.n_warmup_epochs // 10 + 1e-8),
            0.5 * (math.cos(epoch / train_cfg.n_epochs * math.pi) + 1)
        )
    return LambdaLR(optimizer, lr_lambda=lr_func)

def mk_optimizer(model: nn.Module, train_cfg: cfg.TrainingConfig) -> Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        # TODO: Understand the scaling of the max_lr
        lr=train_cfg.max_lr,
        betas=(0.9, 0.95),
    )
