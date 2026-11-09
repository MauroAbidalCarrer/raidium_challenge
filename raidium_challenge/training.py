import os
import warnings
from typing import Any, Optional
from collections import defaultdict
from dataclasses import is_dataclass

import torch
import wandb
import numpy as np
import pandas as pd
from torch import nn, Tensor
from monai import metrics as monai_metrics
from rich.progress import track
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from torch.optim.lr_scheduler import (
    LRScheduler,
    ConstantLR
)

from raidium_challenge import (
    dataset,
    metrics,
    timing,
    utils,
    models,
)
from raidium_challenge import config_classes as cfg


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_cfg: cfg.TrainingConfig,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
            wandb_run: wandb.Run,
        ):
        self.model = model
        self.cfg = train_cfg
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = 0
        self.epoch = 0
        self.training_samples_seen = 0
        self.wandb_run = wandb_run
        dataset.mk_dataset(verbose=False)
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        utils.setup_seed(train_cfg.random_state)
        self.records: list[dict[str, Any]] = []
        self.last_target_metric_val = None
        self.best_target_metric_val = None

    def train_model(
            self,
            data_loaders: dict[str, DataLoader],
            criterion: cfg.criterion_t,
        ) -> list[dict[str, Any]]:
        warnings.filterwarnings(
            "ignore",
            message="RandomErasing.*tv_tensors.Mask",
            category=UserWarning,
        )
        epoch_it = track(
            range(self.epoch, self.cfg.n_epochs),
            description="Training model",
        )
        with timing.time_to_run("evaluation/total"):
            self.evaluate_model(data_loaders, criterion=criterion)
        for _ in epoch_it:
            is_last_epoch = self.epoch == self.cfg.n_epochs - 1
            with timing.time_to_run("training/total"):
                training_dict = self.train_model_for_single_epoch(data_loaders["train"], criterion)
                self.wandb_log_dict_with_prefix(training_dict, "training")
            if is_last_epoch:
                with timing.time_to_run("evaluation/total"):
                    self.evaluate_model(data_loaders, criterion=criterion)
            self.save_checkpoint(os.path.join(self.cfg.ckpt_dir, "last_checkpoint.pt"))
            if self.target_objective_improved():
                print(f"Epoch {self.epoch}, has best new {self.cfg.target_metric} increased from {self.best_target_metric_val} to {self.last_target_metric_val}.")
                self.save_checkpoint(os.path.join(self.cfg.ckpt_dir, "best_checkpoint.pt"))
                self.best_target_metric_val = self.last_target_metric_val
        return self.records

    def target_objective_improved(self) -> bool:
        if self.last_target_metric_val is None:
            raise ValueError("last_target_metric_val is was not logged, please make sure that it's present in the wandb logged metrics.")
        return (
            self.best_target_metric_val is None
            or
            self.cfg.target_improved_fn(
                self.last_target_metric_val,
                self.best_target_metric_val,
            )
        )

    def train_model_for_single_epoch(
            self,
            train_loader: torch.utils.data.DataLoader,
            criterion: cfg.criterion_t,
        ) -> dict[str, float]:
        self.model.train()
        n_batches = len(train_loader)
        mk_epoch_buff = lambda : torch.empty(n_batches, device=cfg.DEVICE)
        epochs_steps_dicts: dict[str, Tensor] = defaultdict(mk_epoch_buff)
        for batch_i, batch_dict in enumerate(train_loader):
            if isinstance(batch_dict, list):
                batch_dict = {"x": batch_dict[0], "y_true": batch_dict[1]}
            batch_dict = dataset.preprocess_batch(batch_dict)
            with timing.time_to_run("training/step"):
                step_dict = self.perform_training_step(batch_dict, criterion)
            for k, v in step_dict.items():
                epochs_steps_dicts[k][batch_i] = v.detach()
        epoch_dict = {k: v.mean().item() for k, v in epochs_steps_dicts.items()}
        epoch_dict["training_samples_seen"] = self.training_samples_seen
        epoch_dict["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        self.epoch += 1
        return epoch_dict

    def perform_training_step(
            self,
            batch_dict: dict[str, Any],
            criterion: cfg.criterion_t,
        ) -> dict[str, Tensor]:
        if self.cfg.transform:
            batch_dict = self.cfg.transform(batch_dict)
        self.optimizer.zero_grad()
        with torch.autocast(cfg.DEVICE.type, torch.bfloat16):
            with timing.time_to_run("training/forward"):
                model_output_dict = self.model(batch_dict)
            with timing.time_to_run("training/loss"):
                loss_dict = criterion(batch_dict | model_output_dict)
        with timing.time_to_run("training/backprop"):
            loss_dict["loss"].backward()
        loss_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg.max_loss_norm,
        )
        self.optimizer.step()
        self.training_samples_seen += len(batch_dict["x"])
        self.lr_scheduler.step()
        self.step += 1
        return {**loss_dict, "loss_norm": loss_norm}

    @torch.no_grad
    def evaluate_model(self, data_loaders: dict[str, DataLoader], criterion: cfg.criterion_t):
        self.model = self.model.eval()
        valid_eval_dict = self.evaluate_model_on_single_split(data_loaders["valid"], criterion)
        self.wandb_log_dict_with_prefix(valid_eval_dict, "validation")

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
        for batch_i, batch in enumerate(data_loader):
            if isinstance(batch, list):
                batch = {"x": batch[0], "y_true": batch[1]}
            batch = dataset.preprocess_batch(batch)
            with torch.autocast(cfg.DEVICE.type, torch.bfloat16):
                model_output_dict = self.model(batch)
                loss_dict = criterion(batch | model_output_dict)
            loss_dict["loss_norm"] = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                1.0, # TODO: Hypertune this
            )
            for k, v in loss_dict.items():
                eval_dict[k][batch_i] = v.detach()
            if "y_pred" in model_output_dict:
                pred = torch.argmax(model_output_dict["y_pred"], dim=1)
                seg_y_true.append(batch["y_true"].squeeze().cpu().numpy())
                seg_preds.append(pred.squeeze().cpu().numpy())
        eval_dict = {k: v.mean().item() for k, v in eval_dict.items()}
        eval_dict["training_samples_seen"] = self.training_samples_seen
        if len(data_loader) and "y_pred" in model_output_dict:
            with timing.time_to_run("evaluation/dice_score"):
                seg_preds = np.concat(seg_preds).reshape(-1 , 256 * 256)
                seg_y_true = np.concat(seg_y_true).reshape(-1, 256 * 256)
                eval_dict["dice_score"] = metrics.dice_pandas(seg_y_true, seg_preds)
        return eval_dict

    def wandb_log_dict_with_prefix(self, data: dict[str, Any], prefix: str):
        data_with_prefix = {prefix + "/" + k: v for k, v in data.items()}
        trainer_data = {
            "training/epoch": self.epoch,
            "training/step": self.step,
            "training/traing_samples_seen": self.training_samples_seen,
            "training/samples_seen": self.training_samples_seen, # for backward compatibility with legacy code
        }
        new_record = data_with_prefix | trainer_data
        wandb.log(data=new_record, step=self.step)
        self.records.append(new_record)
        if self.cfg.target_metric in new_record:
            self.last_target_metric_val = new_record[self.cfg.target_metric]

    def save_checkpoint(self, pth: str):
        chkpt_dict = {
            "model": self.model.state_dict(),
            "model_cfg": vars(self.model.cfg),
            "train_cfg": vars(self.cfg),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "lr_scheduler_cfg": getattr(self.lr_scheduler, "cfg", None),
            "optimizer_cfg": getattr(self.optimizer.state_dict(), "cfg", None),
            "step": self.step,
            "epoch": self.epoch,
            "training_samples_seen": self.training_samples_seen,
        }
        dir_path = os.path.dirname(pth)
        if dir_path:  # handle case where pth has no directory component
            os.makedirs(dir_path, exist_ok=True)
        # Save checkpoint
        torch.save(chkpt_dict, pth)
        # Very ugly code below...
        model = self.model
        if isinstance(model, models.DownScalingWrapper):
            model = model.model
        if issubclass(type(model), PreTrainedModel):
            pretrained_pth = pth[:-3]
            model.save_pretrained(pretrained_pth) # remove .pt
            print("Saved checkpoint to", pth, "and as pretrained model at", pretrained_pth)
        else:
            print("Saved checkpoint to", pth)


def wandb_init(
        *configs: list[Any],
        tags: Optional[list[str]]=[],
        group: Optional[str]=None
    ) -> wandb.Run:
    cfgs_vars = {}
    for cfg in configs:
        cfg_vars = vars(cfg)
        for k, v in cfg_vars.items(): # Should be recursive
            if is_dataclass(v):
                cfg_vars[k] = vars(v)
        cfgs_vars |= cfg_vars
    return wandb.init(
        project="raidium-challenge",
        config={**cfgs_vars,},
        tags=tags,
        group=group,
    )

def mk_lr_scheduler(train_cfg: cfg.TrainingConfig, optimizer: Optimizer) -> LRScheduler:
    return ConstantLR(optimizer, factor=1)

def mk_optimizer(model: nn.Module, optimizer_cfg: cfg.OptimizationConfig) -> Optimizer:
    optim = torch.optim.AdamW(
        model.parameters(),
        # TODO: Understand the scaling of the max_lr
        lr=optimizer_cfg.starting_lr,
        betas=(optimizer_cfg.beta0, optimizer_cfg.beta1),
    )
    optim.cfg = optimizer_cfg
    return optim
