import os
import warnings
from itertools import product
from typing import Any, Optional
from collections import defaultdict

import torch
import wandb
import numpy as np
import pandas as pd
from torch import nn, Tensor
from monai import metrics as monai_metrics
from rich.progress import track
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    LRScheduler,
    ConstantLR
)

from src import (
    dataset,
    metrics,
    timing,
    utils,
)
from src import configs as cfg


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
        self.confuse_mat_metric = monai_metrics.ConfusionMatrixMetric(
            metric_name=cfg.CONFUSE_MAT_METRICS_NAMES,
            reduction="mean_batch",
        )

    def train_model(
            self,
            data_loaders: dict[str, DataLoader],
            criterion: cfg.criterion_t,
            chkpt_pth_format: str,
        ) -> dict[str, Tensor]:
        warnings.filterwarnings(
            "ignore",
            message="RandomErasing.*tv_tensors.Mask",
            category=UserWarning,
        )
        epoch_it = track(
            range(self.epoch, self.cfg.n_epochs),
            description="Training model",
        )
        for _ in epoch_it:
            is_last_epoch = self.epoch == self.cfg.n_epochs - 1
            if self.epoch % self.cfg.eval_interval == 0 or is_last_epoch:
                with timing.time_to_run("evaluation/total"):
                    self.evaluate_model(data_loaders, criterion=criterion)
            with timing.time_to_run("training/total"):
                training_dict = self.train_model_for_single_epoch(data_loaders["train"], criterion)
                self.wandb_log_dict_with_prefix(training_dict, "training")
            if (self.epoch % self.cfg.checkpointing_interval == 0 and self.epoch != 0) or is_last_epoch:
                self.save_checkpoint(chkpt_pth_format)

    def save_checkpoint(self, chkpt_pth_format: str):
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
        pth = chkpt_pth_format.format(epoch=self.epoch, wandb_run_name=self.wandb_run.name)
        dir_path = os.path.dirname(pth)
        if dir_path:  # handle case where pth has no directory component
            os.makedirs(dir_path, exist_ok=True)
        # Save checkpoint
        torch.save(chkpt_dict, pth)
        # create wandb artifact
        artifact_name = f"checkpoint-epoch-{self.epoch}"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
        )
        # add file to artifact
        artifact.add_file(pth)
        # log artifact to run
        self.wandb_run.log_artifact(artifact, artifact_name)
        print("Uploaded checkpoint as artifact", artifact_name, "and to", pth)

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
            batch_dict = dataset.preprocess_batch(batch_dict)
            with timing.time_to_run("training/step"):
                step_dict = self.perform_training_step(batch_dict, criterion)
            for k, v in step_dict.items():
                epochs_steps_dicts[k][batch_i] = v.detach()
        epoch_dict = {k: v.mean().item() for k, v in epochs_steps_dicts.items()}
        epoch_dict["training_samples_seen"] = self.training_samples_seen
        epoch_dict["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        self.epoch += 1
        self.confuse_mat_metric.reset()
        #confuse_mat_metric_agg = self.confuse_mat_metric.aggregate()
        #epoch_dict["confuse_mat_metric"] = confuse_mat_metric_agg
        return epoch_dict

    def perform_training_step(
            self,
            batch_dict: dict[str, Any],
            criterion: cfg.criterion_t,
        ) -> dict[str, Tensor]:
        if self.cfg.transform:
#            batch_dict["x"], batch_dict["y_true"] = self.cfg.transform(
#                batch_dict["x"],
#                batch_dict["y_true"],
#            )
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
        self.confuse_mat_metric_step(batch_dict, model_output_dict)
        self.optimizer.step()
        self.training_samples_seen += len(batch_dict["x"])
        self.lr_scheduler.step()
        self.step += 1
        return {**loss_dict, "loss_norm": loss_norm}

    def confuse_mat_metric_step(self, batch: dict[str, Tensor], model_output: dict[str, Tensor]):
        if "y_pred" in model_output and "y_true" in batch:
            self.confuse_mat_metric(
                torch.nn.functional.one_hot(model_output["y_pred"].argmax(dim=1), cfg.N_CLASSES).permute(0, 3, 1, 2),
                torch.nn.functional.one_hot(batch["y_true"], cfg.N_CLASSES).permute(0, 3, 1, 2),
            )

    @torch.no_grad
    def evaluate_model(self, data_loaders: dict[str, DataLoader], criterion: cfg.criterion_t):
        """
        Evaluates the reconstruction and seg losses of validation split in eval mode.
        Evaluates the seg losses of validation split in inference mode.
        Evaluates the recon of test split in eval mode.
        """
        self.model = self.model.eval()
        valid_eval_dict = self.evaluate_model_on_single_split(data_loaders["valid"], criterion)
        # valid_infer_dict = self.evaluate_model_on_single_split(data_loaders["valid"], criterion)
        # with torch.inference_mode():
        #     test_infer_dict  = self.evaluate_model_on_single_split(data_loaders["test"],  criterion)
        self.wandb_log_dict_with_prefix(valid_eval_dict, "validation")
        # self.wandb_log_dict_with_prefix(test_infer_dict,  "inference_on_test")
        # self.wandb_log_dict_with_prefix(valid_infer_dict, "inference_on_valid")

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
        for batch_i, batch_dict in enumerate(data_loader):
            batch_dict = dataset.preprocess_batch(batch_dict)
            with torch.autocast(cfg.DEVICE.type, torch.bfloat16):
                model_output_dict = self.model(batch_dict)
                loss_dict = criterion(batch_dict | model_output_dict)
            loss_dict["loss_norm"] = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                1.0, # TODO: Hypertune this
            )
            for k, v in loss_dict.items():
                eval_dict[k][batch_i] = v.detach()
            if "y_pred" in model_output_dict:
                pred = torch.argmax(model_output_dict["y_pred"], dim=1)
                seg_y_true.append(batch_dict["y_true"].squeeze().cpu().numpy())
                seg_preds.append(pred.squeeze().cpu().numpy())
                self.confuse_mat_metric_step(batch_dict, model_output_dict)
        eval_dict = {k: v.mean().item() for k, v in eval_dict.items()}
        eval_dict["training_samples_seen"] = self.training_samples_seen
        if len(data_loader) and "y_pred" in model_output_dict:
            eval_dict["confuse_mat_metric"] = self.confuse_mat_metric.aggregate()
            with timing.time_to_run("evaluation/dice_score"):
                seg_preds = np.concat(seg_preds).reshape(-1 , 256 * 256)
                seg_y_true = np.concat(seg_y_true).reshape(-1, 256 * 256)
                eval_dict["dice_score"] = metrics.dice_pandas(seg_y_true, seg_preds)
        self.confuse_mat_metric.reset()
        return eval_dict

    def wandb_log_dict_with_prefix(self, data: dict[str, Any], prefix: str):
        if "confuse_mat_metric" in data:
            confuse_mat_metrics: list[Tensor] = data["confuse_mat_metric"]
            del data["confuse_mat_metric"]
        #     confuse_mat_metrics = (
        #         torch.stack(confuse_mat_metrics)
        #         .cpu()
        #         .numpy()
        #         .tolist()
        #     )
        #     cls_indices = list(range(cfg.N_CLASSES))
        #     for metric_name, metric_values in zip(cfg.CONFUSE_MAT_METRICS_NAMES, confuse_mat_metrics):
        #         confuse_mat_metric_table = wandb.Table(
        #             data=list(zip(cls_indices, metric_values)),
        #             columns=["cls_idx", f"{metric_name}_value"],
        #         )
        #         data[metric_name] = wandb.plot.bar(
        #             confuse_mat_metric_table,
        #             value="cls_idx",
        #             label=f"{metric_name}_value",
        #             title=metric_name,
        #         )
        data_with_prefix = {prefix + "/" + k: v for k, v in data.items()}
        trainer_data = {
            "training/epoch": self.epoch,
            "training/step": self.step,
            "training/traing_samples_seen": self.training_samples_seen,
            # for backward compatibility with legacy code
            "training/samples_seen": self.training_samples_seen,
        }
        wandb.log(
            data=data_with_prefix | trainer_data,
            step=self.step,
        )


def wandb_init(
        *configs: list[Any],
        tags: Optional[list[str]]=[],
        group: Optional[str]=None
    ) -> wandb.Run:
    cfg_vars = {}
    for cfg in configs:
        cfg_vars |= vars(cfg)
    return wandb.init(
        project="raidium-challenge",
        config={**cfg_vars,},
        tags=tags,
        group=group,
    )

def mk_lr_scheduler(train_cfg: cfg.TrainingConfig, optimizer: Optimizer) -> LRScheduler:
    return ConstantLR(optimizer, factor=1)

def mk_optimizer(model: nn.Module, optimizer_cfg: cfg.OptimizerConfig) -> Optimizer:
    optim = torch.optim.AdamW(
        model.parameters(),
        # TODO: Understand the scaling of the max_lr
        lr=optimizer_cfg.starting_lr,
        betas=(optimizer_cfg.beta0, optimizer_cfg.beta1),
    )
    optim.cfg = optimizer_cfg
    return optim
