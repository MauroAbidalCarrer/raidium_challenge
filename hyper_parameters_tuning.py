import os 
from copy import deepcopy
from pytz import timezone
from datetime import datetime
from functools import partial

import torch
import wandb
import optuna
import pandas as pd
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split

import src.configs as cfg
from src import dataset, models, training, metrics


def main():
    os.environ["WANDB_SILENT"] = "true"
    base_model_cfg = cfg.MODELS_CFGS["downscaled_swin_vit"]
    base_train_cfg = cfg.TRAIN_CONFIGS["swin_pretraining"]
    base_train_cfg = deepcopy(cfg.TRAIN_CONFIGS["swin_pretraining"])
    base_train_cfg.n_epochs = 35
    print(base_model_cfg)

    data_loaders = dataset.mk_ssl_loaders(base_train_cfg)

    france_date = datetime.now(timezone('Europe/Paris'))
    hp_tune_run_grp = "hp_tuning-start_lr+betas" + france_date.strftime("%y-%m-%d:%H%M")
    print("Starting hp tuning group", hp_tune_run_grp)

    def objective(trial: optuna.trial.Trial) -> float:
        optim_cfg = deepcopy(cfg.OPTIM_CFGS["downscaled_vit_pretraining"])
        optim_cfg.beta0 = trial.suggest_float("beta_0", 0.85, 0.9999)
        optim_cfg.beta1 = trial.suggest_float("beta_1", 0.85, 0.9999)
        optim_cfg.start_lr = trial.suggest_float("start_lr", 5e-5, 1e-3)
        train_cfg = deepcopy(base_train_cfg)
        train_cfg.max_loss_norm = trial.suggest_float("max_loss_norm", 0.1, 4)
        model_cfg = deepcopy(base_model_cfg)
        model_cfg.constructor_kwargs["embed_dim"] = trial.suggest_int("embed_dim", 128, 512, step=128)
        model_cfg.constructor_kwargs["per_layer_depth"] = trial.suggest_int("per_layer_depth", 2, 4)
        model_cfg.constructor_kwargs["n_layers"] = trial.suggest_int("n_layers", 4, 8)
        model_cfg.constructor_kwargs["patch_size"] = trial.suggest_int("patch_size", 4, 32, step=2)


        
        wandb_run = training.wandb_init(
            base_model_cfg,
            train_cfg,
            optim_cfg,
            tags=cfg.WANDB_RUN_TAGS["downscaled_swin_pretraining"],
            group=hp_tune_run_grp,
        )
        model = models.mk_model_from_cfg(base_model_cfg)
        optim = training.mk_optimizer(model, optim_cfg)
        lr_scheduler = training.mk_lr_scheduler(train_cfg, optim)
        trainer = training.Trainer(
            model,
            train_cfg,
            optim,
            lr_scheduler,
            wandb_run
        )
        records = trainer.train_model(
            data_loaders,
            metrics.ssl_loss,
            chkpt_pth_format=None,
        )
        records = pd.DataFrame.from_records(records)
        min_val_loss = records["validation/rec_l2_loss"].min()
        print("min_val_loss:", min_val_loss)

        return min_val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=100,
        timeout=60 * 60 * 8,
        n_jobs=1,
    )
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    current_trial = study.best_trial
    print("  Value: ", current_trial.value)
    print("  Params: ")
    for key, value in current_trial.params.items():
        print("    {}: {}".format(key, value))

def run_to_df(run: wandb.sdk.wandb_run.Run) -> pd.DataFrame:
    api = wandb.Api()
    api_run = api.run(f"{run.entity}/{run.project}/{run.id}")
    return pd.DataFrame(api_run.scan_history())

if __name__ == "__main__":
    main()
