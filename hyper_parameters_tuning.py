from pytz import timezone
from datetime import datetime
from functools import partial

import torch
import optuna
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split

from src.models import mk_model
from src.training import train_model
from src.metrics import SegmentationLoss
from src import dataset
from src.dataset import load_preprocessed_dataset
from src.configs import TrainingConfig, ModelConfig, DatasetConfig


def main():
    model_cfg = ModelConfig()
    dataset_cfg = DatasetConfig(test_size=0.1)

    dataset.mk_dataset(verbose=False)
    x_train, y_train, x_test = load_preprocessed_dataset()

    france_date = datetime.now(timezone('Europe/Paris'))
    hp_tuning_group = "start_lr+beta1_hp_" + france_date.strftime("%y-%m-%d:%H%M")
    def objective(trial: optuna.trial.Trial, x_train, y_train) -> float:
        train_cfg = TrainingConfig(
            starting_lr=trial.suggest_float("starting_lr", low=1e-5, high=1e-2),
            n_epochs=50,
        )
        model = torch.compile(mk_model(train_cfg, model_cfg))
        criterion = SegmentationLoss(train_cfg)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train,
            y_train,
            test_size=dataset_cfg.test_size,
            random_state=train_cfg.random_state,
        )
        train_loader, valid_loader = dataset.get_data_loaders(
            x_train,
            y_train,
            x_valid,
            y_valid,
            train_cfg,
        )
        valid_dice_score = train_model(
            model,
            dataset_cfg,
            train_cfg,
            train_loader,
            valid_loader,
            criterion,
            save_checkpoint=False,
            print_time_to_run=False,
            hp_tuning_group=hp_tuning_group,
        )
        print("starting_lr:", valid_dice_score)
        return valid_dice_score

    study = optuna.create_study(direction="maximize")
    study.optimize(
        partial(objective, x_train=x_train, y_train=y_train),
        n_trials=100,
        timeout=60 * 60 * 8,
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

if __name__ == "__main__":
    main()
