from sklearn.model_selection import train_test_split

from src.models import mk_model
from src.training import train_unet
from src.metrics import SegmentationLoss
from src.dataset import get_data_loaders
from src.dataset import load_preprocessed_dataset
from src.configs import TrainingConfig, ModelConfig


def main():
    train_cfg = TrainingConfig()
    model_cfg = ModelConfig()

    x_train, y_train, x_test = load_preprocessed_dataset()
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=train_cfg.test_size,
        random_state=train_cfg.random_state,
    )
    x_test = x_test.cpu()
    criterion = SegmentationLoss(train_cfg)
    model = mk_model(train_cfg, model_cfg)
    train_loader, valid_loader = get_data_loaders(
        x_train,
        y_train,
        x_valid,
        y_valid,
        batch_size=train_cfg.batch_size,
    )    
    train_unet(
        model,
        train_cfg,
        train_loader,
        valid_loader,
        criterion,
        save_checkpoint=True,
        plt_preds=False,
        x_test=x_test,
    )

if __name__ == "__main__":
    main()