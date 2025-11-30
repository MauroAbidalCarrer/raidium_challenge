
from sklearn.model_selection import train_test_split

from src.plotting import plt_sample
from src.dataset import load_preprocessed_dataset


x_train, y_train, x_test = load_preprocessed_dataset()
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train,
    y_train,
    test_size=0.1,
    random_state=1,
)
x_test = x_test.cpu()
# SAMPLE_IDX = 0
import torch
from torch import nn, Tensor
from monai.networks.nets import SwinUNETR, UNet

from src.training import train_unet
from src.dataset import get_data_loaders
from src.metrics import SegmentationLoss
from src.configs import TrainingConfig, ModelConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_cfg = TrainingConfig()
model_cfg = ModelConfig()
def mk_model(train_cfg: TrainingConfig, model_cfg: ModelConfig) -> nn.Module:
    model = (
        UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=train_cfg.n_classes,
            channels=model_cfg.channels,
            strides=model_cfg.strides,
            kernel_size=model_cfg.kernel_size,
            num_res_units=model_cfg.num_res_units,
            act=model_cfg.act,
            norm=model_cfg.norm,
            dropout=model_cfg.dropout,
            bias=model_cfg.bias,
        )
        .to(device)
    )
    model.cfg = model_cfg
    return model

model = mk_model(train_cfg, model_cfg)
train_loader, valid_loader = get_data_loaders(
    x_train,
    y_train,
    x_valid,
    y_valid,
    batch_size=train_cfg.batch_size,
)
cirterion = SegmentationLoss(train_cfg)
train_unet(
    model,
    train_cfg,
    train_loader,
    valid_loader,
    cirterion,
    save_checkpoint=True,
    plt_preds=False,
    x_test=x_test,
)
