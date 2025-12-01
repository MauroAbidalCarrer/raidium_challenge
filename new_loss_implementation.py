import torch
import monai
import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt
from torch.nn.functional import one_hot
from sklearn.model_selection import train_test_split

from src.models import mk_model
from src.dataset import load_preprocessed_dataset
from src.configs import TrainingConfig, ModelConfig


x_train, y_train, x_test = load_preprocessed_dataset()

x_train_split, x_valid_split, y_train_split, y_valid_split = train_test_split(
    x_train,
    y_train,
    test_size=0.3,
    # TODO stratify
    random_state=0
)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_train_split:", x_train_split.shape)
print("x_valid_split:", x_valid_split.shape)
print("y_train_split:", y_train_split.shape)
print("y_valid_split:", y_valid_split.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_cfg, model_cfg = TrainingConfig(), ModelConfig()


# NUM_CLASSES = 55
# INCLUDE_BACKGROUND = True
# criterion = monai.losses.DiceLoss(
#     include_background=INCLUDE_BACKGROUND,
#     to_onehot_y=True,
#     softmax=True,
# )
# metric = monai.metrics.DiceMetric(
#     include_background=INCLUDE_BACKGROUND,
#     ignore_empty=True,
#     num_classes=NUM_CLASSES,
# )

# critertion_t = callable[tuple[Tensor, Tensor], Tensor]
# def masked_ce(y_pred: Tensor, y_true: Tensor) -> Tensor:
#     y_true_one_hot


# def get_random_samples_idx(n_samples: int, max_idx: int) -> np.ndarray:
#     return np.random.choice(
#         np.arange(max_idx),
#         size=min(n_samples, max_idx),
#         replace=False,
#     )

# def test_dice_loss_and_metric(b_size: int):
#     x = x_train_split[:b_size].to(device)
#     y_true = y_train_split[:b_size].to(device)
#     model_device = next(model.parameters()).device
#     with torch.no_grad():
#         y_pred = model(x.to(model_device))
#     y_true_one_hot = one_hot_y_true(y_true)
#     y_true_with_channel = y_true.unsqueeze(1)
#     print("x:", x.shape)
#     print("y_pred:", y_pred.shape)
#     print("y_true:", y_true.shape)
#     print("y_true_one_hot:", y_true_one_hot.shape)
#     print("y_true == y_true_one_hot.argmax", (y_true == y_true_one_hot.argmax(dim=1)).all().item())
#     # pred_loss = criterion(y_pred, y_true_with_channel).item()
#     # print("pred loss:", pred_loss)
#     true_loss = criterion(y_true_one_hot, y_true_with_channel) #.item()
#     print("true loss:", true_loss)
#     print("true loss:", true_loss.shape)
#     # print("loss diff:", pred_loss - true_loss)

def one_hot_y_true(y_true: Tensor) -> Tensor:
    return (
        one_hot(
            y_true.type(torch.long),
            num_classes=55,
        )
        .permute(0, 3, 1, 2)
        .type(torch.float32)
    )



BATCH_SIZES_TO_TEST = [int(2 ** i) for i in range(4)]
# N_SAMPLES_TO_TRAIN_ON = 1000
EPOCHS = 3

from src.dataset import get_data_loaders

model = mk_model(train_cfg, model_cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters())

for batch_size in BATCH_SIZES_TO_TEST:
    print("batch size:", batch_size)
    train_loader, _ = get_data_loaders(
        x_train_split,
        y_train_split,
        x_valid_split,
        y_valid_split,
        batch_size,
    )
    # test_loss_and_metric(b_size, criterion, metric)
    # x = x_train_split[:b_size].to(device)
    # y_true = y_train_split[:b_size].to(device)
    model_device = next(model.parameters()).device
    # n_steps = N_SAMPLES_TO_TRAIN_ON // b_size
    for _ in range(EPOCHS):
        for x, y_true in train_loader:
            # y_true_one_hot = one_hot_y_true(y_true)
            # true_loss = torch.nn.functional.cross_entropy(
            #     y_true_one_hot,
            #     y_true.type(torch.long),
            # ) #.item()
            optimizer.zero_grad()
            y_pred = model(x.to(model_device))
            loss = my_awesome_loss(y_pred, y_true.to(model_device))
            print("loss:", loss.item())
            loss.backward()
            optimizer.step()
            # print("loss diff:", pred_loss - true_loss)
    print("=============")
    print()