import os
from tqdm.notebook import tqdm
from typing import Callable, Dict, Optional

import torch
import pandas as pd
import numpy as np
from torch import nn, Tensor
from torch.utils.data import DataLoader

from src.plotting import plt_pred
from src.metrics import dice_pandas


criterion_type = Callable[[Tensor, Tensor], Dict[str, Tensor]]


def train_unet(
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        n_epochs: int,
        n_classes: int,
        criterion: criterion_type,
        save_checkpoint: bool=True,
        plt_preds: bool=False,
        x_test: Optional[Tensor]=None
    ):

    if plt_preds and x_test is None:
        print("plt_preds", plt_preds)
        print("x_test", x_test)
        raise ValueError("Did not provide a value for x_test when setting plt_preds to true.")

    torch.backends.cuda.matmul.fp32_precision = 'ieee'

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model = model.to(device)

    for epoch in tqdm(range(n_epochs)):
        train_loss = train_model_for_single_epoch(
            model,
            optimizer,
            device,
            train_loader,
            criterion,
        )
        test_loss, score = evaluate_model(
            model,
            device,
            valid_loader,
            criterion,
            n_classes,
        )

        print(f"Epoch : {epoch} \t Training Loss : {train_loss / len(train_loader):.3f} \t Test Loss : {test_loss / len(valid_loader):.3f} \t Score: {score:.3f}")

        if save_checkpoint:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/checkpoint_epoch{epoch}.pth')

        if plt_preds:
            x_train, y_train = next(iter(train_loader))
            plt_pred(model, 0, x_train, y_train)
            x_valid, y_valid = next(iter(valid_loader))
            plt_pred(model, 0, x_valid, y_valid)
            plt_pred(model, 20, x_test)

def train_model_for_single_epoch(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        train_loader: DataLoader,
        criterion: criterion_type,
    ) -> float:
    model.train()
    train_loss = 0
    for (image, y_true) in train_loader:
        image = image.to(device=device)
        y_true = y_true.to(device=device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            y_pred_logits = model(image)
            losses = criterion(y_pred_logits, y_true)
            loss = losses["average_loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
    return train_loss

@torch.no_grad
def evaluate_model(
        model: nn.Module,
        device: torch.device,
        valid_loader: DataLoader,
        criterion: criterion_type,    
        n_classes: int,
    ) -> tuple[float, float]:
    model.eval()
    test_loss = 0
    predictions = []
    true_masks = []

    for (image, y_true) in valid_loader:
        true_masks.append(y_true.cpu().numpy().squeeze())
        image = image.to(device=device)
        y_true = y_true.to(device=device)

        y_pred_logits = model(image)

        losses = criterion(y_pred_logits, y_true)
        loss = losses["average_loss"]
        test_loss += loss.item()

        pred = torch.argmax(y_pred_logits, dim=1)
        predictions.append(pred.squeeze().cpu().numpy())

    predictions = pd.DataFrame(np.concat(predictions).reshape(-1 , 256 * 256))
    valid = pd.DataFrame(np.concat(true_masks).reshape((-1, 256*256)))
    score = dice_pandas(valid, predictions, n_classes)
    return test_loss, score
