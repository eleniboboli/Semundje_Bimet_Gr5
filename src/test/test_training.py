import torch.nn as nn
import torch
from src.train import batch_gd


def test_training_runs_one_epoch(model, loaders, device):
    train_loader, val_loader, _ = loaders
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses = batch_gd(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        device=device,
    )

    assert len(train_losses) == 1
    assert len(val_losses) == 1
    assert train_losses[0] >= 0
    assert val_losses[0] >= 0
