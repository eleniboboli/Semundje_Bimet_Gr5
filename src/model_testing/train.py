import numpy as np
import torch
from datetime import datetime


def batch_gd(model, criterion, optimizer, train_loader, val_loader, epochs: int, device: str = "cpu"):
   def batch_gd(model, criterion, train_loader, test_laoder, epochs):
    train_losses = np.zeros(epochs)
    validation_losses = np.zeros(epochs)

    for e in range(epochs):
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            output = model(inputs)

            loss = criterion(output, targets)

            train_loss.append(loss.item())  # torch to numpy world

            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_loss)

        validation_loss = []

        for inputs, targets in validation_loader:

            inputs, targets = inputs.to(device), targets.to(device)

            output = model(inputs)

            loss = criterion(output, targets)

            validation_loss.append(loss.item())  # torch to numpy world

        validation_loss = np.mean(validation_loss)

        train_losses[e] = train_loss
        validation_losses[e] = validation_loss

        dt = datetime.now() - t0

        print(
            f"Epoch : {e+1}/{epochs} Train_loss:{train_loss:.3f} Test_loss:{validation_loss:.3f} Duration:{dt}"
        )

    return train_losses, validation_losses
    train_losses = np.zeros(epochs, dtype=np.float32)
    val_losses = np.zeros(epochs, dtype=np.float32)

    model.to(device)

    for e in range(epochs):
        t0 = datetime.now()

        # Train
        model.train()
        batch_losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_losses.append(float(loss.item()))
            loss.backward()
            optimizer.step()

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0

        # Validate
        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                batch_val_losses.append(float(loss.item()))

        val_loss = float(np.mean(batch_val_losses)) if batch_val_losses else 0.0

        train_losses[e] = train_loss
        val_losses[e] = val_loss

        dt = datetime.now() - t0
        print(
            f"Epoch {e+1}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={dt}"
        )

    return train_losses, val_losses
