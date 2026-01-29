import torch


def accuracy(model, loader, device: str = "cpu"):
    model.eval()
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            n_correct += (preds == targets).sum().item()
            n_total += targets.shape[0]

    return (n_correct / n_total) if n_total > 0 else 0.0
