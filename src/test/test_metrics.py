from src.metrics import accuracy


def test_accuracy_range(model, loaders, device):
    train_loader, _, _ = loaders
    acc = accuracy(model, train_loader, device=device)
    assert 0.0 <= acc <= 1.0
