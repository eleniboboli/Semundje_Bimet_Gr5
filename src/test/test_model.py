import torch


def test_model_forward_shape(model, device):
    x = torch.randn(2, 3, 224, 224).to(device)
    y = model(x)
    assert y.shape == (2, 5)
