import pytest
import torch
from torchvision.datasets import FakeData
from torchvision import transforms
from torch.utils.data import DataLoader

from src.model import CNN


@pytest.fixture(scope="session")
def device():
    return "cpu"


@pytest.fixture(scope="session")
def fake_dataset():
    tfm = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
    return FakeData(size=60, image_size=(3, 224, 224), num_classes=5, transform=tfm)


@pytest.fixture()
def loaders(fake_dataset):
    train_loader = DataLoader(fake_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(fake_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(fake_dataset, batch_size=4, shuffle=False)
    return train_loader, val_loader, test_loader


@pytest.fixture()
def model(fake_dataset, device):
    m = CNN(K=5).to(device)
    return m
