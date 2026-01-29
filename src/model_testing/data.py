import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


def build_transform():
    return transforms.Compose(
        [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]
    )


def load_imagefolder(dataset_path: str):
    transform = build_transform()
    return datasets.ImageFolder(dataset_path, transform=transform)


def make_samplers(dataset_len: int, train_ratio: float = 0.85, train_inside_ratio: float = 0.70, seed: int = 123):
 
      split = floor(0.85 * N)
      validation = floor(0.70 * split)
      train = indices[:validation]
      validation = indices[validation:split]
      test = indices[split:]
 
    indices = list(range(dataset_len))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    split = int(np.floor(train_ratio * dataset_len))
    train_end = int(np.floor(train_inside_ratio * split))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:split]
    test_indices = indices[split:]

    return (
        SubsetRandomSampler(train_indices),
        SubsetRandomSampler(val_indices),
        SubsetRandomSampler(test_indices),
        (len(train_indices), len(val_indices), len(test_indices)),
    )
