import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_test_loader(batch_size: int = 8):
    transform = get_transforms()
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return testloader


def get_train_val_loaders(batch_size: int = 8, val_split: float = 0.1):
    transform = get_transforms()
    full_trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    val_size = int(val_split * len(full_trainset))
    train_size = len(full_trainset) - val_size

    train_subset, val_subset = random_split(full_trainset, [train_size, val_size])

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader
