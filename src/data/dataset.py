import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_test_loader(batch_size: int=8):
    # Define transformation (resize to 224x224 as required by ViT)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load CIFAR-100 dataset
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return testloader