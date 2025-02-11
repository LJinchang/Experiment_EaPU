import torch
import torchvision
import torchvision.transforms as transforms


def get_loader(root='./data', cifar10=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if cifar10:
        train_set = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_test)
    else:
        train_set = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(
            root=root, train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=100, shuffle=False, num_workers=0)

    return train_loader, test_loader
