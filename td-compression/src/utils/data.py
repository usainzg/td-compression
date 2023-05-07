from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import torch

def prepare_data():
    print('==> Preparing data..')

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

    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)

    train_ds, val_ds = random_split(trainset, [45000, 5000])

    trainloader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(
        val_ds, batch_size=128, shuffle=False, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    
    return {
        'train': trainloader, 
        'val': valloader, 
        'test': testloader, 
        'classes': classes
    }