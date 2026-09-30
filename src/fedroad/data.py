import random

import torch
import torchvision
import torchvision.transforms as transforms


def create_class_indices(trainset):
    class_indices = {i: [] for i in range(10)}
    for idx, label in enumerate(trainset.targets):
        class_indices[label].append(idx)
    return class_indices

def sample_local_data(trainset, class_indices, n_data, n_sub_classes):
    selected_classes = random.sample(range(10), n_sub_classes)
    available_indices = []
    for cls in selected_classes:
        available_indices.extend(class_indices[cls])
    if len(available_indices) < n_data:
        sampled_indices = available_indices
    else:
        sampled_indices = random.sample(available_indices, n_data)
    local_data = torch.utils.data.Subset(trainset, sampled_indices)
    return local_data

def get_transform_train():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5))
    ])
    return transform_train

def get_trainset(root):
    return torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=get_transform_train(),
    )

def get_transform_test():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5))
    ])
    return transform_test

def get_testset(root):
    return torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=get_transform_test(),
    )

def get_test_loader(root):
    return torch.utils.data.DataLoader(
        get_testset(root), batch_size=64, shuffle=False, num_workers=0
    )
