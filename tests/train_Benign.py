import os

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
import core


global_seed = 666
deterministic = True
torch.manual_seed(global_seed)


# ============== Train Benign BaselineMNISTNetwork on MNIST ==============
dataset = torchvision.datasets.MNIST

transform_train = Compose([
    ToTensor()
])
trainset = dataset('data', train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset('data', train=False, transform=transform_test, download=True)


badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    pattern=torch.zeros((28, 28), dtype=torch.uint8),
    weight=torch.zeros((28, 28), dtype=torch.float32),
    seed=global_seed,
    deterministic=deterministic
)

# Train Benign Model (schedule is set by yamengxi.)
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': True, # Train Benign Model
    'batch_size': 128,
    'num_workers': 2,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'MNIST_BaselineMNISTNetwork_Benign'
}
badnets.train(schedule)


# ============== Train Benign ResNet-18 on CIFAR-10 ==============
dataset = torchvision.datasets.CIFAR10

transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip()
])
trainset = dataset('data', train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset('data', train=False, transform=transform_test, download=True)


badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    pattern=torch.zeros((32, 32), dtype=torch.uint8),
    weight=torch.zeros((32, 32), dtype=torch.float32),
    seed=global_seed,
    deterministic=deterministic
)

# Train Benign Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_standard.py)
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': True, # Train Benign Model
    'batch_size': 128,
    'num_workers': 2,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'CIFAR-10_ResNet-18_Benign'
}
badnets.train(schedule)


# ============== Train Benign ResNet-18 on GTSRB ==============
transform_train = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    RandomHorizontalFlip(),
    ToTensor()
])
trainset = DatasetFolder(
    root='/data/yamengxi/Backdoor/datasets/GTSRB/train', # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

transform_test = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor()
])
testset = DatasetFolder(
    root='/data/yamengxi/Backdoor/datasets/GTSRB/testset', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)


badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, 43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    pattern=torch.zeros((32, 32), dtype=torch.uint8),
    weight=torch.zeros((32, 32), dtype=torch.float32),
    poisoned_transform_train_index=2,
    poisoned_transform_test_index=2,
    seed=global_seed,
    deterministic=deterministic
)

# Train Benign Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/GTSRB/train_standard.py)
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': True, # Train Benign Model
    'batch_size': 128,
    'num_workers': 2,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [20],

    'epochs': 30,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'GTSRB_ResNet-18_Benign'
}
badnets.train(schedule)