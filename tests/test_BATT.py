'''
This is the test code of benign training and poisoned training under BadNets.
'''


import os

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms as transforms

import core

global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

# ============== cifar10 ==============
# Define Benign Training and Testing Dataset
dataset = torchvision.datasets.CIFAR10

transform_train = Compose([
    ToTensor(),
])
trainset = dataset('data', train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset('data', train=False, transform=transform_test, download=True)

batt = core.BATT(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    seed=global_seed,
    deterministic=deterministic
)

poisoned_train_dataset, poisoned_test_dataset = batt.get_poisoned_dataset()

# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False, # Train Infected Model
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': './result/',
    'experiment_name': 'batt_cifar10'
}

batt.train(schedule)
infected_model = batt.get_model()


# ============== mnist ==============
# Define Benign Training and Testing Dataset
dataset = torchvision.datasets.MNIST


transform_train = Compose([
    ToTensor(),
])
trainset = dataset('data', train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset('data', train=False, transform=transform_test, download=True)

batt = core.BATT(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    seed=global_seed,
    deterministic=deterministic
)

poisoned_train_dataset, poisoned_test_dataset = batt.get_poisoned_dataset()

# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False, # Train Infected Model
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': './result/',
    'experiment_name': 'batt_mnist'
}

batt.train(schedule)
infected_model = batt.get_model()



# ============== GTSRB ==============
# Define Benign Training and Testing Dataset
dataset = torchvision.datasets.DatasetFolder

transform_train = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32,32)),
    ToTensor()
])
trainset = dataset(
    root='/data1/xutong/dataset/GTSRB/Train',
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

transform_test = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32,32)),
    ToTensor()
])
testset = dataset(
    root='/data1/xutong/dataset/GTSRB/testset',
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

batt = core.BATT(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18,43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    poisoned_transform_train_index=2,
    poisoned_transform_test_index=2,
    schedule=None,
    seed=666,
)

poisoned_train_dataset, poisoned_test_dataset = batt.get_poisoned_dataset()

# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False, # Train Infected Model
    'batch_size': 64,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 30,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': './result/',
    'experiment_name': 'batt_gtsrb'
}

batt.train(schedule)
infected_model = batt.get_model()
