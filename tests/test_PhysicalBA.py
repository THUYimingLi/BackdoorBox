'''
This is the test code of poisoned training under PhysicalBA.
Using dataset class of torchvision.datasets.DatasetFolder, torchvision.datasets.MNIST and torchvision.datasets.CIFAR10.
Default physical transformations is Compose([RandomHorizontalFlip(),ColorJitter(), RandomAffine()])
Choose other transformations from torchvsion.transforms if you need
'''


import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, ColorJitter, RandomAffine
import torchvision.transforms as transforms
import core

global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

# ============== GTSRB ==============

dataset = torchvision.datasets.DatasetFolder

transform_train = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor()
])
trainset = dataset(
    root='/data/xutong/test/BackdoorBox/tests/data/GTSRB/train',
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
testset = dataset(
    root='/data/xutong/test/BackdoorBox/tests/data/GTSRB/testset',
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
weight = torch.zeros((1, 32, 32), dtype=torch.float32)
weight[0, -3:, -3:] = 1.0

PhysicalBA = core.PhysicalBA(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18,43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    pattern=pattern,
    weight=weight,
    poisoned_transform_train_index=2,
    poisoned_transform_test_index=2,
    schedule=None,
    seed=666,
    # modify other transformations from torchvsion.transforms if you want 
    physical_transformations = Compose([
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.2,contrast=0.2), 
        RandomAffine(degrees=10,translate=(0.1, 0.1), scale=(0.8, 0.9))
    ])
)

poisoned_train_dataset, poisoned_test_dataset = PhysicalBA.get_poisoned_dataset()

# train attacked model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 16,

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
    'experiment_name': 'train_poisoned_DatasetFolder_PhysicalBA'
}

PhysicalBA.train(schedule)
infected_model = PhysicalBA.get_model()

# Test Infected Model
test_schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'batch_size': 128,
    'num_workers': 16,

    'save_dir': 'experiments',
    'experiment_name': 'test_poisoned_DatasetFolder_PhysicalBA'
}
PhysicalBA.test(test_schedule)

# ============== mnist ==============

dataset = torchvision.datasets.MNIST

transform_train = Compose([
    ToTensor(),
])
trainset = dataset('data', train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset('data', train=False, transform=transform_test, download=True)

PhysicalBA = core.PhysicalBA(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    seed=global_seed,
    deterministic=deterministic,
    # modify other transformations from torchvsion.transforms if you want 
    physical_transformations = Compose([
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.2), 
        RandomAffine(degrees=10,translate=(0.1, 0.1), scale=(0.8, 0.9))
    ])
)

poisoned_train_dataset, poisoned_test_dataset = PhysicalBA.get_poisoned_dataset()

# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False, # Train Infected Model
    'batch_size': 128,
    'num_workers': 4,

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
    'experiment_name': 'train_poisoned_MNIST_PhysicalBA'
}

PhysicalBA.train(schedule)
infected_model = PhysicalBA.get_model()

# Test Infected Model
test_schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'batch_size': 128,
    'num_workers': 4,

    'save_dir': 'experiments',
    'experiment_name': 'test_poisoned_MNIST_PhysicalBA'
}
PhysicalBA.test(test_schedule)

# ============== cifar10 ==============

dataset = torchvision.datasets.CIFAR10

transform_train = Compose([
        ToTensor(),
    ])
trainset = dataset('data', train=True, transform=transform_train, download=True)

transform_test = Compose([
        ToTensor()
    ])
testset = dataset('data', train=False, transform=transform_test, download=True)

PhysicalBA = core.PhysicalBA(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    seed=global_seed,
    deterministic=deterministic,
    # modify other transformations from torchvsion.transforms if you want 
    physical_transformations = Compose([
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.2,contrast=0.2), 
        RandomAffine(degrees=10,translate=(0.1, 0.1), scale=(0.8, 0.9))
    ])
)

poisoned_train_dataset, poisoned_test_dataset = PhysicalBA.get_poisoned_dataset()

# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False, # Train Infected Model
    'batch_size': 128,
    'num_workers': 4,

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
    'experiment_name': 'train_poisoned_CIFAR10_PhysicalBA'
}

PhysicalBA.train(schedule)
infected_model = PhysicalBA.get_model()

# Test Infected Model
test_schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'batch_size': 128,
    'num_workers': 4,

    'save_dir': 'experiments',
    'experiment_name': 'test_poisoned_CIFAR10_PhysicalBA'
}
PhysicalBA.test(test_schedule)