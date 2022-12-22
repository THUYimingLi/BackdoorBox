'''
This is the test code of benign training and poisoned training under Blended Attack.
'''


import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import core


global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

# Define Benign Training and Testing Dataset
# dataset = torchvision.datasets.CIFAR10
dataset = torchvision.datasets.MNIST


transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip()
])
trainset = dataset('data', train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset('data', train=False, transform=transform_test, download=True)


# Show an Example of Benign Training Samples
index = 44

x, y = trainset[index]
print(y)
for a in x[0]:
    for b in a:
        print("%-4.2f" % float(b), end=' ')
    print()


# Settings of Pattern and Weight
'''
pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
weight = torch.zeros((1, 32, 32), dtype=torch.float32)
weight[0, -3:, -3:] = 0.2
'''
pattern = torch.zeros((1, 28, 28), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
weight = torch.zeros((1, 28, 28), dtype=torch.float32)
weight[0, -3:, -3:] = 0.2


blended = core.Blended(
    train_dataset=trainset,
    test_dataset=testset,
    # model=core.models.ResNet(18),
    model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    pattern=pattern,
    weight=weight,
    y_target=1,
    poisoned_rate=0.05,
    seed=global_seed,
    deterministic=deterministic
)

poisoned_train_dataset, poisoned_test_dataset = blended.get_poisoned_dataset()


# Show an Example of Poisoned Training Samples
x, y = poisoned_train_dataset[index]
print(y)
for a in x[0]:
    for b in a:
        print("%-4.2f" % float(b), end=' ')
    print()


# Show an Example of Poisoned Testing Samples
x, y = poisoned_test_dataset[index]
print(y)
for a in x[0]:
    for b in a:
        print("%-4.2f" % float(b), end=' ')
    print()


# Train Benign Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

    'benign_training': True,
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
    # 'experiment_name': 'train_benign_CIFAR10_Blended'
    'experiment_name': 'train_benign_MNIST_Blended'
}

blended.train(schedule)
benign_model = blended.get_model()


# Test Benign Model
test_schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

    'batch_size': 128,
    'num_workers': 4,

    'save_dir': 'experiments',
    # 'experiment_name': 'test_benign_CIFAR10_Blended'
    'experiment_name': 'test_benign_MNIST_Blended'
}
blended.test(test_schedule)

blended.model = core.models.BaselineMNISTNetwork()
# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

    'benign_training': False,
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
    # 'experiment_name': 'train_poisoned_CIFAR10_Blended'
    'experiment_name': 'train_poisoned_MNIST_Blended'
}

blended.train(schedule)
infected_model = blended.get_model()


# Test Infected Model
test_schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

    'batch_size': 128,
    'num_workers': 4,

    'save_dir': 'experiments',
    # 'experiment_name': 'test_poisoned_CIFAR10_Blended'
    'experiment_name': 'test_poisoned_MNIST_Blended'
}
blended.test(test_schedule)
