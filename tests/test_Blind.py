
'''
This is the test code of poisoned training on GTSRB, CIFAR10, MNIST, using dataset class of torchvision.datasets.DatasetFolder torchvision.datasets.CIFAR10 torchvision.datasets.MNIST.
The attack method is Blind.
'''


import os
from typing import Pattern
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
import core


global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

alpha = torch.zeros(3,32,32)
alpha[:,-3:,-3:]=1.
pattern = torch.zeros(3,32,32)
pattern[:,-3:,-3:]=1.

def show_dataset(dataset, num, path_to_save):
    """Each image in dataset should be torch.Tensor, shape (C,H,W)"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,20))
    for i in range(num):
        ax = plt.subplot(num,1,i+1)
        img = (dataset[i][0]).permute(1,2,0).cpu().detach().numpy()
        ax.imshow(img)
    plt.savefig(path_to_save)

# ===== Train backdoored model on GTSRB using with DatasetFolder ======

# Prepare datasets
transform_train = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    RandomHorizontalFlip(),
    ToTensor(),
])
transform_test = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor(),
])

trainset = DatasetFolder(
    root='/data/ganguanhao/datasets/GTSRB/train', # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

testset = DatasetFolder(
    root='/data/ganguanhao/datasets/GTSRB/testset', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

# Configure the attack scheme
blind= core.Blind(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, 43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    pattern=pattern,
    alpha=alpha,
    schedule=None,
    seed=global_seed,
    deterministic=deterministic,
    use_neural_cleanse=False,
)


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [50, 75],

    'epochs': 100,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_DataFolder_GTSRB_Blind'
}

# Train backdoored model
blind.train(schedule)
poisoned_trainset, poisoned_testset = blind.get_poisoned_dataset()
show_dataset(poisoned_trainset, 5, 'gtsrb_train_poison.png')
show_dataset(poisoned_testset, 5, 'gtsrb_test_poison.png')
# # ===== Train backdoored model on GTSRB using with DatasetFolder (done) ======

# # ===== Train backdoored model on CIFAR10 using with CIFAR10 ===== 

# Prepare datasets
transform_train = Compose([
    transforms.Resize((32, 32)),
    RandomHorizontalFlip(),
    ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406),
    #                     (0.229, 0.224, 0.225))
])
transform_test = Compose([
    transforms.Resize((32, 32)),
    ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406),
    #                     (0.229, 0.224, 0.225))
])
trainset = CIFAR10(
    root='/data/ganguanhao/datasets', # please replace this with path to your dataset
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
testset = CIFAR10(
    root='/data/ganguanhao/datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)


blind= core.Blind(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    pattern=pattern,
    alpha=alpha,
    schedule=None,
    seed=global_seed,
    deterministic=deterministic,
    use_neural_cleanse=False,
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    # 'batch_size': 64,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [50, 75],

    'epochs': 100,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_CIFAR10_Blind'
}



# Train backdoored model
blind.train(schedule)
poisoned_trainset, poisoned_testset = blind.get_poisoned_dataset()
show_dataset(poisoned_trainset, 5, 'cifar_train_poison.png')
show_dataset(poisoned_testset, 5, 'cifar_test_poison.png')


        

# ===== Train backdoored model on CIFAR10 using with CIFAR10 (done)===== 

# ===== Train backdoored model on MNIST using with MNIST ===== 
# Prepare datasets
transform_train = Compose([
    transforms.Resize((28, 28)),
    RandomHorizontalFlip(),
    ToTensor(),
])
transform_test = Compose([
    transforms.Resize((28, 28)),
    ToTensor(),
])
trainset = MNIST(
    root='/data/ganguanhao/datasets', # please replace this with path to your dataset
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
testset = MNIST(
    root='/data/ganguanhao/datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)

loader = DataLoader(trainset,)

alpha = torch.zeros(1,28,28)
alpha[:,-3:,-3:]=1.
pattern = torch.zeros(1,28,28)
pattern[:,-3:,-3:]=1.
# Configure the attack scheme
blind= core.Blind(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    pattern=pattern,
    alpha=alpha,
    schedule=None,
    seed=global_seed,
    deterministic=deterministic,
    use_neural_cleanse=False,
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [10, 15],

    'epochs': 20,

    'log_iteration_interval': 100,
    'test_epoch_interval': 5,
    'save_epoch_interval': 5,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_MNIST_Blind'
}
# Train backdoored model
blind.train(schedule)
poisoned_trainset, poisoned_testset = blind.get_poisoned_dataset()
show_dataset(poisoned_trainset, 5, 'mnist_train_poison.png')
show_dataset(poisoned_testset, 5, 'mnist_test_poison.png')
# # ===== Train backdoored model on MNIST using with MNIST (done)===== 
