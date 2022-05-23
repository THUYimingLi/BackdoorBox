'''
This is the test code of poisoned training under TUAP.
'''


import os
import os.path as osp

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder

import core
from core.attacks.TUAP import TUAP


CUDA_VISIBLE_DEVICES = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
datasets_root_dir = '/data/yamengxi/Backdoor/datasets/'
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)


# ============== TUAP attack BaselineMNISTNetwork on MNIST ==============
dataset = torchvision.datasets.MNIST

transform_train = Compose([
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
print("trainset",len(trainset))
transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
print("testset",len(testset))


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': False, # Train Attacked Model
    'batch_size': 1024,
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

    'save_dir': 'experiments',
    'experiment_name': 'BaselineMNISTNetwork_MNIST_TUAP'
}


UAP_benign_model = core.models.BaselineMNISTNetwork()
UAP_benign_PATH = './benign_MNIST/MNIST/ckpt_epoch_20.pth'
checkpoint = torch.load(UAP_benign_PATH)
UAP_benign_model.load_state_dict(checkpoint)

poisoned_rate = 0.25
# epsilon = 76.0/255
epsilon = 0.3
delta = 0.2
max_iter_uni = np.inf
p_norm = np.inf
num_classes = 10
overshoot = 0.02
max_iter_df = 50
p_samples = 0.01
mask = np.ones((1, 28, 28))

tuap = core.TUAP(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),

    benign_model=UAP_benign_model,
    y_target=2,
    poisoned_rate=poisoned_rate,
    epsilon = epsilon,
    delta=delta,
    max_iter_uni=max_iter_uni,
    p_norm=p_norm,
    num_classes=num_classes,
    overshoot=overshoot,
    max_iter_df=max_iter_df,
    p_samples=p_samples,
    mask=mask,
    
    poisoned_transform_train_index=0,
    poisoned_transform_test_index=0,
    poisoned_target_transform_index=0,
    schedule=schedule,
    seed=global_seed,
    deterministic=True
)
tuap.train()


# ==============TUAP attack ResNet-18 on CIFAR-10 ==============
dataset = torchvision.datasets.CIFAR10
transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
print("trainset",len(trainset))
transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
print("testset",len(testset))


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': False, # Train Attacked Model
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

    'save_dir': 'experiments',
    'experiment_name': 'ResNet-18_CIFAR-10_TUAP'
}

UAP_benign_model = core.models.ResNet(18)
UAP_benign_PATH = './benign_CIFAR10/ckpt_epoch_100.pth'
checkpoint = torch.load(UAP_benign_PATH)
UAP_benign_model.load_state_dict(checkpoint)
poisoned_rate = 0.25
# epsilon = 10
epsilon = 0.031
delta = 0.2
max_iter_uni = np.inf
p_norm = np.inf
num_classes = 10
overshoot = 0.02
max_iter_df = 50
p_samples = 0.01
mask = np.ones((3, 32, 32))


tuap = core.TUAP(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),

    benign_model=UAP_benign_model,
    y_target=2,
    poisoned_rate=poisoned_rate,
    epsilon = epsilon,
    delta=delta,
    max_iter_uni=max_iter_uni,
    p_norm=p_norm,
    num_classes=num_classes,
    overshoot=overshoot,
    max_iter_df=max_iter_df,
    p_samples=p_samples,
    mask=mask,

    poisoned_transform_train_index=0,
    poisoned_transform_test_index=0,
    poisoned_target_transform_index=0,
    schedule=schedule,
    seed=global_seed,
    deterministic=True
)

tuap.train()



# ============== TUAP attack ResNet-18 on GTSRB ==============
transform_train = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    RandomHorizontalFlip(),
    ToTensor()
])
trainset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB', 'train'), # please replace this with path to your training set
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
    root=osp.join(datasets_root_dir, 'GTSRB', 'testset'), # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': False, # Train Attacked Model
    'batch_size': 256,
    'num_workers': 8,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [20],

    'epochs': 50,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'ResNet-18_GTSRB_TUAP'
}

UAP_benign_model = core.models.ResNet(18,43)
UAP_benign_PATH = './benign_GTSRB/ckpt_epoch_100.pth'
checkpoint = torch.load(UAP_benign_PATH)
UAP_benign_model.load_state_dict(checkpoint)
poisoned_rate = 0.5
epsilon = 0.031
delta = 0.2
max_iter_uni =20
p_norm = np.inf
num_classes = 10
overshoot = 0.02
max_iter_df = 50
p_samples = 0.02
mask = np.ones((3, 32, 32))


tuap = core.TUAP(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, 43),
    loss=nn.CrossEntropyLoss(),

    benign_model=UAP_benign_model,
    y_target=2,
    poisoned_rate=poisoned_rate,
    epsilon = epsilon,
    delta=delta,
    max_iter_uni=max_iter_uni,
    p_norm=p_norm,
    num_classes=num_classes,
    overshoot=overshoot,
    max_iter_df=max_iter_df,
    p_samples=p_samples,
    mask=mask,

    poisoned_transform_train_index=2,
    poisoned_transform_test_index=2,
    poisoned_target_transform_index=0,
    schedule=schedule,
    seed=global_seed,
    deterministic=True
)

tuap.train()
