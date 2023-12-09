'''
This is the test code of Spectral defense.
'''
import os
from copy import deepcopy
import os.path as osp
from cv2 import transform

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, ToPILImage, Resize
import core


# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

datasets_root_dir = '../dataset/'
CUDA_VISIBLE_DEVICES = '1'

# BadNets Configs
pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0

# WaNet Config
def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid

identity_grid, noise_grid = gen_grid(32, 4)


# # ========== ResNet-18_CIFAR-10_Attack_Spectral ==========

# CIFAR-10 dataset
dataset = torchvision.datasets.CIFAR10

transform_train = Compose([
    RandomHorizontalFlip(),
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)


transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

# ===================== BadNets ======================

torch.manual_seed(global_seed)

model = core.models.ResNet(18)

# Get BadNets poisoned dataset
badnets = core.BadNets(
    train_dataset = trainset, 
    test_dataset = testset,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target = 1,
    poisoned_rate=0.05,
    pattern=pattern, 
    weight=weight,   
)

poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()

# defend against BadNets attack
defense = core.Spectral(
    model=model,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=poisoned_train_dataset,
    clean_trainset=trainset,
    seed=global_seed,
    deterministic=deterministic
)
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
    'experiment_name': 'spectral_badnets_cifar10'
}

defense.test(poisoned_location = badnets.poisoned_train_dataset.poisoned_set, schedule = schedule)

# # # # ===================== WaNet ======================

torch.manual_seed(global_seed)

model = core.models.ResNet(18)

# Get BadNets poisoned dataset
wanet = core.WaNet(
    train_dataset = trainset, 
    test_dataset = testset,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target = 1,
    poisoned_rate=0.05,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=True,
)
poisoned_train_dataset, poisoned_test_dataset = wanet.get_poisoned_dataset()
 
# defend against BadNets attack
defense = core.Spectral(
    model=model,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=poisoned_train_dataset,
    clean_trainset=trainset,
    seed=global_seed,
    deterministic=deterministic
)
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
    'experiment_name': 'spectral_wanet_cifar10'
}

defense.test(poisoned_location = wanet.poisoned_train_dataset.poisoned_set, schedule = schedule)

# ===================== Label Consistent ======================

adv_model = core.models.ResNet(18)
adv_ckpt = torch.load('/data1/xx/BackdoorBox/tests/experiments/ResNet-18_CIFAR-10_Benign_2023-02-26_22:33:37/ckpt_epoch_200.pth')
adv_model.load_state_dict(adv_ckpt)

pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-1, -1] = 255
pattern[-1, -3] = 255
pattern[-3, -1] = 255
pattern[-2, -2] = 255

pattern[0, -1] = 255
pattern[1, -2] = 255
pattern[2, -3] = 255
pattern[2, -1] = 255

pattern[0, 0] = 255
pattern[1, 1] = 255
pattern[2, 2] = 255
pattern[2, 0] = 255

pattern[-1, 0] = 255
pattern[-1, 2] = 255
pattern[-2, 1] = 255
pattern[-3, 0] = 255

weight = torch.zeros((32, 32), dtype=torch.float32)
weight[:3,:3] = 1.0
weight[:3,-3:] = 1.0
weight[-3:,:3] = 1.0
weight[-3:,-3:] = 1.0


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
    'experiment_name': 'ResNet-18_CIFAR-10_LabelConsistent'
}


eps = 8
alpha = 1.5
steps = 100
max_pixel = 255
poisoned_rate = 0.25

label_consistent = core.LabelConsistent(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    adv_model=adv_model,
    adv_dataset_dir=f'./adv_dataset/CIFAR-10_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
    loss=nn.CrossEntropyLoss(),
    y_target = 1,
    poisoned_rate=poisoned_rate,
    pattern=pattern,
    weight=weight,
    eps=eps,
    alpha=alpha,
    steps=steps,
    max_pixel=max_pixel,
    poisoned_transform_train_index=0,
    poisoned_transform_test_index=0,
    poisoned_target_transform_index=0,
    schedule=schedule,
    seed=global_seed,
    deterministic=True
)

poisoned_train_dataset, poisoned_test_dataset = label_consistent.get_poisoned_dataset()

model = core.models.ResNet(18)

# defend against BadNets attack
defense = core.Spectral(
    model = model,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=poisoned_train_dataset,
    clean_trainset=trainset,
    seed=global_seed,
    deterministic=deterministic
)

defense.test(poisoned_location = label_consistent.poisoned_test_dataset.poisoned_set, schedule = schedule)


# ========== ResNet-18_GTSRB_Attack_Spectral ==========

# GTSRB dataset

transform_train = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor()
])
trainset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB', 'Train'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

transform_test = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor()
])
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB', 'testset'), # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

# # ===================== BadNets ======================

torch.manual_seed(global_seed)
model = core.models.ResNet(18,43)

# Get BadNets poisoned dataset
badnets = core.BadNets(
    train_dataset = trainset, 
    test_dataset = testset,
    model=core.models.ResNet(18,43),
    loss=nn.CrossEntropyLoss(),
    y_target = 1,
    poisoned_rate=0.05,
    pattern=pattern, 
    weight=weight, 
    poisoned_transform_train_index=2,  
    poisoned_transform_test_index=2,  
)
poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()
 
# defend against BadNets attack
defense = core.Spectral(
    model=model,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=poisoned_train_dataset,
    clean_trainset=trainset,
    seed=global_seed,
    deterministic=deterministic
)
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
    'experiment_name': 'spectral_badnets_gtsrb'
}

defense.test(poisoned_location = badnets.poisoned_test_dataset.poisoned_set, schedule = schedule)

# ===================== WaNet ======================

torch.manual_seed(global_seed)
model = core.models.ResNet(18,43)

# Get BadNets poisoned dataset
wanet = core.WaNet(
    train_dataset = trainset, 
    test_dataset = testset,
    model = None,
    loss = None,
    y_target = 1,
    poisoned_rate=0.05,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=True,
    poisoned_transform_train_index=2,  
    poisoned_transform_test_index=2,  
)
poisoned_train_dataset, poisoned_test_dataset = wanet.get_poisoned_dataset()
 
# defend against BadNets attack
defense = core.Spectral(
    model=model,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=poisoned_train_dataset,
    clean_trainset=trainset,
    seed=global_seed,
    deterministic=deterministic
)
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
    'experiment_name': 'spectral_wanet_gtsrb'
}

defense.test(poisoned_location = wanet.poisoned_train_dataset.poisoned_set, schedule = schedule)


# # ===================== Label Consistent ======================

torch.manual_seed(global_seed)
adv_model = core.models.ResNet(18, 43)
adv_ckpt = torch.load('/data1/xx/BackdoorBox/tests/experiments/ResNet-18_GTSRB_BadNets_2023-03-20_01:40:11/ckpt_epoch_30.pth')
adv_model.load_state_dict(adv_ckpt)

pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-1, -1] = 255
pattern[-1, -3] = 255
pattern[-3, -1] = 255
pattern[-2, -2] = 255

pattern[0, -1] = 255
pattern[1, -2] = 255
pattern[2, -3] = 255
pattern[2, -1] = 255

pattern[0, 0] = 255
pattern[1, 1] = 255
pattern[2, 2] = 255
pattern[2, 0] = 255

pattern[-1, 0] = 255
pattern[-1, 2] = 255
pattern[-2, 1] = 255
pattern[-3, 0] = 255

weight = torch.zeros((32, 32), dtype=torch.float32)
weight[:3,:3] = 1.0
weight[:3,-3:] = 1.0
weight[-3:,:3] = 1.0
weight[-3:,-3:] = 1.0

eps = 8
alpha = 1.5
steps = 100
max_pixel = 255
poisoned_rate = 0.25

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

    'save_dir': './result',
    'experiment_name': 'spectral_lc_gtsrb'
}

labelconsistent = core.LabelConsistent(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, 43),
    adv_model=adv_model,
    # adv_dataset_dir=f'./adv_dataset/CIFAR-10_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
    adv_dataset_dir=f'./adv_dataset/GTSRB_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
    adv_transform=Compose([transforms.ToPILImage(), transforms.Resize((32, 32)), ToTensor()]),
    loss=nn.CrossEntropyLoss(),
    
    y_target = 1,
    poisoned_rate=poisoned_rate,
    pattern=pattern,
    weight=weight,
    eps=eps,
    alpha=alpha,
    steps=steps,
    max_pixel=max_pixel,
    poisoned_transform_train_index=2,
    poisoned_transform_test_index=2,
    # poisoned_target_transform_index=0,
    schedule=schedule,
    seed=global_seed,
    deterministic=True
)

model = core.models.ResNet(18, 43)
poisoned_train_dataset, poisoned_test_dataset = labelconsistent.get_poisoned_dataset()
 
# defend against BadNets attack
defense = core.Spectral(
    model=model,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=poisoned_train_dataset,
    clean_trainset=trainset,
    seed=global_seed,
    deterministic=deterministic
)

defense.test(poisoned_location = labelconsistent.poisoned_test_dataset.poisoned_set, schedule = schedule)