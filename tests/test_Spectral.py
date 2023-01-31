'''
This is the test code of Spectral defense.
'''
import os
import sys
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core


# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
datasets_root_dir = '../datasets'
CUDA_VISIBLE_DEVICES = '1'
portion = 0.05
batch_size = 128
num_workers = 4


def test_spectral(defense, defend, model_name, dataset_name, attack_name, defense_name, transform):
    if defend:
        pre_epoch, retrain_epoch,  = 100, 100
    else:
        pre_epoch, retrain_epoch = 0, 0
    
    first_train_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        'epochs': pre_epoch, 
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'schedule': [],
        'gamma': 0.1,

        'log_iteration_interval': 100,
        'test_epoch_interval': 1,
    }

    retrain_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        'epochs': retrain_epoch, 
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'schedule': [30, 55],
        'gamma': 0.1,

        'log_iteration_interval': 100,
        'test_epoch_interval': 1,
    }


    test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,
    }
    
    filter_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,
    }

    schedule = {
        'save_dir': 'experiments/Spectral-defense',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}',

        'first_train_schedule': first_train_schedule,
        'filter_schedule': filter_schedule,
        'retrain_schedule': retrain_schedule,
        'test_schedule': test_schedule,
    }

    defense.repair(schedule=schedule, transform=transform)

    return 

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

# ========== ResNet-18_CIFAR-10_Attack_Spectral ==========

# CIFAR-10 dataset
dataset = torchvision.datasets.CIFAR10

transform_train = Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

# # ===================== No Attack ======================
# torch.manual_seed(global_seed)
# model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'CIFAR-10', 'No Attack', 'Spectral'
# model = core.models.ResNet(18)
# defense = core.Spectral(
#     model=model,
#     loss=nn.CrossEntropyLoss(),
#     poisoned_trainset=trainset,
#     poisoned_testset=testset,
#     clean_trainset=trainset,
#     clean_testset=testset,
#     seed=global_seed,
#     deterministic=deterministic
# )
# test_spectral(defense=defense,
#      defend=True,
#      model_name=model_name,
#      dataset_name=dataset_name, 
#      attack_name=attack_name, 
#      defense_name=defense_name, 
#      transform=ToTensor())

# ===================== BadNets ======================

torch.manual_seed(global_seed)
model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'CIFAR-10', 'BadNets', 'Spectral'

model = core.models.ResNet(18)

# Get BadNets poisoned dataset
attack = core.BadNets(
    train_dataset = trainset, 
    test_dataset = testset,
    model = None,
    loss = None,
    y_target = 0,
    poisoned_rate=0.05,
    pattern=pattern, 
    weight=weight,   
)
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()

# defend against BadNets attack
defense = core.Spectral(
    model=model,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=poisoned_trainset,
    poisoned_testset=poisoned_testset,
    clean_trainset=trainset,
    clean_testset=testset,
    seed=global_seed,
    deterministic=deterministic
)
test_spectral(defense=defense,
     defend=True,
     model_name=model_name,
     dataset_name=dataset_name, 
     attack_name=attack_name, 
     defense_name=defense_name, 
     transform=ToTensor())

# # ===================== WaNet ======================

model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'CIFAR-10', 'WaNet', 'Spectral'
torch.manual_seed(global_seed)

model = core.models.ResNet(18)

# Get BadNets poisoned dataset
attack = core.WaNet(
    train_dataset = trainset, 
    test_dataset = testset,
    model = None,
    loss = None,
    y_target = 0,
    poisoned_rate=0.1,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=True,
)
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()

# defend against BadNets attack
defense = core.Spectral(
    model=model,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=poisoned_trainset,
    poisoned_testset=poisoned_testset,
    clean_trainset=trainset,
    clean_testset=testset,
    seed=global_seed,
    deterministic=deterministic
)
test_spectral(defense=defense,
     defend=True,
     model_name=model_name,
     dataset_name=dataset_name, 
     attack_name=attack_name, 
     defense_name=defense_name, 
     transform=ToTensor())

# ===================== Label Consistent ======================

adv_model = core.models.ResNet(18)
adv_ckpt = torch.load('/data/xutong/BackdoorBox/experiments/train_CIFAR10_Benign_2022-07-02_18:52:00/ckpt_epoch_100.pth')
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

    'save_dir': 'experiments',
    'experiment_name': 'ResNet-18_CIFAR-10_LabelConsistent'
}

trainset.transform = transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip()
])
attack = core.LabelConsistent(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    adv_model=adv_model,
    adv_dataset_dir=f'./adv_dataset/CIFAR-10_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
    # adv_dataset_dir=f'./adv_dataset/adv_CIFAR-10_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
    loss=nn.CrossEntropyLoss(),
    y_target=2,
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
trainset.transform = transform_train

model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'CIFAR-10', 'Label-Consistent', 'Spectral'

poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()

torch.manual_seed(global_seed)
model = core.models.ResNet(18)
defense = core.Spectral(
    model=model,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=poisoned_trainset,
    poisoned_testset=poisoned_testset,
    clean_trainset=trainset,
    clean_testset=testset,
    seed=global_seed,
    deterministic=deterministic
)
test_spectral(defense=defense,
     defend=True,
     model_name=model_name,
     dataset_name=dataset_name, 
     attack_name=attack_name, 
     defense_name=defense_name, 
     transform=ToTensor())


# ========== ResNet-18_GTSRB_Attack_Spectral ==========

# GTSRB dataset

transform_train = Compose([
    ToPILImage(),
    Resize((32, 32)),
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

# ===================== No Attack ======================
model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'GTSRB', 'No Attack', 'Spectral'
torch.manual_seed(global_seed)
model = core.models.ResNet(18,43)

# defend against BadNets attack
defense = core.Spectral(
    model=model,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=trainset,
    poisoned_testset=testset,
    clean_trainset=trainset,
    clean_testset=testset,
    seed=global_seed,
    deterministic=deterministic
)
test_spectral(defense=defense,
     defend=True,
     model_name=model_name,
     dataset_name=dataset_name, 
     attack_name=attack_name, 
     defense_name=defense_name, 
     transform=ToTensor())
# ===================== BadNets ======================

model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'GTSRB', 'BadNets', 'Spectral'


torch.manual_seed(global_seed)
model = core.models.ResNet(18,43)

# Get BadNets poisoned dataset
attack = core.BadNets(
    train_dataset = trainset, 
    test_dataset = testset,
    model = None,
    loss = None,
    y_target = 0,
    poisoned_rate=0.05,
    pattern=pattern, 
    weight=weight, 
    poisoned_transform_train_index=2,  
    poisoned_transform_test_index=2,  
)
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()

# defend against BadNets attack
defense = core.Spectral(
    model=model,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=poisoned_trainset,
    poisoned_testset=poisoned_testset,
    clean_trainset=trainset,
    clean_testset=testset,
    seed=global_seed,
    deterministic=deterministic
)
test_spectral(defense=defense,
     defend=True,
     model_name=model_name,
     dataset_name=dataset_name, 
     attack_name=attack_name, 
     defense_name=defense_name, 
     transform=ToTensor())

# ===================== WaNet ======================

model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'GTSRB', 'WaNet', 'Spectral'

torch.manual_seed(global_seed)
model = core.models.ResNet(18,43)

# Get BadNets poisoned dataset
attack = core.WaNet(
    train_dataset = trainset, 
    test_dataset = testset,
    model = None,
    loss = None,
    y_target = 0,
    poisoned_rate=0.1,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=True,
    poisoned_transform_train_index=2,  
    poisoned_transform_test_index=2,  
)
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()

# defend against BadNets attack
defense = core.Spectral(
    model=model,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=poisoned_trainset,
    poisoned_testset=poisoned_testset,
    clean_trainset=trainset,
    clean_testset=testset,
    seed=global_seed,
    deterministic=deterministic
)
test_spectral(defense=defense,
     defend=True,
     model_name=model_name,
     dataset_name=dataset_name, 
     attack_name=attack_name, 
     defense_name=defense_name, 
     transform=ToTensor())


# ===================== Label Consistent ======================

torch.manual_seed(global_seed)
adv_model = core.models.ResNet(18, 43)
adv_ckpt = torch.load('/data/ganguanhao/BackdoorBox/experiments/train_GTSRB_Benign_2022-07-04_10:34:40/ckpt_epoch_100.pth')
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

    'save_dir': 'experiments',
    'experiment_name': 'ResNet-18_GTSRB_LabelConsistent'
}

attack = core.LabelConsistent(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, 43),
    adv_model=adv_model,
    # adv_dataset_dir=f'./adv_dataset/CIFAR-10_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
    adv_dataset_dir=f'./adv_dataset/GTSRB_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
    adv_transform=Compose([transforms.ToPILImage(), transforms.Resize((32, 32)), ToTensor()]),
    loss=nn.CrossEntropyLoss(),
    
    y_target=2,
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

model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'GTSRB', 'Label-Consistent', 'Spectral'

poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()

model = core.models.ResNet(18, 43)
defense = core.Spectral(
    model=model,
    loss=nn.CrossEntropyLoss(),
    poisoned_trainset=poisoned_trainset,
    poisoned_testset=poisoned_testset,
    clean_trainset=trainset,
    clean_testset=testset,
    seed=global_seed,
    deterministic=deterministic
)

test_spectral(defense=defense,
     defend=True,
     model_name=model_name,
     dataset_name=dataset_name, 
     attack_name=attack_name, 
     defense_name=defense_name, 
     transform=ToTensor())