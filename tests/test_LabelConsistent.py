'''
This is the test code of benign training and poisoned training under LabelConsistent.
'''


import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import core
from tests.model.network.resnet import resnet18


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

# Define Benign Training and Testing Dataset
dataset = torchvision.datasets.CIFAR10
# dataset = torchvision.datasets.MNIST


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip()
])
trainset = dataset('data', train=True, transform=transform_train, download=True)


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])
testset = dataset('data', train=False, transform=transform_test, download=True)


adv_model = core.models.ResNet(18)
adv_ckpt = torch.load('/data/yamengxi/Backdoor/BackdoorBox/experiments/train_benign_CIFAR10_BadNets_2022-01-12_20:50:36/ckpt_epoch_200.pth')
adv_model.load_state_dict(adv_ckpt)
# adv_model = resnet18()
# adv_ckpt = torch.load('/data/yamengxi/Backdoor/BackdoorBox/tests/model/adv_models/cifar_resnet_e8_a2_s10.pth')
# adv_model.load_state_dict(adv_ckpt)

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


# pattern = cv2.imread('/data/yamengxi/Backdoor/label_consistent_attacks_pytorch/data/trigger/cifar_1.png')
# pattern = pattern.transpose(2, 0, 1)
# weight = (pattern > 0).astype(np.float32)

# pattern = torch.from_numpy(pattern)
# weight = torch.from_numpy(weight)

# breakpoint()

# breakpoint()

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '2',
    'GPU_num': 1,

    'benign_training': False, # Train Attacked Model
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 2e-4,
    'gamma': 0.1,
    'schedule': [100, 150],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 5,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poisioned_CIFAR10_LabelConsistent'
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
    y_target=3,
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

label_consistent.train()