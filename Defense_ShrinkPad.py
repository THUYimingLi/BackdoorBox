'''
This is the example code of defending the BadNets attack.
Dataset is CIFAR-10.
Defense method is ShrinkPad.
'''

import os

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, Lambda

import core

shrinkpad = core.ShrinkPad(size_map=32, pad=4)

dataset = torchvision.datasets.DatasetFolder

pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255

# The targeting models that have been poisoned and trained by the BadNet attack, the specific watermark is a small 3x3 square, which you can see in example.py.

transform_test = Compose([
    ToTensor() ,
    Lambda(lambda img: img + pattern) ,
    Lambda(lambda img: shrinkpad.preprocess(img))
])

testset = dataset(
    root='./data/cifar10/test',
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

schedule = {
    'test_model': './experiments/train_poisoned_DatasetFolder-CIFAR10_2025-02-24_18:20:13/ckpt_epoch_50.pth',
    'save_dir': './experiments', 
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,
    'experiment_name': 'CIFAR10_test', 
    'device': 'GPU',
    'metric': 'ASR_NoTarget',
    'y_target': 0, 
    'batch_size': 64,
    'num_workers': 4,
}
model=core.models.ResNet(18)

predictions = shrinkpad.test(model, testset, schedule=schedule, size_map=32, pad=4)

