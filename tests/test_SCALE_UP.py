'''
This is the test code of SCALE-UP defense.
SCALE-UP: An Efficient Black-box Input-level Backdoor Detection via Analyzing Scaled Prediction Consistency [ICLR, 2023] (https://arxiv.org/abs/2302.03251)
'''


from copy import deepcopy
import os.path as osp

import numpy as np
import random
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import Subset
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, ToPILImage, Resize
from PIL import Image
import core


# ========== Set global settings ==========
global_seed = 0
# deterministic = True
deterministic = False
torch.manual_seed(global_seed)
# ========== Set global settings ==========
datasets_root_dir = os.path.expanduser('~/data/dataset')
CUDA_VISIBLE_DEVICES = '0'
portion = 0.1
batch_size = 128
num_workers = 4


def test(model_name, dataset_name, attack_name, defense_name, benign_dataset, attacked_dataset, defense, y_target):
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        'metric': 'BA',

        'save_dir': 'experiments/SCALE-UP-defense',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_BA'
    }
    defense.test_acc(benign_dataset, schedule)
    if not attack_name == 'Benign':
        schedule = {
            'device': 'GPU',
            'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
            'GPU_num': 1,

            'batch_size': batch_size,
            'num_workers': num_workers,

            # 1. ASR: the attack success rate calculated on all poisoned samples
            # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
            # 3. BA: the accuracy on all benign samples
            # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
            # In other words, ASR or BA does not influence the computation of the metric.
            # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
            'metric': 'ASR_NoTarget',
            'y_target': y_target,

            'save_dir': 'experiments/SCALE-UP-defense',
            'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_ASR'
        }
        defense.test_acc(attacked_dataset, schedule)


# ========== ResNet-18_CIFAR-10_Attack_SCALE-UP ==========
# ===================== Benign ======================
model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'CIFAR-10', 'Benign', 'SCALE-UP'

benign_model = core.models.ResNet(18, num_classes=10)
model_path = '/home/ay2/data/experiments/List_ResNet-18_CIFAR-10_Benign/_2023-08-01_19:05:02/ckpt_999.pth'
benign_model.load_state_dict(torch.load(model_path), strict=False)

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
# Construct Shift Set for Defensive Purpose
num_img = len(testset)
indices = list(range(0, num_img))
random.shuffle(indices)
val_budget = 2000
val_indices = indices[:val_budget]
test_indices = indices[val_budget:]
# Construct Shift Set for Defensive Purpose
sub_testset = Subset(testset, test_indices)
val_set = Subset(testset, val_indices)

defense = core.SCALE_UP(model=benign_model)
test(model_name, dataset_name, attack_name, defense_name, testset, None, defense, None)

# # ========== ResNet-18_CIFAR-10_Attack_Defense SCALE-UP ==========
poisoning_rate = 0.1
target_label = 0
# ===================== BadNets ======================
model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'CIFAR-10', 'BadNets', 'SCALE-UP'
badnet_model = core.models.ResNet(18, num_classes=10)
model_path = '/home/ay2/data/experiments/CIFAR10/ResNet-18/BadNets/pratio_0.1/2024-05-23_16:07:13/ckpt_ random.pth'
def load_dict(model_path):
    state_dict = torch.load(model_path)
    # print(state_dict)
    if 'model' in list(state_dict.keys()):
        return state_dict['model']
    else:
        return state_dict
badnet_model.load_state_dict(load_dict(model_path))

pattern_path = '/home/ay2/data/experiments/CIFAR10/ResNet-18/BadNets/pratio_0.1/2024-05-23_16:07:13/triggers/random_pattern.pt'
pattern = torch.load(pattern_path)
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0
# Get BadNets poisoned dataset
attack = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, num_classes=10),
    loss=nn.CrossEntropyLoss(),
    y_target=target_label,
    poisoned_rate=poisoning_rate,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()
# poisoned_testset = Subset(poisoned_testset, test_indices)
defense = core.SCALE_UP(model=badnet_model)
print(f'the BA and ASR of the original BadNets model: ............. ')
# test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, None)
print('---------data-free scenario----------')
defense.test(testset, poisoned_testset)
print('---------data-limited scenario----------')
defense_with_val = core.SCALE_UP(model=badnet_model, valset=val_set)
defense_with_val.test(testset, poisoned_testset)

# `detect` function is used to check if the first batch of samples in the dataset is poisoned.
# Users can assemble their data into a batch of shape [num_samples, n, w, h] and call `defense._detect(batch)`
# for online detection of the input.
# `preds_benign` contains the detection results for the original test dataset.
# `preds_poison` contains the detection results for the poisoned test dataset.
preds_benign = defense.detect(testset)
preds_poison = defense.detect(poisoned_testset)
print(f'Is poisoned for real benign batch: {preds_benign}')
print(f'Is poisoned for real poisoned batch: {preds_poison}')


# ===================== WaNet ======================
attack_name = 'WaNet'
wanet_model = core.models.ResNet(18, num_classes=10)
model_path = '/home/ay2/data/experiments/CIFAR-10/ResNet-18/Wanet/ay3_wanet/0th_k_4_s_0.5_target_0_morph.pth.tar'
state_dict = torch.load(model_path)
wanet_model.load_state_dict(state_dict['netC'])
identity_grid, noise_grid = state_dict["identity_grid"].cpu(), state_dict["noise_grid"].cpu()
# target_label = state_dict['target']

attack = core.WaNet(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, num_classes=10),
    loss=nn.CrossEntropyLoss(),
    y_target=target_label,
    poisoned_rate=poisoning_rate,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=True,
    seed=global_seed,
    deterministic=deterministic
)
poisoned_testset = attack.poisoned_test_dataset
defense = core.SCALE_UP(model=wanet_model)
print(f'the BA and ASR of the original WaNet model: ............. ')
test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, None)
print('---------data-free scenario----------')
defense.test(testset, poisoned_testset)
print('---------data-limited scenario----------')
defense_with_val = core.SCALE_UP(model=wanet_model, valset=val_set)
defense_with_val.test(testset, poisoned_testset)

# `detect` function is used to check if the first batch of samples in the dataset is poisoned.
# Users can assemble their data into a batch of shape [num_samples, n, w, h] and call `defense._detect(batch)`
# for online detection of the input.
# `preds_benign` contains the detection results for the original test dataset.
# `preds_poison` contains the detection results for the poisoned test dataset.
preds_benign = defense.detect(testset)
preds_poison = defense.detect(poisoned_testset)
print(f'Is poisoned for real benign batch: {preds_benign}')
print(f'Is poisoned for real poisoned batch: {preds_poison}')
