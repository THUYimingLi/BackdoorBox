'''
This is the test code of AutoEncoder defense.
'''


from copy import deepcopy
import os.path as osp

from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, ToPILImage, Resize

import core
from core.utils import any2tensor


# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
datasets_root_dir = '../datasets'
CUDA_VISIBLE_DEVICES = '1'
batch_size = 128
num_workers = 4
autoencoder_model_path = '/data/yamengxi/Backdoor/BackdoorBox/experiments/AutoEncoder3x32x32_CIFAR-10_train_2022-07-07_21:17:42/ckpt_epoch_100.pth'


def test(model_name, dataset_name, attack_name, defense_name, model, model_path, benign_dataset, attacked_dataset, defense, y_target):
    if dataset_name == 'CIFAR-10':
        data = any2tensor(benign_dataset.data)
        data = data.permute((0, 3, 1, 2))
        data = defense.preprocess(data.float() / 255)

        schedule = {
            'device': 'GPU',
            'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
            'GPU_num': 1,

            'test_model': model_path,
            'batch_size': batch_size,
            'num_workers': num_workers,
        }
        res = defense.predict(model, data.float() / 255, schedule)

    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'test_model': model_path,
        'batch_size': batch_size,
        'num_workers': num_workers,

        'metric': 'BA',

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_BA'
    }
    defense.test(model, benign_dataset, schedule)

    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'test_model': model_path,
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

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_ASR'
    }
    defense.test(model, attacked_dataset, schedule)


# ========== ResNet-18_CIFAR-10_Attack_AutoEncoder ==========
model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'CIFAR-10', 'Benign', 'AutoEncoder'

model = core.models.ResNet(18)

model_path = '/data/yamengxi/Backdoor/experiments/ResNet-18_CIFAR-10_Benign_2022-03-29_16:27:15/ckpt_epoch_200.pth'

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

defense = core.AutoEncoderDefense(
    autoencoder=core.models.AutoEncoder(img_size=(3, 32, 32)),
    pretrain=autoencoder_model_path,
    seed=global_seed,
    deterministic=deterministic
)
test(model_name, dataset_name, attack_name, defense_name, model, model_path, testset, testset, defense, None)


attack_name = 'BadNets'
model_path = '/data/yamengxi/Backdoor/experiments/ResNet-18_CIFAR-10_BadNets_2022-03-29_16:27:40/ckpt_epoch_200.pth'

pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0

attack = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()
test(model_name, dataset_name, attack_name, defense_name, model, model_path, testset, poisoned_testset, defense, 1)


attack_name = 'WaNet'
model_path = '/data/yamengxi/Backdoor/experiments/ResNet-18_CIFAR-10_WaNet_2022-03-29_19:18:08/ckpt_epoch_200.pth'

identity_grid, noise_grid = torch.load('/data/yamengxi/Backdoor/experiments/ResNet-18_CIFAR-10_WaNet_identity_grid.pth'), torch.load('/data/yamengxi/Backdoor/experiments/ResNet-18_CIFAR-10_WaNet_noise_grid.pth')
attack = core.WaNet(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=0,
    poisoned_rate=0.1,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=False,
    seed=global_seed,
    deterministic=deterministic
)
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()
test(model_name, dataset_name, attack_name, defense_name, model, model_path, testset, poisoned_testset, defense, 0)


attack_name = 'LabelConsistent'
model_path = '/data/yamengxi/Backdoor/experiments/ResNet-18_CIFAR-10_LabelConsistent_2022-03-30_01:20:03/ckpt_epoch_200.pth'

adv_model = deepcopy(model)
adv_ckpt = torch.load('/data/yamengxi/Backdoor/experiments/ResNet-18_CIFAR-10_Benign_2022-03-29_16:27:15/ckpt_epoch_200.pth')
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

attack = core.LabelConsistent(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    adv_model=adv_model,
    adv_dataset_dir=f'./adv_dataset/CIFAR-10_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
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
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()
test(model_name, dataset_name, attack_name, defense_name, model, model_path, testset, poisoned_testset, defense, 2)


# ========== ResNet-18_GTSRB_Attack_AutoEncoder ==========
model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'GTSRB', 'Benign', 'AutoEncoder'

model = core.models.ResNet(18, 43)

model_path = '/data/yamengxi/Backdoor/experiments/ResNet-18_GTSRB_Benign_2022-03-29_19:59:05/ckpt_epoch_30.pth'

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

defense = core.AutoEncoderDefense(
    autoencoder=core.models.AutoEncoder(img_size=(3, 32, 32)),
    pretrain=autoencoder_model_path,
    seed=global_seed,
    deterministic=deterministic
)
test(model_name, dataset_name, attack_name, defense_name, model, model_path, testset, testset, defense, None)


attack_name = 'BadNets'
model_path = '/data/yamengxi/Backdoor/experiments/ResNet-18_GTSRB_BadNets_2022-03-29_19:57:40/ckpt_epoch_30.pth'

pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0

attack = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    pattern=pattern,
    weight=weight,
    poisoned_transform_train_index=2,
    poisoned_transform_test_index=2,
    seed=global_seed,
    deterministic=deterministic
)
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()
test(model_name, dataset_name, attack_name, defense_name, model, model_path, testset, poisoned_testset, defense, 1)


attack_name = 'WaNet'
model_path = '/data/yamengxi/Backdoor/experiments/ResNet-18_GTSRB_WaNet_2022-03-29_16:18:05/ckpt_epoch_200.pth'

transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip(),
    ToPILImage(),
    Resize((32, 32)),
    ToTensor()
])
transform_test = Compose([
    ToTensor(),
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
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB', 'testset'), # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

identity_grid, noise_grid = torch.load('/data/yamengxi/Backdoor/experiments/ResNet-18_GTSRB_WaNet_identity_grid.pth'), torch.load('/data/yamengxi/Backdoor/experiments/ResNet-18_GTSRB_WaNet_noise_grid.pth')
attack = core.WaNet(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=0,
    poisoned_rate=0.1,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=True,
    seed=global_seed,
    deterministic=deterministic
)
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()
test(model_name, dataset_name, attack_name, defense_name, model, model_path, testset, poisoned_testset, defense, 0)


attack_name = 'LabelConsistent'
model_path = '/data/yamengxi/Backdoor/experiments/ResNet-18_GTSRB_LabelConsistent_2022-03-30_06:05:46/ckpt_epoch_50.pth'

transform_train = Compose([
    ToPILImage(),
    Resize((32, 32)),
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

adv_model = deepcopy(model)
adv_ckpt = torch.load('/data/yamengxi/Backdoor/experiments/ResNet-18_GTSRB_Benign_2022-03-29_19:59:05/ckpt_epoch_30.pth')
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

eps = 16
alpha = 1.5
steps = 100
max_pixel = 255
poisoned_rate = 0.5

attack = core.LabelConsistent(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    adv_model=adv_model,
    adv_dataset_dir=f'./adv_dataset/GTSRB_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
    loss=nn.CrossEntropyLoss(),
    y_target=2,
    poisoned_rate=poisoned_rate,
    adv_transform=Compose([ToPILImage(), Resize((32, 32)), ToTensor()]),
    pattern=pattern,
    weight=weight,
    eps=eps,
    alpha=alpha,
    steps=steps,
    max_pixel=max_pixel,
    poisoned_transform_train_index=2,
    poisoned_transform_test_index=2,
    poisoned_target_transform_index=0,
    schedule=schedule,
    seed=global_seed,
    deterministic=True
)
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()
test(model_name, dataset_name, attack_name, defense_name, model, model_path, testset, poisoned_testset, defense, 2)
