'''
Train an antoencoder network for antoencoder defense.
'''


import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip

import core


# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = '1'
datasets_root_dir = '../datasets'


# ========== AutoEncoder3x32x32_CIFAR-10 ==========
autoencoder_defense = core.AutoEncoderDefense(
    autoencoder=core.models.AutoEncoder(img_size=(3, 32, 32)),
    pretrain=None,
    seed=global_seed,
    deterministic=deterministic
)

dataset = torchvision.datasets.CIFAR10
transform_train = Compose([
    # RandomHorizontalFlip(),
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

# Train AutoEncoder Network (schedule is modified from https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder/blob/master/main.py)
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'batch_size': 16,
    'num_workers': 2,

    'lr': 0.001,
    'betas': (0.9, 0.999),
    'eps': 1e-08,
    'weight_decay': 0,
    'amsgrad': False,

    'schedule': [],
    'gamma': 0.1,

    'epochs': 100,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'AutoEncoder3x32x32_CIFAR-10_train'
}

autoencoder_defense.train_autoencoder(trainset, testset, schedule)




