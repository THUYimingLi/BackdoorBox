'''
This is the test code of MCR defense.
TODO: training with part of the benign test samples.
'''

from copy import deepcopy
from email.mime import base
import os.path as osp
from tracemalloc import start

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
from core.models import curves
from core.models.resnet_curve import *
from core.models.vgg_curve import *

# ========== Set global settings ==========
global_seed = 666
deterministic = False             #TODO: 'deterministic=True' works on 3th, May. But it meets a Runtime Error on 5th, May.
torch.manual_seed(global_seed)
datasets_root_dir = '../datasets'
CUDA_VISIBLE_DEVICES = '7'
batch_size = 128
num_workers = 8

# ========== MCR settings ===============
fix_start = True
fix_end = True
num_bends = 3
fix_points = [fix_start] + [False] * (num_bends - 2) + [fix_end]
curve_type =  'Bezier'            # 'PolyChain' | 'Bezier'   
portion = 0.1                     # portion of training dataset to train the curve model.
l2_regularizer = True             # whether use l2-regularizer
coeffs_t = np.arange(0, 1.1, 0.1) # float or list. hyperparam for MCR testing, in range(0,1)
initialize_curve = True

settings = dict(
    global_seed=global_seed,
    deterministic = deterministic,
    datasets_root_dir  = datasets_root_dir,
    CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES,
    batch_size = batch_size,
    num_workers = num_workers,
    fix_start = fix_start,
    fix_end = fix_end,
    num_bends = num_bends,
    fix_points = fix_points,
    curve_type = curve_type,
    portion = portion,
    l2_regularizer = l2_regularizer,
    coeffs_t = coeffs_t,
    initialize_curve = initialize_curve
)

def test_model_without_defense(start_model, end_model, model_name, dataset_name, attack_name, benign_dataset, attacked_dataset, y_target):
    """test BA and ASR before MCR.

    Args:
        start_model (nn.Module): Start model of MCR, params needs to be loaded beforehand.
        end_model (nn.Module): End model of MCR, params needs to be loaded beforehand.
        model_name (str): name of the network
        dataset_name (str): name of the dataset
        attack_name (str): name of attack method
        benign_dataset (types in support_list): Benign dataset.
        attacked_dataset (types in support_list): Attacked dataset.
        y_target (int or None): target attack label.
    """

    print("===> Start testing start model and end model before MCR")
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        'metric': 'BA',

        'save_dir': 'experiments/MCR-defense',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_MCR_Start_BA'
    }
    core.utils.test(start_model, benign_dataset, schedule)

    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        'metric': 'ASR_NoTarget',
        'y_target': 0,

        'save_dir': 'experiments/MCR-defense',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_MCR_Start_ASR'
    }
    core.utils.test(start_model, attacked_dataset, schedule)

    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        'metric': 'BA',

        'save_dir': 'experiments/MCR-defense',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_MCR_End_BA'
    }
    core.utils.test(end_model, benign_dataset, schedule)

    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        'metric': 'ASR_NoTarget',
        'y_target': 0,

        'save_dir': 'experiments/MCR-defense',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_MCR_End_ASR'
    }
    core.utils.test(end_model, attacked_dataset, schedule)


def test(model_name, dataset_name, attack_name, defense_name, benign_dataset, attacked_dataset, defense, y_target, portion, coeffs_t):
    """test BA and ASR after MCR.
    
    Args:
        model_name (str): name of the network
        dataset_name (str): name of the dataset
        attack_name (str): name of the attack method
        defense_name (str): name of the defense method
        benign_dataset (types in support_list): Benign dataset.
        attacked_dataset (types in support_list): Attacked dataset.
        defense (core.defense.Base): object of defense class.
        y_target (int or None): target attack label.
        portion (float): proportion of the training set for model repairing training.
        coeffs_t (float or list): Hyperparam for the curve, in range(0,1).
    """
    print(f"===> Start testing repaired Model from {defense_name}")
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        'metric': 'BA',

        'save_dir': 'experiments/MCR-defense',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_{curve_type}_p{portion}_BA'
    }
    defense.test(benign_dataset, schedule, coeffs_t)

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

        'save_dir': 'experiments/MCR-defense',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_{curve_type}_p{portion}_ASR'
    }
    defense.test(attacked_dataset, schedule, coeffs_t)




# ========== ResNet-18_CIFAR-10_Attack_MCR ========== #
# 1. Benign
model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'CIFAR-10', 'Benign', 'MCR'
print(f"===> Running {defense_name} on {dataset_name}, {attack_name}, {model_name}")

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

start_model = core.models.ResNet(18)
start_model_path = '/data/yamengxi/Backdoor/experiments/ResNet-18_CIFAR-10_Benign_2022-03-29_16:27:15/ckpt_epoch_200.pth'
start_model.load_state_dict(torch.load(start_model_path), strict=False)

end_model = core.models.ResNet(18)
end_model_path = '/data/zhonghaoxiang/BackdoorBox/experiments/ResNet-18_CIFAR-10_Benign_2022-04-29_10:38:25/ckpt_epoch_200.pth'
end_model.load_state_dict(torch.load(end_model_path), strict=False)

test_model_without_defense(start_model, end_model, model_name, dataset_name, attack_name, testset, testset, None)

base_model = ResNetCurve(18, fix_points, initialize=initialize_curve)

# use "pretrained" during MCR initialize, you can load a trained MCR model and avoid training during repairing.
# although "defense.repair()" is still required to be called for possible BN layer handling.
defense = core.MCR(
    start_model,
    end_model,
    base_model,
    num_bends,
    curve_type,
    loss=nn.CrossEntropyLoss(),
    fix_start=fix_start,
    fix_end=fix_end,
    init_linear=True,
    # pretrained = '/data/zhonghaoxiang/BackdoorBox/experiments/MCR/ResNet-18_CIFAR-10_Benign_MCR_Bezier_p0.1_2022-05-03_18:04:27/ckpt_epoch_600.pth',
    seed=global_seed,
    deterministic=deterministic
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': True,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.03,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule':  [450, 540], 

    'l2_regularizer': l2_regularizer,

    'epochs': 600,

    'log_iteration_interval': 20,
    'test_epoch_interval': 20,
    'save_epoch_interval': 20,

    'save_dir': 'experiments/MCR-defense',
    'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_{curve_type}_p{portion}'
}

# 'settings' could be ignored. Only for logging purpose.
defense.repair(dataset=trainset, portion=portion, schedule=schedule, settings=settings)
repaired_model = defense.get_model()
test(model_name, dataset_name, attack_name, defense_name, testset, testset, defense, None, portion, coeffs_t)


# 2. BadNet
attack_name = 'BadNets'
print(f"===> Running {defense_name} on {dataset_name}, {attack_name}, {model_name}")

start_model = core.models.ResNet(18)
start_model_path = '/data/zhonghaoxiang/BackdoorBox/experiments/ResNet-18_CIFAR-10_BadNets_2022-04-28_16:06:15/ckpt_epoch_200.pth'
start_model.load_state_dict(torch.load(start_model_path), strict=False)

end_model = core.models.ResNet(18)
end_model_path = '/data/zhonghaoxiang/BackdoorBox/experiments/ResNet-18_CIFAR-10_BadNets_2022-04-29_15:54:24/ckpt_epoch_200.pth'
end_model.load_state_dict(torch.load(end_model_path), strict=False)

base_model = ResNetCurve(18, fix_points, initialize=initialize_curve)

pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0

attack = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=start_model,
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()

test_model_without_defense(start_model, end_model, model_name, dataset_name, attack_name, testset, poisoned_testset, 1)

defense = core.MCR(
    start_model,
    end_model,
    base_model,
    num_bends,
    curve_type,
    loss=nn.CrossEntropyLoss(),
    fix_start=fix_start,
    fix_end=fix_end,
    init_linear=True,
    # pretrained="/data/zhonghaoxiang/BackdoorBox/experiments/MCR/ResNet-18_CIFAR-10_BadNets_MCR_Bezier_p0.1_2022-05-03_19:16:25/ckpt_epoch_600.pth",
    seed=global_seed,
    deterministic=deterministic
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': True,
    'batch_size': 128,
    'num_workers': 16,

    'lr': 0.03,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [450, 540],

    'l2_regularizer': l2_regularizer,

    'epochs': 600,

    'log_iteration_interval': 20,
    'test_epoch_interval': 50,
    'save_epoch_interval': 50,

    'save_dir': 'experiments/MCR-defense',
    'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_{curve_type}_p{portion}'
}

defense.repair(dataset=trainset, portion=portion, schedule=schedule, settings=settings)
repaired_model = defense.get_model()
test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, 1, portion, coeffs_t)


#3. LabelConsistant
attack_name = 'LabelConsistent'
print(f"===> Running {defense_name} on {dataset_name}, {attack_name}, {model_name}")

start_model = core.models.ResNet(18)
start_model_path = '/data/yamengxi/Backdoor/experiments/ResNet-18_CIFAR-10_LabelConsistent_2022-03-30_01:20:03/ckpt_epoch_200.pth'
start_model.load_state_dict(torch.load(start_model_path), strict=False)

end_model = core.models.ResNet(18)
end_model_path = '/data/zhonghaoxiang/BackdoorBox/experiments/ResNet-18_CIFAR-10_LabelConsistent_2022-04-28_17:50:23/ckpt_epoch_200.pth'
end_model.load_state_dict(torch.load(end_model_path), strict=False)

base_model = ResNetCurve(18, fix_points, initialize=initialize_curve)


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

    'log_iteration_interval': 20,
    'test_epoch_interval': 50,
    'save_epoch_interval': 50,

    'save_dir': 'experiments/MCR-defense',
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
    model=start_model,
    adv_model=start_model,
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
    deterministic=deterministic
)
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()

test_model_without_defense(start_model, end_model, model_name, dataset_name, attack_name, testset, poisoned_testset, 2)

defense = core.MCR(
    start_model,
    end_model,
    base_model,
    num_bends,
    curve_type,
    loss=nn.CrossEntropyLoss(),
    fix_start=fix_start,
    fix_end=fix_end,
    init_linear=True,
    # pretrained="/data/zhonghaoxiang/BackdoorBox/experiments/MCR/ResNet-18_CIFAR-10_LabelConsistent_MCR_Bezier_p0.1_2022-05-03_20:31:42/ckpt_epoch_600.pth",
    seed=global_seed,
    deterministic=deterministic
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': True,
    'batch_size': 128,
    'num_workers': 16,

    'lr': 0.03,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [450, 540],

    'l2_regularizer': l2_regularizer,

    'epochs': 600,

    'log_iteration_interval': 20,
    'test_epoch_interval': 50,
    'save_epoch_interval': 50,

    'save_dir': 'experiments/MCR-defense',
    'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_{curve_type}_p{portion}'
}

defense.repair(dataset=trainset, portion=portion, schedule=schedule, settings=settings)
repaired_model = defense.get_model()
test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, 2, portion, coeffs_t)


#4. WaNet
attack_name = 'WaNet'
print(f"===> Running {defense_name} on {dataset_name}, {attack_name}, {model_name}")

start_model = core.models.ResNet(18)
start_model_path = '/data/zhonghaoxiang/BackdoorBox/experiments/ResNet-18_CIFAR-10_WaNet_2022-04-29_18:18:21/ckpt_epoch_200.pth'
start_model.load_state_dict(torch.load(start_model_path), strict=False)

end_model = core.models.ResNet(18)
end_model_path = '/data/zhonghaoxiang/BackdoorBox/experiments/ResNet-18_CIFAR-10_WaNet_2022-04-30_12:16:19/ckpt_epoch_200.pth'
end_model.load_state_dict(torch.load(start_model_path), strict=False)

base_model = ResNetCurve(18, fix_points, initialize=initialize_curve)

identity_grid, noise_grid = torch.load('/data/zhonghaoxiang/BackdoorBox/experiments/ResNet-18_CIFAR-10_WaNet_identity_grid.pth'), torch.load('/data/zhonghaoxiang/BackdoorBox/experiments/ResNet-18_CIFAR-10_WaNet_noise_grid.pth')
attack = core.WaNet(
    train_dataset=trainset,
    test_dataset=testset,
    model=start_model,
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

test_model_without_defense(start_model, end_model, model_name, dataset_name, attack_name, testset, poisoned_testset, 0)

defense = core.MCR(
    start_model,
    end_model,
    base_model,
    num_bends,
    curve_type,
    loss=nn.CrossEntropyLoss(),
    fix_start=fix_start,
    fix_end=fix_end,
    init_linear=True,
    # pretrained="/data/zhonghaoxiang/BackdoorBox/experiments/MCR/ResNet-18_CIFAR-10_WaNet_MCR_Bezier_p0.1_2022-05-03_21:46:36/ckpt_epoch_600.pth",
    seed=global_seed,
    deterministic=deterministic
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': True,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.03,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [450, 540],

    'l2_regularizer': l2_regularizer,

    'epochs': 600,

    'log_iteration_interval': 20,
    'test_epoch_interval': 50,
    'save_epoch_interval': 50,

    'save_dir': 'experiments/MCR-defense',
    'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_{curve_type}_p{portion}'
}

defense.repair(dataset=trainset, portion=portion, schedule=schedule, settings=settings)
repaired_model = defense.get_model()
test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, 0, portion, coeffs_t)




# ========== ResNet-18_GTSRB_Attack_MCR ========== #
# 1. Benign
model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'GTSRB', 'Benign', 'MCR'
print(f"===> Running {defense_name} on {dataset_name}, {attack_name}, {model_name}")

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

start_model = core.models.ResNet(18, num_classes=43)
start_model_path = '/data/yamengxi/Backdoor/experiments/ResNet-18_GTSRB_Benign_2022-03-29_19:59:05/ckpt_epoch_30.pth'
start_model.load_state_dict(torch.load(start_model_path), strict=False)

end_model = core.models.ResNet(18, num_classes=43)
end_model_path = '/data/zhonghaoxiang/BackdoorBox/experiments/ResNet-18_GTSRB_Benign_2022-04-29_16:22:16/ckpt_epoch_30.pth'
end_model.load_state_dict(torch.load(end_model_path), strict=False)

test_model_without_defense(start_model, end_model, model_name, dataset_name, attack_name, testset, testset, None)

base_model = ResNetCurve(18, fix_points, num_classes=43, initialize=initialize_curve)

defense = core.MCR(
    start_model,
    end_model,
    base_model,
    num_bends,
    curve_type,
    loss=nn.CrossEntropyLoss(),
    fix_start=fix_start,
    fix_end=fix_end,
    init_linear=True,
    # pretrained="/data/zhonghaoxiang/BackdoorBox/experiments/MCR/ResNet-18_GTSRB_Benign_MCR_Bezier_p0.1_2022-05-03_19:58:51/ckpt_epoch_100.pth",
    seed=global_seed,
    deterministic=deterministic
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': True,
    'batch_size': 128,  #128,
    'num_workers': 8, #16,

    'lr': 0.03, #
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [50,],

    'epochs': 100,

    'l2_regularizer': l2_regularizer,

    'log_iteration_interval': 20,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments/MCR-defense',
    'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_{curve_type}_p{portion}'
}

defense.repair(dataset=trainset, portion=portion, schedule=schedule)
repaired_model = defense.get_model()
test(model_name, dataset_name, attack_name, defense_name, testset, testset, defense, None, portion, coeffs_t)


#2. BadNets
attack_name = "BadNets"
print(f"===> Running {defense_name} on {dataset_name}, {attack_name}, {model_name}")

start_model = core.models.ResNet(18, 43)
start_model_path = '/data/zhonghaoxiang/BackdoorBox/experiments/ResNet-18_GTSRB_BadNets_2022-04-28_19:37:09/ckpt_epoch_30.pth'
start_model.load_state_dict(torch.load(start_model_path), strict=False)

end_model = core.models.ResNet(18, 43)
end_model_path = '/data/zhonghaoxiang/BackdoorBox/experiments/ResNet-18_GTSRB_BadNets_2022-04-29_19:28:21/ckpt_epoch_30.pth'
end_model.load_state_dict(torch.load(end_model_path), strict=False)

base_model = ResNetCurve(18, fix_points, num_classes=43, initialize=initialize_curve)

pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0

attack = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=start_model,
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

test_model_without_defense(start_model, end_model, model_name, dataset_name, attack_name, testset, poisoned_testset, 1)

defense = core.MCR(
    start_model,
    end_model,
    base_model,
    num_bends,
    curve_type,
    loss=nn.CrossEntropyLoss(),
    fix_start=fix_start,
    fix_end=fix_end,
    init_linear=True,
    # pretrained="/data/zhonghaoxiang/BackdoorBox/experiments/MCR/ResNet-18_GTSRB_BadNets_MCR_Bezier_p0.1_2022-05-03_20:10:41/ckpt_epoch_100.pth",
    seed=global_seed,
    deterministic=deterministic
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': True,
    'batch_size': 128,  #128,
    'num_workers': 8, #16,

    'lr': 0.03, #
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [50,],

    'epochs': 100,

    'l2_regularizer': l2_regularizer,

    'log_iteration_interval': 20,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments/MCR-defense',
    'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_{curve_type}_p{portion}'
}

defense.repair(dataset=trainset, portion=portion, schedule=schedule)
repaired_model = defense.get_model()
test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, 1, portion, coeffs_t)


# 3. LabelConsistent
attack_name = 'LabelConsistent'
print(f"===> Running {defense_name} on {dataset_name}, {attack_name}, {model_name}")

start_model = core.models.ResNet(18, 43)
start_model_path = '/data/yamengxi/Backdoor/experiments/ResNet-18_GTSRB_LabelConsistent_2022-03-30_06:05:46/ckpt_epoch_50.pth'
start_model.load_state_dict(torch.load(start_model_path), strict=False)

end_model = core.models.ResNet(18, 43)
end_model_path = '/data/zhonghaoxiang/BackdoorBox/experiments/ResNet-18_GTSRB_LabelConsistent_2022-04-28_22:49:39/ckpt_epoch_50.pth'
end_model.load_state_dict(torch.load(end_model_path), strict=False)

base_model = ResNetCurve(18, fix_points, 43, initialize=initialize_curve)

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
    model=start_model,
    adv_model=start_model,
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
    deterministic=deterministic
)
poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()

test_model_without_defense(start_model, end_model, model_name, dataset_name, attack_name, testset, poisoned_testset, 2)

defense = core.MCR(
    start_model,
    end_model,
    base_model,
    num_bends,
    curve_type,
    loss=nn.CrossEntropyLoss(),
    fix_start=fix_start,
    fix_end=fix_end,
    init_linear=True,
    # pretrained="/data/zhonghaoxiang/BackdoorBox/experiments/MCR/ResNet-18_GTSRB_LabelConsistent_MCR_Bezier_p0.1_2022-05-03_20:22:37/ckpt_epoch_100.pth",
    seed=global_seed,
    deterministic=deterministic
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': True,
    'batch_size': 128,
    'num_workers': 16,

    'lr': 0.03,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [50,],

    'l2_regularizer': l2_regularizer,

    'epochs': 100,

    'log_iteration_interval': 20,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments/MCR-defense',
    'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_{curve_type}_p{portion}'
}

defense.repair(dataset=trainset, portion=portion, schedule=schedule, settings=settings)
repaired_model = defense.get_model()
test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, 2, portion, coeffs_t)


# 4. WaNet
attack_name = 'WaNet'
print(f"===> Running {defense_name} on {dataset_name}, {attack_name}, {model_name}")

start_model = core.models.ResNet(18, 43)
start_model_path = '/data/zhonghaoxiang/BackdoorBox/experiments/ResNet-18_GTSRB_WaNet_2022-04-29_15:30:11/ckpt_epoch_200.pth'
start_model.load_state_dict(torch.load(start_model_path), strict=False)

end_model = core.models.ResNet(18, 43)
end_model_path = '/data/zhonghaoxiang/BackdoorBox/experiments/ResNet-18_GTSRB_WaNet_2022-04-30_09:26:54/ckpt_epoch_200.pth'
end_model.load_state_dict(torch.load(end_model_path), strict=False)

base_model = ResNetCurve(18, fix_points, 43, initialize=initialize_curve)

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

identity_grid, noise_grid = torch.load('/data/zhonghaoxiang/BackdoorBox/experiments/ResNet-18_GTSRB_WaNet_identity_grid.pth'), torch.load('/data/zhonghaoxiang/BackdoorBox/experiments/ResNet-18_GTSRB_WaNet_noise_grid.pth')
attack = core.WaNet(
    train_dataset=trainset,
    test_dataset=testset,
    model=start_model,
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

test_model_without_defense(start_model, end_model, model_name, dataset_name, attack_name, testset, poisoned_testset, 0)

defense = core.MCR(
    start_model,
    end_model,
    base_model,
    num_bends,
    curve_type,
    loss=nn.CrossEntropyLoss(),
    fix_start=fix_start,
    fix_end=fix_end,
    init_linear=True,
    # pretrained="/data/zhonghaoxiang/BackdoorBox/experiments/MCR/ResNet-18_GTSRB_WaNet_MCR_Bezier_p0.1_2022-05-03_20:35:03/ckpt_epoch_100.pth",
    seed=global_seed,
    deterministic=deterministic
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': True,
    'batch_size': 128,
    'num_workers': 16,

    'lr': 0.03,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [50,],

    'l2_regularizer': l2_regularizer,

    'epochs': 100,

    'log_iteration_interval': 20,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments/MCR-defense',
    'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_{curve_type}_p{portion}'
}

defense.repair(dataset=trainset, portion=portion, schedule=schedule, settings=settings)
repaired_model = defense.get_model()
test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, 0, portion, coeffs_t)

