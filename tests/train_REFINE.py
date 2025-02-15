'''
Train an input transformation module for REFINE defense.
'''


import sys
import os
sys.path.append(os.getcwd())
import torch
import core
import argparse
from getmodel import GetModel
from getdataset import GetDataset
 
parser = argparse.ArgumentParser(description='PyTorch REFINE')
parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--model', default='ResNet18', type=str)
parser.add_argument('--attack', default='BadNets', type=str)
parser.add_argument('--tlabel', default=0, type=int)
 
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# ========== Set training model ==========
getmodel = GetModel(args.dataset, args.model, args.attack)
my_model = getmodel.get_model()

# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = args.gpu_id
save_path = f'REFINE/{args.dataset}/{args.model}/{args.attack}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

batch_size = 128
num_workers = 4

getdataset = GetDataset(args.dataset, args.model, args.attack, args.tlabel)
trainset, testset = getdataset.get_benign_dataset()
poisoned_trainset, poisoned_testset = getdataset.get_poisoned_dataset()

if args.dataset == 'CIFAR10':
    size_map = 32
    first_channel = 64
elif args.dataset == 'ImageNet50':
    size_map = 224
    first_channel = 32


defense = core.REFINE(
    unet=core.models.UNetLittle(args=None, n_channels=3, n_classes=3, first_channels=first_channel),
    model=my_model,
    num_classes=10,
    seed=global_seed,
    deterministic=deterministic
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    # 'test_model': model_path,
    'batch_size': batch_size,
    'num_workers': num_workers,

    'lr': 0.01,
    'betas': (0.9, 0.999),
    'eps': 1e-08,
    'weight_decay': 0,
    'amsgrad': False,

    'schedule': [100, 130],
    'gamma': 0.1,

    'epochs': 150,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': save_path,
    'experiment_name': f'REFINE_train'
}
defense.train_unet(trainset, testset, schedule)

unet_trained = defense.unet
arr_shuffle_defined = defense.arr_shuffle