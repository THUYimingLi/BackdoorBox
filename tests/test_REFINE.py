'''
This is the test code of REFINE defense. You need to train an input transformation module before testing.
You can refer to `train_REFINE.py` for more details.
'''

import sys
import os
sys.path.append(os.getcwd())
import torch
import core
import argparse
from getmodel import GetModel
from getdataset import GetDataset
 
parser = argparse.ArgumentParser(description='PyTorch ShrinkPad')
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

# If you have trained 'unet' and defined 'arr_shuffle' in core.REFINE, you can also get them (instead of loading them directly locally) as follows:
# unet_trained = defense.unet
# arr_shuffle_defined = defense.arr_shuffle

defense = core.REFINE(
    unet=core.models.UNetLittle(args=None, n_channels=3, n_classes=3, first_channels=first_channel),
    pretrain='/data/home/Yukun/BackdoorBox/REFINE/CIFAR10/ResNet18/BadNets/REFINE_train_2025-02-14_17:24:44/ckpt_epoch_150.pth',
    arr_path='/data/home/Yukun/BackdoorBox/REFINE/CIFAR10/ResNet18/BadNets/REFINE_train_2025-02-14_17:24:44/label_shuffle.pth',
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

    'metric': 'BA',

    'save_dir': save_path,
    'experiment_name': f'REFINE_BA'
}
defense.test(testset, schedule)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    # 'test_model': model_path,
    'batch_size': batch_size,
    'num_workers': num_workers,

    # 1. ASR: the attack success rate calculated on all poisoned samples
    # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
    # 3. BA: the accuracy on all benign samples
    # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
    # In other words, ASR or BA does not influence the computation of the metric.
    # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
    'metric': 'ASR_NoTarget',
    'y_target': args.tlabel,

    'save_dir': save_path,
    'experiment_name': f'REFINE_ASR'
}
defense.test(poisoned_testset, schedule)
