import sys
import os
sys.path.append(os.getcwd())
import cv2
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, dataloader
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
import torchvision.models as models
import core
import argparse

parser = argparse.ArgumentParser(description='PyTorch Attack')
parser.add_argument('--gpu', default='0', type=str, choices=[str(x) for x in range(8)])
parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'ImageNet50']) 
parser.add_argument('--model', default='ResNet18', type=str, choices=['ResNet18', 'ResNet50', 'VGG16', 'InceptionV3', 'DenseNet121', 'ViT']) 
parser.add_argument('--attack', default='Adaptive', type=str, choices=['Benign', 'BadNets', 'Blended', 'WaNet', 'BATT', 'Physical', 'LC','Adaptive']) 

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
 
# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = args.gpu
dataset=args.dataset
model=args.model
attack=args.attack
datasets_root_dir = f'./data/{dataset}'
save_path = f'./{dataset}/{model}/{attack}'

if dataset == 'CIFAR10':
    img_size = 32
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    num_classes = 10
elif dataset == 'ImageNet50':
    img_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    num_classes = 50
else:
    raise NotImplementedError

input_size = img_size

if model == 'ResNet18':
    my_model = core.models.ResNet(18, num_classes=num_classes)
    if dataset == 'ImageNet50':
        my_model = models.resnet18(weights=None, num_classes=num_classes)
    lr = 0.1
elif model == 'ResNet50':
    my_model = core.models.ResNet(50, num_classes=num_classes)
    if dataset == 'ImageNet50':
        my_model = models.resnet50(pretrained=False, num_classes=num_classes)
    lr = 0.1
elif model == 'VGG16':
    deterministic = False
    my_model = core.models.vgg16(num_classes=num_classes)
    if dataset == 'ImageNet50':
        my_model = models.vgg16(pretrained=False, num_classes=num_classes)
    lr = 0.01
elif model == 'InceptionV3':
    my_model = models.inception_v3(pretrained=False, num_classes=num_classes, aux_logits = False)   # 299*299
    if dataset == 'CIFAR10':
        input_size = 96
    lr = 0.1
elif model == 'DenseNet121':
    my_model = models.densenet121(pretrained=False, num_classes=num_classes)    # 224*224
    lr = 0.1
elif model == 'ViT':
    # my_model = models.vit_b_16(weights=None, num_classes=num_classes)    # 224*224
    my_model = core.models.ViT(
        image_size = img_size,
        patch_size = int(img_size / 8),
        num_classes = num_classes,
        dim = int(512),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    lr = 1e-3
    # input_size = 224
else:
    raise NotImplementedError

my_model = my_model.to('cuda' if torch.cuda.is_available() else 'cpu')
 
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(input_size),
    transforms.Normalize(mean, std),
])
 
trainset = DatasetFolder(root=os.path.join(datasets_root_dir, 'train'),
                         transform=transform_train,
                         loader=cv2.imread,
                         extensions=('png','jpeg',),
                         target_transform=None,
                         is_valid_file=None,
                         )
 
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(input_size),
    transforms.Normalize(mean, std),
])
 
testset = DatasetFolder(root=os.path.join(datasets_root_dir, 'test'),
                         transform=transform_test,
                         loader=cv2.imread,
                         extensions=('png','jpeg',),
                         target_transform=None,
                         is_valid_file=None,
                         )

trigger_dir = './adaptive_triggers'
trigger_names = [
    f'phoenix_corner_{img_size}.png',
    f'badnet_patch4_{img_size}.png',
    f'firefox_corner_{img_size}.png',
    f'trojan_square_{img_size}.png',
]
trigger_path = [os.path.join(trigger_dir, name) for name in trigger_names]

patterns = []
for path in trigger_path:
    pattern = cv2.imread(path)
    pattern = transforms.ToTensor()(pattern)
    patterns.append(pattern)

alphas = [0.5, 0.5, 0.2, 0.3]
if model == 'ResNet18':
    poisoned_rate = 0.01
    covered_rate = 0.02
elif model in ['VGG16', 'DenseNet121', 'ViT']:
    poisoned_rate = 0.03
    covered_rate = 0.06

attacker = core.AdaptivePatch(
    train_dataset=trainset,
    test_dataset=testset,
    model=my_model,
    loss=nn.CrossEntropyLoss(),
    y_target=0,
    poisoned_rate=poisoned_rate,
    covered_rate=covered_rate,
    patterns=patterns,
    alphas=alphas,
    seed=global_seed,
    deterministic=deterministic,
)


benign_training = False
if attack == 'Benign':
    benign_training = True

# Train Attacked Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,
 
    'benign_training': benign_training,
    'batch_size': 128,
    'num_workers': 8,
 
    'lr': lr,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [100, 130],
 
    'epochs': 150,
 
    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 20,
 
    'save_dir': save_path,
    'experiment_name': f'Normalize_{model}_{dataset}_{attack}'
}
attacker.train(schedule)
