
'''
This is the test code of poisoned training on GTSRB, CIFAR10, MNIST, using dataset class of torchvision.datasets.DatasetFolder torchvision.datasets.CIFAR10 torchvision.datasets.MNIST.
The attack method is IAD.
'''


import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
import core


global_seed = 666
deterministic = False
torch.manual_seed(global_seed)


def read_image(img_path, type=None):
    img = cv2.imread(img_path)
    if type is None:        
        return img
    elif isinstance(type,str) and type.upper() == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(type,str) and type.upper() == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise NotImplementedError
    

# # ===== Train backdoored model on GTSRB using with DatasetFolder ======

# # Prepare datasets
# transform_train = Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((32, 32)),
#     RandomHorizontalFlip(),
#     ToTensor()
# ])
# transform_test = Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((32, 32)),
#     ToTensor()
# ])

# trainset = DatasetFolder(
#     root='/data/gaokuofeng/datasets/GTSRB/train', # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_train,
#     target_transform=None,
#     is_valid_file=None)

# testset = DatasetFolder(
#     root='/data/gaokuofeng/datasets/GTSRB/testset', # please replace this with path to your test set
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)
# # Configure the attack scheme
# IAD = core.IAD(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=core.models.ResNet(18, 43),
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.05,
#     poisoned_transform_train_index=0,
#     poisoned_transform_test_index=0,
#     poisoned_target_transform_index=0,
#     schedule=None,
#     seed=global_seed,
#     deterministic=deterministic
# )


# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '0',
#     'GPU_num': 1,

#     'benign_training': False,
#     'batch_size': 128,
#     'num_workers': 8,

#     'lr': 0.01,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'milestones': [100, 200, 300, 400],
#     'lambda': 0.1,
    
#     'lr_G': 0.01,
#     'betas_G': (0.5, 0.9),
#     'milestones_G': [200, 300, 400, 500],
#     'lambda_G': 0.1,

#     'lr_M': 0.01,
#     'betas_M': (0.5, 0.9),
#     'milestones_M': [10, 20],
#     'lambda_M': 0.1,
    
#     'epochs': 600,
#     'epochs_M': 25,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 10,

#     'save_dir': 'experiments',
#     'experiment_name': 'train_poison_DataFolder_GTSRB_IAD'
# }

# # Train backdoored model
# refool.train(schedule)

# # ===== Train backdoored model on GTSRB using with DatasetFolder (done) ======

# ===== Train backdoored model on CIFAR10 using with CIFAR10 ===== 

# Prepare datasets
transform_train = Compose([
    transforms.Resize((32, 32)),
    RandomHorizontalFlip(),
    ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.247, 0.243, 0.261))
])
transform_test = Compose([
    transforms.Resize((32, 32)),
    ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.247, 0.243, 0.261))
])
trainset = CIFAR10(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
trainset1 = CIFAR10(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
testset = CIFAR10(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)
testset1 = CIFAR10(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)


# Configure the attack scheme
IAD = core.IAD(
    dataset_name="cifar10",
    train_dataset=trainset,
    test_dataset=testset,
    train_dataset1=trainset1,
    test_dataset1=testset1,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,      # follow the default configure in the original paper
    cross_rate=0.1,
    lambda_div=1,
    lambda_norm=100,
    mask_density=0.032,
    EPSILON=1e-7,
    schedule=None,
    seed=global_seed,
    deterministic=deterministic
)


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'milestones': [100, 200, 300, 400],
    'lambda': 0.1,
    
    'lr_G': 0.01,
    'betas_G': (0.5, 0.9),
    'milestones_G': [200, 300, 400, 500],
    'lambda_G': 0.1,

    'lr_M': 0.01,
    'betas_M': (0.5, 0.9),
    'milestones_M': [10, 20],
    'lambda_M': 0.1,
    
    'epochs': 600,
    'epochs_M': 25,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_DataFolder_CIFAR10_IAD'
}


# Train backdoored model
IAD.train(schedule)

# ===== Train backdoored model on CIFAR10 using with CIFAR10 (done)===== 

# # ===== Train backdoored model on MNIST using with MNIST ===== 
# # Prepare datasets
# transform_train = Compose([
#     transforms.Resize((28, 28)),
#     RandomHorizontalFlip(),
#     ToTensor(),
# ])
# transform_test = Compose([
#     transforms.Resize((28, 28)),
#     ToTensor(),
# ])
# trainset = MNIST(
#     root='/data/ganguanhao/datasets', # please replace this with path to your dataset
#     transform=transform_train,
#     target_transform=None,
#     train=True,
#     download=True)
# testset = MNIST(
#     root='/data/ganguanhao/datasets', # please replace this with path to your dataset
#     transform=transform_test,
#     target_transform=None,
#     train=False,
#     download=True)

# loader = DataLoader(trainset,)


# # Configure the attack scheme
# refool= core.Refool(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=core.models.BaselineMNISTNetwork(),
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.05,
#     poisoned_transform_train_index=0,
#     poisoned_transform_test_index=0,
#     poisoned_target_transform_index=0,
#     schedule=None,
#     seed=global_seed,
#     deterministic=deterministic,
#     reflection_candidates = reflection_images,
# )

# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '0',
#     'GPU_num': 1,

#     'benign_training': False,
#     'batch_size': 128,
#     'num_workers': 8,

#     'lr': 0.1,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [10, 15],

#     'epochs': 20,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 5,
#     'save_epoch_interval': 5,

#     'save_dir': 'experiments',
#     'experiment_name': 'train_poison_MNIST_Refool'
# }
# # Train backdoored model
# refool.train(schedule)
# # ===== Train backdoored model on MNIST using with MNIST (done)===== 
