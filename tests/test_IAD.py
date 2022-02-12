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
deterministic = True
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
    

# ===== Train backdoored model on GTSRB using with DatasetFolder ======

# Prepare datasets and follow the default data augmentation in the original paper
transform_train = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.RandomCrop((32, 32), padding=5),
    transforms.RandomRotation(10),
    ToTensor()
])
transform_test = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor()
])

trainset = DatasetFolder(
    root='/data/gaokuofeng/datasets/GTSRB/train', # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
trainset1 = DatasetFolder(
    root='/data/gaokuofeng/datasets/GTSRB/train', # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
testset = DatasetFolder(
    root='/data/gaokuofeng/datasets/GTSRB/testset', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
testset1 = DatasetFolder(
    root='/data/gaokuofeng/datasets/GTSRB/testset', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
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
    'experiment_name': 'train_poison_DataFolder_GTSRB_IAD'
}


# Configure the attack scheme
IAD = core.IAD(
    dataset_name="gtsrb",
    train_dataset=trainset,
    test_dataset=testset,
    train_dataset1=trainset1,
    test_dataset1=testset1,
    model=core.models.ResNet(18, 43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,      # follow the default configure in the original paper
    cross_rate=0.1,         # follow the default configure in the original paper
    lambda_div=1,
    lambda_norm=100,
    mask_density=0.032,
    EPSILON=1e-7,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)


# Train backdoored model
IAD.train()


# ===== Train backdoored model on CIFAR10 using with CIFAR10 ===== 

# Prepare datasets and follow the default data augmentation in the original paper
transform_train = Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop((32, 32), padding=5),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.5),
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


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
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
    cross_rate=0.1,         # follow the default configure in the original paper
    lambda_div=1,
    lambda_norm=100,
    mask_density=0.032,
    EPSILON=1e-7,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)


# Train backdoored model
IAD.train()

# ===== Train backdoored model on CIFAR10 using with CIFAR10 (done)===== 

# ===== Train backdoored model on MNIST using with MNIST ===== 

# Prepare datasets and follow the default data augmentation in the original paper
transform_train = Compose([
    transforms.Resize((28, 28)),
    transforms.RandomCrop((28, 28), padding=5),
    ToTensor(),
    transforms.Normalize((0.5),
                    (0.5))
])
transform_test = Compose([
    transforms.Resize((28, 28)),
    ToTensor(),
    transforms.Normalize((0.5),
                    (0.5))
])

trainset = MNIST(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
trainset1 = MNIST(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
testset = MNIST(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)
testset1 = MNIST(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
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
    'experiment_name': 'train_poison_DataFolder_MNIST_IAD'
}


# Configure the attack scheme
IAD = core.IAD(
    dataset_name="mnist",
    train_dataset=trainset,
    test_dataset=testset,
    train_dataset1=trainset1,
    test_dataset1=testset1,
    model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.15,      # it may not reach the best if we follow the default configure in the original paper
    cross_rate=0.15,         # because we set the target label to 1
    lambda_div=1,
    lambda_norm=100,
    mask_density=0.032,
    EPSILON=1e-7,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)


# Train backdoored model
IAD.train()

# Get the victim model
victim_model = IAD.get_model()

# Get the backdoor trigger mask generator
modelM = IAD.get_modelM()

# Get the backdoor trigger pattern generator
modelG = IAD.get_modelG()

# Get the poisoned dataset
poisoned_train_dataset, poisoned_test_dataset = IAD.get_poisoned_dataset()

# Show an Example of Poisoned Training Samples
index = 44
x, y = poisoned_train_dataset[index]
print(y)
for a in x[0]:
    for b in a:
        print("%-4.2f" % float(b), end=' ')
    print()

# Show an Example of Poisoned Testing Samples
x, y = poisoned_test_dataset[index]
print(y)
for a in x[0]:
    for b in a:
        print("%-4.2f" % float(b), end=' ')
    print()

print("The length of poisoned train dataset is: ", len(poisoned_train_dataset))
print("The length of poisoned test dataset is: ", len(poisoned_test_dataset))

# ===== Train backdoored model on MNIST using with MNIST (done)===== 
