'''
This is the test code of poisoned training on GTSRB, CIFAR10, MNIST, using dataset class of torchvision.datasets.DatasetFolder torchvision.datasets.CIFAR10 torchvision.datasets.MNIST.
The attack method is ISSBA.
'''


import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
import numpy as np
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


class GetPoisonedDataset(torch.utils.data.Dataset):
    """Construct a dataset.

    Args:
        data_list (list): the list of data.
        labels (list): the list of label.
    """
    def __init__(self, data_list, labels):
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = torch.FloatTensor(self.data_list[index])
        label = torch.FloatTensor(self.labels[index])
        return img, label


# ===== Train backdoored model on GTSRB using with GTSRB ===== 

# Prepare datasets and follow the default data augmentation in the original paper
transform_train = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor(),
])
transform_test = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor(),
])

trainset = DatasetFolder(
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


secret_size = 20

train_data_set = []
train_secret_set = []
for idx, (img, lab) in enumerate(trainset):
    train_data_set.append(img.tolist())
    secret = np.random.binomial(1, .5, secret_size).tolist()
    train_secret_set.append(secret)


for idx, (img, lab) in enumerate(testset):
    train_data_set.append(img.tolist())
    secret = np.random.binomial(1, .5, secret_size).tolist()
    train_secret_set.append(secret)


train_steg_set = GetPoisonedDataset(train_data_set, train_secret_set)


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

    'benign_training': False,
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
    'save_epoch_interval': 100,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_DataFolder_GTSRB_ISSBA'
}

encoder_schedule = {
    'secret_size': secret_size,
    'enc_height': 32,
    'enc_width': 32,
    'enc_in_channel': 3,
    'enc_total_epoch': 20,
    'enc_secret_only_epoch': 2,
    'enc_use_dis': False,
}

# Configure the attack scheme
ISSBA = core.ISSBA(
    dataset_name="gtsrb",
    train_dataset=trainset,
    test_dataset=testset,
    train_steg_set=train_steg_set,
    model=core.models.ResNet(18, 43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,      # follow the default configure in the original paper
    encoder_schedule=encoder_schedule,
    encoder=None,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)

ISSBA.train(schedule=schedule)

# ===== Train backdoored model on GTSRB using with GTSRB (done) ===== 


# ===== Train backdoored model on CIFAR10 using with CIFAR10 ===== 

# Prepare datasets and follow the default data augmentation in the original paper
transform_train = Compose([
    transforms.Resize((32, 32)),
    RandomHorizontalFlip(),
    ToTensor(),
])
transform_test = Compose([
    transforms.Resize((32, 32)),
    ToTensor(),
])

trainset = CIFAR10(
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


secret_size = 20

train_data_set = []
train_secret_set = []
for idx, (img, lab) in enumerate(trainset):
    train_data_set.append(img.tolist())
    secret = np.random.binomial(1, .5, secret_size).tolist()
    train_secret_set.append(secret)


for idx, (img, lab) in enumerate(testset):
    train_data_set.append(img.tolist())
    secret = np.random.binomial(1, .5, secret_size).tolist()
    train_secret_set.append(secret)


train_steg_set = GetPoisonedDataset(train_data_set, train_secret_set)


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

    'benign_training': False,
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
    'save_epoch_interval': 100,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_DataFolder_CIFAR10_ISSBA'
}

encoder_schedule = {
    'secret_size': secret_size,
    'enc_height': 32,
    'enc_width': 32,
    'enc_in_channel': 3,
    'enc_total_epoch': 20,
    'enc_secret_only_epoch': 2,
    'enc_use_dis': False,
}

# Configure the attack scheme
ISSBA = core.ISSBA(
    dataset_name="cifar10",
    train_dataset=trainset,
    test_dataset=testset,
    train_steg_set=train_steg_set,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,      # follow the default configure in the original paper
    encoder_schedule=encoder_schedule,
    encoder=None,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)

ISSBA.train(schedule=schedule)

# ===== Train backdoored model on CIFAR10 using with CIFAR10 (done) ===== 


# ===== Train backdoored model on MNIST using with MNIST ===== 

# Prepare datasets and follow the default data augmentation in the original paper
transform_train = Compose([
    transforms.Resize((28, 28)),
    RandomHorizontalFlip(),
    ToTensor(),
])
transform_test = Compose([
    transforms.Resize((28, 28)),
    ToTensor(),
])
trainset = MNIST(
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

secret_size = 20

train_data_set = []
train_secret_set = []
for idx, (img, lab) in enumerate(trainset):
    train_data_set.append(img.tolist())
    secret = np.random.binomial(1, .5, secret_size).tolist()
    train_secret_set.append(secret)


for idx, (img, lab) in enumerate(testset):
    train_data_set.append(img.tolist())
    secret = np.random.binomial(1, .5, secret_size).tolist()
    train_secret_set.append(secret)


train_steg_set = GetPoisonedDataset(train_data_set, train_secret_set)


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [30, 50],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 100,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_DataFolder_MNIST_ISSBA'
}

encoder_schedule = {
    'secret_size': secret_size,
    'enc_height': 28,
    'enc_width': 28,
    'enc_in_channel': 1,
    'enc_total_epoch': 20,
    'enc_secret_only_epoch': 2,
    'enc_use_dis': False,
}

# Configure the attack scheme
ISSBA = core.ISSBA(
    dataset_name="mnist",
    train_dataset=trainset,
    test_dataset=testset,
    train_steg_set=train_steg_set,
    model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.2,      # follow the default configure in the original paper
    encoder_schedule=encoder_schedule,
    encoder=None,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)

ISSBA.train(schedule=schedule)

# Get the poisoned dataset
poisoned_train_dataset, poisoned_test_dataset = ISSBA.get_poisoned_dataset()

print("The length of poisoned train dataset is: ", len(poisoned_train_dataset))
print("The length of poisoned test dataset is: ", len(poisoned_test_dataset))

# ===== Train backdoored model on MNIST using with MNIST (done) ===== 


# ===== Train encoder and decoder model on 224 * 224 *3 ===== 

transform_train = Compose([
    transforms.Resize((224, 224)),
    ToTensor(),
])
transform_test = Compose([
    transforms.Resize((224, 224)),
    ToTensor(),
])

trainset = CIFAR10(
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

secret_size = 20
train_data_set = np.random.random((1, 3, 224, 224)).tolist()
train_secret_set = np.random.random((1, 20)).tolist()
train_steg_set = GetPoisonedDataset(train_data_set, train_secret_set)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

    'benign_training': False,
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
    'save_epoch_interval': 100,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_DataFolder_Other_ISSBA'
}

encoder_schedule = {
    'secret_size': secret_size,
    'enc_height': 224,
    'enc_width': 224,
    'enc_in_channel': 3,
    'enc_total_epoch': 20,
    'enc_secret_only_epoch': 2,
    'enc_use_dis': False,
}

# Configure the attack scheme
ISSBA = core.ISSBA(
    dataset_name="other",
    train_dataset=trainset,
    test_dataset=testset,
    train_steg_set=train_steg_set,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,      # follow the default configure in the original paper
    encoder=None,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)

ISSBA.train_encoder_decoder(train_only=True)

# ===== Train encoder and decoder model on 224 * 224 *3 (done) ===== 
