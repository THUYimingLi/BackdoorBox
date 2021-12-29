
'''
This is the test code of poisoned training on GTSRB ,CIFAR10, MNIST, using dataset class of torchvision.datasets.DatasetFolder torchvision.datasets.CIFAR10 torchvision.datasets.MNIST.
Attack method is Refool.
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



reflection_images = []
reflection_data_dir = "/data/ganguanhao/datasets/VOCdevkit/VOC2012/JPEGImages/"
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
    
reflection_image_path = os.listdir(reflection_data_dir)
reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]


# Dataset Folder GTSRB
transform_train = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32,32)),
    RandomHorizontalFlip(),
    ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
])
transform_test = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32,32)),
    ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
])

trainset = DatasetFolder(
    root='/data/ganguanhao/datasets/GTSRB/train',
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

testset = DatasetFolder(
    root='/data/ganguanhao/datasets/GTSRB/testset',
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

refool= core.Refool(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18,43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    poisoned_transform_index=0,
    poisoned_target_transform_index=0,
    schedule=None,
    seed=666,
    reflection_candidates = reflection_images,
)


poisoned_train_dataset, poisoned_test_dataset = refool.get_poisoned_dataset()
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 16,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [50, 75],

    'epochs': 100,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_DataFolder_GTSRB'
}

refool.train(schedule)

# CIFAR10
transform_train = Compose([
    transforms.Resize((32,32)),
    RandomHorizontalFlip(),
    ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
])
transform_test = Compose([
    transforms.Resize((32,32)),
    ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
])
trainset = CIFAR10(
    root='/data/ganguanhao/datasets',
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
testset = CIFAR10(
    root='/data/ganguanhao/datasets',
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)

refool= core.Refool(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    poisoned_transform_index=0,
    poisoned_target_transform_index=0,
    schedule=None,
    seed=666,
    reflection_candidates = reflection_images,
)


poisoned_train_dataset, poisoned_test_dataset = refool.get_poisoned_dataset()
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 16,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [50, 75],

    'epochs': 100,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_CIFAR10'
}

refool.train(schedule)

# MNIST
transform_train = Compose([
    transforms.Resize((28,28)),
    RandomHorizontalFlip(),
    ToTensor(),
])
transform_test = Compose([
    transforms.Resize((28,28)),
    ToTensor(),
])
trainset = MNIST(
    root='/data/ganguanhao/datasets',
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
testset = MNIST(
    root='/data/ganguanhao/datasets',
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)

loader = DataLoader(trainset,)

refool= core.Refool(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    poisoned_transform_index=0,
    poisoned_target_transform_index=0,
    schedule=None,
    seed=666,
    reflection_candidates = reflection_images,
)


poisoned_train_dataset, poisoned_test_dataset = refool.get_poisoned_dataset()
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 16,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [10,15,],

    'epochs': 20,

    'log_iteration_interval': 100,
    'test_epoch_interval': 5,
    'save_epoch_interval': 5,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_MNIST'
}

refool.train(schedule)
