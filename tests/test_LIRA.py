'''
This is the test code of poisoned training on GTSRB, CIFAR10, MNIST, using dataset class of torchvision.datasets.DatasetFolder torchvision.datasets.CIFAR10 torchvision.datasets.MNIST.
The attack method is LIRA.
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
import torch.nn.functional as F
import core


global_seed = 666
deterministic = True
torch.manual_seed(global_seed)


cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, feature_dim=512):
        """
        for image size 32, feature_dim = 512
        for other sizes, feature_dim = 512 * (size//32)**2
        """
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


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
testset = DatasetFolder(
    root='/data/gaokuofeng/datasets/GTSRB/testset', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.01,
    'lr_atk': 0.0001,
    'momentum': 0.9,
    
    'epochs': 50,
    'train_epoch': 1,
    'cls_test_epoch': 5,


    'tune_test_epochs': 250,
    'tune_test_lr': 0.01,
    'tune_momentum': 0.9,
    'tune_weight_decay': 5e-4,
    'tune_test_epoch_interval': 1,

    'schedulerC_lambda': 0.1,
    'schedulerC_milestones': '50,100,150,200',

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_DataFolder_GTSRB_LIRA'
}


# Configure the attack scheme
LIRA = core.LIRA(
    dataset_name="gtsrb",
    train_dataset=trainset,
    test_dataset=testset,
    model=VGG('VGG11', num_classes=43), #core.models.vgg11(num_classes=43), #core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    eps=0.01,
    alpha=0.5,
    tune_test_eps=0.01,
    tune_test_alpha=0.5,
    best_threshold=0.1,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)


# Train backdoored model
LIRA.train()

# Get the poisoned dataset
poisoned_train_dataset, poisoned_test_dataset = LIRA.get_poisoned_dataset()

print("The length of poisoned train dataset is: ", len(poisoned_train_dataset))
print("The length of poisoned test dataset is: ", len(poisoned_test_dataset))

# ===== Train backdoored model on GTSRB using with DatasetFolder (done)======

# ===== Train backdoored model on CIFAR10 using with CIFAR10 ===== 

# Prepare datasets and follow the default data augmentation in the original paper
transform_train = Compose([
    transforms.Resize((32, 32)),
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
testset = CIFAR10(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '3',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.01,
    'lr_atk': 0.0001,
    'momentum': 0.9,
    
    'epochs': 50,
    'train_epoch': 1,
    'cls_test_epoch': 5,


    'tune_test_epochs': 250,
    'tune_test_lr': 0.01,
    'tune_momentum': 0.9,
    'tune_weight_decay': 5e-4,
    'tune_test_epoch_interval': 1,

    'schedulerC_lambda': 0.1,
    'schedulerC_milestones': '50,100,150,200',

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_DataFolder_CIFAR10_LIRA'
}


# Configure the attack scheme
LIRA = core.LIRA(
    dataset_name="cifar10",
    train_dataset=trainset,
    test_dataset=testset,
    model=VGG('VGG11', num_classes=10), #core.models.vgg11(num_classes=10), #core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    eps=0.01,
    alpha=0.5,
    tune_test_eps=0.01,
    tune_test_alpha=0.5,
    best_threshold=0.1,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)


# Train backdoored model
LIRA.train()

# Get the poisoned dataset
poisoned_train_dataset, poisoned_test_dataset = LIRA.get_poisoned_dataset()

print("The length of poisoned train dataset is: ", len(poisoned_train_dataset))
print("The length of poisoned test dataset is: ", len(poisoned_test_dataset))

# ===== Train backdoored model on CIFAR10 using with CIFAR10 (done)===== 

# ===== Train backdoored model on MNIST using with MNIST ===== 
class MNISTBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(MNISTBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.ind = None

    def forward(self, x):
        return self.conv1(F.relu(self.bn1(x)))


class BaselineMNISTNetwork(nn.Module):
    def __init__(self):
        super(BaselineMNISTNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 2, 1)  # 14
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = MNISTBlock(32, 64, 2)  # 7
        self.layer3 = MNISTBlock(64, 64, 2)  # 4
        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.3)
        self.linear9 = nn.Linear(512, 10)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


# Prepare datasets and follow the default data augmentation in the original paper
transform_train = Compose([
    transforms.Resize((28, 28)),
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
testset = MNIST(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '3',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.01,
    'lr_atk': 0.0001,
    'momentum': 0.9,
    
    'epochs': 10,
    'train_epoch': 1,
    'cls_test_epoch': 5,


    'tune_test_epochs': 50,
    'tune_test_lr': 0.01,
    'tune_momentum': 0.9,
    'tune_weight_decay': 5e-4,
    'tune_test_epoch_interval': 1,

    'schedulerC_lambda': 0.1,
    'schedulerC_milestones': '10,20,30,40',

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_DataFolder_MNIST_LIRA'
}


# Configure the attack scheme
LIRA = core.LIRA(
    dataset_name="mnist",
    train_dataset=trainset,
    test_dataset=testset,
    model=BaselineMNISTNetwork(), # core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    eps=0.01,
    alpha=0.5,
    tune_test_eps=0.01,
    tune_test_alpha=0.5,
    best_threshold=0.1,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)


# Train backdoored model
LIRA.train()

# Get the poisoned dataset
poisoned_train_dataset, poisoned_test_dataset = LIRA.get_poisoned_dataset()

print("The length of poisoned train dataset is: ", len(poisoned_train_dataset))
print("The length of poisoned test dataset is: ", len(poisoned_test_dataset))

# Get the victim model
victim_model = LIRA.get_model()

# ===== Train backdoored model on MNIST using with MNIST (done)===== 
