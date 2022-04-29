'''
This is the test code of Finetuning.


'''

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, ToPILImage, Resize
from torch.utils.data import random_split
import core
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = '4'
batch_size = 128
num_workers = 4


def test(model,p,trainset,testset,poisoned_testset,layer,y_target=1):
    num1=int(len(trainset)*p)
    num2=int(len(trainset)-num1)
    fttrainset, fttestset = random_split(trainset, [num1, num2])
    finetuning=core.Finetuning(
        train_dataset=fttrainset,
        test_dataset=testset,
        model=model,
        layer=layer,
        loss=nn.CrossEntropyLoss(),
        seed=global_seed,
        deterministic=deterministic
    )
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '1',
        'GPU_num': 1,

        'batch_size': 128,
        'num_workers': 4,

        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [],

        'epochs': 10,
        'log_iteration_interval': 100,
    }
    finetuning.finetuning(schedule)

    test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '1',
        'GPU_num': 1,

        'batch_size': 128,
        'num_workers': 4,
        'metric': 'BA',

        'save_dir': 'experiments',
        'experiment_name': 'finetuning_benign_CIFAR10_BadNets'
    }
    finetuning.test(test_schedule)

    #更换测试集
    finetuning.test_dataset=poisoned_testset
    test_schedule2 = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '1',
        'GPU_num': 1,

        'batch_size': 128,
        'num_workers': 4,
        'metric': 'ASR_NoTarget',
        'y_target': y_target,

        'save_dir': 'experiments',
        'experiment_name': 'finetuning_benign_CIFAR10_BadNets'
    }
    finetuning.test(test_schedule2)



# ========== ResNet-18_CIFAR-10_Benign_Finetuing ==========
model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'CIFAR-10', 'Benign', 'Finetuning'

model = core.models.ResNet(18)

model_path = '/data/yangsheng/graduationproject/Mybenigndmodel1_666.pth.tar'

# Define Benign Training and Testing Dataset
dataset = torchvision.datasets.CIFAR10
# dataset = torchvision.datasets.MNIST

transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip()
])
trainset = dataset('data', train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset('data', train=False, transform=transform_test, download=True)

model.load_state_dict(torch.load(model_path))

#p为选用的干净样本所占比例
p=0.1
#选择需要finetuning的层
layer=["layer3","layer2"]
#ASR攻击的目标label
y_target=1
test(model,p,testset,testset,testset,layer,y_target)


# ========== ResNet-18_CIFAR-10_Badnets_Finetuing ==========
model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'CIFAR-10', 'Benign', 'Finetuning'

model = core.models.ResNet(18)

model_path = '/data/yangsheng/graduationproject/Myinfectedmodel.pth.tar'

# Define Benign Training and Testing Dataset
dataset = torchvision.datasets.CIFAR10
# dataset = torchvision.datasets.MNIST

transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip()
])
trainset = dataset('data', train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset('data', train=False, transform=transform_test, download=True)

model.load_state_dict(torch.load(model_path))

#p为选用的干净样本所占比例
p=0.1
#选择需要finetuning的层
layer=["layer3"]
#ASR攻击的目标label
y_target=1

badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    seed=global_seed,
    deterministic=deterministic
)

poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()


test(model,p,testset,testset,poisoned_test_dataset,layer,y_target)



# ========== ResNet-18_CIFAR-10_Blended_Finetuing ==========
model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'CIFAR-10', 'Benign', 'Finetuning'

model = core.models.ResNet(18)

model_path = '/data/yangsheng/graduationproject/Mynewinfectedmodel_blended_666.pth.tar'

# Define Benign Training and Testing Dataset
dataset = torchvision.datasets.CIFAR10
# dataset = torchvision.datasets.MNIST

transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip()
])
trainset = dataset('data', train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset('data', train=False, transform=transform_test, download=True)

model.load_state_dict(torch.load(model_path))

#p为选用的干净样本所占比例
p=0.1
#选择需要finetuning的层
layer=["layer3"]
#ASR攻击的目标label
y_target=1

blended = core.Blended(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    seed=global_seed,
    deterministic=deterministic
)

poisoned_train_dataset, poisoned_test_dataset = blended.get_poisoned_dataset()


test(model,p,testset,testset,poisoned_test_dataset,layer,y_target)