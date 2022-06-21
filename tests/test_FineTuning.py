'''
This is the test code of Finetuning.
'''

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip,ToPILImage, Resize
from torch.utils.data import random_split
import core
import os
from copy import deepcopy
from torchvision.datasets import DatasetFolder
import cv2
from core.utils import test

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = '1'
batch_size = 128
num_workers = 4


def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid



def test_finetuning(model,p,trainset,testset,poisoned_testset,layer,y_target):
    num1=int(len(trainset)*p)
    num2=int(len(trainset)-num1)
    fttrainset, fttestset = random_split(trainset, [num1, num2])
    mytestset=deepcopy(testset)
    mypoisoned_testset=deepcopy(poisoned_testset)
    finetuning=core.FineTuning(
        train_dataset=fttrainset,
        test_dataset=mytestset,
        model=model,
        layer=layer,
        loss=nn.CrossEntropyLoss(),
        seed=global_seed,
        deterministic=deterministic
    )
    print("with defense")
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '3',
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
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': 'finetuning_GTSRB_BadNets'
    }
    finetuning.repair(schedule)

    test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '1',
        'GPU_num': 1,

        'batch_size': 128,
        'num_workers': 4,
        'metric': 'BA',

        'save_dir': 'experiments',
        'experiment_name': 'finetuning_CIFAR10_BadNets'
    }
    finetuning.test(test_schedule)

    #change the set
    finetuning.test_dataset=mypoisoned_testset
    test_schedule2 = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '1',
        'GPU_num': 1,

        'batch_size': 128,
        'num_workers': 4,
        'metric': 'ASR_NoTarget',
        'y_target': y_target,

        'save_dir': 'experiments',
        'experiment_name': 'finetuning_CIFAR10_BadNets'
    }
    finetuning.test(test_schedule2)

    #test the get_model
    repair_model=finetuning.get_model()
    test(repair_model,mypoisoned_testset,test_schedule2)


    del finetuning



# # ========== ResNet-18_CIFAR-10_Wanet_Finetuing ==========
# print("wanet")
# dataset = torchvision.datasets.CIFAR10
#
#
#
# transform_train = Compose([
#     ToTensor(),
#     RandomHorizontalFlip()
# ])
# trainset = dataset('data', train=True, transform=transform_train, download=False)
#
# transform_test = Compose([
#     ToTensor()
# ])
# testset = dataset('data', train=False, transform=transform_test, download=False)
#
#
#
# identity_grid,noise_grid=gen_grid(32,4)
# torch.save(identity_grid, 'ResNet-18_CIFAR-10_WaNet_identity_grid.pth')
# torch.save(noise_grid, 'ResNet-18_CIFAR-10_WaNet_noise_grid.pth')
# model=core.models.ResNet(18)
# model.load_state_dict(torch.load('Wanet_Resnet18_Cifar10_666.pth.tar'))
# wanet = core.WaNet(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=model,
#     # model=core.models.BaselineMNISTNetwork(),
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.1,
#     identity_grid=identity_grid,
#     noise_grid=noise_grid,
#     noise=False,
#     seed=global_seed,
#     deterministic=deterministic
# )
#
# poisoned_train_dataset, poisoned_test_dataset = wanet.get_poisoned_dataset()
#
#
#
# test_finetuning(model,0.1,testset,testset,poisoned_test_dataset,["full layers"],1)
#
#
# # ========== ResNet-18_CIFAR-10_Benign_Finetuing ==========
# print("benign")
# model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'CIFAR-10', 'Benign', 'FineTuning'
#
# model = core.models.ResNet(18)
#
# model_path = 'Benign_Resnet18_Cifar10_666.pth.tar'
#
# # Define Benign Training and Testing Dataset
# dataset = torchvision.datasets.CIFAR10
# # dataset = torchvision.datasets.MNIST
#
# transform_train = Compose([
#     ToTensor(),
#     RandomHorizontalFlip()
# ])
# trainset = dataset('data', train=True, transform=transform_train, download=True)
#
# transform_test = Compose([
#     ToTensor()
# ])
# testset = dataset('data', train=False, transform=transform_test, download=True)
#
# model.load_state_dict(torch.load(model_path))
#
# #p is the rate of choosing the clean sample
# p=0.1
# #choose finetuning layers
# layer=["full layers"]
# #set the target label
# y_target=1
# # test2(model,p,testset,testset,testset,layer,y_target)
# test_finetuning(model,p,testset,testset,testset,layer,y_target)
#
#
# # ========== ResNet-18_CIFAR-10_Badnets_Finetuing ==========
# print("badnets")
# model = core.models.ResNet(18)
#
# attack_name = 'BadNets'
# model_path = 'Badnets_Resnet18_Cifar10_666.pth.tar'
#
# pattern = torch.zeros((32, 32), dtype=torch.uint8)
# pattern[-3:, -3:] = 255
# weight = torch.zeros((32, 32), dtype=torch.float32)
# weight[-3:, -3:] = 1.0
#
# attack = core.BadNets(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=model,
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.05,
#     pattern=pattern,
#     weight=weight,
#     seed=global_seed,
#     deterministic=deterministic
# )
# poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()
#
# #p is the rate of choosing the clean sample
# p=0.1
# #choose finetuning layers
# layer=["full layers"]
# #set the target label
# y_target=1
# model.load_state_dict(torch.load(model_path))
#
#
# # test2(model,p,testset,testset,poisoned_testset,layer,y_target)
# test_finetuning(model,p,testset,testset,poisoned_testset,layer,y_target)
#
#
#
# # ========== ResNet-18_CIFAR-10_Labelconsistent_Finetuing ==========
# print("labelconsistent")
# model = core.models.ResNet(18)
#
# attack_name = 'LabelConsistent'
# model_path = 'Labelconsistent_Resnet18_Cifar10_666.pth.tar'
#
# adv_model = deepcopy(model)
# adv_ckpt = torch.load('Benign_Resnet18_Cifar10_666.pth.tar')
# adv_model.load_state_dict(adv_ckpt)
#
# pattern = torch.zeros((32, 32), dtype=torch.uint8)
# pattern[-1, -1] = 255
# pattern[-1, -3] = 255
# pattern[-3, -1] = 255
# pattern[-2, -2] = 255
#
# pattern[0, -1] = 255
# pattern[1, -2] = 255
# pattern[2, -3] = 255
# pattern[2, -1] = 255
#
# pattern[0, 0] = 255
# pattern[1, 1] = 255
# pattern[2, 2] = 255
# pattern[2, 0] = 255
#
# pattern[-1, 0] = 255
# pattern[-1, 2] = 255
# pattern[-2, 1] = 255
# pattern[-3, 0] = 255
#
# weight = torch.zeros((32, 32), dtype=torch.float32)
# weight[:3,:3] = 1.0
# weight[:3,-3:] = 1.0
# weight[-3:,:3] = 1.0
# weight[-3:,-3:] = 1.0
#
# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
#     'GPU_num': 1,
#
#     'benign_training': False, # Train Attacked Model
#     'batch_size': 128,
#     'num_workers': 8,
#
#     'lr': 0.1,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [150, 180],
#
#     'epochs': 200,
#
#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 10,
#
#     'save_dir': 'experiments',
#     'experiment_name': 'ResNet-18_CIFAR-10_LabelConsistent'
# }
#
# eps = 8
# alpha = 1.5
# steps = 100
# max_pixel = 255
# poisoned_rate = 0.25
#
# attack = core.LabelConsistent(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=model,
#     adv_model=adv_model,
#     adv_dataset_dir=f'./adv_dataset/CIFAR-10_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
#     loss=nn.CrossEntropyLoss(),
#     y_target=2,
#     poisoned_rate=poisoned_rate,
#     pattern=pattern,
#     weight=weight,
#     eps=eps,
#     alpha=alpha,
#     steps=steps,
#     max_pixel=max_pixel,
#     poisoned_transform_train_index=0,
#     poisoned_transform_test_index=0,
#     poisoned_target_transform_index=0,
#     schedule=schedule,
#     seed=global_seed,
#     deterministic=True
# )
# poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()
#
# #p is the rate of choosing the clean sample
# p=0.1
# #choose finetuning layers
# layer=["full layers"]
# #set the target label
# y_target=2
# model.load_state_dict(torch.load(model_path))
#
# # test2(model,p,testset,testset,poisoned_testset,layer,y_target)
# test_finetuning(model,p,testset,testset,poisoned_testset,layer,y_target)


# ========== ResNet-18_GTSRB_Wanet_Finetuing ==========
print("wanet")
dataset = torchvision.datasets.DatasetFolder


# image file -> cv.imread -> numpy.ndarray (H x W x C) -> ToTensor -> torch.Tensor (C x H x W) -> RandomHorizontalFlip -> resize (32) -> torch.Tensor -> network input
transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip(),
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor()
])
transform_test = Compose([
    ToTensor(),
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor()

])


trainset = dataset(
    root='/data/ganguanhao/datasets/GTSRB/train', # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

testset = dataset(
    root='/data/ganguanhao/datasets/GTSRB/testset', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

index = 44

x, y = trainset[index]
print(y)
for a in x[0]:
    for b in a:
        print("%-4.2f" % float(b), end=' ')
    print()

identity_grid,noise_grid=gen_grid(32, 4)
torch.save(identity_grid, 'ResNet-18_GTSRB_WaNet_identity_grid.pth')
torch.save(noise_grid, 'ResNet-18_GTSRB_WaNet_noise_grid.pth')
wanet = core.WaNet(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18,43),
    loss=nn.CrossEntropyLoss(),
    y_target=0,
    poisoned_rate=0.1,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=True,
    seed=global_seed,
    deterministic=deterministic
)

poisoned_train_dataset, poisoned_test_dataset = wanet.get_poisoned_dataset()

x, y = poisoned_train_dataset[index]
print(y)
for a in x[0]:
    for b in a:
        print("%-4.2f" % float(b), end=' ')
    print()

x, y = poisoned_test_dataset[index]
print(y)
for a in x[0]:
    for b in a:
        print("%-4.2f" % float(b), end=' ')
    print()

model=core.models.ResNet(18,43)
model.load_state_dict(torch.load("Wanet_Resnet18_GTSRB_666.pth.tar"))


test_finetuning(model,0.1,testset,testset,poisoned_test_dataset,["full layers"],0)


# ========== ResNet-18_GTSRB_Benign_Finetuing ==========
print("benign")
model_name, dataset_name, attack_name, defense_name = 'ResNet-18', 'CIFAR-10', 'Benign', 'FineTuning'

model = core.models.ResNet(18,43)

model_path = 'Benign_Resnet18_GTSRB_666.pth.tar'

transform_train = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor()
])
trainset = DatasetFolder(
    root='/data/ganguanhao/datasets/GTSRB/train', # please replace this with path to your training set
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
    root='/data/ganguanhao/datasets/GTSRB/testset', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)


model.load_state_dict(torch.load(model_path))

#p is the rate of choosing the clean sample
p=0.1
#choose finetuning layers
layer=["full layers"]
#set the target label
y_target=1
# test2(model,p,testset,testset,testset,layer,y_target)
test_finetuning(model,p,testset,testset,testset,layer,y_target)


# ========== ResNet-18_GTSRB_Badnets_Finetuing ==========
print("badnets")
model = core.models.ResNet(18,43)

transform_train = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor()
])
trainset = DatasetFolder(
    root='/data/ganguanhao/datasets/GTSRB/train', # please replace this with path to your training set
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
    root='/data/ganguanhao/datasets/GTSRB/testset', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

attack_name = 'BadNets'
model_path = 'Badnets_Resnet18_GTSRB_666.pth.tar'

pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0

badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, 43),
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
poisoned_trainset, poisoned_testset = badnets.get_poisoned_dataset()

#p is the rate of choosing the clean sample
p=0.1
#choose finetuning layers
layer=["full layers"]
#set the target label
y_target=1
model.load_state_dict(torch.load(model_path))


# test2(model,p,testset,testset,poisoned_testset,layer,y_target)
test_finetuning(model,p,testset,testset,poisoned_testset,layer,y_target)



# ========== ResNet-18_GTSRB_Labelconsistent_Finetuing ==========
print("labelconsistent")
model = core.models.ResNet(18,43)

attack_name = 'LabelConsistent'
model_path = 'Labelconsistent_Resnet18_GTSRB_666.pth.tar'


transform_train = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor()
])
transform_train = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    RandomHorizontalFlip(),
    ToTensor()
])
trainset = DatasetFolder(
    root='/data/ganguanhao/datasets/GTSRB/train', # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

transform_test = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor()
])
testset = DatasetFolder(
    root='/data/ganguanhao/datasets/GTSRB/testset', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)


adv_model = core.models.ResNet(18, 43)
adv_ckpt = torch.load('Benign_Resnet18_GTSRB_666.pth.tar')
adv_model.load_state_dict(adv_ckpt)

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


# pattern = torch.zeros((32, 32), dtype=torch.uint8)

# k = 5

# pattern[:k,:k] = 255
# pattern[:k,-k:] = 255
# pattern[-k:,:k] = 255
# pattern[-k:,-k:] = 255

# weight = torch.zeros((32, 32), dtype=torch.float32)
# weight[:k,:k] = 1.0
# weight[:k,-k:] = 1.0
# weight[-k:,:k] = 1.0
# weight[-k:,-k:] = 1.0

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': False, # Train Attacked Model
    'batch_size': 256,
    'num_workers': 8,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [20],

    'epochs': 50,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'ResNet-18_GTSRB_LabelConsistent'
}


eps = 16
alpha = 1.5
steps = 100
max_pixel = 255
poisoned_rate = 0.5

label_consistent = core.LabelConsistent(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, 43),
    adv_model=adv_model,
    adv_dataset_dir=f'./adv_dataset/GTSRB_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
    loss=nn.CrossEntropyLoss(),
    y_target=2,
    poisoned_rate=poisoned_rate,
    adv_transform=Compose([transforms.ToPILImage(), transforms.Resize((32, 32)), ToTensor()]),
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
    deterministic=True
)
poisoned_trainset, poisoned_testset = label_consistent.get_poisoned_dataset()

#p is the rate of choosing the clean sample
p=0.1
#choose finetuning layers
layer=["full layers"]
#set the target label
y_target=2
model.load_state_dict(torch.load(model_path))

# test2(model,p,testset,testset,poisoned_testset,layer,y_target)
test_finetuning(model,p,testset,testset,poisoned_testset,layer,y_target)