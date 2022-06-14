
'''
This is the test code of poisoned training on GTSRB, CIFAR10, MNIST, using dataset class of torchvision.datasets.DatasetFolder torchvision.datasets.CIFAR10 torchvision.datasets.MNIST.
The attack method is SleeperAgent.
'''


from typing import Pattern
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
temp_patch = 0.5 * torch.ones(3, 8, 8)
patch = torch.bernoulli(temp_patch)


# trigger = torch.Tensor([[0,0,1],[0,1,0],[1,0,1]])
# patch = trigger.repeat((3, 1, 1))

def show_dataset(dataset, num, path_to_save):
    """Each image in dataset should be torch.Tensor, shape (C,H,W)"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,20))
    for i in range(num):
        ax = plt.subplot(num,1,i+1)
        img = (dataset[i][0]).permute(1,2,0).cpu().detach().numpy()
        ax.imshow(img)
    plt.savefig(path_to_save)

def init_model(model):
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    return model

# ===== Train backdoored model on GTSRB using with DatasetFolder ======
torch.manual_seed(global_seed)
# Prepare datasets
transform_train = Compose([ # the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
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
    root='/data/ganguanhao/datasets/GTSRB/train', # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

testset = DatasetFolder(
    root='/data/ganguanhao/datasets/GTSRB/testset', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

# # Configure the attack scheme


sleeper_agent= core.SleeperAgent(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, 43),
    loss=nn.CrossEntropyLoss(),
    patch=patch,
    random_patch=True,
    eps=16./255, # 16/255. 2e-2 pr. leads to 20 asr, 32/255 2e-2 pr. leads to 60 asr
    y_target=12, # 12 is the class with most samples, choose 12 to retain performance
    y_source=1, 
    poisoned_rate=0.02,
    source_num=1000,
    schedule=None,
    seed=global_seed,
    deterministic=deterministic,
)
# def init_model(model):
#     return torch.nn.DataParallel(core.models.ResNet(18,43).cuda())

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [50, 75],

    # 'pretrain': 'pretrain/pretrain_gtsrb.pth',

    'epochs': 100,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_DataFolder_GTSRB_SleeperAgent',
    
    'pretrain_schedule': {'epochs':100, 'lr':0.1, 'weight_decay': 5e-4,  'gamma':0.1, 'milestones':[50,75], 'batch_size':128, 'num_workers':8, 'momentum': 0.9,},
    'retrain_schedule': {'epochs':40, 'lr':0.1, 'weight_decay': 5e-4,  'gamma':0.1, 'milestones':[14,24,35], 'batch_size':128, 'num_workers':8, 'momentum': 0.9,},
    'craft_iters': 250, # total iterations to craft the poisoned trainset
    'retrain_iter_interval': 50, # retrain the model after #retrain_iter_interval crafting iterations
}

# Train backdoored model
print("train on GTSRB")
sleeper_agent.train(init_model, schedule)
print("GTSRB done")
poisoned_trainset, poisoned_testset = sleeper_agent.get_poisoned_dataset()
show_dataset(poisoned_trainset, 5, 'gtsrb_train_poison.png')
show_dataset(poisoned_testset, 5, 'gtsrb_test_poison.png')
# # # ===== Train backdoored model on GTSRB using with DatasetFolder (done) ======

# # # ===== Train backdoored model on CIFAR10 using with CIFAR10 ===== 

torch.manual_seed(global_seed)
# Prepare datasets
transform_train = Compose([ # the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
    transforms.Resize((32, 32)),
    ToTensor(),
])
transform_test = Compose([
    transforms.Resize((32, 32)),
    ToTensor(),
])


trainset = CIFAR10(
    root='/data/ganguanhao/datasets', # please replace this with path to your dataset
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
testset = CIFAR10(
    root='/data/ganguanhao/datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)


sleeper_agent = core.SleeperAgent(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    patch=patch,
    random_patch=True,
    eps=16./255,
    y_target=1,
    y_source=2,    
    poisoned_rate=0.01,
    schedule=None,
    seed=global_seed,
    deterministic=deterministic,
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [50, 75],
    
    # 'pretrain': 'pretrain/pretrain_cifar.pth',

    'epochs': 100,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_CIFAR10_SleeperAgent',


    'pretrain_schedule': {'epochs':100, 'lr':0.1, 'weight_decay': 5e-4,  'gamma':0.1, 'milestones':[50,75], 'batch_size':128, 'num_workers':8, 'momentum': 0.9,},
    'retrain_schedule': {'epochs':40, 'lr':0.1, 'weight_decay': 5e-4,  'gamma':0.1, 'milestones':[14,24,35], 'batch_size':128, 'num_workers':8, 'momentum': 0.9,},
    'craft_iters': 250, # total iterations to craft the poisoned trainset
    'retrain_iter_interval': 50, # retrain the model after #retrain_iter_interval crafting iterations
    # milestones for retrain: [epochs // 2.667, epochs // 1.6, epochs // 1.142]
}
print("train on CIFAR10")
# def init_model(model):
#     return torch.nn.DataParallel(core.models.ResNet(18,10).cuda())
sleeper_agent.train(init_model, schedule)
print("CIFAR10 done")
poisoned_trainset, poisoned_testset = sleeper_agent.get_poisoned_dataset()
show_dataset(poisoned_trainset, 5, 'cifar_train_poison.png')
show_dataset(poisoned_testset, 5, 'cifar_test_poison.png')


        

# # ===== Train backdoored model on CIFAR10 using with CIFAR10 (done)===== 

# # ===== Train backdoored model on MNIST using with MNIST ===== 
torch.manual_seed(global_seed)
# # Prepare datasets
transform_train = Compose([ # the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
    transforms.Resize((28, 28)),
    ToTensor(),
])
transform_test = Compose([
    transforms.Resize((28, 28)),
    ToTensor(),
])
trainset = MNIST(
    root='/data/ganguanhao/datasets', # please replace this with path to your dataset
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
testset = MNIST(
    root='/data/ganguanhao/datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)



mnist_patch = patch[1].unsqueeze(0)
# Configure the attack scheme
sleeper_agent= core.SleeperAgent(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=4,
    y_source=8,
    patch=mnist_patch,
    random_patch=True,
    eps=16/255.,   
    poisoned_rate=0.05,
    schedule=None,
    seed=global_seed,
    deterministic=deterministic,
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [10, 15],

    # 'pretrain': 'pretrain/pretrain_mnist.pth',

    'epochs': 20,

    'log_iteration_interval': 100,
    'test_epoch_interval': 5,
    'save_epoch_interval': 5,

    'save_dir': 'experiments',
    'experiment_name': 'train_poison_MNIST_SleeperAgent',
    'pretrain_schedule': {'epochs':20, 'lr':0.1, 'weight_decay': 5e-4,  'gamma':0.1, 'milestones':[10,15], 'batch_size':128, 'num_workers':8, 'momentum': 0.9,},
    'retrain_schedule': {'epochs':20, 'lr':0.1, 'weight_decay': 5e-4,  'gamma':0.1, 'milestones':[10,15], 'batch_size':128, 'num_workers':8, 'momentum': 0.9,},
    'craft_iters': 250, # total iterations to craft the poisoned trainset
    'retrain_iter_interval': 50, # retrain the model after #retrain_iter_interval crafting iterations
}
# def init_model(model):
#     return torch.nn.DataParallel(core.models.BaselineMNISTNetwork().cuda())
# # Train backdoored model
print("train on MNIST")
sleeper_agent.train(init_model, schedule)
print("MNIST done")
poisoned_trainset, poisoned_testset = sleeper_agent.get_poisoned_dataset()
show_dataset(poisoned_trainset, 5, 'mnist_train_poison.png')
show_dataset(poisoned_testset, 5, 'mnist_test_poison.png')
# # # ===== Train backdoored model on MNIST using with MNIST (done)===== 
