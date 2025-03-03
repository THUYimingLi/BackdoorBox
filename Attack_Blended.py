'''
This is the example code of benign training and poisoned training on torchvision.datasets.DatasetFolder.
Dataset is ImageNet50.
Attack method is Blended.
'''

import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomResizedCrop, RandomHorizontalFlip, Normalize
import core

global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

# Change dataset to ImageNet50
dataset = torchvision.datasets.ImageFolder

transform_train = Compose([
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
trainset = dataset(root='./data/ImageNet50/train', transform=transform_train)

transform_test = Compose([
    RandomResizedCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
testset = dataset(root='./data/ImageNet50/val', transform=transform_test)

# Update pattern and weight for ImageNet50 (224x224 images)
pattern = torch.zeros((3, 224, 224), dtype=torch.uint8)
pattern[:, -3:, -3:] = 255
weight = torch.zeros((3, 224, 224), dtype=torch.float32)
weight[:, -3:, -3:] = 0.2

blended = core.Blended(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, num_classes=50),  # Use ResNet for ImageNet50
    loss=nn.CrossEntropyLoss(),
    pattern=pattern,
    weight=weight,
    y_target=1,
    poisoned_rate=0.1,  # 设置投毒率为10%
    seed=global_seed,
    deterministic=deterministic
)

poisoned_train_dataset, poisoned_test_dataset = blended.get_poisoned_dataset()

index = 0  # 确保在使用之前定义 index
# Show an Example of Poisoned Training Samples
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


# Train Benign Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '3',
    'GPU_num': 1,

    'benign_training': True,
    'batch_size': 128,
    'num_workers': 4,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [30, 60, 90],

    'epochs': 100,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_benign_ImageNet50_Blended'
}

blended.train(schedule)
benign_model = blended.get_model()
torch.save(benign_model.state_dict(), 'benign_model_resnet18.pt')

# Test Benign Model
test_schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '3',
    'GPU_num': 1,

    'batch_size': 128,
    'num_workers': 4,

    'save_dir': 'experiments',
    'experiment_name': 'test_benign_ImageNet50_Blended'
}
blended.test(test_schedule)
torch.save(benign_model.state_dict(), 'benign_model_resnet18_tested.pt')


blended.model = core.models.ResNet(18, num_classes=50)
# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '3',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 4,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [30, 60, 90],

    'epochs': 100,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poisoned_ImageNet50_Blended'
}

blended.train(schedule)
infected_model = blended.get_model()
torch.save(infected_model.state_dict(), 'infected_model_resnet18.pt')

# Test Infected Model
test_schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '3',
    'GPU_num': 1,

    'batch_size': 128,
    'num_workers': 4,

    'save_dir': 'experiments',
    'experiment_name': 'test_poisoned_ImageNet50_Blended'
}
blended.test(test_schedule)
torch.save(infected_model.state_dict(), 'infected_model_resnet18_tested.pt')