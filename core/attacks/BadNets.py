import random

from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import Compose

from .base import *


class AddMNISTTrigger:
    def __init__(self):
        pass

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = img.squeeze()
        img[-3:, -3:] = 255
        img = Image.fromarray(img.numpy(), mode='L')
        return img


class AddCIFAR10Trigger:
    def __init__(self):
        pass

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img[:, -3:, -3:] = 255
        img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img


class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target


class PoisonedMNIST(MNIST):
    def __init__(self, benign_dataset, y_target, poisoned_rate, seed=0):
        super(PoisonedMNIST, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True
        )
        random.seed(seed)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])
        self.poisoned_transform = Compose([AddMNISTTrigger()])
        self.poisoned_target_transform = Compose([ModifyTarget(y_target)])

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class PoisonedCIFAR10(CIFAR10):
    def __init__(self, benign_dataset, y_target, poisoned_rate, seed=0):
        super(PoisonedCIFAR10, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True
        )
        random.seed(seed)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])
        self.poisoned_transform = Compose([AddCIFAR10Trigger()])
        self.poisoned_target_transform = Compose([ModifyTarget(y_target)])

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def CreatePoisonedDataset(benign_dataset, y_target, poisoned_rate, seed=0):
    class_name = type(benign_dataset)
    if class_name == MNIST:
        return PoisonedMNIST(benign_dataset, y_target, poisoned_rate, seed)
    elif class_name == CIFAR10:
        return PoisonedCIFAR10(benign_dataset, y_target, poisoned_rate, seed)
    else:
        raise NotImplementedError


class BadNets(Base):
    def __init__(self,
                 train_dataset,
                 test_dataset,
                 model,
                 loss,
                 y_target,
                 poisoned_rate,
                 schedule=None,
                 seed=0):
        super(BadNets, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule)
        self.poisoned_train_dataset = CreatePoisonedDataset(train_dataset, y_target, poisoned_rate, seed)
        self.poisoned_test_dataset = CreatePoisonedDataset(test_dataset, y_target, 1.0, seed)
    
    def get_poisoned_dataset(self):
        return self.poisoned_train_dataset, self.poisoned_test_dataset





    

