'''
This is the implement of BadNets-based physical backdoor attack proposed in [1].

Reference:
[1] Backdoor Attack in the Physical World. ICLR Workshop, 2021.
'''

import os
import sys
import copy
import cv2
import random
import numpy as np
import PIL
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
from .BadNets import *
from .BadNets import CreatePoisonedDataset as CreatePoisonedTestDataset

class PoisonedDatasetFolder(DatasetFolder):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 pattern,
                 weight,
                 poisoned_transform_index,
                 poisoned_target_transform_index,
                 physical_transformations):
        super(PoisonedDatasetFolder, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(pattern, weight))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))
      
        # Add physical transformations
        if physical_transformations is None:
            raise ValueError("physical_transformations can not be None.")
        else:
            self.physical_transformations = physical_transformations

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if index in self.poisoned_set:
            sample = self.poisoned_transform(sample)
            sample = self.physical_transformations(sample)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
                sample = self.physical_transformations(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target

class PoisonedMNIST(MNIST):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 pattern,
                 weight,
                 poisoned_transform_index,
                 poisoned_target_transform_index,
                 physical_transformations):
        super(PoisonedMNIST, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddMNISTTrigger(pattern, weight))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))
        
        # Add physical transformations
        if physical_transformations is None:
            raise ValueError("physical_transformations can not be None.")
        else:
            self.physical_transformations = physical_transformations

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            img = self.physical_transformations(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)
                img = self.physical_transformations(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target

class PoisonedCIFAR10(CIFAR10):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 pattern,
                 weight,
                 poisoned_transform_index,
                 poisoned_target_transform_index,
                 physical_transformations
                 ):
        super(PoisonedCIFAR10, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddCIFAR10Trigger(pattern, weight))
       
        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

        # Add physical transformations
        if physical_transformations is None:
            raise ValueError("physical_transformations can not be None.")
        else:
            self.physical_transformations = physical_transformations

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            img = self.physical_transformations(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)
                img = self.physical_transformations(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


def CreatePoisonedTrainDataset(benign_dataset, y_target, poisoned_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index, physical_transformations):
    class_name = type(benign_dataset)
    if class_name == DatasetFolder:
        return PoisonedDatasetFolder(benign_dataset, y_target, poisoned_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index,physical_transformations)
    elif class_name == MNIST:
        return PoisonedMNIST(benign_dataset, y_target, poisoned_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index,physical_transformations)
    elif class_name == CIFAR10:
        return PoisonedCIFAR10(benign_dataset, y_target, poisoned_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index,physical_transformations)
    else:
        raise NotImplementedError


class PhysicalBA(BadNets):
    """Construct poisoned datasets with PhysicalBA method.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        y_target (int): N-to-1 attack target label.
        poisoned_rate (float): Ratio of poisoned samples.
        pattern (None | torch.Tensor): Trigger pattern, shape (C, H, W) or (H, W).
        weight (None | torch.Tensor): Trigger pattern weight, shape (C, H, W) or (H, W).
        poisoned_transform_train_index (int): The position index that poisoned transform will be inserted in train dataset. Default: 0.
        poisoned_transform_test_index (int): The position index that poisoned transform will be inserted in test dataset. Default: 0.
        poisoned_target_transform_index (int): The position that poisoned target transform will be inserted. Default: 0.
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Random seed for poisoned set. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
        physical_transformations (types in torchvsion.transforms): Transformations used to approximate the physical world. Choose transformation from torchvsion.transforms or use default
    """

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 model,
                 loss,
                 y_target,
                 poisoned_rate,
                 pattern=None,
                 weight=None,
                 poisoned_transform_train_index=0,
                 poisoned_transform_test_index=0,
                 poisoned_target_transform_index=0,
                 schedule=None,
                 seed=0,
                 deterministic=False,
                 physical_transformations=None):
        assert pattern is None or (isinstance(pattern, torch.Tensor) and ((0 < pattern) & (pattern < 1)).sum() == 0), 'pattern should be None or 0-1 torch.Tensor.'

        super(PhysicalBA, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            y_target=y_target,
            poisoned_rate=poisoned_rate,
            pattern=pattern,
            weight=weight,
            poisoned_transform_train_index=0,
            poisoned_transform_test_index=0,
            poisoned_target_transform_index=0,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)

        self.poisoned_train_dataset = CreatePoisonedTrainDataset(
            train_dataset,
            y_target,
            poisoned_rate,
            pattern,
            weight,
            poisoned_transform_train_index,
            poisoned_target_transform_index,
            physical_transformations)

        self.poisoned_test_dataset = CreatePoisonedTestDataset(
            test_dataset,
            y_target,
            1.0,
            pattern,
            weight,
            poisoned_transform_test_index,
            poisoned_target_transform_index)
