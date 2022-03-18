'''
This is the implement of Label-consistent backdoor attacks [1].

Reference:
[1] Label-consistent backdoor attacks. arXiv preprint arXiv:1912.02771, 2019.
'''

import copy
from copy import deepcopy
import random
import os.path as osp

import cv2
import numpy as np
import PIL
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from .base import *
from ..utils import PGD


def my_imread(file_path):
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)


class AddTrigger:
    def __init__(self):
        pass

    def add_trigger(self, img):
        """Add watermarked trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        return (self.weight * img + self.res).type(torch.uint8)


class AddDatasetFolderTrigger(AddTrigger):
    """Add watermarked trigger to DatasetFolder images.

    Args:
        pattern (torch.Tensor): shape (C, H, W) or (H, W).
        weight (torch.Tensor): shape (C, H, W) or (H, W).
    """

    def __init__(self, pattern, weight):
        super(AddDatasetFolderTrigger, self).__init__()

        if pattern is None:
            raise ValueError("Pattern can not be None.")
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if weight is None:
            raise ValueError("Weight can not be None.")
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        """Get the poisoned image.

        Args:
            img (PIL.Image.Image | numpy.ndarray | torch.Tensor): If img is numpy.ndarray or torch.Tensor, the shape should be (H, W, C) or (H, W).

        Returns:
            torch.Tensor: The poisoned image.
        """

        def add_trigger(img):
            if img.dim() == 2:
                img = img.unsqueeze(0)
                img = self.add_trigger(img)
                img = img.squeeze()
            else:
                img = self.add_trigger(img)
            return img

        if type(img) == PIL.Image.Image:
            img = F.pil_to_tensor(img)
            img = add_trigger(img)
            # 1 x H x W
            if img.size(0) == 1:
                img = Image.fromarray(img.squeeze().numpy(), mode='L')
            # 3 x H x W
            elif img.size(0) == 3:
                img = Image.fromarray(img.permute(1, 2, 0).numpy())
            else:
                raise ValueError("Unsupportable image shape.")
            return img
        elif type(img) == np.ndarray:
            # H x W
            if len(img.shape) == 2:
                img = torch.from_numpy(img)
                img = add_trigger(img)
                img = img.numpy()
            # H x W x C
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)
                img = add_trigger(img)
                img = img.permute(1, 2, 0).numpy()
            return img
        elif type(img) == torch.Tensor:
            # H x W
            if img.dim() == 2:
                img = add_trigger(img)
            # H x W x C
            else:
                img = img.permute(2, 0, 1)
                img = add_trigger(img)
                img = img.permute(1, 2, 0)
            return img
        else:
            raise TypeError('img should be PIL.Image.Image or numpy.ndarray or torch.Tensor. Got {}'.format(type(img)))


class AddMNISTTrigger(AddTrigger):
    """Add watermarked trigger to MNIST image.

    Args:
        pattern (None | torch.Tensor): shape (1, 28, 28) or (28, 28).
        weight (None | torch.Tensor): shape (1, 28, 28) or (28, 28).
    """

    def __init__(self, pattern, weight):
        super(AddMNISTTrigger, self).__init__()

        if pattern is None:
            self.pattern = torch.zeros((1, 28, 28), dtype=torch.uint8)
            self.pattern[0, -2, -2] = 255
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if weight is None:
            self.weight = torch.zeros((1, 28, 28), dtype=torch.float32)
            self.weight[0, -2, -2] = 1.0
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = img.squeeze()
        img = Image.fromarray(img.numpy(), mode='L')
        return img


class AddCIFAR10Trigger(AddTrigger):
    """Add watermarked trigger to MNIST image.

    Args:
        pattern (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
        weight (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
    """

    def __init__(self, pattern, weight):
        super(AddCIFAR10Trigger, self).__init__()

        if pattern is None:
            self.pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
            self.pattern[0, -3:, -3:] = 255
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if weight is None:
            self.weight = torch.zeros((1, 32, 32), dtype=torch.float32)
            self.weight[0, -3:, -3:] = 1.0
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img


class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target


class PoisonedDatasetFolder(DatasetFolder):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 pattern,
                 weight,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
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
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
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
                 poisoned_target_transform_index):
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

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

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
                 poisoned_target_transform_index):
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

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


def CreatePoisonedDataset(benign_dataset, y_target, poisoned_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index):
    class_name = type(benign_dataset)
    if class_name == DatasetFolder:
        return PoisonedDatasetFolder(benign_dataset, y_target, poisoned_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index)
    elif class_name == MNIST:
        return PoisonedMNIST(benign_dataset, y_target, poisoned_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index)
    elif class_name == CIFAR10:
        return PoisonedCIFAR10(benign_dataset, y_target, poisoned_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index)
    else:
        raise NotImplementedError


class CreatePoisonedTargetDataset(DatasetFolder):
    def __init__(self,
                 target_adv_dataset,
                 poisoned_set,
                 pattern,
                 weight,
                 poisoned_transform_index):
        super(CreatePoisonedTargetDataset, self).__init__(
            target_adv_dataset.root,
            target_adv_dataset.loader,
            target_adv_dataset.extensions,
            target_adv_dataset.transform,
            target_adv_dataset.target_transform,
            None)
        self.poisoned_set = poisoned_set

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(pattern, weight))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if len(sample.shape) == 2:
            sample = sample.reshape((sample.shape[0], sample.shape[1], 1))
        
        img_index = int(path.split('/')[-1].split('.')[0])
        if img_index in self.poisoned_set:
            sample = self.poisoned_transform(sample) # add trigger to image
        else:
            if self.transform is not None:
                sample = self.transform(sample)

        if self.target_transform is not None: # The process of target transform is the same. 
            target = self.target_transform(target)

        return sample, target


class LabelConsistent(Base):
    """Construct poisoned datasets with Label-consistent method.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        adv_model (torch.nn.Module): Adversarial model to attack to generate adversarial samples.
        adv_dataset_dir (str): The directory to save adversarial dataset.
        loss (torch.nn.Module): Loss.
        y_target (int): N-to-1 attack target label.
        poisoned_rate (float): Ratio of poisoned samples.
        adv_transform (Compose): The data transform for generating adversarial samples, Default: Compose([transforms.ToTensor()]).
        pattern (None | torch.Tensor): Trigger pattern, shape (C, H, W) or (H, W), Default: None.
        weight (None | torch.Tensor): Trigger pattern weight, shape (C, H, W) or (H, W), Default: None.
        eps (float): Maximum perturbation for PGD adversarial attack. Default: 8.
        alpha (float): Step size for PGD adversarial attack. Default: 1.5.
        steps (int): Number of steps for PGD adversarial attack. Default: 100.
        max_pixel (int): Maximum image pixel value. Default: 255.
        poisoned_transform_train_index (int): The position index that poisoned transform will be inserted in train dataset. Default: 0.
        poisoned_transform_test_index (int): The position index that poisoned transform will be inserted in test dataset. Default: 0.
        poisoned_target_transform_index (int): The position that poisoned target transform will be inserted. Default: 0.
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 model,
                 adv_model,
                 adv_dataset_dir,
                 loss,
                 y_target,
                 poisoned_rate,
                 adv_transform=Compose([transforms.ToTensor()]),
                 pattern=None,
                 weight=None,
                 eps=8,
                 alpha=1.5,
                 steps=100,
                 max_pixel=255,
                 poisoned_transform_train_index=0,
                 poisoned_transform_test_index=0,
                 poisoned_target_transform_index=0,
                 schedule=None,
                 seed=0,
                 deterministic=False):

        super(LabelConsistent, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)
        
        if poisoned_rate > 0:
            self.whole_adv_dataset, self.target_adv_dataset, poisoned_set = self._get_adv_dataset(
                train_dataset,
                adv_model=adv_model,
                adv_dataset_dir=adv_dataset_dir,
                adv_transform=adv_transform,
                eps=eps/max_pixel,
                alpha=alpha/max_pixel,
                steps=steps,
                y_target=y_target,
                poisoned_rate=poisoned_rate)

            self.poisoned_train_dataset = CreatePoisonedTargetDataset(
                self.target_adv_dataset,
                poisoned_set,
                pattern,
                weight,
                poisoned_transform_train_index)
        else:
            self.poisoned_train_dataset = train_dataset

        self.poisoned_test_dataset = CreatePoisonedDataset(
            test_dataset,
            y_target,
            1.0,
            pattern,
            weight,
            poisoned_transform_test_index,
            poisoned_target_transform_index)

    def _get_adv_dataset(self, dataset, adv_model, adv_dataset_dir, adv_transform, eps, alpha, steps, y_target, poisoned_rate):

        def _generate_adv_dataset(dataset, adv_model, adv_dataset_dir, adv_transform, eps, alpha, steps, y_target, poisoned_rate):
            if self.current_schedule is None and self.global_schedule is None:
                self.current_schedule = {
                    'device': 'CPU',
                    'batch_size': 128,
                    'num_workers': 8
                }
            elif self.current_schedule is None and self.global_schedule is not None:
                self.current_schedule = deepcopy(self.global_schedule)

            if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
                if 'CUDA_VISIBLE_DEVICES' in self.current_schedule:
                    os.environ['CUDA_VISIBLE_DEVICES'] = self.current_schedule['CUDA_VISIBLE_DEVICES']

                assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
                assert self.current_schedule['GPU_num'] >0, 'GPU_num should be a positive integer'
                print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {self.current_schedule['GPU_num']} of them to train.")

                if self.current_schedule['GPU_num'] == 1:
                    device = torch.device("cuda:0")
                else:
                    gpus = list(range(self.current_schedule['GPU_num']))
                    self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                    # TODO: DDP training
                    pass
            # Use CPU
            else:
                device = torch.device("cpu")


            adv_model = adv_model.to(device)

            backup_transform = deepcopy(dataset.transform)
            dataset.transform = adv_transform

            data_loader = DataLoader(
                dataset,
                batch_size=self.current_schedule['batch_size'],
                shuffle=False,
                num_workers=self.current_schedule['num_workers'],
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )

            attacker = PGD(adv_model, eps, alpha, steps, random_start=False)
            attacker.set_return_type("int")

            original_imgs = []
            perturbed_imgs = []
            targets = []

            for batch in tqdm(data_loader):
                # Adversarially perturb image. Note that torchattacks will automatically
                # move `img` and `target` to the gpu where the attacker.model is located.
                batch_img = batch[0]
                batch_label = batch[1]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                img = attacker(batch_img, batch_label)
                original_imgs.append(torch.round(batch_img * 255).to(dtype=torch.uint8).permute(0, 2, 3, 1).detach().cpu())
                perturbed_imgs.append(img.permute(0, 2, 3, 1).detach().cpu())
                targets.append(batch_label.cpu())

            dataset.transform = backup_transform

            original_imgs = torch.cat(original_imgs, dim=0).numpy()
            perturbed_imgs = torch.cat(perturbed_imgs, dim=0).numpy()
            targets = torch.cat(targets, dim=0).numpy()

            y_target_index_list = np.squeeze(np.argwhere(targets == y_target))
            total_target_num = len(y_target_index_list)
            poisoned_num = int(total_target_num * poisoned_rate)
            assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
            random.shuffle(y_target_index_list)
            poisoned_set = frozenset(list(y_target_index_list[:poisoned_num]))

            for target in np.unique(targets):
                os.makedirs(osp.join(adv_dataset_dir, 'whole_adv_dataset', str(target).zfill(2)), exist_ok=True)
                os.makedirs(osp.join(adv_dataset_dir, 'target_adv_dataset', str(target).zfill(2)), exist_ok=True)

            np.save(osp.join(adv_dataset_dir, 'poisoned_set.npy'), y_target_index_list[:poisoned_num])

            for index, item in enumerate(zip(original_imgs, perturbed_imgs, targets)):
                original_img, perturbed_img, target = item
                cv2.imwrite(osp.join(adv_dataset_dir, 'whole_adv_dataset', str(target).zfill(2), str(index).zfill(8) + '.png'), perturbed_img)

                if index in poisoned_set:
                    cv2.imwrite(osp.join(adv_dataset_dir, 'target_adv_dataset', str(target).zfill(2), str(index).zfill(8) + '.png'), perturbed_img)
                else:
                    cv2.imwrite(osp.join(adv_dataset_dir, 'target_adv_dataset', str(target).zfill(2), str(index).zfill(8) + '.png'), original_img)


        if not osp.exists(osp.join(adv_dataset_dir, 'whole_adv_dataset')) or not osp.exists(osp.join(adv_dataset_dir, 'target_adv_dataset')):
            _generate_adv_dataset(dataset, adv_model, adv_dataset_dir, adv_transform, eps, alpha, steps, y_target, poisoned_rate)

        whole_adv_dataset = DatasetFolder(
            root=osp.join(adv_dataset_dir, 'whole_adv_dataset'),
            loader=my_imread,
            extensions=('png',),
            transform=deepcopy(dataset.transform),
            target_transform=deepcopy(dataset.target_transform),
            is_valid_file=None
        )

        target_adv_dataset = DatasetFolder(
            root=osp.join(adv_dataset_dir, 'target_adv_dataset'),
            loader=my_imread,
            extensions=('png',),
            transform=deepcopy(dataset.transform),
            target_transform=deepcopy(dataset.target_transform),
            is_valid_file=None
        )

        poisoned_set = np.load(osp.join(adv_dataset_dir, 'poisoned_set.npy'))
        poisoned_set = frozenset(list(poisoned_set))

        return whole_adv_dataset, target_adv_dataset, poisoned_set
