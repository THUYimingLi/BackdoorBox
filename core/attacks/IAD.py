'''
This is the implement of IAD [1]. 
This code is developed based on its official codes (https://github.com/VinAIResearch/input-aware-backdoor-attack-release)

Reference:
[1] Input-Aware Dynamic Backdoor Attack. NeurIPS 2020.
'''


import warnings
warnings.filterwarnings("ignore")
import os
import os.path as osp
import time
from copy import deepcopy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import DatasetFolder, MNIST, CIFAR10

from ..utils import Log
from .base import *


class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, targets):
        return torch.ones_like(targets) * self.y_target


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
        img = torch.tensor(self.data_list[index])
        label = torch.tensor(self.labels[index])
        return img, label


class Normalize:
    """Normalization of images.

    Args:
        dataset_name (str): the name of the dataset to be normalized.
        expected_values (float): the normalization expected values.
        variance (float): the normalization variance.
    """
    def __init__(self, dataset_name, expected_values, variance):
        if dataset_name == "cifar10" or dataset_name == "gtsrb":
            self.n_channels = 3
        elif dataset_name == "mnist":
            self.n_channels = 1
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    """Denormalization of images.

    Args:
        dataset_name (str): the name of the dataset to be denormalized.
        expected_values (float): the denormalization expected values.
        variance (float): the denormalization variance.
    """
    def __init__(self, dataset_name, expected_values, variance):
        if dataset_name == "cifar10" or dataset_name == "gtsrb":
            self.n_channels = 3
        elif dataset_name == "mnist":
            self.n_channels = 1
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


# ===== The generator of dynamic backdoor trigger ===== 
class Conv2dBlock(nn.Module):
    """The Conv2dBlock in the generator of dynamic backdoor trigger."""
    def __init__(self, in_c, out_c, ker_size=(3, 3), stride=1, padding=1, batch_norm=True, relu=True):
        super(Conv2dBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_c, out_c, ker_size, stride, padding)
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.05, affine=True)
        if relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class DownSampleBlock(nn.Module):
    """The DownSampleBlock in the generator of dynamic backdoor trigger."""
    def __init__(self, ker_size=(2, 2), stride=2, dilation=(1, 1), ceil_mode=False, p=0.0):
        super(DownSampleBlock, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=ker_size, stride=stride, dilation=dilation, ceil_mode=ceil_mode)
        if p:
            self.dropout = nn.Dropout(p)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class UpSampleBlock(nn.Module):
    """The UpSampleBlock in the generator of dynamic backdoor trigger."""
    def __init__(self, scale_factor=(2, 2), mode="nearest", p=0.0):
        super(UpSampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        if p:
            self.dropout = nn.Dropout(p)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class Generator(nn.Sequential):
    """The generator of dynamic backdoor trigger.
    
    Args:
        dataset_name (str): the name of the dataset.
        out_channels (int): the output channel of the generator. 
    """
    def __init__(self, dataset_name, out_channels=None):
        super(Generator, self).__init__()
        if dataset_name == "mnist":
            channel_init = 16
            steps = 2
            input_channel = 1
            channel_current = 1
        else:
            channel_init = 32
            steps = 3
            input_channel = 3
            channel_current = 3

        channel_next = channel_init
        for step in range(steps):
            self.add_module("convblock_down_{}".format(2 * step), Conv2dBlock(channel_current, channel_next))
            self.add_module("convblock_down_{}".format(2 * step + 1), Conv2dBlock(channel_next, channel_next))
            self.add_module("downsample_{}".format(step), DownSampleBlock())
            if step < steps - 1:
                channel_current = channel_next
                channel_next *= 2

        self.add_module("convblock_middle", Conv2dBlock(channel_next, channel_next))

        channel_current = channel_next
        channel_next = channel_current // 2
        for step in range(steps):
            self.add_module("upsample_{}".format(step), UpSampleBlock())
            self.add_module("convblock_up_{}".format(2 * step), Conv2dBlock(channel_current, channel_current))
            if step == steps - 1:
                self.add_module(
                    "convblock_up_{}".format(2 * step + 1), Conv2dBlock(channel_current, channel_next, relu=False)
                )
            else:
                self.add_module("convblock_up_{}".format(2 * step + 1), Conv2dBlock(channel_current, channel_next))
            channel_current = channel_next
            channel_next = channel_next // 2
            if step == steps - 2:
                if out_channels is None:
                    channel_next = input_channel
                else:
                    channel_next = out_channels

        self._EPSILON = 1e-7
        self._normalizer = self._get_normalize(dataset_name)
        self._denormalizer = self._get_denormalize(dataset_name)

    def _get_denormalize(self, dataset_name):
        if dataset_name == "cifar10":
            denormalizer = Denormalize(dataset_name, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif dataset_name == "mnist":
            denormalizer = Denormalize(dataset_name, [0.5], [0.5])
        elif dataset_name == "gtsrb":
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, dataset_name):
        if dataset_name == "cifar10":
            normalizer = Normalize(dataset_name, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif dataset_name == "mnist":
            normalizer = Normalize(dataset_name, [0.5], [0.5])
        elif dataset_name == "gtsrb":
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        return normalizer

    def forward(self, x):
        for module in self.children():
            x = module(x)
        x = nn.Tanh()(x) / (2 + self._EPSILON) + 0.5
        return x

    def normalize_pattern(self, x):
        if self._normalizer:
            x = self._normalizer(x)
        return x

    def denormalize_pattern(self, x):
        if self._denormalizer:
            x = self._denormalizer(x)
        return x

    def threshold(self, x):
        return nn.Tanh()(x * 20 - 10) / (2 + self._EPSILON) + 0.5

# ===== The generator of dynamic backdoor trigger (done) ===== 


class IAD(Base):
    """Construct backdoored model with IAD method.

    Args:
        dataset_name (str): the name of the dataset.
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        train_dataset1 (types in support_list): Another benign training dataset to implement the diversity loss in [1].
        test_dataset1 (types in support_list): Another benign testing dataset to implement the diversity loss in [1].
        model (torch.nn.Module): Victim model.
        loss (torch.nn.Module): Loss.
        y_target (int): N-to-1 attack target label.
        poisoned_rate (float): Ratio of poisoned samples.
        cross_rate (float): Ratio of samples in diversity loss.
        lambda_div (float): Hyper-parameter in diversity loss.
        lambda_norm (float): Hyper-parameter in mask generation loss.
        mask_density (float): Magnitude of the generated mask in the backdoor trigger.
        EPSILON (float): Preventing divisor 0 errors in diversity loss.
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """
    def __init__(self, 
                 dataset_name,
                 train_dataset, 
                 test_dataset, 
                 train_dataset1, 
                 test_dataset1, 
                 model, 
                 loss, 
                 y_target,
                 poisoned_rate,
                 cross_rate,
                 lambda_div,
                 lambda_norm,
                 mask_density,
                 EPSILON,
                 schedule=None, 
                 seed=0, 
                 deterministic=False,
                 ):
        super(IAD, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)
        
        self.dataset_name = dataset_name
        self.train_dataset1 = train_dataset1
        self.test_dataset1 = test_dataset1
        self.y_target = y_target
        self.poisoned_rate = poisoned_rate
        self.cross_rate = cross_rate
        self.lambda_div = lambda_div
        self.lambda_norm = lambda_norm
        self.mask_density = mask_density
        self.EPSILON = EPSILON
        self.create_targets_bd = ModifyTarget(self.y_target)
        self.train_poisoned_data = []
        self.train_poisoned_label = []
        self.test_poisoned_data = []
        self.test_poisoned_label = []

    def train(self, schedule=None):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        if 'pretrain' in self.current_schedule:
            self.model.load_state_dict(torch.load(self.current_schedule['pretrain']), strict=False)

        # Use GPU
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
        self.device = device

        # Prepare the dataset and construct the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.current_schedule['batch_size'],
            shuffle=True,
            num_workers=self.current_schedule['num_workers'],
            drop_last=True,
            worker_init_fn=self._seed_worker
        )
        train_loader1 = DataLoader(
            self.train_dataset1,
            batch_size=self.current_schedule['batch_size'],
            shuffle=True,
            num_workers=self.current_schedule['num_workers'],
            drop_last=True,
            worker_init_fn=self._seed_worker
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.current_schedule['batch_size'],
            shuffle=False,
            num_workers=self.current_schedule['num_workers'],
            drop_last=False,
            worker_init_fn=self._seed_worker
        )
        test_loader1 = DataLoader(
            self.test_dataset1,
            batch_size=self.current_schedule['batch_size'],
            shuffle=True,
            num_workers=self.current_schedule['num_workers'],
            drop_last=False,
            worker_init_fn=self._seed_worker
        )

        # Prepare the victim classification model
        self.model = self.model.to(device)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_schedule['lr'], momentum=self.current_schedule['momentum'], weight_decay=self.current_schedule['weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.current_schedule['milestones'], self.current_schedule['lambda'])

        # Prepare the backdoor trigger pattern generator
        self.modelG = Generator(self.dataset_name).to(device)
        optimizerG = torch.optim.Adam(self.modelG.parameters(), lr=self.current_schedule['lr_G'], betas=self.current_schedule['betas_G'])
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, self.current_schedule['milestones_G'], self.current_schedule['lambda_G'])

        # Prepare the backdoor trigger mask generator
        self.modelM = Generator(self.dataset_name, out_channels=1).to(device)
        optimizerM = torch.optim.Adam(self.modelM.parameters(), lr=self.current_schedule['lr_M'], betas=self.current_schedule['betas_M'])
        schedulerM = torch.optim.lr_scheduler.MultiStepLR(optimizerM, self.current_schedule['milestones_M'], self.current_schedule['lambda_M'])

        # The path to save log files and checkpoints
        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))
        self.work_dir = work_dir

        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint

        self.iteration = 0
        epoch = 1
        last_time = time.time()
        msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.test_dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(self.train_dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
        log(msg)

        # The backdoor trigger mask generator will be trained independently first in early epochs 
        if epoch == 1:
            self.modelM.train()
            for i in range(25):
                msg = "Epoch {} | mask_density: {} | - {}  - lambda_div: {}  - lambda_norm: {}\n".format(
                        epoch, self.mask_density, self.dataset_name, self.lambda_div, self.lambda_norm
                    )
                log(msg)
                
                total_loss, loss_norm, loss_div = self.train_mask_step(self.modelM, optimizerM, schedulerM, train_loader, train_loader1)
                msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + "Train Mask loss: {:.4f} | Norm: {:.3f} | Diversity: {:.3f}\n".format(total_loss, loss_norm, loss_div)
                log(msg)
                
                loss_norm_eval, loss_div_eval = self.eval_mask(self.modelM, test_loader, test_loader1)
                if epoch % self.current_schedule['test_epoch_interval'] == 0:
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + "Test Norm: {:.3f} | Diversity: {:.3f}\n".format(loss_norm_eval, loss_div_eval)
                    log(msg)
                epoch += 1

        # The backdoor trigger mask generator will been frozen
        self.modelM.eval()
        self.modelM.requires_grad_(False)

        best_acc_clean = -1
        best_acc_bd = -1
        best_acc_cross = -1
        best_epoch = 1
        for i in range(self.current_schedule['epochs']):
            msg = f"Epoch {epoch} | mask_density: {self.mask_density} | - {self.dataset_name} - lambda_div: {self.lambda_div}\n"
            log(msg)

            # Train the victim model and the backdoor trigger pattern generator jointly
            avg_loss, acc_clean, acc_bd, acc_cross = self.train_step(
                self.model,
                self.modelG,
                self.modelM,
                optimizer,
                optimizerG,
                scheduler,
                schedulerG,
                train_loader,
                train_loader1
            )

            msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + "Train CE loss: {:.4f} | BA: {:.3f} | ASR: {:.3f} | Cross Accuracy: {:3f}\n".format(
                avg_loss, acc_clean, acc_bd, acc_cross
            )
            log(msg)

            if epoch % self.current_schedule['test_epoch_interval'] == 0:
                last_time = time.time()
                # Test the victim model
                avg_acc_clean, avg_acc_bd, avg_acc_cross = self.test(
                    test_loader,
                    test_loader1,
                    self.model,
                    self.modelG,
                    self.modelM
                )
                msg = "==========Test result on benign test dataset==========\n" + \
                        time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                        f"Accuracy: {avg_acc_clean}, time: {time.time()-last_time}\n"
                log(msg)
                msg = "==========Test result on poisoned test dataset==========\n" + \
                        time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                        f"Accuracy: {avg_acc_bd}, time: {time.time()-last_time}\n"
                log(msg)
                msg = "==========Test result on cross test dataset==========\n" + \
                        time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                        f"Accuracy: {avg_acc_cross}, time: {time.time()-last_time}\n"
                log(msg)

                # Record the best checkpoints
                if (avg_acc_clean + avg_acc_bd + avg_acc_cross) > (best_acc_clean + best_acc_bd + best_acc_cross):
                    best_acc_clean = avg_acc_clean
                    best_acc_bd = avg_acc_bd
                    best_acc_cross = avg_acc_cross
                    best_epoch = epoch
                    best_state_dict = {
                        "model": self.model.state_dict(),
                        "modelG": self.modelG.state_dict(),
                        "modelM": self.modelM.state_dict(),
                        "optimizerC": optimizer.state_dict(),
                        "optimizerG": optimizerG.state_dict(),
                        "schedulerC": scheduler.state_dict(),
                        "schedulerG": schedulerG.state_dict(),
                        "best_acc_clean": best_acc_clean,
                        "best_acc_bd": best_acc_bd,
                        "best_acc_cross": best_acc_cross,
                        "best_epoch": best_epoch
                    }
            
            # Save the checkpoints
            if epoch % self.current_schedule['save_epoch_interval'] == 0:
                print(" Saving!!")
                state_dict = {
                    "model": self.model.state_dict(),
                    "modelG": self.modelG.state_dict(),
                    "modelM": self.modelM.state_dict(),
                    "optimizerC": optimizer.state_dict(),
                    "optimizerG": optimizerG.state_dict(),
                    "schedulerC": scheduler.state_dict(),
                    "schedulerG": schedulerG.state_dict(),
                    "avg_acc_clean": avg_acc_clean,
                    "avg_acc_bd": avg_acc_bd,
                    "avg_acc_cross": avg_acc_cross,
                    "epoch": epoch
                }
                ckpt_model_filename = "ckpt_epoch_" + str(epoch) + ".pth"
                ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                torch.save(state_dict, ckpt_model_path)
            
            epoch += 1
            if epoch > self.current_schedule['epochs']:
                print(" Saving!!")
                ckpt_model_filename = "best_ckpt_epoch_" + str(epoch) + ".pth"
                ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                # Save the best checkpoints
                torch.save(best_state_dict, ckpt_model_path)
                msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    "Best Epoch {}: | Best BA: {:.3f} | Best ASR: {:.3f} | Best Cross Accuracy: {:3f}\n".format(
                        best_epoch, best_acc_clean, best_acc_bd, best_acc_cross
                    )
                log(msg)
                break   
       


    def train_step(
        self, 
        model, 
        modelG, 
        modelM, 
        optimizerC, 
        optimizerG, 
        schedulerC, 
        schedulerG, 
        train_dl1, 
        train_dl2
    ):
        """Train the victim model and the backdoor trigger pattern generator jointly.
        
        Args:
            model (torch.nn.Module): Victim model.
            modelG (torch.nn.Module): Backdoor trigger pattern generator.
            modelM (torch.nn.Module): Backdoor trigger mask generator.
            optimizerC (torch.optim.Optimizer): Optimizer of the victim model.
            optimizerG (torch.optim.Optimizer): Optimizer of the backdoor trigger pattern generator.
            schedulerC (torch.optim.lr_scheduler._LRScheduler): Scheduler of the victim model.
            schedulerG (torch.optim.lr_scheduler._LRScheduler): Scheduler of the backdoor trigger pattern generator.
            train_dl1 (torch.utils.data.DataLoader): Benign training dataloader.
            train_dl2 (torch.utils.data.DataLoader): Another benign training dataloader to implement the diversity loss in [1].
        """
        model.train()
        modelG.train()
        total = 0
        total_cross = 0
        total_bd = 0
        total_clean = 0

        total_correct_clean = 0
        total_cross_correct = 0
        total_bd_correct = 0

        # Construct the classification loss and the diversity loss
        total_loss = 0
        criterion = self.loss
        criterion_div = nn.MSELoss(reduction="none")
        self.train_poisoned_data, self.train_poisoned_label = [], []
        for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(train_dl1)), train_dl1, train_dl2):
            optimizerC.zero_grad()

            inputs1, targets1 = inputs1.to(self.device), targets1.to(self.device)
            inputs2, targets2 = inputs2.to(self.device), targets2.to(self.device)

            # Construct the benign samples, backdoored samples and cross samples
            bs = inputs1.shape[0]
            num_bd = int(self.poisoned_rate * bs)
            num_cross = int(self.cross_rate * bs)

            inputs_bd, targets_bd, patterns1, masks1 = self.create_bd(inputs1[:num_bd], targets1[:num_bd], modelG, modelM)
            inputs_cross, patterns2, masks2 = self.create_cross(
                inputs1[num_bd : num_bd + num_cross], inputs2[num_bd : num_bd + num_cross], modelG, modelM
            )

            total_inputs = torch.cat((inputs_bd, inputs_cross, inputs1[num_bd + num_cross :]), 0)
            total_targets = torch.cat((targets_bd, targets1[num_bd:]), 0)

            self.train_poisoned_data += total_inputs.detach().cpu().numpy().tolist()
            self.train_poisoned_label += total_targets.detach().cpu().numpy().tolist()

            # Calculating the classification loss
            preds = model(total_inputs)
            loss_ce = criterion(preds, total_targets)

            # Calculating diversity loss
            distance_images = criterion_div(inputs1[:num_bd], inputs2[num_bd : num_bd + num_bd])
            distance_images = torch.mean(distance_images, dim=(1, 2, 3))
            distance_images = torch.sqrt(distance_images)

            distance_patterns = criterion_div(patterns1, patterns2)
            distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
            distance_patterns = torch.sqrt(distance_patterns)

            loss_div = distance_images / (distance_patterns + self.EPSILON)
            loss_div = torch.mean(loss_div) * self.lambda_div

            # Total loss
            total_loss = loss_ce + loss_div
            total_loss.backward()
            optimizerC.step()
            optimizerG.step()

            total += bs
            total_bd += num_bd
            total_cross += num_cross
            total_clean += bs - num_bd - num_cross

            # Calculating the clean accuracy
            total_correct_clean += torch.sum(
                torch.argmax(preds[num_bd + num_cross :], dim=1) == total_targets[num_bd + num_cross :]
            )
            # Calculating the diversity accuracy
            total_cross_correct += torch.sum(
                torch.argmax(preds[num_bd : num_bd + num_cross], dim=1) == total_targets[num_bd : num_bd + num_cross]
            )
            # Calculating the backdoored accuracy
            total_bd_correct += torch.sum(torch.argmax(preds[:num_bd], dim=1) == targets_bd)
            total_loss += loss_ce.detach() * bs
            avg_loss = total_loss / total

            acc_clean = total_correct_clean * 100.0 / total_clean
            acc_bd = total_bd_correct * 100.0 / total_bd
            acc_cross = total_cross_correct * 100.0 / total_cross

            # Saving images for debugging
            if batch_idx == len(train_dl1) - 2:
                images = modelG.denormalize_pattern(torch.cat((inputs1[:num_bd], inputs_bd), dim=2))
                file_name = "{}_images.png".format(self.dataset_name)
                file_path = os.path.join(self.work_dir, file_name)
                torchvision.utils.save_image(images, file_path, normalize=True, pad_value=1)
        schedulerC.step()
        schedulerG.step()
        return avg_loss, acc_clean, acc_bd, acc_cross


    def test(
        self, 
        test_dl1,
        test_dl2,
        model=None,
        modelG=None,
        modelM=None,
        schedule=None
    ):
        """Test the victim model.
        
        Args:
            test_dl1 (torch.utils.data.DataLoader): Benign testing dataloader
            test_dl2 (torch.utils.data.DataLoader): Another benign testing dataloader to implement the diversity loss in [1].
            model (torch.nn.Module): Victim model. Default: None.
            modelG (torch.nn.Module): Backdoor trigger pattern generator. Default: None.
            modelM (torch.nn.Module): Backdoor trigger mask generator. Default: None.
            schedule (dict): Testing schedule. Default: None.
        """
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Test schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.schedule is not None:
            self.current_schedule = deepcopy(schedule)

        if model is None:
            model = self.model

        if modelG is None:
            modelG = self.modelG
        
        if modelM is None:
            modelM = self.modelM

        if 'test_model' in self.current_schedule:
            model.load_state_dict(torch.load(self.current_schedule['test_model']), strict=False)

        if 'test_modelG' in self.current_schedule:
            model.load_state_dict(torch.load(self.current_schedule['test_modelG']), strict=False)

        if 'test_modelM' in self.current_schedule:
            model.load_state_dict(torch.load(self.current_schedule['test_modelM']), strict=False)

        # Use GPU
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
                model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")
        
        device = self.device if self.device else device
        model.eval()
        modelG.eval()
        total = 0.0

        total_correct_clean = 0.0
        total_correct_bd = 0.0
        total_correct_cross = 0.0
        self.test_poisoned_data, self.test_poisoned_label = [], []
        for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(test_dl1)), test_dl1, test_dl2):
            with torch.no_grad():
                inputs1, targets1 = inputs1.to(device), targets1.to(device)
                inputs2, targets2 = inputs2.to(device), targets2.to(device)
                bs = inputs1.shape[0]

                # Construct the benign samples and calculate the clean accuracy
                preds_clean = model(inputs1)
                correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets1)
                total_correct_clean += correct_clean

                # Construct the backdoored samples and calculate the backdoored accuracy
                inputs_bd, targets_bd, _, _ = self.create_bd(inputs1, targets1, modelG, modelM)

                self.test_poisoned_data += inputs_bd.detach().cpu().numpy().tolist()
                self.test_poisoned_label += targets_bd.detach().cpu().numpy().tolist()

                preds_bd = model(inputs_bd)
                correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
                total_correct_bd += correct_bd

                # Construct the cross samples and calculate the diversity accuracy
                inputs_cross, _, _ = self.create_cross(inputs1, inputs2, modelG, modelM)
                preds_cross = model(inputs_cross)
                correct_cross = torch.sum(torch.argmax(preds_cross, 1) == targets1)
                total_correct_cross += correct_cross

                total += bs
                avg_acc_clean = total_correct_clean * 100.0 / total
                avg_acc_cross = total_correct_cross * 100.0 / total
                avg_acc_bd = total_correct_bd * 100.0 / total
        
        return avg_acc_clean, avg_acc_bd, avg_acc_cross


    def train_mask_step(
        self, 
        modelM, 
        optimizerM, 
        schedulerM, 
        train_dl1, 
        train_dl2
    ):
        """Train the backdoor trigger mask generator.
        
        Args:
            modelM (torch.nn.Module): Backdoor trigger mask generator.
            optimizerM (torch.optim.Optimizer): Optimizer of the backdoor trigger mask generator.
            schedulerM (torch.optim.lr_scheduler._LRScheduler): Scheduler of backdoor trigger mask generator.
            train_dl1 (torch.utils.data.DataLoader): Benign training dataloader
            train_dl2 (torch.utils.data.DataLoader): Another benign training dataloader to implement the diversity loss in [1].
        """
        modelM.train()

        total_loss = 0
        criterion_div = nn.MSELoss(reduction="none")
        for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(train_dl1)), train_dl1, train_dl2):
            optimizerM.zero_grad()

            inputs1, targets1 = inputs1.to(self.device), targets1.to(self.device)
            inputs2, targets2 = inputs2.to(self.device), targets2.to(self.device)

            # Generate the mask of data
            masks1 = modelM(inputs1)
            masks1, masks2 = modelM.threshold(modelM(inputs1)), modelM.threshold(modelM(inputs2))

            # Calculating diversity loss
            distance_images = criterion_div(inputs1, inputs2)
            distance_images = torch.mean(distance_images, dim=(1, 2, 3))
            distance_images = torch.sqrt(distance_images)

            distance_patterns = criterion_div(masks1, masks2)
            distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
            distance_patterns = torch.sqrt(distance_patterns)

            loss_div = distance_images / (distance_patterns + self.EPSILON)
            loss_div = torch.mean(loss_div) * self.lambda_div

            # Calculating mask magnitude loss
            loss_norm = torch.mean(F.relu(masks1 - self.mask_density))

            total_loss = self.lambda_norm * loss_norm + self.lambda_div * loss_div
            total_loss.backward()
            optimizerM.step()

        schedulerM.step()
        return total_loss, loss_norm, loss_div


    def eval_mask(
        self, 
        modelM, 
        test_dl1, 
        test_dl2
    ):
        """Test the backdoor trigger mask generator.
        
        Args:
            modelM (torch.nn.Module): Backdoor trigger mask generator.
            test_dl1 (torch.utils.data.DataLoader): Benign testing dataloader
            test_dl2 (torch.utils.data.DataLoader): Another benign testing dataloader to implement the diversity loss in [1].
        """
        modelM.eval()

        criterion_div = nn.MSELoss(reduction="none")
        for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(test_dl1)), test_dl1, test_dl2):
            with torch.no_grad():
                inputs1, targets1 = inputs1.to(self.device), targets1.to(self.device)
                inputs2, targets2 = inputs2.to(self.device), targets2.to(self.device)

                # Generate the mask of data
                bs = inputs1.shape[0]
                masks1, masks2 = modelM.threshold(modelM(inputs1)), modelM.threshold(modelM(inputs2))

                # Calculating diversity loss
                distance_images = criterion_div(inputs1, inputs2)
                distance_images = torch.mean(distance_images, dim=(1, 2, 3))
                distance_images = torch.sqrt(distance_images)

                distance_patterns = criterion_div(masks1, masks2)
                distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
                distance_patterns = torch.sqrt(distance_patterns)

                loss_div = distance_images / (distance_patterns + self.EPSILON)
                loss_div = torch.mean(loss_div) * self.lambda_div

                # Calculating mask magnitude loss
                loss_norm = torch.mean(F.relu(masks1 - self.mask_density))

        return loss_norm, loss_div


    def create_bd(
        self, 
        inputs, 
        targets, 
        modelG, 
        modelM
    ):
        """Construct the backdoored samples by the backdoor trigger mask generator and backdoor trigger pattern generator.
        
        Args:
            inputs (torch.Tensor): Benign samples to be attached with the backdoor trigger.
            targets (int): The attacker-specified target label.
            modelG (torch.nn.Module): Backdoor trigger pattern generator.
            modelM (torch.nn.Module): Backdoor trigger mask generator.
        """
        bd_targets = self.create_targets_bd(targets).to(self.device)
        patterns = modelG(inputs)
        patterns = modelG.normalize_pattern(patterns)
        masks_output = modelM.threshold(modelM(inputs))
        bd_inputs = inputs + (patterns - inputs) * masks_output
        return bd_inputs, bd_targets, patterns, masks_output


    def create_cross(
        self, 
        inputs1, 
        inputs2, 
        modelG, 
        modelM
    ):
        """Construct the cross samples to implement the diversity loss in [1].
        
        Args:
            inputs1 (torch.Tensor): Benign samples.
            inputs2 (torch.Tensor): Benign samples different from inputs1.
            modelG (torch.nn.Module): Backdoor trigger pattern generator.
            modelM (torch.nn.Module): Backdoor trigger mask generator.
        """
        patterns2 = modelG(inputs2)
        patterns2 = modelG.normalize_pattern(patterns2)
        masks_output = modelM.threshold(modelM(inputs2))
        inputs_cross = inputs1 + (patterns2 - inputs1) * masks_output
        return inputs_cross, patterns2, masks_output


    def get_model(self):
        """
            Return the victim model.
        """
        return self.model


    def get_modelM(self):
        """
            Return the backdoor trigger mask generator.
        """
        return self.modelM


    def get_modelG(self):
        """
            Return the backdoor trigger pattern generator.
        """
        return self.modelG


    def get_poisoned_dataset(self):
        """
            Return the poisoned dataset.
        """
        warnings.warn("IAD is implemented by controlling the training process so that the poisoned dataset is dynamic.")
        if len(self.train_poisoned_data) == 0 and len(self.test_poisoned_data) == 0:
            return None, None
        elif len(self.train_poisoned_data) == 0 and len(self.test_poisoned_data) != 0:
            poisoned_test_dataset = GetPoisonedDataset(self.test_poisoned_data, self.test_poisoned_label)
            return None, poisoned_test_dataset
        elif len(self.train_poisoned_data) != 0 and len(self.test_poisoned_data) == 0:
            poisoned_train_dataset = GetPoisonedDataset(self.train_poisoned_data, self.train_poisoned_label)
            return poisoned_train_dataset, None
        else:
            poisoned_train_dataset = GetPoisonedDataset(self.train_poisoned_data, self.train_poisoned_label)
            poisoned_test_dataset = GetPoisonedDataset(self.test_poisoned_data, self.test_poisoned_label)
            return poisoned_train_dataset, poisoned_test_dataset
