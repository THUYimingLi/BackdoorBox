'''
This is the implement of ABL [1]. 
This code is developed based on its official codes. (https://github.com/bboylyg/ABL)

Reference:
[1] Anti-backdoor Learning: Training Clean Models on Poisoned Data. NeurIPS, 2021.
'''


from multiprocessing.sharedctypes import Value
import os
import os.path as osp
from copy import deepcopy
import time
from turtle import forward
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Base

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, MNIST, DatasetFolder

from ..utils.accuracy import accuracy
from ..utils.log import Log


class LGALoss(nn.Module):
    def __init__(self, loss, gamma):
        """The local gradient ascent (LGA) loss used in first phrase (called pre-isolation phrase) in ABL.

        Args:
            loss (nn.Module): Loss for repaired model training. Please specify the reduction augment in the loss.
            gamma (float): Lower Bound for repairing model    
        """
        super().__init__()
        self.loss = loss
        self.gamma = gamma
        
        if not hasattr(loss, 'reduction'):
            raise ValueError("Loss module must have the attribute named reduction!")
        if loss.reduction not in ['none', 'mean']:
            raise NotImplementedError("This loss only support loss.reduction='mean' or loss.reduction='none'")
    
    def forward(self, logits, targets):
        loss = self.loss(logits, targets)
        loss = torch.sign(loss-self.gamma) * loss
        if self.loss.reduction=='none':
            return loss.mean()            

class ABL(Base):
    """Repair a model via Anti-backdoor Learning (ABL).

    Args:
        model (nn.Module): Repaired model.
        loss (nn.Module): Loss for repaired model training.
        poisoned_trainset (types in support_list): Poisoned trainset.
        poisoned_testset (types in support_list): Poisoned testset.
        clean_testset (types in support_list): Clean testset.
        seed (int): Global seed for random numbers. Default: 0.
        split_ratio (float): Ratio of samples that are considered as poisoned
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    
    """
    def __init__(self,   
                 model, 
                 loss, 
                 poisoned_trainset, 
                 poisoned_testset, 
                 clean_testset, 
                 seed=0,                  
                 deterministic=False):
        super(ABL, self).__init__(seed, deterministic)
        """Repair a model via Anti-backdoor Learning (ABL).

        Args:
            model (nn.Module): Repaired model.
            loss (nn.Module): Loss for repaired model training.
            poisoned_trainset (torch.utils.data.Dataset): Poisoned trainset
            poisoned_testset (torch.utils.data.Dataset): Poisoned testset
            clean_testset (torch.utils.data.Dataset): Clean testset
            seed (int): Global seed for random numbers. Default: 0.
            deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
                That is, algorithms which, given the same input, and when run on the same software and hardware,
                always produce the same output. When enabled, operations will use deterministic algorithms when available,
                and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.       
        """
        self.model = model
        self.loss = loss
        self.seed = seed
        self.poisoned_trainset = poisoned_trainset
        self.poisoned_testset = poisoned_testset
        self.clean_testset = clean_testset

    def _seed_worker(self, worker_id):
        """Assign seed to workers of dataloader. Make the results reproduceable."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def save_ckpt(self, ckpt_name):
        ckpt_model_path = os.path.join(self.work_dir, ckpt_name)
        torch.save(self.model.cpu().state_dict(), ckpt_model_path)
        return 

    def get_filtered_poisoned_dataset(self):
        "Return filtered poisoned dataset. Must call ABL.train() first to filter out poisoned samples."
        if hasattr(self, "poisoned_dataset"):
            return self.poisoned_dataset
        else:
            raise ValueError("Must call ABL.train() first to filter out poisoned samples")
        
    def get_model(self):
        """Return trained model. Should call ABL.train() first to train model. """
        return self.model

    def train(self, split_ratio, isolation_criterion, gamma, schedule, transform, selection_criterion):
        """Perform ABL defense method based on attacked models. 
        The repaired model will be stored in self.model
        
        Args:
            split_ratio (float): Ratio of trainset that are used in unlearning.
            gamma (float): The threshold of loss in first phrase. Model is optimized that the minimal loss won't be lower than gamma.
            schedule (dict): Schedule for ABL. Contraining sub-schedule for pre-isolatoin training, clean training, unlearning and test phrase.
            transform (classes in torchvison.transforms): Transform for poisoned trainset in splitting phrase
            selection_criterion (nn.Module): The criterion to select poison samples. Outputs loss values of each samples in the batch. 
        """
        
        # get logger
        work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        self.work_dir = work_dir
        self.log = log
        self.test_schedule = schedule['test_schedule']

        # First, train the model with all data
        log("\n\n\n===> Start training with poisoned data...")
        isolation_loss = LGALoss(isolation_criterion, gamma)
        self._train(self.poisoned_trainset, schedule['pre_isolation_schedule'], loss=isolation_loss)
        self.save_ckpt("pre-isolation.pth")

        # Then, filter out the samples with the lowest loss. These samples are deemed as poisoned samples.
        log("\n\n\n===> Start filtering out the poisoned data from the clean data...")       
        train_transform = self.poisoned_trainset.transform
        self.poisoned_trainset.transform = transform # set transform to have no data augmentation
        poisoned_indices, other_indices = self.split_dataset(self.poisoned_trainset, split_ratio, selection_criterion, schedule['split_schedule'])
        self.poisoned_trainset.transform = train_transform # restore data augmentation
        poisoned_dataset = Subset(self.poisoned_trainset, poisoned_indices) # select poisoned data
        other_dataset = Subset(self.poisoned_trainset, other_indices) # select other data
        
        self.poisoned_dataset = poisoned_dataset
        torch.save(poisoned_dataset, os.path.join(work_dir, "selected_poison.pth"))
        log("\n\n\nSelect %d poisoned data"%len(poisoned_indices))

        # Train the model with clean data (and some poisoned data that were not filtered out in previous training phrase).
        log("\n\n\n===> Training with selected clean data...")
        self._train(other_dataset, schedule['clean_schedule'])
        self.save_ckpt("after-clean.pth")

        # Unleanrn the seleted poisoned data. 
        # This will remove the inserted backdoor if the selected poisoned samples are really poisoned. 
        # Otherwise, this step will greatly tamper the performance of the model on clean samples.
        log("\n\n\n===> Unlearning the backdoor with selected poisoned data...")
        self._train(poisoned_dataset, schedule['unlearning_schedule'], unlearning=True)   
        self.save_ckpt("after-unlearning.pth")    

    def split_dataset(self, dataset, split_ratio, criterion, schedule):
        """Split dataset into poisoned part and clean part. The ratio of poisoned part is controlled by split_ratio.
        
        Args:
            dataset (torch.utils.data.Dataset): The dataset to split.
            split_ratio (float): Ratio of trainset that are used in unlearning.
            criterion (nn.Module): The criterion to select poison samples. Outputs loss values of each samples in the batch. 
            schedule (dict): schedule for spliting the dataset.            
        """

        if schedule is None:
            raise AttributeError("Reparing Training schedule is None, please check your schedule setting.")
        elif schedule is not None: 
            self.current_schedule = deepcopy(schedule)

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
                if not isinstance(self.model, nn.DataParallel):
                    gpus = list(range(self.current_schedule['GPU_num']))
                    self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        self.model = self.model.to(device)

        model = self.model
        dataloader = DataLoader(dataset, batch_size=schedule['batch_size'], shuffle=False, num_workers=schedule['num_workers'])
        losses = []
        with torch.no_grad():
            for data, label in dataloader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                loss = criterion(output, label)
                losses.append(loss)
        originial_device = dataset[0][0].device
        losses = torch.cat(losses, dim=0)
        indices = torch.argsort(losses)
        num_poisoned = int(split_ratio * len(losses))
        return indices[:num_poisoned].to(originial_device), indices[num_poisoned:].to(originial_device)

    def _test(self, model, dataset, device, batch_size=16, num_workers=8):
        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )

            model = model.to(device)
            model.eval()

            predict_digits = []
            labels = []
            for batch in test_loader:
                batch_img, batch_label = batch
                batch_img = batch_img.to(device)
                batch_img = model(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)
                labels.append(batch_label)

            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            return predict_digits, labels     

    def test(self, model, dataset, schedule=None):
        """Uniform test API for any model and any dataset.

        Args:
            model (torch.nn.Module): Network.
            dataset (torch.utils.data.Dataset): Dataset.
            schedule (dict): Testing schedule.
        """
        if schedule is None:
            schedule = self.test_schedule
        # Use GPU
        if 'device' in schedule and schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert schedule['GPU_num'] >0, 'GPU_num should be a positive integer'
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {schedule['GPU_num']} of them to test.")

            if schedule['GPU_num'] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(schedule['GPU_num']))
                model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        predict_digits, labels = self._test(model, dataset, device, schedule['batch_size'], schedule['num_workers'])
        total_num = labels.size(0)
        prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
        loss = self.loss(predict_digits, labels)
        return loss, prec1, prec5, total_num

    def _train(self, dataset, schedule, loss=None, unlearning=False):
        """
        The basic training function, 
        """
        log = self.log
        
        if schedule is None:
            raise AttributeError("Reparing Training schedule is None, please check your schedule setting.")
        elif schedule is not None: 
            self.current_schedule = deepcopy(schedule)

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
                if not isinstance(self.model, nn.DataParallel):
                    gpus = list(range(self.current_schedule['GPU_num']))
                    self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        self.model = self.model.to(device)

        if loss:
            criterion = loss
        else:
            criterion = self.loss
        
        if unlearning:
            factor = -1
        else:
            factor = 1
        

        train_loader = DataLoader(
            dataset,
            batch_size=self.current_schedule['batch_size'],
            shuffle=True,
            num_workers=self.current_schedule['num_workers'],
            drop_last=False,
            pin_memory=True,
            worker_init_fn=self._seed_worker
        )

        model = self.model
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.current_schedule['lr'], momentum=self.current_schedule['momentum'], weight_decay=self.current_schedule['weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.current_schedule['schedule'],gamma=self.current_schedule['gamma'])
        iteration = 0
        last_time = time.time()

        msg = f"Total train samples: {len(dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
        log(msg)

        for i in range(self.current_schedule['epochs']):
            for batch_id, batch in enumerate(train_loader):
                batch_img = batch[0]
                batch_label = batch[1]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                optimizer.zero_grad()
                predict_digits = model(batch_img)
                loss = criterion(predict_digits, batch_label)
                (factor * loss).backward()
                optimizer.step()

                iteration += 1

                if iteration % self.current_schedule['log_iteration_interval'] == 0:
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{self.current_schedule['epochs']}, iteration:{batch_id + 1}/{len(dataset)//self.current_schedule['batch_size']}, lr: {self.current_schedule['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                    last_time = time.time()
                    log(msg)
                    model.train()
            scheduler.step()

            if (i + 1) % self.current_schedule['test_epoch_interval'] == 0:
                poison_loss, asr, _, _ = self.test(model, self.poisoned_testset, self.test_schedule)
                clean_loss, acc, _, _ = self.test(model, self.clean_testset, self.test_schedule)
                model.train()
                msg = "==========Test results ==========\n" + \
                            time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{self.current_schedule['epochs']}" +\
                            " ASR: %.2f Acc: %.2f poison_loss: %.3f clean_loss: %.3f\n"%(asr, acc, poison_loss, clean_loss)
                log(msg)

        
