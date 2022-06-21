'''
This is the implement of fine-tuning proposed in [1].
[1] Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks. RAID, 2018.
'''


import os

import torch
import torch.nn as nn
import numpy as np

from .base import Base
from ..utils import test
from torch.utils.data import DataLoader
import random
import time
import os.path as osp
from ..utils import Log


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def adjust_learning_rate(lr, optimizer, epoch):
    if epoch in [20]:
        lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



class FineTuning(Base):
    """FineTuning process.
    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        layer(list): The layers to fintune
        loss (torch.nn.Module): Loss.
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """
    def __init__(self,
                 train_dataset=None,
                 test_dataset=None,
                 model=None,
                 layer=None,
                 loss=None,
                 schedule=None,
                 seed=0,
                 deterministic=False):

        super(FineTuning, self).__init__(seed=seed, deterministic=deterministic)

        self.train_dataset=train_dataset
        self.test_dataset=test_dataset
        self.model=model
        self.layer=layer
        self.loss=loss
        self.schedule=schedule


    def frozen(self):
        """Frozen the layers which don't need to fine tuning.
        """
        if self.layer==None or self.layer[0]=="full layers":
            return
        else:
            for name, child in self.model.named_children():
                if not name in self.layer:
                    for param in child.parameters():
                        param.requires_grad = False

    def repair(self,schedule=None):
        """Finetuning.
        Args:
            schedule (dict): Schedule for testing.
        """
        self.frozen()
        print("--------fine tuning-------")
        if schedule==None:
            raise AttributeError("Schedule is None, please check your schedule setting.")
        current_schedule =schedule

        # Use GPU
        if 'device' in current_schedule and current_schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in current_schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = current_schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert current_schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            print(
                f"This machine has {torch.cuda.device_count()} cuda devices, and use {current_schedule['GPU_num']} of them to train.")

            if current_schedule['GPU_num'] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(current_schedule['GPU_num']))
                self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        if self.train_dataset==None:
            raise AttributeError("Train set is None, please check your setting.")
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=current_schedule['batch_size'],
            shuffle=True,
            num_workers=current_schedule['num_workers'],
            drop_last=True,
            pin_memory=True,
            worker_init_fn=_seed_worker
        )

        model = self.model.to(device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=current_schedule['lr'],
                                    momentum=current_schedule['momentum'],
                                    weight_decay=current_schedule['weight_decay'])
        work_dir = osp.join(current_schedule['save_dir'], current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint

        iteration = 0
        last_time = time.time()

        for i in range(current_schedule['epochs']):
            adjust_learning_rate(current_schedule['lr'],optimizer, i)
            for batch_id, batch in enumerate(train_loader):
                batch_img = batch[0]
                batch_label = batch[1]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                optimizer.zero_grad()
                predict_digits = self.model(batch_img)
                loss = self.loss(predict_digits, batch_label)
                loss.backward()
                optimizer.step()

                iteration += 1
                if iteration % current_schedule['log_iteration_interval'] == 0:
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ",
                                        time.localtime()) + f"Epoch:{i + 1}/{current_schedule['epochs']}, iteration:{batch_id + 1}/{len(self.train_dataset) // current_schedule['batch_size']}, lr: {current_schedule['lr']}, loss: {float(loss)}, time: {time.time() - last_time}\n"
                    last_time = time.time()
                    log(msg)

            if (i + 1) % self.current_schedule['save_epoch_interval'] == 0:
                self.model.eval()
                self.model = self.model.cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(i+1) + ".pth"
                ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                torch.save(self.model.state_dict(), ckpt_model_path)
                self.model = self.model.to(device)
                self.model.train()

    def test(self, schedule=None):
        """Test the finetuning model.
        Args:
            schedule (dict): Schedule for testing.
        """
        if schedule==None:
            raise AttributeError("Schedule is None, please check your schedule setting.")
        if self.test_dataset==None:
            raise AttributeError("Test set is None, please check your setting.")
        test(self.model,self.test_dataset,schedule)

    def get_model(self):
        return self.model