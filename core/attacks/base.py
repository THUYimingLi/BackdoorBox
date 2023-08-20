from copy import deepcopy
import math
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, MNIST, CIFAR10

from ..utils import Log


support_list = (
    DatasetFolder,
    MNIST,
    CIFAR10
)


def check(dataset):
    return isinstance(dataset, support_list)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Base(object):
    """Base class for backdoor training and testing.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        schedule (dict): Training or testing global schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self, train_dataset, test_dataset, model, loss, schedule=None, seed=0, deterministic=False):
        assert isinstance(train_dataset, support_list), 'train_dataset is an unsupported dataset type, train_dataset should be a subclass of our support list.'
        self.train_dataset = train_dataset

        assert isinstance(test_dataset, support_list), 'test_dataset is an unsupported dataset type, test_dataset should be a subclass of our support list.'
        self.test_dataset = test_dataset
        self.model = model
        self.loss = loss
        self.global_schedule = deepcopy(schedule)
        self.current_schedule = None
        self._set_seed(seed, deterministic)

    def _set_seed(self, seed, deterministic):
        # Use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA).
        torch.manual_seed(seed)

        # Set python seed
        random.seed(seed)

        # Set numpy seed (However, some applications and libraries may use NumPy Random Generator objects,
        # not the global RNG (https://numpy.org/doc/stable/reference/random/generator.html), and those will
        # need to be seeded consistently as well.)
        np.random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            # torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            # Hint: In some versions of CUDA, RNNs and LSTM networks may have non-deterministic behavior.
            # If you want to set them deterministic, see torch.nn.RNN() and torch.nn.LSTM() for details and workarounds.

    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_model(self):
        return self.model

    def get_poisoned_dataset(self):
        return self.poisoned_train_dataset, self.poisoned_test_dataset

    def adjust_learning_rate(self, optimizer, epoch, step, len_epoch):
        factor = (torch.tensor(self.current_schedule['schedule']) <= epoch).sum()

        lr = self.current_schedule['lr']*(self.current_schedule['gamma']**factor)

        """Warmup"""
        if 'warmup_epoch' in self.current_schedule and epoch < self.current_schedule['warmup_epoch']:
            lr = lr*float(1 + step + epoch*len_epoch)/(self.current_schedule['warmup_epoch']*len_epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, schedule=None):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        # log and output:
        # 1. experiment config
        # 2. ouput loss and time
        # 3. test and output statistics
        # 4. save checkpoint

        log('==========Schedule parameters==========\n')
        log(str(self.current_schedule)+'\n')

        if 'pretrain' in self.current_schedule:
            self.model.load_state_dict(torch.load(self.current_schedule['pretrain'], map_location='cpu'), strict=False)
            log(f"Load pretrained parameters: {self.current_schedule['pretrain']}\n")

        # Use GPU
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            log('==========Use GPUs to train==========\n')

            CUDA_VISIBLE_DEVICES = ''
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
            else:
                CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in range(torch.cuda.device_count())])
            log(f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}\n')

            if CUDA_VISIBLE_DEVICES == '':
                raise ValueError(f'This machine has no visible cuda devices!')

            CUDA_SELECTED_DEVICES = ''
            if 'CUDA_SELECTED_DEVICES' in self.current_schedule:
                CUDA_SELECTED_DEVICES = self.current_schedule['CUDA_SELECTED_DEVICES']
            else:
                CUDA_SELECTED_DEVICES = CUDA_VISIBLE_DEVICES
            log(f'CUDA_SELECTED_DEVICES={CUDA_SELECTED_DEVICES}\n')

            CUDA_VISIBLE_DEVICES_LIST = sorted(CUDA_VISIBLE_DEVICES.split(','))
            CUDA_SELECTED_DEVICES_LIST = sorted(CUDA_SELECTED_DEVICES.split(','))

            CUDA_VISIBLE_DEVICES_SET = set(CUDA_VISIBLE_DEVICES_LIST)
            CUDA_SELECTED_DEVICES_SET = set(CUDA_SELECTED_DEVICES_LIST)
            if not (CUDA_SELECTED_DEVICES_SET <= CUDA_VISIBLE_DEVICES_SET):
                raise ValueError(f'CUDA_VISIBLE_DEVICES should be a subset of CUDA_VISIBLE_DEVICES!')

            GPU_num = len(CUDA_SELECTED_DEVICES_SET)
            device_ids = [CUDA_VISIBLE_DEVICES_LIST.index(CUDA_SELECTED_DEVICE) for CUDA_SELECTED_DEVICE in CUDA_SELECTED_DEVICES_LIST]
            device = torch.device(f'cuda:{device_ids[0]}')
            self.model = self.model.to(device)

            if GPU_num > 1:
                self.model = nn.DataParallel(self.model, device_ids=device_ids, output_device=device_ids[0])
        # Use CPU
        else:
            device = torch.device("cpu")

        if self.current_schedule['benign_training'] is True:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.current_schedule['batch_size'],
                shuffle=True,
                num_workers=self.current_schedule['num_workers'],
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )
        elif self.current_schedule['benign_training'] is False:
            train_loader = DataLoader(
                self.poisoned_train_dataset,
                batch_size=self.current_schedule['batch_size'],
                shuffle=True,
                num_workers=self.current_schedule['num_workers'],
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )
        else:
            raise AttributeError("self.current_schedule['benign_training'] should be True or False.")

        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_schedule['lr'], momentum=self.current_schedule['momentum'], weight_decay=self.current_schedule['weight_decay'])

        iteration = 0
        last_time = time.time()

        msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.test_dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(self.train_dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
        log(msg)

        for i in range(self.current_schedule['epochs']):
            for batch_id, batch in enumerate(train_loader):
                self.adjust_learning_rate(optimizer, i, batch_id, int(math.ceil(len(self.train_dataset) / self.current_schedule['batch_size'])))
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

                if iteration % self.current_schedule['log_iteration_interval'] == 0:
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{self.current_schedule['epochs']}, iteration:{batch_id + 1}/{len(self.poisoned_train_dataset)//self.current_schedule['batch_size']}, lr: {optimizer.param_groups[0]['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                    last_time = time.time()
                    log(msg)

            if (i + 1) % self.current_schedule['test_epoch_interval'] == 0:
                # test result on benign test dataset
                predict_digits, labels, mean_loss = self._test(self.test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on benign test dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
                log(msg)

                # test result on poisoned test dataset
                # if self.current_schedule['benign_training'] is False:
                predict_digits, labels, mean_loss = self._test(self.poisoned_test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on poisoned test dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
                log(msg)

                self.model.train()

            if (i + 1) % self.current_schedule['save_epoch_interval'] == 0:
                ckpt_model_filename = "ckpt_epoch_" + str(i+1) + ".pth"
                ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                self.model.eval()
                torch.save(self.model.state_dict(), ckpt_model_path)
                self.model.train()

        self.model.eval()
        self.model = self.model.cpu()

    def _test(self, dataset, device, batch_size=16, num_workers=8, model=None, test_loss=None):
        if model is None:
            model = self.model
        else:
            model = model

        if test_loss is None:
            test_loss = self.loss
        else:
            test_loss = test_loss

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

            model.eval()

            predict_digits = []
            labels = []
            losses = []
            for batch in test_loader:
                batch_img, batch_label = batch
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                batch_img = model(batch_img)
                loss = test_loss(batch_img, batch_label)

                predict_digits.append(batch_img.cpu()) # (B, self.num_classes)
                labels.append(batch_label.cpu()) # (B)
                if loss.ndim == 0: # scalar
                    loss = torch.tensor([loss])
                losses.append(loss.cpu()) # (B) or (1)

            predict_digits = torch.cat(predict_digits, dim=0) # (N, self.num_classes)
            labels = torch.cat(labels, dim=0) # (N)
            losses = torch.cat(losses, dim=0) # (N)
            return predict_digits, labels, losses.mean().item()

    def test(self, schedule=None, model=None, test_dataset=None, poisoned_test_dataset=None, test_loss=None):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Test schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        if model is None:
            model = self.model

        if 'test_model' in self.current_schedule:
            model.load_state_dict(torch.load(self.current_schedule['test_model']), strict=False)

        if test_dataset is None and poisoned_test_dataset is None:
            test_dataset = self.test_dataset
            poisoned_test_dataset = self.poisoned_test_dataset

        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        # Use GPU
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            log('==========Use GPUs to train==========\n')

            CUDA_VISIBLE_DEVICES = ''
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
            else:
                CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in range(torch.cuda.device_count())])
            log(f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}\n')

            if CUDA_VISIBLE_DEVICES == '':
                raise ValueError(f'This machine has no visible cuda devices!')

            CUDA_SELECTED_DEVICES = ''
            if 'CUDA_SELECTED_DEVICES' in self.current_schedule:
                CUDA_SELECTED_DEVICES = self.current_schedule['CUDA_SELECTED_DEVICES']
            else:
                CUDA_SELECTED_DEVICES = CUDA_VISIBLE_DEVICES
            log(f'CUDA_SELECTED_DEVICES={CUDA_SELECTED_DEVICES}\n')

            CUDA_VISIBLE_DEVICES_LIST = sorted(CUDA_VISIBLE_DEVICES.split(','))
            CUDA_SELECTED_DEVICES_LIST = sorted(CUDA_SELECTED_DEVICES.split(','))

            CUDA_VISIBLE_DEVICES_SET = set(CUDA_VISIBLE_DEVICES_LIST)
            CUDA_SELECTED_DEVICES_SET = set(CUDA_SELECTED_DEVICES_LIST)
            if not (CUDA_SELECTED_DEVICES_SET <= CUDA_VISIBLE_DEVICES_SET):
                raise ValueError(f'CUDA_VISIBLE_DEVICES should be a subset of CUDA_VISIBLE_DEVICES!')

            GPU_num = len(CUDA_SELECTED_DEVICES_SET)
            device_ids = [CUDA_VISIBLE_DEVICES_LIST.index(CUDA_SELECTED_DEVICE) for CUDA_SELECTED_DEVICE in CUDA_SELECTED_DEVICES_LIST]
            device = torch.device(f'cuda:{device_ids[0]}')
            self.model = self.model.to(device)

            if GPU_num > 1:
                self.model = nn.DataParallel(self.model, device_ids=device_ids, output_device=device_ids[0])
        # Use CPU
        else:
            device = torch.device("cpu")

        if test_dataset is not None:
            last_time = time.time()
            # test result on benign test dataset
            predict_digits, labels, mean_loss = self._test(test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], model, test_loss)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

        if poisoned_test_dataset is not None:
            last_time = time.time()
            # test result on poisoned test dataset
            predict_digits, labels, mean_loss = self._test(poisoned_test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], model, test_loss)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

        return top1_correct, top5_correct, total_num, mean_loss
