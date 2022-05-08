'''
This is the part of the implement of model-repairing-based backdoor defense with MCR proposed in [1].

Reference:
[1] Bridging Mode Connectivity in Loss Landscapes and Adversarial Robustness. ICLR, 2020.
'''

import os
import os.path as osp
from copy import deepcopy
import time
import numpy as np
import random

import torch
import torchvision.transforms as transforms

from .base import Base
from ..utils import test
from ..models import curves
from ..models.resnet_curve import *
from ..models.vgg_curve import *

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, DatasetFolder

from ..utils.accuracy import accuracy
from ..utils.log import Log

def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
        issubclass(module.__class__, curves._BatchNorm)

def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()

def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]

def update_bn(loader, model, device, **kwargs):
    if not check_bn(model):
        return
    model.to(device)
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for input, _ in loader:
        input = input.to(device)
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))


class MCR(Base):
    """Repair a repaired curve model via model connectivity.

    Args:
        start_point (nn.Module): Start point model of connection curve.
        end_point (nn.Module): End point model of connection curve.
        base_model (nn.Module): Repaired curve model. (ResNetCurve or VGGCurve)
        num_bends (int): Number of bends in the curve, num_bends>=3.
        curve_type (str): Type of connection curve. (Only support 'Bezier' or 'PolyChain')
        loss (nn.Module): Loss for repaired model training.
        fix_start (bool): Sets whether params of start model are fixed. Default: True.
        fix_end (bool): Sets whether params of end model are fixed. Default: True.
        init_linear (bool): Sets whether to initialize the linear layer of the base_model. Default: True.
        pretrained (str): Path of pretrained MCR repaired model. If provided, repair training process will be skipped. Default: ''.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.

    
    """
    def __init__(self,
                 start_point,
                 end_point,
                 base_model,
                 num_bends,
                 curve_type,
                 loss, 
                 fix_start=True,
                 fix_end=True,
                 init_linear=True,
                 pretrained = '',
                 seed=0,
                 deterministic=False):
        super(MCR, self).__init__(seed, deterministic)
        self.seed = seed
        self.start_point = start_point
        self.end_point = end_point
        self.loss = loss
        self.train_loader = None
        
        assert curve_type in ('Bezier', 'PolyChain'), "Curve type *{}* not supported".format(curve_type) 
        curve = getattr(curves, curve_type)
        self.curve_model = curve(num_bends)
        self.base_model = base_model
        self.model = curves.CurveNet(
            self.curve_model,
            self.base_model,
            num_bends,
            fix_start,
            fix_end,
        )
        self.pretrained = pretrained

        print('===> Loading start_point as {}, end_point as {}.'.format(0, num_bends - 1))
        for cur_point, k in [(start_point, 0), (end_point, num_bends - 1)]:
            if cur_point is not None:
                self.model.import_base_parameters(cur_point, k)

        if init_linear:
            self.model.init_linear()
        
        if self.pretrained:
            print('===> Loading pretrained model: {}'.format(pretrained))
            self.model.load_state_dict(torch.load(pretrained), strict=False)

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch in self.current_schedule['schedule']:
            self.current_schedule['lr'] *= self.current_schedule['gamma']
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.current_schedule['lr']

    def _train(self, dataset, portion, schedule=None, settings=None):
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
                gpus = list(range(self.current_schedule['GPU_num']))
                self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        if self.current_schedule['benign_training'] is True:
            # get a portion of the repairing training dataset
            print("===> Loading {:.1f}% of traing samples.".format(portion*100))
            idxs = np.random.permutation(len(dataset))[:int(portion*len(dataset))]
            dataset = torch.utils.data.Subset(dataset, idxs)

            train_loader = DataLoader(
                dataset,
                batch_size=self.current_schedule['batch_size'],
                shuffle=True,
                num_workers=self.current_schedule['num_workers'],
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )
            self.train_loader=train_loader
        else:
            raise AttributeError("self.current_schedule['benign_training'] should be True during model repairing.")

        if self.pretrained:
            print("===> Skip repairing as pretrained model is loaded.")
            return

        self.model = self.model.to(device)
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_schedule['lr'], momentum=self.current_schedule['momentum'], weight_decay=self.current_schedule['weight_decay'])

        regularizer = curves.l2_regularizer(self.current_schedule['weight_decay']) if self.current_schedule['l2_regularizer']==True else None
        
        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        if settings is not None:
            settings.update(self.current_schedule)
            log("Current Schedule and settings:\n{}".format(settings))

        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint

        iteration = 0
        last_time = time.time()

        msg = f"Total train samples: {len(dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
        log(msg)

        for i in range(self.current_schedule['epochs']):
            self.adjust_learning_rate(optimizer, i)
            for batch_id, batch in enumerate(train_loader):
                batch_img = batch[0]
                batch_label = batch[1]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                optimizer.zero_grad()
                predict_digits = self.model(batch_img)
                loss = self.loss(predict_digits, batch_label)
                #TODO: check regularizer
                if regularizer is not None:
                    loss += regularizer(self.model)
                loss.backward()
                optimizer.step()

                iteration += 1

                if iteration % self.current_schedule['log_iteration_interval'] == 0:
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{self.current_schedule['epochs']}, iteration:{batch_id + 1}/{len(dataset)//self.current_schedule['batch_size']}, lr: {self.current_schedule['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
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
        


    def repair(self, dataset, portion, schedule, settings=None):
        """Perform MCR defense method based on attacked models (staring point and the end point). 
        The repaired model will be stored in self.model (CurveModel)
        
        Args:
            dataset (types in support_list): Dataset.
            portion (float): in range(0,1), proportion of training dataset.
            schedule (dict): Schedule for Training the curve model.
            settings (dict): Globbal settings, only for logs.
        """
        print("===> Start training repaired model...")
        self._train(dataset, portion, schedule, settings)
    

    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    
    def _test(self, model, dataset, coeffs_t, device, batch_size=16, num_workers=8):
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
                batch_img = model(batch_img, coeffs_t)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)
                labels.append(batch_label)

            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            return predict_digits, labels


    def test(self, dataset, schedule, coeffs_t):
        """Test repaired curve model on dataset

        Args:
            dataset (types in support_list): Dataset.
            schedule (dict): Schedule for testing.
            coeffs_t (float or list): Hyperparam for the curve, in range(0,1). 
        """
        model  = self.model
        # Use GPU
        if 'device' in schedule and schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert schedule['GPU_num'] >0, 'GPU_num should be a positive integer'
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {schedule['GPU_num']} of them to train.")

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

        if schedule['metric'] == 'ASR_NoTarget':
            if isinstance(dataset, CIFAR10):
                data = []
                targets = []
                for i, target in enumerate(dataset.targets):
                    if target != schedule['y_target']:
                        data.append(dataset.data[i])
                        targets.append(target)
                data = np.stack(data, axis=0)
                dataset.data = data
                dataset.targets = targets
            elif isinstance(dataset, MNIST):
                data = []
                targets = []
                for i, target in enumerate(dataset.targets):
                    if int(target) != schedule['y_target']:
                        data.append(dataset.data[i])
                        targets.append(target)
                data = torch.stack(data, dim=0)
                dataset.data = data
                dataset.targets = targets
            elif isinstance(dataset, DatasetFolder):
                samples = []
                for sample in dataset.samples:
                    if sample[1] != schedule['y_target']:
                        samples.append(sample)
                dataset.samples = samples
            else:
                raise NotImplementedError

        work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        if isinstance(coeffs_t, float):
            coeffs_t = [coeffs_t]
        assert isinstance(coeffs_t, (list, np.ndarray)), f'coeffs_t is a type of {type(coeffs_t)}, list or np.ndarray supported.'
        assert self.train_loader is not None, "MCR.repair() should be called before MCR.test()."

        for t in coeffs_t:
            print(f'===> Update BN for t={t}')
            update_bn(self.train_loader, model.eval(), device, t=t)
            last_time = time.time()
            predict_digits, labels = self._test(model, dataset, t, device, schedule['batch_size'], schedule['num_workers'])

            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = f"==========Test result on {schedule['metric']}, coeffs_t {t} ==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num} time: {time.time()-last_time}\n"
            log(msg)

    def get_model(self):
        return self.model

