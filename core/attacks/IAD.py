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


class Normalize:
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


# ---------------------------- Generators ----------------------------#

class Conv2dBlock(nn.Module):
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
    def __init__(self, scale_factor=(2, 2), mode="bilinear", p=0.0):
        super(UpSampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        if p:
            self.dropout = nn.Dropout(p)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class Generator(nn.Sequential):
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


class IAD(Base):
    """Construct poisoned datasets with IAD method.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        y_target (int): N-to-1 attack target label.
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Random seed for poisoned set. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
        poisoned_rate (float): Ratio of poisoned samples.
        cross_rate (float): Ratio of 
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
            shuffle=False,
            num_workers=self.current_schedule['num_workers'],
            drop_last=False,
            worker_init_fn=self._seed_worker
        )

        self.model = self.model.to(device)
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_schedule['lr'], momentum=self.current_schedule['momentum'], weight_decay=self.current_schedule['weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.current_schedule['milestones'], self.current_schedule['lambda'])

        self.modelG = Generator(self.dataset_name).to(device)
        optimizerG = torch.optim.Adam(self.modelG.parameters(), lr=self.current_schedule['lr_G'], betas=self.current_schedule['betas_G'])
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, self.current_schedule['milestones_G'], self.current_schedule['lambda_G'])

        self.modelM = Generator(self.dataset_name, out_channels=1).to(device)
        optimizerM = torch.optim.Adam(self.modelM.parameters(), lr=self.current_schedule['lr_M'], betas=self.current_schedule['betas_M'])
        schedulerM = torch.optim.lr_scheduler.MultiStepLR(optimizerM, self.current_schedule['milestones_M'], self.current_schedule['lambda_M'])

        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))
        self.work_dir = work_dir

        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint

        iteration = 0
        epoch = 1
        last_time = time.time()

        msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.test_dataset)}\nBatch size:{self.current_schedule['batch_size']}\niteration every epoch:{len(self.train_dataset) // self.current_schedule['batch_size']}\nInitial learning rate:{self.current_schedule['lr']}\n"
        log(msg)

        if epoch == 1:
            self.modelM.train()
            for i in range(25):
                msg = f"Epoch {epoch} - {self.dataset_name} | mask_density: {self.mask_density} - lambda_div: {self.lambda_div}  - lambda_norm: {self.lambda_norm}\n"
                log(msg)
                self.train_mask_step(self.modelM, optimizerM, schedulerM, train_loader, train_loader1, epoch)
                loss_div, loss_norm, epoch = self.eval_mask(self.modelM, optimizerM, schedulerM, test_loader, test_loader1, epoch)
                msg = "Norm: {:.3f} | Diversity: {:.3f}\n".format(loss_norm, loss_div)
                log(msg)
                epoch += 1
        self.modelM.eval()
        self.modelM.requires_grad_(False)

        for i in range(self.current_schedule['epochs']):
            msg = f"Epoch {epoch} - {self.dataset_name} | mask_density: {self.mask_density} - lambda_div: {self.lambda_div}\n"
            log(msg)

            avg_loss, acc_clean, acc_bd, acc_cross = self.train_step(
                self.model,
                self.modelG,
                self.modelM,
                optimizer,
                optimizerG,
                scheduler,
                schedulerG,
                train_loader,
                train_loader1,
                epoch
            )
            msg = f"Train CE loss: {avg_loss} - Accuracy: {acc_clean} | BD Accuracy: {acc_bd} | Cross Accuracy: {acc_cross}"
            log(msg)

            last_time = time.time()
            avg_acc_clean, avg_acc_bd, avg_acc_cross, epoch = self.eval(
                self.model,
                self.modelG,
                self.modelM,
                optimizer,
                optimizerG,
                scheduler,
                schedulerG,
                test_loader,
                test_loader1,
                epoch
            )
            msg = "==========Test result on benign test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 accuracy:{avg_acc_clean}, time: {time.time()-last_time}\n"
            log(msg)
            msg = "==========Test result on poisoned test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 accuracy:{avg_acc_bd}, time: {time.time()-last_time}\n"
            log(msg)
            msg = "==========Test result on cross test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 accuracy:{avg_acc_cross}, time: {time.time()-last_time}\n"
            log(msg)
            epoch += 1
            if epoch > self.current_schedule['epochs']:
                break            




            # for batch_id, batch in enumerate(train_loader):
            #     batch_img = batch[0]
            #     batch_label = batch[1]
            #     batch_img = batch_img.to(device)
            #     batch_label = batch_label.to(device)
            #     optimizer.zero_grad()
            #     predict_digits = self.model(batch_img)
            #     loss = self.loss(predict_digits, batch_label)
            #     loss.backward()
            #     optimizer.step()

            #     iteration += 1

            #     if iteration % self.current_schedule['log_iteration_interval'] == 0:
            #         msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{self.current_schedule['epochs']}, iteration:{batch_id + 1}/{len(self.poisoned_train_dataset)//self.current_schedule['batch_size']}, lr: {self.current_schedule['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
            #         last_time = time.time()
            #         log(msg)



            # if (i + 1) % self.current_schedule['test_epoch_interval'] == 0:
            #     # test result on benign test dataset
            #     predict_digits, labels = self._test(self.test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
            #     total_num = labels.size(0)
            #     prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            #     top1_correct = int(round(prec1.item() / 100.0 * total_num))
            #     top5_correct = int(round(prec5.item() / 100.0 * total_num))
            #     msg = "==========Test result on benign test dataset==========\n" + \
            #           time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
            #           f"Top-1 correct / Total:{top1_correct}/{total_num}, Top-1 accuracy:{top1_correct/total_num}, Top-5 correct / Total:{top5_correct}/{total_num}, Top-5 accuracy:{top5_correct/total_num} time: {time.time()-last_time}\n"
            #     log(msg)

            #     # test result on poisoned test dataset
            #     # if self.current_schedule['benign_training'] is False:
            #     predict_digits, labels = self._test(self.poisoned_test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
            #     total_num = labels.size(0)
            #     prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            #     top1_correct = int(round(prec1.item() / 100.0 * total_num))
            #     top5_correct = int(round(prec5.item() / 100.0 * total_num))
            #     msg = "==========Test result on poisoned test dataset==========\n" + \
            #           time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
            #           f"Top-1 correct / Total:{top1_correct}/{total_num}, Top-1 accuracy:{top1_correct/total_num}, Top-5 correct / Total:{top5_correct}/{total_num}, Top-5 accuracy:{top5_correct/total_num}, time: {time.time()-last_time}\n"
            #     log(msg)

            #     self.model = self.model.to(device)
            #     self.model.train()

            # if (i + 1) % self.current_schedule['save_epoch_interval'] == 0:
            #     self.model.eval()
            #     self.model = self.model.cpu()
            #     ckpt_model_filename = "ckpt_epoch_" + str(i+1) + ".pth"
            #     ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
            #     torch.save(self.model.state_dict(), ckpt_model_path)
            #     self.model = self.model.to(device)
            #     self.model.train()


    def train_step(
        self, netC, netG, netM, optimizerC, optimizerG, schedulerC, schedulerG, train_dl1, train_dl2, epoch
    ):
        netC.train()
        netG.train()
        print(" Training:")
        total = 0
        total_cross = 0
        total_bd = 0
        total_clean = 0

        total_correct_clean = 0
        total_cross_correct = 0
        total_bd_correct = 0

        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        criterion_div = nn.MSELoss(reduction="none")
        for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(train_dl1)), train_dl1, train_dl2):
            optimizerC.zero_grad()

            inputs1, targets1 = inputs1.to(self.device), targets1.to(self.device)
            inputs2, targets2 = inputs2.to(self.device), targets2.to(self.device)

            bs = inputs1.shape[0]
            num_bd = int(self.poisoned_rate * bs)
            num_cross = int(self.cross_rate * bs)

            inputs_bd, targets_bd, patterns1, masks1 = self.create_bd(inputs1[:num_bd], targets1[:num_bd], netG, netM)
            inputs_cross, patterns2, masks2 = self.create_cross(
                inputs1[num_bd : num_bd + num_cross], inputs2[num_bd : num_bd + num_cross], netG, netM
            )

            total_inputs = torch.cat((inputs_bd, inputs_cross, inputs1[num_bd + num_cross :]), 0)
            total_targets = torch.cat((targets_bd, targets1[num_bd:]), 0)

            preds = netC(total_inputs)
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

            total_loss = loss_ce + loss_div
            total_loss.backward()
            optimizerC.step()
            optimizerG.step()

            total += bs
            total_bd += num_bd
            total_cross += num_cross
            total_clean += bs - num_bd - num_cross

            total_correct_clean += torch.sum(
                torch.argmax(preds[num_bd + num_cross :], dim=1) == total_targets[num_bd + num_cross :]
            )
            total_cross_correct += torch.sum(
                torch.argmax(preds[num_bd : num_bd + num_cross], dim=1) == total_targets[num_bd : num_bd + num_cross]
            )
            total_bd_correct += torch.sum(torch.argmax(preds[:num_bd], dim=1) == targets_bd)
            total_loss += loss_ce.detach() * bs
            avg_loss = total_loss / total

            acc_clean = total_correct_clean * 100.0 / total_clean
            acc_bd = total_bd_correct * 100.0 / total_bd
            acc_cross = total_cross_correct * 100.0 / total_cross

            # Saving images for debugging

            if batch_idx == len(train_dl1) - 2:
                images = netG.denormalize_pattern(torch.cat((inputs1[:num_bd], patterns1, inputs_bd), dim=2))
                file_name = "{}_images.png".format(self.dataset_name)
                file_path = os.path.join(self.work_dir, file_name)
                torchvision.utils.save_image(images, file_path, normalize=True, pad_value=1)
        schedulerC.step()
        schedulerG.step()
        return avg_loss, acc_clean, acc_bd, acc_cross


    def eval(
        self,
        netC,
        netG,
        netM,
        optimizerC,
        optimizerG,
        schedulerC,
        schedulerG,
        test_dl1,
        test_dl2,
        epoch
    ):
        netC.eval()
        netG.eval()
        print(" Eval:")
        total = 0.0

        total_correct_clean = 0.0
        total_correct_bd = 0.0
        total_correct_cross = 0.0
        for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(test_dl1)), test_dl1, test_dl2):
            with torch.no_grad():
                inputs1, targets1 = inputs1.to(self.device), targets1.to(self.device)
                inputs2, targets2 = inputs2.to(self.device), targets2.to(self.device)
                bs = inputs1.shape[0]

                preds_clean = netC(inputs1)
                correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets1)
                total_correct_clean += correct_clean

                inputs_bd, targets_bd, _, _ = self.create_bd(inputs1, targets1, netG, netM)
                preds_bd = netC(inputs_bd)
                correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
                total_correct_bd += correct_bd

                inputs_cross, _, _ = self.create_cross(inputs1, inputs2, netG, netM)
                preds_cross = netC(inputs_cross)
                correct_cross = torch.sum(torch.argmax(preds_cross, 1) == targets1)
                total_correct_cross += correct_cross

                total += bs
                avg_acc_clean = total_correct_clean * 100.0 / total
                avg_acc_cross = total_correct_cross * 100.0 / total
                avg_acc_bd = total_correct_bd * 100.0 / total

                # infor_string = "Clean Accuracy: {:.3f} | Backdoor Accuracy: {:.3f} | Cross Accuracy: {:3f}".format(
                #     avg_acc_clean, avg_acc_bd, avg_acc_cross
                # )
                # progress_bar(batch_idx, len(test_dl1), infor_string)

        # print(
        #     " Result: Best Clean Accuracy: {:.3f} - Best Backdoor Accuracy: {:.3f} - Best Cross Accuracy: {:.3f}| Clean Accuracy: {:.3f}".format(
        #         best_acc_clean, best_acc_bd, best_acc_cross, avg_acc_clean
        #     )
        # )
        # print(" Saving!!")
        # best_acc_clean = avg_acc_clean
        # best_acc_bd = avg_acc_bd
        # best_acc_cross = avg_acc_cross
        # state_dict = {
        #     "netC": netC.state_dict(),
        #     "netG": netG.state_dict(),
        #     "netM": netM.state_dict(),
        #     "optimizerC": optimizerC.state_dict(),
        #     "optimizerG": optimizerG.state_dict(),
        #     "schedulerC": schedulerC.state_dict(),
        #     "schedulerG": schedulerG.state_dict(),
        #     "best_acc_clean": best_acc_clean,
        #     "best_acc_bd": best_acc_bd,
        #     "best_acc_cross": best_acc_cross,
        #     "epoch": epoch,
        #     "opt": opt,
        # }
        # ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode)
        # if not os.path.exists(ckpt_folder):
        #     os.makedirs(ckpt_folder)
        # ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
        # torch.save(state_dict, ckpt_path)
        return avg_acc_clean, avg_acc_bd, avg_acc_cross, epoch


    # -------------------------------------------------------------------------------------
    def train_mask_step(self, netM, optimizerM, schedulerM, train_dl1, train_dl2, epoch):
        netM.train()
        print(" Training:")

        total_loss = 0
        criterion_div = nn.MSELoss(reduction="none")
        for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(train_dl1)), train_dl1, train_dl2):
            optimizerM.zero_grad()

            inputs1, targets1 = inputs1.to(self.device), targets1.to(self.device)
            inputs2, targets2 = inputs2.to(self.device), targets2.to(self.device)

            masks1 = netM(inputs1)
            masks1, masks2 = netM.threshold(netM(inputs1)), netM.threshold(netM(inputs2))

            # Calculating diversity loss
            distance_images = criterion_div(inputs1, inputs2)
            distance_images = torch.mean(distance_images, dim=(1, 2, 3))
            distance_images = torch.sqrt(distance_images)

            distance_patterns = criterion_div(masks1, masks2)
            distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
            distance_patterns = torch.sqrt(distance_patterns)

            loss_div = distance_images / (distance_patterns + self.EPSILON)
            loss_div = torch.mean(loss_div) * self.lambda_div

            loss_norm = torch.mean(F.relu(masks1 - self.mask_density))

            total_loss = self.lambda_norm * loss_norm + self.lambda_div * loss_div
            total_loss.backward()
            optimizerM.step()
            # infor_string = "Mask loss: {:.4f} - Norm: {:.3f} | Diversity: {:.3f}".format(total_loss, loss_norm, loss_div)
            # progress_bar(batch_idx, len(train_dl1), infor_string)

        schedulerM.step()


    def eval_mask(self, netM, optimizerM, schedulerM, test_dl1, test_dl2, epoch):
        netM.eval()
        print(" Eval:")

        criterion_div = nn.MSELoss(reduction="none")
        for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(test_dl1)), test_dl1, test_dl2):
            with torch.no_grad():
                inputs1, targets1 = inputs1.to(self.device), targets1.to(self.device)
                inputs2, targets2 = inputs2.to(self.device), targets2.to(self.device)
                bs = inputs1.shape[0]
                masks1, masks2 = netM.threshold(netM(inputs1)), netM.threshold(netM(inputs2))

                # Calculating diversity loss
                distance_images = criterion_div(inputs1, inputs2)
                distance_images = torch.mean(distance_images, dim=(1, 2, 3))
                distance_images = torch.sqrt(distance_images)

                distance_patterns = criterion_div(masks1, masks2)
                distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
                distance_patterns = torch.sqrt(distance_patterns)

                loss_div = distance_images / (distance_patterns + self.EPSILON)
                loss_div = torch.mean(loss_div) * self.lambda_div

                loss_norm = torch.mean(F.relu(masks1 - self.mask_density))

                # infor_string = "Norm: {:.3f} | Diversity: {:.3f}".format(loss_norm, loss_div)
                # progress_bar(batch_idx, len(test_dl1), infor_string)

        return loss_div, loss_norm, epoch


    def create_targets_bd(self, targets):
        # if opt.attack_mode == "all2one":
        #     bd_targets = torch.ones_like(targets) * opt.target_label
        # elif opt.attack_mode == "all2all_mask":
        #     bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
        # else:
        #     raise Exception("{} attack mode is not implemented".format(opt.attack_mode))

        bd_targets = torch.ones_like(targets) * self.y_target
        return bd_targets.to(self.device)


    def create_bd(self, inputs, targets, netG, netM):
        bd_targets = self.create_targets_bd(targets)
        patterns = netG(inputs)
        patterns = netG.normalize_pattern(patterns)

        masks_output = netM.threshold(netM(inputs))
        bd_inputs = inputs + (patterns - inputs) * masks_output
        return bd_inputs, bd_targets, patterns, masks_output


    def create_cross(self, inputs1, inputs2, netG, netM):
        patterns2 = netG(inputs2)
        patterns2 = netG.normalize_pattern(patterns2)
        masks_output = netM.threshold(netM(inputs2))
        inputs_cross = inputs1 + (patterns2 - inputs1) * masks_output
        return inputs_cross, patterns2, masks_output



    def _test(self, dataset, device, batch_size=16, num_workers=8):
        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                worker_init_fn=self._seed_worker
            )

            self.model = self.model.to(device)
            self.model.eval()

            predict_digits = []
            labels = []
            for batch in test_loader:
                batch_img, batch_label = batch
                batch_img = batch_img.to(device)
                batch_img = self.model(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)
                labels.append(batch_label)

            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            return predict_digits, labels

    def test(self, schedule=None, model=None, test_dataset=None, poisoned_test_dataset=None):
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

        if 'test_model' in self.current_schedule:
            model.load_state_dict(torch.load(self.current_schedule['test_model']), strict=False)

        if test_dataset is None and poisoned_test_dataset is None:
            test_dataset = self.test_dataset
            poisoned_test_dataset = self.poisoned_test_dataset

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

        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))





        if test_dataset is not None:
            last_time = time.time()
            # test result on benign test dataset
            predict_digits, labels = self._test(test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total:{top1_correct}/{total_num}, Top-1 accuracy:{top1_correct/total_num}, Top-5 correct / Total:{top5_correct}/{total_num}, Top-5 accuracy:{top5_correct/total_num} time: {time.time()-last_time}\n"
            log(msg)

        if poisoned_test_dataset is not None:
            last_time = time.time()
            # test result on poisoned test dataset
            predict_digits, labels = self._test(poisoned_test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total:{top1_correct}/{total_num}, Top-1 accuracy:{top1_correct/total_num}, Top-5 correct / Total:{top5_correct}/{total_num}, Top-5 accuracy:{top5_correct/total_num}, time: {time.time()-last_time}\n"
            log(msg)
