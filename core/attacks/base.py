import os
import os.path as osp
import time

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
        schedule (dict): Training or testing schedule. Default: None.
    """

    def __init__(self, train_dataset, test_dataset, model, loss, schedule=None):
        assert isinstance(train_dataset, support_list), 'train_dataset is an unsupported dataset type, train_dataset should be a subclass of our support list.'
        self.train_dataset = train_dataset

        assert isinstance(test_dataset, support_list), 'test_dataset is an unsupported dataset type, test_dataset should be a subclass of our support list.'
        self.test_dataset = test_dataset
        self.model = model
        self.loss = loss
        self.schedule = schedule

    def get_model(self):
        return self.model

    def get_poisoned_dataset(self):
        return self.poisoned_train_dataset, self.poisoned_test_dataset

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch in self.schedule['schedule']:
            self.schedule['lr'] *= self.schedule['gamma']
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.schedule['lr']

    def train(self, schedule=None):
        if schedule is None and self.schedule is None:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        elif schedule is not None and self.schedule is None:
            self.schedule = schedule
        elif schedule is None and self.schedule is not None:
            schedule = self.schedule
        elif schedule is not None and self.schedule is not None:
            self.schedule = schedule

        if 'pretrain' in schedule:
            self.model.load_state_dict(torch.load(schedule['pretrain']), strict=False)

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
                self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        if schedule['benign_training'] is True:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=schedule['batch_size'],
                shuffle=True,
                num_workers=schedule['num_workers'],
                drop_last=True
            )
        elif schedule['benign_training'] is False:
            train_loader = DataLoader(
                self.poisoned_train_dataset,
                batch_size=schedule['batch_size'],
                shuffle=True,
                num_workers=schedule['num_workers'],
                drop_last=True
            )
        else:
            raise AttributeError("schedule['benign_training'] should be True or False.")

        self.model = self.model.to(device)
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=schedule['lr'], momentum=schedule['momentum'], weight_decay=schedule['weight_decay'])

        work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint

        iteration = 0
        last_time = time.time()

        msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.test_dataset)}\nBatch size:{schedule['batch_size']}\niteration every epoch:{len(self.train_dataset) // schedule['batch_size']}\nInitial learning rate:{schedule['lr']}\n"
        log(msg)

        for i in range(schedule['epochs']):
            self.adjust_learning_rate(optimizer, i)
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

                # breakpoint()
                if iteration % schedule['log_iteration_interval'] == 0:
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{schedule['epochs']}, iteration:{batch_id + 1}/{len(self.poisoned_train_dataset)//schedule['batch_size']}, lr: {schedule['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                    last_time = time.time()
                    log(msg)

            if (i + 1) % schedule['test_epoch_interval'] == 0:
                # test result on benign test dataset
                predict_digits, labels = self._test(self.test_dataset, device, schedule['batch_size'], schedule['num_workers'])
                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                msg = "==========Test result on benign test dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total:{int(prec1.item() / 100.0 * total_num)}/{total_num}, Top-1 accuracy:{prec1.item()}, Top-5 correct / Total:{int(prec5.item() / 100.0 * total_num)}/{total_num}, Top-5 accuracy:{prec5.item()} time: {time.time()-last_time}\n"
                log(msg)

                # test result on poisoned test dataset
                # if schedule['benign_training'] is False:
                predict_digits, labels = self._test(self.poisoned_test_dataset, device, schedule['batch_size'], schedule['num_workers'])
                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                msg = "==========Test result on poisoned test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total:{int(prec1.item() / 100.0 * total_num)}/{total_num}, Top-1 accuracy:{prec1.item()}, Top-5 correct / Total:{int(prec5.item() / 100.0 * total_num)}/{total_num}, Top-5 accuracy:{prec5.item()}, time: {time.time()-last_time}\n"
                log(msg)

                self.model = self.model.to(device)
                self.model.train()

            if (i + 1) % schedule['save_epoch_interval'] == 0:
                self.model.eval()
                self.model = self.model.cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(i+1) + ".pth"
                ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                torch.save(self.model.state_dict(), ckpt_model_path)
                self.model = self.model.to(device)
                self.model.train()

    def _test(self, dataset, device, batch_size=16, num_workers=8):
        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False
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
        if schedule is None and self.schedule is None:
            raise AttributeError("Test schedule is None, please check your schedule setting.")
        elif schedule is not None and self.schedule is None:
            self.schedule = schedule
        elif schedule is None and self.schedule is not None:
            schedule = self.schedule
        elif schedule is not None and self.schedule is not None:
            self.schedule = schedule

        if model is None:
            model = self.model

        if 'test_model' in schedule:
            model.load_state_dict(torch.load(schedule['test_model']), strict=False)

        if test_dataset is None and poisoned_test_dataset is None:
            test_dataset = self.test_dataset
            poisoned_test_dataset = self.poisoned_test_dataset

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

        work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        if test_dataset is not None:
            last_time = time.time()
            # test result on benign test dataset
            predict_digits, labels = self._test(test_dataset, device, schedule['batch_size'], schedule['num_workers'])
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            msg = "==========Test result on benign test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total:{int(prec1.item() / 100.0 * total_num)}/{total_num}, Top-1 accuracy:{prec1.item()}, Top-5 correct / Total:{int(prec5.item() / 100.0 * total_num)}/{total_num}, Top-5 accuracy:{prec5.item()} time: {time.time()-last_time}\n"
            log(msg)

        if poisoned_test_dataset is not None:
            last_time = time.time()
            # test result on poisoned test dataset
            predict_digits, labels = self._test(poisoned_test_dataset, device, schedule['batch_size'], schedule['num_workers'])
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                f"Top-1 correct / Total:{int(prec1.item() / 100.0 * total_num)}/{total_num}, Top-1 accuracy:{prec1.item()}, Top-5 correct / Total:{int(prec5.item() / 100.0 * total_num)}/{total_num}, Top-5 accuracy:{prec5.item()}, time: {time.time()-last_time}\n"
            log(msg)
