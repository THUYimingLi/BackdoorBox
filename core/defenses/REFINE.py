

'''
This is the implement of pre-processing-based backdoor defense with REFINE proposed in [1].

Reference:
[1] REFINE: Inversion-Free Backdoor Defense via Model Reprogramming. ICLR, 2025.
'''


from copy import deepcopy
import os
import os.path as osp
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, MNIST, DatasetFolder

from .base import Base
from ..utils import Log, SupConLoss


class REFINE(Base):
    """Backdoor defense with REFINE.

    Args:
        unet (torch.nn.Module): Input transformation module.
        model (torch.nn.Module): Backdoored model to be defended.
        pretrain (str): Pretrained "unet" path. Default: None.
        arr_path (str): Predefined "arr_shuffle" path. Default: None.
        num_classes (int): Class number of dataset. Default: 10.
        lmd (float): Scalar temperature parameter. Default: 0.1.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self,
                 unet,
                 model,
                 pretrain=None,
                 arr_path=None,
                 num_classes=10,
                 lmd=0.1,
                 seed=0,
                 deterministic=False):
        super(REFINE, self).__init__(seed=seed, deterministic=deterministic)
        self.unet = unet
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.num_classes = num_classes
        self.lmd = lmd
        if pretrain is not None:
            self.unet.load_state_dict(torch.load(pretrain), strict=False)
        if arr_path is not None:
            self.arr_shuffle = np.array(torch.load(arr_path))
        else:
            self.init_label_shuffle()
        
    def init_label_shuffle(self):
        start = 0
        end = self.num_classes
        arr = np.array([i for i in range(self.num_classes)])
        arr_shuffle = np.array([i for i in range(self.num_classes)])
        while True:
            num = sum(arr[start:end] == arr_shuffle[start:end])
            if num == 0:
                break
            np.random.shuffle(arr_shuffle[start:end])
        self.arr_shuffle = arr_shuffle

    def label_shuffle(self, label):
        label_new = torch.zeros_like(label)
        index = torch.from_numpy(self.arr_shuffle).repeat(label.shape[0], 1).cuda()
        label_new = label_new.scatter(1, index, label)
        return label_new
    
    def forward(self, image):
        self.X_adv = torch.clamp(self.unet(image), 0, 1)
        # self.X_adv = F.normalize(self.X_adv)
        self.Y_adv = self.model(self.X_adv)
        Y_adv = F.softmax(self.Y_adv, 1)
        return self.label_shuffle(Y_adv)
    
    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _test(self, dataset, device, batch_size=16, num_workers=8, loss_func=torch.nn.BCELoss(reduction='none'), supconloss_func=SupConLoss()):
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

            self.unet = self.unet.to(device)
            self.unet.eval()
            self.model = self.model.to(device)

            losses = []
            for batch in test_loader:
                batch_img, _ = batch
                batch_img = batch_img.to(device)

                bsz = batch_img.shape[0]
                # Get the pseudo-labels of images
                f_logit = self.model(batch_img)
                f_index = f_logit.argmax(1)
                f_label = torch.zeros_like(f_logit).to(device).scatter_(1, f_index.view(-1, 1), 1)

                logit = self.forward(batch_img)

                # Calculate the supervised constractive loss
                features = self.X_adv.view(bsz, -1)
                features = F.normalize(features, dim=1)
                f1, f2 = features, features
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                supconloss = supconloss_func(features, f_index)

                loss = loss_func(logit, f_label) + self.lmd * supconloss

                losses.append(loss.cpu())

            losses = torch.cat(losses, dim=0)
            return losses.mean()

    def train_unet(self, train_dataset, test_dataset, schedule):
        if 'pretrain' in schedule:
            self.unet.load_state_dict(torch.load(schedule['pretrain']), strict=False)
        if 'arr_path' in schedule:
            self.arr_shuffle.load_state_dict(torch.load(schedule['arr_path']), strict=False)

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
                self.unet = nn.DataParallel(self.unet.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        train_loader = DataLoader(
            train_dataset,
            batch_size=schedule['batch_size'],
            shuffle=True,
            num_workers=schedule['num_workers'],
            drop_last=False,
            pin_memory=True,
            worker_init_fn=self._seed_worker
        )


        self.unet = self.unet.to(device)
        self.unet.train()
        self.model = self.model.to(device)

        loss_func = torch.nn.BCELoss(reduction='mean')
        supconloss_func = SupConLoss()
        optimizer = torch.optim.Adam(self.unet.parameters(), schedule['lr'], schedule['betas'], schedule['eps'], schedule['weight_decay'], schedule['amsgrad'])

        work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))
        torch.save(self.arr_shuffle, os.path.join(work_dir, 'label_shuffle.pth'))

        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint

        iteration = 0
        last_time = time.time()

        msg = f"Total train samples: {len(train_dataset)}\nTotal test samples: {len(test_dataset)}\nBatch size: {schedule['batch_size']}\niteration every epoch: {len(train_dataset) // schedule['batch_size']}\nInitial learning rate: {schedule['lr']}\n"
        log(msg)

        for i in range(schedule['epochs']):
            if i in schedule['schedule']:
                schedule['lr'] *= schedule['gamma']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = schedule['lr']
            for batch_id, batch in enumerate(train_loader):
                batch_img, _ = batch
                batch_img = batch_img.to(device)

                bsz = batch_img.shape[0]
                # Get the pseudo-labels of images
                f_logit = self.model(batch_img)
                f_index = f_logit.argmax(1)
                # print('f_logit:', f_logit.shape)
                # print('f_index:', f_index.shape)
                f_label = torch.zeros_like(f_logit).to(device).scatter_(1, f_index.view(-1, 1), 1)

                logit = self.forward(batch_img)

                # Calculate the supervised constractive loss
                features = self.X_adv.view(bsz, -1)
                features = F.normalize(features, dim=1)
                f1, f2 = features, features
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                supconloss = supconloss_func(features, f_index)

                loss = loss_func(logit, f_label) + self.lmd * supconloss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iteration += 1

                if iteration % schedule['log_iteration_interval'] == 0:
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{schedule['epochs']}, iteration:{batch_id + 1}/{len(train_dataset)//schedule['batch_size']}, lr: {schedule['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                    last_time = time.time()
                    log(msg)

            if (i + 1) % schedule['test_epoch_interval'] == 0:
                loss = self._test(test_dataset, device, schedule['batch_size'], schedule['num_workers'])
                msg = "==========Test result on test dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"loss: {loss}, time: {time.time()-last_time}\n"
                log(msg)

                self.unet = self.unet.to(device)
                self.unet.train()

            if (i + 1) % schedule['save_epoch_interval'] == 0:
                self.unet.eval()
                self.unet = self.unet.cpu()
                ckpt_unet_filename = "ckpt_epoch_" + str(i+1) + ".pth"
                ckpt_unet_path = os.path.join(work_dir, ckpt_unet_filename)
                torch.save(self.unet.state_dict(), ckpt_unet_path)
                self.unet = self.unet.to(device)
                self.unet.train()

    def preprocess(self, data):
        """Perform unet defense method on data and return the preprocessed data.

        Args:
            data (torch.Tensor): Input data (between 0.0 and 1.0), shape: (N, C, H, W) or (C, H, W), dtype: torch.float32.

        Returns:
            torch.Tensor: The preprocessed data.
        """
        with torch.no_grad():
            self.unet.eval()
            if data.ndim == 3:
                preprocessed_data = self.unet(data.unsqueeze(0))
                preprocessed_data = torch.clamp(preprocessed_data, 0, 1)
                return preprocessed_data[0]
            else:
                return torch.clamp(self.unet(data), 0, 1)

    def _predict(self, data, device, batch_size):
        with torch.no_grad():
            self.unet = self.unet.to(device)
            self.unet.eval()
            self.model = self.model.to(device)
            self.model.eval()
            predict_digits = []
            for i in range(data.shape[0] // batch_size):
                # breakpoint()
                batch_img = data[i*batch_size:(i+1)*batch_size, ...]
                batch_img = batch_img.to(device)
                batch_img = self.forward(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)

            if data.shape[0] % batch_size != 0:
                batch_img = data[(data.shape[0] // batch_size) * batch_size:, ...]
                batch_img = batch_img.to(device)
                batch_img = self.forward(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)

            predict_digits = torch.cat(predict_digits, dim=0)
            return predict_digits

    def predict(self, data, schedule):
        """Apply unet defense method to input data and get the predicts.

        Args:
            model (torch.nn.Module): Network.
            data (torch.Tensor): Input data (between 0.0 and 1.0), shape: (N, C, H, W), dtype: torch.float32.
            schedule (dict): Schedule for predicting.

        Returns:
            torch.Tensor: The predicts.
        """
        preprocessed_data = self.preprocess(data)

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

        return self._predict(model, preprocessed_data, device, schedule['batch_size'], schedule['num_workers'])

    def test(self, dataset, schedule):
        """Test unet on dataset.

        Args:
            model (torch.nn.Module): Network.
            dataset (types in support_list): Dataset.
            schedule (dict): Schedule for testing.
        """
        work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)

        # print('saving...')
        # ckpt_model_path = os.path.join(work_dir, 'finetuning_ckpt_epoch_200.pth')
        # torch.save(model.state_dict(), ckpt_model_path)
        print('done')    
        
        log = Log(osp.join(work_dir, 'log.txt'))
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

        defense_dataset = deepcopy(dataset)
        # defense_dataset.transform.transforms.append(transforms.ToPILImage())
        defense_dataset.transform.transforms.append(self.preprocess)
        # defense_dataset.transform.transforms.append(transforms.ToTensor())

        if hasattr(defense_dataset, 'poisoned_transform'):
            # defense_dataset.poisoned_transform.transforms.append(transforms.ToPILImage())
            defense_dataset.poisoned_transform.transforms.append(self.preprocess)
            # defense_dataset.poisoned_transform.transforms.append(transforms.ToTensor())

        last_time = time.time()
        batch_size=16
        num_workers=8
        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
            )

            self.unet = self.unet.to(device)
            self.unet.eval()
            self.model = self.model.to(device)
            self.model.eval()

            predict_digits = []
            labels = []
            for batch in test_loader:
                batch_img, batch_label = batch
                batch_img = batch_img.to(device)
                batch_img = self.forward(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)
                labels.append(batch_label)

            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = f"==========Test result on {schedule['metric']}==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
            log(msg)


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