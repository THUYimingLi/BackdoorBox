'''
This is the implement of LIRA [1]. 
This code is developed based on its official codes (https://github.com/sunbelbd/invisible_backdoor_attacks)

Reference:
[1] LIRA: Learnable, Imperceptible and Robust Backdoor Attacks. ICCV 2021.
'''


import warnings
warnings.filterwarnings("ignore")
import os
import os.path as osp
import time
from copy import deepcopy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from ..utils import Log
from .base import *


class MNISTBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(MNISTBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.ind = None

    def forward(self, x):
        return self.conv1(F.relu(self.bn1(x)))


class BaselineMNISTNetwork(nn.Module):
    def __init__(self):
        super(BaselineMNISTNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 2, 1)  # 14
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = MNISTBlock(32, 64, 2)  # 7
        self.layer3 = MNISTBlock(64, 64, 2)  # 4
        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.3)
        self.linear9 = nn.Linear(512, 10)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, feature_dim=512):
        """
        for image size 32, feature_dim = 512
        for other sizes, feature_dim = 512 * (size//32)**2
        """
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

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


class MNISTAutoencoder(nn.Module):
    """The generator of backdoor trigger on MNIST."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 64, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 3, stride=2),  # b, 16, 5, 5
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    """The generator of backdoor trigger on CIFAR10."""
    def __init__(self, out_channel):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, out_channel, 1),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        out = F.tanh(out)

        return out


class Autoencoder(nn.Module):
    """The generator of backdoor trigger on GTSRB."""
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ProbTransform(torch.nn.Module):
    """The data augmentation transform by the probability.
    
    Args:
        f (nn.Module): the data augmentation transform operation.
        p (float): the probability of the data augmentation transform. 
    """
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTensorTransform(torch.nn.Module):
    """The data augmentation transform.
    
    Args:
        dataset_name (str): the name of the dataset.
    """
    def __init__(self, dataset_name):
        super(PostTensorTransform, self).__init__()
        if dataset_name == 'mnist':
            input_height, input_width = 28, 28
        elif dataset_name == 'cifar10':
            input_height, input_width = 32, 32
        elif dataset_name == 'gtsrb':
            input_height, input_width = 32, 32
        self.random_crop = ProbTransform(transforms.RandomCrop((input_height, input_width), padding=5), p=0.8) # ProbTransform(A.RandomCrop((input_height, input_width), padding=5), p=0.8)
        self.random_rotation = ProbTransform(transforms.RandomRotation(10), p=0.5) # ProbTransform(A.RandomRotation(10), p=0.5)
        if dataset_name == "cifar10":
            self.random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5) # A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class LIRA(Base):
    """Construct backdoored model with LIRA method.

    Args:
        dataset_name (str): the name of the dataset.
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Victim model.
        loss (torch.nn.Module): Loss.
        y_target (int): N-to-1 attack target label.
        eps (float): The magnitude of backdoor trigger in training process.
        alpha (float): The hyperparameter to balance the clean loss and backdoor loss in training process.
        tune_test_eps (float): The magnitude of backdoor trigger in finetuning process.
        tune_test_alpha (float): The hyperparameter to balance the clean loss and backdoor loss in finetuning process.
        best_threshold (float): The threshold to decide whether the model should be saved.
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
                 model, 
                 loss, 
                 y_target,
                 eps,
                 alpha,
                 tune_test_eps,
                 tune_test_alpha,
                 best_threshold,
                 schedule=None, 
                 seed=0, 
                 deterministic=False,
                 ):
        super(LIRA, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)
        
        self.dataset_name = dataset_name
        self.y_target = y_target
        self.eps = eps
        self.alpha = alpha
        self.tune_test_eps = tune_test_eps
        self.tune_test_alpha = tune_test_alpha
        self.best_threshold = best_threshold

        self.create_targets_bd = ModifyTarget(self.y_target)
        self.train_poisoned_data = []
        self.train_poisoned_label = []
        self.test_poisoned_data = []
        self.test_poisoned_label = []

    def get_model(self):
        return self.model

    def get_atkmodel(self):
        return self.atkmodel

    def get_poisoned_dataset(self):
        """
            Return the poisoned dataset.
        """
        warnings.warn("LIRA is implemented by controlling the training process so that the poisoned dataset is dynamic.")
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

    def get_target_transform(self):
        """
            Return the attacker-specified target label.
        """
        target_transform = lambda x: self.create_targets_bd(x)
        return target_transform

    def clear_grad(self, model):
        """
            Clear the gradient of model parameters.
        """
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def create_net(self):
        """
            Return the victim model architecture.
        """
        if self.dataset_name == 'mnist':
            def create_mnist():
                return BaselineMNISTNetwork() # core.models.BaselineMNISTNetwork()
            return create_mnist
        elif self.dataset_name == 'cifar10':
            def create_cifar():
                return VGG('VGG11', num_classes=10) # core.models.vgg11(num_classes=10) #core.models.ResNet(18)
            return create_cifar
        elif self.dataset_name == 'gtsrb':
            def create_gtsrb():
                return VGG('VGG11', num_classes=43) # core.models.vgg11(num_classes=43) #core.models.ResNet(18, 43)
            return create_gtsrb

    def create_atkmodel(self):
        """
            Return the generator of backdoor trigger.
        """
        if self.dataset_name == 'mnist':
            atkmodel = MNISTAutoencoder()
            # Copy of attack model
            tgtmodel = MNISTAutoencoder()
        elif self.dataset_name == 'cifar10':
            atkmodel = UNet(3)
            # Copy of attack model
            tgtmodel = UNet(3)
        elif self.dataset_name == 'gtsrb':
            atkmodel = Autoencoder()
            # Copy of attack model
            tgtmodel = Autoencoder()
        return atkmodel, tgtmodel
        
    def clip_image(self, x):
        """
            Return the function of clip image.
        """
        if self.dataset_name == 'mnist':
            return torch.clamp(x, -1.0, 1.0)
        elif self.dataset_name == 'cifar10':
            return x
        elif self.dataset_name == 'gtsrb':
            return torch.clamp(x, 0.0, 1.0)

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
            drop_last=False,
            pin_memory=True,
            worker_init_fn=self._seed_worker
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.current_schedule['batch_size'],
            shuffle=False,
            num_workers=self.current_schedule['num_workers'],
            drop_last=False,
            pin_memory=True,
            worker_init_fn=self._seed_worker
        )

        self.model = self.model.to(device)
        self.model.train()
        atkmodel, tgtmodel = self.create_atkmodel()
        atkmodel, tgtmodel = atkmodel.to(device), tgtmodel.to(device)
        self.atkmodel = atkmodel
        tgtmodel.load_state_dict(atkmodel.state_dict(), strict=True)

        # Optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_schedule['lr'], momentum=self.current_schedule['momentum'])
        tgtoptimizer = torch.optim.Adam(tgtmodel.parameters(), lr=self.current_schedule['lr_atk'])
        post_transforms = PostTensorTransform(self.dataset_name).to(device)
        target_transform = self.get_target_transform()
        create_net = self.create_net()
        clip_image = self.clip_image

        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint

        last_time = time.time()

        msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.test_dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(self.train_dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
        log(msg)

        best_acc_clean = 0
        best_acc_poison = 0
        start_epoch = 1

        # The victim model and trigger generator will be trained jointly.
        for epoch in range(start_epoch, self.current_schedule['epochs'] + 1):
            for i in range(self.current_schedule['train_epoch']):
                atkloss, atkcleanloss, atkpoisonloss, atktriloss = self.train_step(atkmodel, tgtmodel, self.model, 
                                                                        tgtoptimizer, optimizer, target_transform, train_loader,
                                                                        create_net, clip_image, post_transforms=post_transforms)
                msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    'Train [{}] Loss: clean {:.4f} poison {:.4f} total {:.4f} Tri {:.4f}\n'.format(
                     epoch, atkcleanloss, atkpoisonloss, atkloss, atktriloss)
                log(msg)
            atkmodel.load_state_dict(tgtmodel.state_dict())

            scratchmodel = create_net().to(device)
            scratchmodel.load_state_dict(self.model.state_dict()) #transfer from cls to scratch for testing

            if epoch % self.current_schedule['test_epoch_interval'] == 0 or epoch == self.current_schedule['epochs']: 
                predict_digits, labels, tri_predict_digits, tri_labels = self.train_test(atkmodel, scratchmodel, target_transform, 
                    train_loader, test_loader, self.current_schedule['cls_test_epoch'], clip_image, testoptimizer=None)
            else:
                predict_digits, labels, tri_predict_digits, tri_labels = self.train_test(atkmodel, scratchmodel, target_transform, 
                    train_loader, test_loader, self.current_schedule['train_epoch'], clip_image, testoptimizer=None)
            
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num} time: {time.time()-last_time}\n"
            log(msg)
            acc_clean = prec1

            total_num = tri_labels.size(0)
            prec1, prec5 = accuracy(tri_predict_digits, tri_labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
            log(msg)
            acc_poison = prec1

            if acc_clean > best_acc_clean or (acc_clean > (best_acc_clean-self.best_threshold) and best_acc_poison < acc_poison):
                best_acc_poison = acc_poison
                best_acc_clean = acc_clean
                best_atkmodelckpt = atkmodel.state_dict()
                best_clsmodelckpt = self.model.state_dict()
                torch.save({
                    'atkmodel': atkmodel.state_dict(),
                    'clsmodel': self.model.state_dict(),
                }, os.path.join(work_dir, 'total_model.th'))
        
        # The victim will be finetuned and the trigger generator will be fixed.
        self.model = create_net().to(device)
        self.model.load_state_dict(best_clsmodelckpt)
        atkmodel, tgtmodel = self.create_atkmodel()
        atkmodel, tgtmodel = atkmodel.to(device), tgtmodel.to(device)
        self.atkmodel = atkmodel
        atkmodel.load_state_dict(best_atkmodelckpt)
        # Optimizer
        optimizerC = torch.optim.SGD(self.model.parameters(), self.current_schedule['tune_test_lr'], momentum=self.current_schedule['tune_momentum'], weight_decay=self.current_schedule['tune_weight_decay'])
        # Scheduler
        schedulerC_milestones = [int(e) for e in self.current_schedule['schedulerC_milestones'].split(',')]
        schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, schedulerC_milestones, self.current_schedule['schedulerC_lambda'])
        self.log = log
        self.work_dir = work_dir
        self.last_time = last_time
        self.finetune_model(
            atkmodel, self.model, target_transform, train_loader, test_loader, trainepoch=self.current_schedule['tune_test_epochs'],
            alpha=self.tune_test_alpha, optimizerC=optimizerC, schedulerC=schedulerC, 
            data_transforms=post_transforms, start_epoch=1, clip_image=clip_image)
            

    def train_step(self, atkmodel, tgtmodel, model, tgtoptimizer, clsoptimizer, target_transform, 
            train_loader, create_net, clip_image, post_transforms=None):
        """Train the victim model and the backdoor trigger generator jointly.
        
        Args:
            atkmodel (torch.nn.Module): Backdoor trigger generator.
            tgtmodel (torch.nn.Module): Assistant backdoor trigger generator.
            model (torch.nn.Module): Victim model.
            tgtoptimizer (torch.optim.Optimizer): Optimizer of the backdoor trigger generator.
            clsoptimizer (torch.optim.Optimizer): Optimizer of the victim model. 
            target_transform (object): The transform to target label.
            train_loader (torch.utils.data.DataLoader): Benign training dataloader.
            create_net (object): Build a victim model achitecture.
            clip_image (object): Clip images to a certain range.
            post_transforms (object): The data augmentation transform operation.
        """
        model.train()
        atkmodel.eval()
        tgtmodel.train()
        losslist = []
        loss_clean_list = []
        loss_poison_list = []
        loss_tri_list = []
        for batch_idx, (data, target) in enumerate(train_loader):
            if post_transforms is not None:
                data = post_transforms(data)
            
            # Update backdoor trigger generator
            tmpmodel = create_net().to(self.device)
            data, target = data.to(self.device), target.to(self.device)
            noise = tgtmodel(data) * self.eps
            atkdata = clip_image(data + noise)
            
            # Calculate loss
            output = model(data)
            atkoutput = model(atkdata)
            loss_clean = self.loss(output, target)
            loss_poison = self.loss(atkoutput, target_transform(target))
            loss = loss_clean * self.alpha + (1-self.alpha) * loss_poison
            
            losslist.append(loss.item())
            loss_clean_list.append(loss_clean.item())
            loss_poison_list.append(loss_poison.item())

            self.clear_grad(model)
            paragrads = torch.autograd.grad(loss, model.parameters(),
                                            create_graph=True)
            for i, (layername, layer) in enumerate(model.named_parameters()):
                modulenames, weightname = \
                    layername.split('.')[:-1], layername.split('.')[-1]
                module = tmpmodel._modules[modulenames[0]]
                # TODO: could be potentially faster if we save the intermediate mappings
                for name in modulenames[1:]:
                    module = module._modules[name]
                module._parameters[weightname] = \
                    layer - clsoptimizer.param_groups[0]['lr'] * paragrads[i]        
            tgtoptimizer.zero_grad()
            
            noise = tgtmodel(data) * self.eps
            atkdata = clip_image(data + noise)
            tgtloss = self.loss(tmpmodel(atkdata), target_transform(target))
            loss_tri_list.append(tgtloss)

            tgtloss.backward()
            tgtoptimizer.step() # This is the slowest step

            # Update the victim model.
            noise = atkmodel(data) * self.eps
            atkdata = clip_image(data + noise)
            output = model(atkdata)
            loss = self.loss(output, target)
            clsoptimizer.zero_grad()
            loss.backward()
            clsoptimizer.step()

        atkloss = sum(losslist) / len(losslist)
        atkcleanloss = sum(loss_clean_list) / len(loss_clean_list)
        atkpoisonloss = sum(loss_poison_list) / len(loss_poison_list)
        atktriloss = sum(loss_tri_list) / len(loss_tri_list)
        
        return atkloss, atkcleanloss, atkpoisonloss, atktriloss


    def train_test(self, atkmodel, model, target_transform, 
        train_loader, test_loader, trainepoch, clip_image, testoptimizer=None):
        """Test the victim model using the backdoor trigger generator in training process.
        
        Args:
            atkmodel (torch.nn.Module): Backdoor trigger generator.
            model (torch.nn.Module): Victim model.
            target_transform (object): The transform to target label.
            train_loader (torch.utils.data.DataLoader): Benign training dataloader.
            test_loader (torch.utils.data.DataLoader): Benign test dataloader.
            trainepoch (int): The finetuning epoch in test process.
            clip_image (object): Clip images to a certain range.
            testoptimizer (torch.optim.Optimizer): Optimizer of the victim model. 
        """
        
        atkmodel.eval()
        if testoptimizer is None:
            testoptimizer = torch.optim.SGD(model.parameters(), lr=self.current_schedule['lr'])
        for cepoch in range(trainepoch):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                testoptimizer.zero_grad()
                with torch.no_grad():
                    noise = atkmodel(data) * self.eps
                    atkdata = clip_image(data + noise)

                atkoutput = model(atkdata)
                output = model(data)
                
                loss_clean = self.loss(output, target)
                loss_poison = self.loss(atkoutput, target_transform(target))
                loss = self.alpha * loss_clean + (1-self.alpha) * loss_poison
                
                loss.backward()
                testoptimizer.step()
                
            if cepoch == trainepoch-1:
                predict_digits = []
                labels = []
                tri_predict_digits = []
                tri_labels = []
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        predict_digits.append(output.cpu())
                        labels.append(target.cpu())

                        noise = atkmodel(data) * self.eps
                        atkdata = clip_image(data + noise)
                        atkoutput = model(atkdata)
                        
                        tri_predict_digits.append(atkoutput.cpu())
                        tri_labels.append(target_transform(target).cpu())

        predict_digits = torch.cat(predict_digits, dim=0)
        labels = torch.cat(labels, dim=0)
        tri_predict_digits = torch.cat(tri_predict_digits, dim=0)
        tri_labels = torch.cat(tri_labels, dim=0)

        return predict_digits, labels, tri_predict_digits, tri_labels


    def finetune_model(self, atkmodel, model, target_transform, train_loader, test_loader, 
            trainepoch, alpha=0.5, optimizerC=None, schedulerC=None, data_transforms=None, start_epoch=1, clip_image=None):
        """Test the victim model using the backdoor trigger generator in training process.
        
        Args:
            atkmodel (torch.nn.Module): Backdoor trigger generator.
            model (torch.nn.Module): Victim model.
            target_transform (object): The transform to target label.
            train_loader (torch.utils.data.DataLoader): Benign training dataloader.
            test_loader (torch.utils.data.DataLoader): Benign test dataloader.
            trainepoch (int): The epoch in finetuning process.
            alpha (float): The hyperparameter to balance the clean loss and backdoor loss in finetuning process.
            optimizerC (torch.optim.Optimizer): Optimizer of the victim model. 
            schedulerC (torch.optim.lr_scheduler._LRScheduler): Scheduler of the victim model.
            data_transforms (object): The data augmentation transform operation.
            start_epoch (int): The start epoch.
            clip_image (object): Clip images to a certain range.
        """
        best_clean_acc, best_poison_acc = 0, 0
        
        atkmodel.eval()
        
        if optimizerC is None:
            print('No optimizer, creating default SGD...')  
            optimizerC = torch.optim.SGD(model.parameters(), lr=self.current_schedule['tune_test_lr'])
        if schedulerC is None:
            print('No scheduler, creating default 100,200,300,400...')  
            schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, [100, 200, 300, 400], self.current_schedule['tune_test_lr'])
            
        for cepoch in range(start_epoch, trainepoch+1):
            model.train()
            self.train_poisoned_data, self.train_poisoned_label = [], []
            self.test_poisoned_data, self.test_poisoned_label = [], []
            losslist = []
            loss_clean_list = []
            loss_poison_list = []
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                if data_transforms is not None:
                    data = data_transforms(data)                                    
                optimizerC.zero_grad()
                
                output = model(data)
                loss_clean = self.loss(output, target)
                
                if alpha < 1:
                    with torch.no_grad():
                        noise = atkmodel(data) * self.tune_test_eps
                        if clip_image is None:
                            atkdata = torch.clamp(data + noise, 0, 1)
                        else:
                            atkdata = clip_image(data + noise)
                    atkoutput = model(atkdata)
                    loss_poison = self.loss(atkoutput, target_transform(target))
                else:
                    loss_poison = torch.tensor(0.0).to(self.device)
                
                loss = alpha * loss_clean + (1-alpha) * loss_poison                
                loss.backward()
                optimizerC.step()
                
                losslist.append(loss.item())
                loss_clean_list.append(loss_clean.item())
                loss_poison_list.append(loss_poison.item())
                
                self.train_poisoned_data += atkdata.detach().cpu().numpy().tolist()
                self.train_poisoned_label += target_transform(target).detach().cpu().numpy().tolist()

            schedulerC.step()
            atkloss = sum(losslist) / len(losslist)
            atkcleanloss = sum(loss_clean_list) / len(loss_clean_list)
            atkpoisonloss = sum(loss_poison_list) / len(loss_poison_list)

            msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                'Finetune [{}] Loss: clean {:.4f} poison {:.4f} total {:.4f}\n'.format(
                    cepoch, atkcleanloss, atkpoisonloss, atkloss)
            self.log(msg)

            if cepoch % self.current_schedule['tune_test_epoch_interval'] == 0 or cepoch == trainepoch-1:
                with torch.no_grad():
                    predict_digits = []
                    labels = []
                    tri_predict_digits = []
                    tri_labels = []
                    for data, target in test_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        predict_digits.append(output.cpu())
                        labels.append(target.cpu())

                        noise = atkmodel(data) * self.tune_test_eps
                        if clip_image is None:
                            atkdata = torch.clamp(data + noise, 0, 1)
                        else:
                            atkdata = clip_image(data + noise)
                        atkoutput = model(atkdata)
                        
                        self.test_poisoned_data += atkdata.detach().cpu().numpy().tolist()
                        self.test_poisoned_label += target_transform(target).detach().cpu().numpy().tolist()
                        
                        tri_predict_digits.append(atkoutput.cpu())
                        tri_labels.append(target_transform(target).cpu())

                    predict_digits = torch.cat(predict_digits, dim=0)
                    labels = torch.cat(labels, dim=0)
                    tri_predict_digits = torch.cat(tri_predict_digits, dim=0)
                    tri_labels = torch.cat(tri_labels, dim=0)

                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on benign test dataset==========\n" + \
                        time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                        f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num} time: {time.time()-self.last_time}\n"
                self.log(msg)
                acc_clean = prec1

                total_num = tri_labels.size(0)
                prec1, prec5 = accuracy(tri_predict_digits, tri_labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on poisoned test dataset==========\n" + \
                        time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                        f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-self.last_time}\n"
                self.log(msg)
                acc_poison = prec1
                
                if acc_clean > best_clean_acc or (acc_clean > best_clean_acc-0.02 and acc_poison > best_poison_acc):
                    best_clean_acc = acc_clean
                    best_poison_acc = acc_poison
                    
                    print(f'Saving current best model in {self.work_dir}\n')
                    if isinstance(model, torch.nn.DataParallel):
                        torch.save({
                            'atkmodel': atkmodel.module.state_dict(),
                            'model': model.module.state_dict(), 
                            'optimizerC': optimizerC.state_dict(), 
                            'clean_schedulerC': schedulerC,
                            'best_clean_acc': best_clean_acc, 
                            'best_poison_acc': best_poison_acc
                        }, os.path.join(self.work_dir, 'best_model.th'))
                    else:
                        torch.save({
                            'atkmodel': atkmodel.state_dict(),
                            'model': model.state_dict(), 
                            'optimizerC': optimizerC.state_dict(), 
                            'clean_schedulerC': schedulerC,
                            'best_clean_acc': best_clean_acc, 
                            'best_poison_acc': best_poison_acc
                        }, os.path.join(self.work_dir, 'best_model.th'))
                if cepoch == trainepoch:
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                        f"Best Clean accuracy: {best_clean_acc} Best Backdoor accuracy: {best_poison_acc} time: {time.time()-self.last_time}\n"
                    self.log(msg)


    def _test(self, dataset, device, batch_size=16, num_workers=8, model=None, atkmodel=None):
        if model is None:
            model = self.model
        else:
            model = model

        target_transform = self.get_target_transform()
        clip_image = self.clip_image

        predict_digits = []
        labels = []
        tri_predict_digits = []
        tri_labels = []
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

            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                predict_digits.append(output.cpu())
                labels.append(target.cpu())

                noise = atkmodel(data) * self.eps
                atkdata = clip_image(data + noise)
                atkoutput = model(atkdata)
                
                tri_predict_digits.append(atkoutput.cpu())
                tri_labels.append(target_transform(target).cpu())

            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            tri_predict_digits = torch.cat(tri_predict_digits, dim=0)
            tri_labels = torch.cat(tri_labels, dim=0)
            return predict_digits, labels, tri_predict_digits, tri_labels


    def test(self, schedule=None, model=None, atkmodel=None, test_dataset=None, poisoned_test_dataset=None):
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

        if atkmodel is None:
            atkmodel = self.atkmodel

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
        
        last_time = time.time()
        predict_digits, labels, tri_predict_digits, tri_labels = self._test(test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], model, atkmodel)        
        total_num = labels.size(0)
        prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
        top1_correct = int(round(prec1.item() / 100.0 * total_num))
        top5_correct = int(round(prec5.item() / 100.0 * total_num))
        msg = "==========Test result on benign test dataset==========\n" + \
                time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num} time: {time.time()-last_time}\n"
        log(msg)

        total_num = tri_labels.size(0)
        prec1, prec5 = accuracy(tri_predict_digits, tri_labels, topk=(1, 5))
        top1_correct = int(round(prec1.item() / 100.0 * total_num))
        top5_correct = int(round(prec5.item() / 100.0 * total_num))
        msg = "==========Test result on poisoned test dataset==========\n" + \
                time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
        log(msg)
