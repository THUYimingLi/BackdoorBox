

'''
This is the implement of pre-processing-based backdoor defense with auto-encoder [1].

Reference:
[1] Neural Trojans. ICCD, 2017.
'''


from copy import deepcopy
import os
import os.path as osp
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import Base
from ..utils import Log, test


class AutoEncoderDefense(Base):
    """Backdoor defense with autoencoder.

    Args:
        autoencoder (torch.nn.Module): Autoencoder network.
        pretrain (str): Pretrained autoencoder network path. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self,
                 autoencoder,
                 pretrain=None,
                 seed=0,
                 deterministic=False):
        super(AutoEncoderDefense, self).__init__(seed=seed, deterministic=deterministic)
        self.autoencoder = autoencoder
        if pretrain is not None:
            self.autoencoder.load_state_dict(torch.load(pretrain), strict=False)

    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _test(self, dataset, device, batch_size=16, num_workers=8, loss_func=torch.nn.BCELoss(reduction='none')):
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

            self.autoencoder = self.autoencoder.to(device)
            self.autoencoder.eval()

            losses = []
            for batch in test_loader:
                batch_img, _ = batch
                batch_img = batch_img.to(device)
                out = self.autoencoder(batch_img)
                loss = loss_func(out, batch_img)
                losses.append(loss.cpu())

            losses = torch.cat(losses, dim=0)
            return losses.mean()

    def train_autoencoder(self, train_dataset, test_dataset, schedule):
        if 'pretrain' in schedule:
            self.autoencoder.load_state_dict(torch.load(schedule['pretrain']), strict=False)

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
                self.autoencoder = nn.DataParallel(self.autoencoder.cuda(), device_ids=gpus, output_device=gpus[0])
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


        self.autoencoder = self.autoencoder.to(device)
        self.autoencoder.train()

        loss_func = torch.nn.BCELoss(reduction='mean')
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), schedule['lr'], schedule['betas'], schedule['eps'], schedule['weight_decay'], schedule['amsgrad'])

        work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

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
                predict_digits = self.autoencoder(batch_img)
                loss = loss_func(predict_digits, batch_img)
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

                self.autoencoder = self.autoencoder.to(device)
                self.autoencoder.train()

            if (i + 1) % schedule['save_epoch_interval'] == 0:
                self.autoencoder.eval()
                self.autoencoder = self.autoencoder.cpu()
                ckpt_autoencoder_filename = "ckpt_epoch_" + str(i+1) + ".pth"
                ckpt_autoencoder_path = os.path.join(work_dir, ckpt_autoencoder_filename)
                torch.save(self.autoencoder.state_dict(), ckpt_autoencoder_path)
                self.autoencoder = self.autoencoder.to(device)
                self.autoencoder.train()

    def preprocess(self, data):
        """Perform AutoEncoder defense method on data and return the preprocessed data.

        Args:
            data (torch.Tensor): Input data (between 0.0 and 1.0), shape: (N, C, H, W) or (C, H, W), dtype: torch.float32.

        Returns:
            torch.Tensor: The preprocessed data.
        """
        with torch.no_grad():
            self.autoencoder.eval()
            if data.ndim == 3:
                preprocessed_data = self.autoencoder(data.unsqueeze(0))
                return preprocessed_data[0]
            else:
                return self.autoencoder(data)

    def _predict(self, model, data, device, batch_size, num_workers):
        with torch.no_grad():
            model = model.to(device)
            model.eval()

            predict_digits = []
            for i in range(data.shape[0] // batch_size):
                # breakpoint()
                batch_img = data[i*batch_size:(i+1)*batch_size, ...]
                batch_img = batch_img.to(device)
                batch_img = model(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)

            if data.shape[0] % batch_size != 0:
                batch_img = data[(data.shape[0] // batch_size) * batch_size:, ...]
                batch_img = batch_img.to(device)
                batch_img = model(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)

            predict_digits = torch.cat(predict_digits, dim=0)
            return predict_digits

    def predict(self, model, data, schedule):
        """Apply AutoEncoder defense method to input data and get the predicts.

        Args:
            model (torch.nn.Module): Network.
            data (torch.Tensor): Input data (between 0.0 and 1.0), shape: (N, C, H, W), dtype: torch.float32.
            schedule (dict): Schedule for predicting.

        Returns:
            torch.Tensor: The predicts.
        """
        preprocessed_data = self.preprocess(data)

        if 'test_model' in schedule:
            model.load_state_dict(torch.load(schedule['test_model']), strict=False)

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

    def test(self, model, dataset, schedule):
        """Test AutoEncoder on dataset.

        Args:
            model (torch.nn.Module): Network.
            dataset (types in support_list): Dataset.
            schedule (dict): Schedule for testing.
        """
        defense_dataset = deepcopy(dataset)
        # defense_dataset.transform.transforms.append(transforms.ToPILImage())
        defense_dataset.transform.transforms.append(self.preprocess)
        # defense_dataset.transform.transforms.append(transforms.ToTensor())

        if hasattr(defense_dataset, 'poisoned_transform'):
            # defense_dataset.poisoned_transform.transforms.append(transforms.ToPILImage())
            defense_dataset.poisoned_transform.transforms.append(self.preprocess)
            # defense_dataset.poisoned_transform.transforms.append(transforms.ToTensor())

        test(model, defense_dataset, schedule)
