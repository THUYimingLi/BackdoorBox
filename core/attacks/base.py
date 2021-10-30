import os
import os.path as osp
import time

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.datasets import MNIST

from ..utils import Log


support_list = (
    DatasetFolder,
    MNIST
)


def check(dataset):
    return isinstance(dataset, support_list)


class Base(object):
    def __init__(self, train_dataset, test_dataset, model, loss, schedule=None):
        assert isinstance(train_dataset, support_list), 'train_dataset is an unsupported dataset type, train_dataset should be a subclass of our support list.'
        self.train_dataset = train_dataset

        assert isinstance(test_dataset, support_list), 'test_dataset is an unsupported dataset type, test_dataset should be a subclass of our support list.'
        self.test_dataset = test_dataset
        self.model = model
        self.loss = loss
        self.schedule = schedule
    
    def train(self, schedule=None):
        if schedule is None:
            schedule = self.schedule
            if schedule is None:
                raise AttributeError("Training schedule is None, please check your schedule setting.")

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
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        train_loader = DataLoader(
            self.train_dataset if schedule['model_type'] == 'benign' else self.poisoned_train_dataset,
            batch_size=schedule['batch_size'],
            shuffle=True,
            num_workers=schedule['num_workers'],
            drop_last=True
        )

        self.model = self.model.to(device)
        self.model.train()

        optimizer = torch.optim.SGD(
            [{'params': self.model.parameters()}],
            lr=schedule['lr']
        )

        work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint

        iteration = 0
        last_time = time.time()
        for i in range(schedule['epochs']):
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
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{schedule['epochs']}, iteration:{batch_id + 1}/{len(self.poisoned_train_dataset)//schedule['batch_size']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                    last_time = time.time()
                    log(msg)

            if (i + 1) % schedule['test_epoch_interval'] == 0:
                predict_digits, labels = self._test(self.test_dataset, device, schedule['batch_size'], schedule['num_workers'])
                correct_num = int((predict_digits == labels).sum())
                total_num = labels.size(0)
                accuracy = correct_num / total_num
                msg = "==========Test result on benign test dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Correct / Total:{correct_num}/{total_num}, Benign accuracy:{accuracy}, time: {time.time()-last_time}\n"
                log(msg)
                if schedule['model_type'] != 'benign':
                    predict_digits, labels = self._test(self.poisoned_test_dataset, device, schedule['batch_size'], schedule['num_workers'])
                    correct_num = int((predict_digits == labels).sum())
                    total_num = labels.size(0)
                    accuracy = correct_num / total_num
                    msg = "==========Test result on poisoned test dataset==========\n" + \
                        time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                        f"Correct / Total:{correct_num}/{total_num}, ASR:{accuracy}, time: {time.time()-last_time}\n"
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
            batch_img = batch[0]
            batch_label = batch[1]
            batch_img = batch_img.to(device)
            batch_label = batch_label.to(device)
            predict_digits.append(self.model(batch_img).cpu())
            labels.append(batch_label.cpu())

        predict_digits = torch.cat(predict_digits, dim=0)
        predict_digits = torch.argmax(predict_digits, dim=1)
        labels = torch.cat(labels, dim=0)
        return predict_digits, labels


    def test(self):
        pass





