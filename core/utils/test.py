import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, DatasetFolder

from .accuracy import accuracy
from .log import Log


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _test(model, dataset, device, batch_size=16, num_workers=8):
    with torch.no_grad():
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=_seed_worker
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


def test(model, dataset, schedule):
    """Uniform test API for any model and any dataset.

    Args:
        model (torch.nn.Module): Network.
        dataset (torch.utils.data.Dataset): Dataset.
        schedule (dict): Testing schedule.
    """

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

    last_time = time.time()
    predict_digits, labels = _test(model, dataset, device, schedule['batch_size'], schedule['num_workers'])
    total_num = labels.size(0)
    prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
    top1_correct = int(round(prec1.item() / 100.0 * total_num))
    top5_correct = int(round(prec5.item() / 100.0 * total_num))
    msg = f"==========Test result on {schedule['metric']}==========\n" + \
            time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
            f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
    log(msg)
