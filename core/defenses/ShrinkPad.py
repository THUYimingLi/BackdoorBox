'''
This is the implement of pre-processing-based backdoor defense with ShrinkPad proposed in [1].

Reference:
[1] Backdoor Attack in the Physical World. ICLR Workshop, 2021.
'''


import os
from copy import deepcopy

import torch
import torchvision.transforms as transforms

from .base import Base
from ..utils import test


def RandomPad(sum_w, sum_h, fill=0):
    transforms_bag=[]
    for i in range(sum_w+1):
        for j in range(sum_h+1):
            transforms_bag.append(transforms.Pad(padding=(i,j,sum_w-i,sum_h-j)))

    return transforms_bag


def build_ShrinkPad(size_map, pad):
    return transforms.Compose([
        transforms.Resize((size_map - pad, size_map - pad)),
        transforms.RandomChoice(RandomPad(sum_w=pad, sum_h=pad))
        ])


class ShrinkPad(Base):
    """Construct defense datasets with ShrinkPad method.

    Args:
        size_map (int): Size of image.
        pad (int): Size of pad.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self,
                 size_map,
                 pad,
                 seed=0,
                 deterministic=False):

        super(ShrinkPad, self).__init__(seed=seed, deterministic=deterministic)

        self.global_size_map = size_map
        self.current_size_map = None

        self.global_pad = pad
        self.current_pad = None

    def preprocess(self, data, size_map=None, pad=None):
        """Perform ShrinkPad defense method on data and return the preprocessed data.

        Args:
            data (torch.Tensor): Input data.
            size_map (int): Size of image. Default: None.
            pad (int): Size of pad. Default: None.

        Returns:
            torch.Tensor: The preprocessed data.
        """
        if size_map is None:
            self.current_size_map = self.global_size_map
        else:
            self.current_size_map = size_map

        if pad is None:
            self.current_pad = self.global_pad
        else:
            self.current_pad = pad

        shrinkpad = build_ShrinkPad(self.current_size_map, self.current_pad)
        return shrinkpad(data)

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

    def predict(self, model, data, schedule, size_map=None, pad=None):
        """Apply ShrinkPad defense method to input data and get the predicts.

        Args:
            model (torch.nn.Module): Network.
            data (torch.Tensor): Input data.
            schedule (dict): Schedule for predicting.
            size_map (int): Size of image. Default: None.
            pad (int): Size of pad. Default: None.

        Returns:
            torch.Tensor: The predicts.
        """
        if size_map is None:
            self.current_size_map = self.global_size_map
        else:
            self.current_size_map = size_map

        if pad is None:
            self.current_pad = self.global_pad
        else:
            self.current_pad = pad

        shrinkpad = build_ShrinkPad(self.current_size_map, self.current_pad)
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

    def test(self, model, dataset, schedule, size_map=None, pad=None):
        """Test ShrinkPad on dataset.

        Args:
            model (torch.nn.Module): Network.
            dataset (types in support_list): Dataset.
            schedule (dict): Schedule for testing.
            size_map (int): Size of image. Default: None.
            pad (int): Size of pad. Default: None.
        """
        if size_map is None:
            self.current_size_map = self.global_size_map
        else:
            self.current_size_map = size_map

        if pad is None:
            self.current_pad = self.global_pad
        else:
            self.current_pad = pad

        defense_dataset = deepcopy(dataset)
        # defense_dataset.transform.transforms.append(transforms.ToPILImage())
        defense_dataset.transform.transforms.append(build_ShrinkPad(self.current_size_map, self.current_pad))
        # defense_dataset.transform.transforms.append(transforms.ToTensor())

        if hasattr(defense_dataset, 'poisoned_transform'):
            # defense_dataset.poisoned_transform.transforms.append(transforms.ToPILImage())
            defense_dataset.poisoned_transform.transforms.append(build_ShrinkPad(self.current_size_map, self.current_pad))
            # defense_dataset.poisoned_transform.transforms.append(transforms.ToTensor())

        test(model, defense_dataset, schedule)
