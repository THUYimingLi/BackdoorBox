'''
This is the implement of pruning proposed in [1].
[1] Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks. RAID, 2018.
'''

import os
import torch
import torch.nn as nn


from .base import Base
from ..utils import test
from torch.utils.data import DataLoader


# Define model pruning
class MaskedLayer(nn.Module):
    def __init__(self, base, mask):
        super(MaskedLayer, self).__init__()
        self.base = base
        self.mask = mask

    def forward(self, input):
        return self.base(input) * self.mask



class Pruning(Base):
    """Pruning process.
    Args:
        train_dataset (types in support_list): forward dataset.
        test_dataset (types in support_list): testing dataset.
        model (torch.nn.Module): Network.
        layer(list): The layers to prune
        prune_rate (double): the pruning rate
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self,
                 train_dataset=None,
                 test_dataset=None,
                 model=None,
                 layer=None,
                 prune_rate=None,
                 schedule=None,
                 seed=0,
                 deterministic=False):
        super(Pruning, self).__init__(seed=seed, deterministic=deterministic)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.layer = layer
        self.prune_rate = prune_rate
        self.schedule = schedule


    def repair(self, schedule=None):
        """pruning.
        Args:
            schedule (dict): Schedule for testing.
        """

        if schedule == None:
            raise AttributeError("Schedule is None, please check your schedule setting.")
        current_schedule = schedule


        # Use GPU
        if 'device' in current_schedule and current_schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in current_schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = current_schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert current_schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            print(
                f"This machine has {torch.cuda.device_count()} cuda devices, and use {current_schedule['GPU_num']} of them to train.")

            if current_schedule['GPU_num'] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(current_schedule['GPU_num']))
                self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        model = self.model.to(device)
        layer_to_prune = self.layer
        tr_loader = DataLoader(self.train_dataset, batch_size=current_schedule['batch_size'],
                               num_workers=current_schedule['num_workers'],
                               drop_last=True, pin_memory=True)
        prune_rate = self.prune_rate


        # prune silent activation
        print("======== pruning... ========")
        with torch.no_grad():
            container = []

            def forward_hook(module, input, output):
                container.append(output)

            hook = getattr(model, layer_to_prune).register_forward_hook(forward_hook)
            print("Forwarding all training set")

            model.eval()
            for data, _ in tr_loader:
                model(data.cuda())
            hook.remove()

        container = torch.cat(container, dim=0)
        activation = torch.mean(container, dim=[0, 2, 3])
        seq_sort = torch.argsort(activation)
        num_channels = len(activation)
        prunned_channels = int(num_channels * prune_rate)
        mask = torch.ones(num_channels).cuda()
        for element in seq_sort[:prunned_channels]:
            mask[element] = 0
        if len(container.shape) == 4:
            mask = mask.reshape(1, -1, 1, 1)
        setattr(model, layer_to_prune, MaskedLayer(getattr(model, layer_to_prune), mask))

        self.model = model
        print("======== pruning complete ========")


    def test(self, schedule=None):
        """Test the pruned model.
        Args:
            schedule (dict): Schedule for testing.
        """
        if schedule == None:
            raise AttributeError("Schedule is None, please check your schedule setting.")
        if self.test_dataset == None:
            raise AttributeError("Test set is None, please check your setting.")
        test(self.model, self.test_dataset, schedule)

    def get_model(self):
        return self.model
