'''
This is the implement of spectral signatures backdoor defense. 
This code is developed based on its official codes, but use the PyTorch instead of TensorFlow. 
(link: https://github.com/MadryLab/backdoor_data_poisoning)

Reference:
[1] Spectral Signatures in Backdoor Attacks. NeurIPS, 2018.
'''

import os
import os.path as osp
from copy import deepcopy
import time
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Base
from torch.utils.data import DataLoader, Subset

from ..utils.accuracy import accuracy
from ..utils.log import Log
from ..utils.compute_metric import compute_confusion_matrix, compute_indexes

from multiprocessing.sharedctypes import Value
from tqdm import trange

class Spectral(Base):
    """Filtering the training samples spectral signatures (Spectral).

    Args:
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        poisoned_trainset (types in support_list): Poisoned trainset.
        poisoned_testset (types in support_list): Poisoned testset.
        clean_trainset (types in support_list): Clean trainset.
        clean_testset (types in support_list): Clean testset.
        target_label (int): N-to-1 attack target label.
        seed (int): Global seed for random numbers. Default: 0.
        percentile(float): 0-to-100. Default: 85
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    
    """
    def __init__(self,   
                 model, 
                 loss,
                 seed=0,
                 target_label = 1,
                 percentile = 85,                  
                 deterministic=False,
                 poisoned_trainset=None, 
                 clean_trainset=None,
                 filtered_poisoned_id=None,
                 filtered_benign_id=None,
                 ):
        super(Spectral, self).__init__(seed, deterministic)

        self.model = model
        self.loss = loss
        self.seed = seed
        self.target_label = target_label
        self.percentile = percentile
        self.poisoned_trainset = poisoned_trainset
        self.clean_trainset = clean_trainset
        self.filtered_poisoned_id=filtered_poisoned_id
        self.filtered_benign_id=filtered_benign_id

    def filter(self, schedule):
        """filter out poisoned samples from poisoned dataset. 
        
        Args:
            schedule (dict): schedule for spliting the dataset.            
        """

        if schedule is None:
            raise AttributeError("schedule is None, please check your schedule setting.")
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
                if not isinstance(self.model, nn.DataParallel):
                    gpus = list(range(self.current_schedule['GPU_num']))
                    self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        self.model = self.model.to(device)

        ### a. prepare the model and dataset
        model = self.model
        poisoned_trainset = self.poisoned_trainset
        filtered_poisoned_id = self.filtered_poisoned_id
        filtered_benign_id = self.filtered_benign_id

        lbl = self.target_label

        poisoned_label = []

        for i in range(len(poisoned_trainset)):
            poisoned_label.append(poisoned_trainset[i][1])

        cur_indices = [i for i,v in enumerate(poisoned_label) if v==lbl]    
        cur_examples = len(cur_indices)
        #print('Target_label, num: ', lbl, cur_examples)
        model.eval()

        ### b. get the activation as representation for each data
        for iex in trange(cur_examples):
            cur_im = cur_indices[iex]
            x_batch = poisoned_trainset[cur_im][0].unsqueeze(0).to(device)
            y_batch = poisoned_trainset[cur_im][1]
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.layer4.register_forward_hook(layer_hook)
            _ = model(x_batch)
            batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
            hook.remove()
            if iex==0:
                full_cov = np.zeros(shape=(cur_examples, len(batch_grads)))
            full_cov[iex] = batch_grads.detach().cpu().numpy()

        ### c. detect the backdoor data by the SVD decomposition
        total_p = self.percentile            
        full_mean = np.mean(full_cov, axis=0, keepdims=True) 
        
        centered_cov = full_cov - full_mean
        u,s,v = np.linalg.svd(centered_cov, full_matrices=False)
        print('Top 7 Singular Values: '+ str(s[0:7]))
        eigs = v[0:1]  
        p = total_p
        #shape num_top, num_active_indices
        corrs = np.matmul(eigs, np.transpose(full_cov)) 
        #shape num_active_indices
        scores = np.linalg.norm(corrs, axis=0) 
        # length of score
        print('Length Scores:'+str(len(scores)) )
        p_score = np.percentile(scores, p)
        top_scores = np.where(scores>p_score)[0]
        #print(top_scores)
    
        #remove
        filtered_poisoned_id = np.copy(top_scores)
        print('removed_inds_length:'+str(len(filtered_poisoned_id)))
        print(filtered_poisoned_id)

        #left
        re = [cur_indices[v] for i,v in enumerate(filtered_poisoned_id)]
        filtered_benign_id = np.delete(range(len(poisoned_trainset)), re)
        print('left_inds_length:'+str(len(filtered_benign_id)))       
        print(filtered_benign_id)

        return filtered_poisoned_id, filtered_benign_id
    
    
    def test(self,poisoned_location,schedule):
        """compute metrics: accuracy, precision, recall, F1. 

            Args:
            poisoned_location (frozenset): poisoned id in clean_dataset
            schedule (dict): schedule for spliting the dataset.            
        """
        # prepare dataset
        filtered_poisoned_id, filtered_benign_id = self.filter(schedule)
        poisoned_id = filtered_poisoned_id
        benign_id = filtered_benign_id
        poisoned_trainset = self.poisoned_trainset

        # tolist
        poisoned_label = []
        for m in poisoned_location:
             poisoned_label.append(m)
        
        # set parameters
        precited = np.zeros(len(poisoned_trainset))
        expected = np.zeros(len(poisoned_trainset))

        # get precited and expected
        for i in range(len(expected)):
            if i in poisoned_label:
                expected[i] = 1
            else:
                expected[i] = 0

        for i in range(len(poisoned_trainset)):
            if i in poisoned_id:
                precited[i] = 1
            elif i in benign_id:
                precited[i] = 0

        precited = [ int(x) for x in precited.tolist() ]
        expected = [ int(x) for x in expected.tolist() ]
        
        tp, fp, tn, fn = compute_confusion_matrix(precited, expected)
        
        print('tp :' + str(tp) + ', fp :' + str(fp) + ', tn : ' + str(tn) + ', fn :' + str(fn))


        accuracy, precision, recall, F1 = compute_indexes(tp, fp, tn, fn)
        
        print('accuracy :' + str(accuracy) + ', precision :' + str(precision) + ', recall : ' + str(recall) + ', F1 :' + str(F1))

    