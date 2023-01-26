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
                 poisoned_trainset, 
                 poisoned_testset, 
                 clean_trainset,
                 clean_testset, 
                 seed=0,
                 target_label = 0,
                 percentile = 85,                  
                 deterministic=False
                 ):
        super(Spectral, self).__init__(seed, deterministic)

        self.model = model
        self.loss = loss
        self.seed = seed
        self.poisoned_trainset = poisoned_trainset
        self.poisoned_testset = poisoned_testset
        self.clean_trainset = clean_trainset
        self.clean_testset = clean_testset
        self.target_label = target_label
        self.percentile = percentile

    def repair(self, schedule, transform):
        """Perform Spectral defense method based on attacked dataset and retrain with the filtered dataset. 
        The repaired model will be stored in self.model
        
        Args:
            schedule (dict): Schedule for Spectral. Contraining sub-schedule for pre-isolatoin training, clean training, unlearning and test phrase.
            transform (classes in torchvison.transforms): Transform for poisoned trainset in filter phrase
        """
        
        # get logger
        work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        self.work_dir = work_dir
        self.log = log
        self.test_schedule = schedule['test_schedule']

        # train the model with poisoned dataset
        log("\n\n\n==========Start training with poisoned dataset==========\n")
        self._train(self.poisoned_trainset, schedule['first_train_schedule'])
        print('the length of poisoned dataset: ' + str(len(self.poisoned_trainset)))
        self.save_ckpt("first_train.pth")

        # filter out poisoned samples
        log("\n\n\n==========Start filtering the poisoned samples==========\n")       
        train_transform = self.poisoned_trainset.transform
        self.poisoned_trainset.transform = transform 
        
        removed_indices, left_indices, cur_indices = self.filter(self.poisoned_trainset, schedule['filter_schedule'])
        self.poisoned_trainset.transform = train_transform
        removed_dataset = Subset(self.poisoned_trainset, removed_indices) 
        left_dataset = Subset(self.poisoned_trainset, left_indices) 
        
        self.poisoned_dataset = removed_dataset
        print('the length of removed samples: '+ str(len(removed_dataset)))
        print('the length of left samples: '+ str(len(left_dataset)))
        torch.save(removed_dataset, os.path.join(work_dir, "selected_poison.pth"))
        log("\n\n\nSelect %d poisoned data"%len(removed_indices))
        
        # compute metrics
        poisoned_id = removed_indices
        benign_id = left_indices

        precited = np.zeros(len(cur_indices))
        expected = np.ones(len(cur_indices))

        for i,v in enumerate(cur_indices):
            if v in poisoned_id:
                precited[i] =1
            elif v  in benign_id:
                precited[i] = 0

        precited = [ int(x) for x in precited.tolist() ]
        expected = [ int(x) for x in expected.tolist() ]
        
        tp, fp, tn, fn = compute_confusion_matrix(precited, expected)

        accuracy, precision, recall, F1 = compute_indexes(tp, fp, tn, fn)
        
        print('accuracy :' + str(accuracy) + ', precision :' + str(precision) + ', recall : ' + str(recall) + ', F1 :' + str(F1))


        # Train the model with filtered dataset
        log("\n\n\n==========Training with selected clean data==========\n")
        self._train(removed_dataset, schedule['retrain_schedule'])
        self.save_ckpt("retrain.pth")



    def filter(self, dataset, schedule):
        """filter out poisoned samples from poisoned dataset. 
        
        Args:
            dataset (torch.utils.data.Dataset): The dataset to filter.
            schedule (dict): schedule for spliting the dataset.            
        """

        if schedule is None:
            raise AttributeError("Reparing Training schedule is None, please check your schedule setting.")
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

        lbl = self.target_label
        dataset_y = []

        for i in range(len(dataset)):
            dataset_y.append(dataset[i][1])
        # print(dataset_y)
        cur_indices = [i for i,v in enumerate(dataset_y) if v==lbl]
        # benign_id = [i for j,v in enumerate(dataset_y) if v!=lbl]
        cur_examples = len(cur_indices)
        # print(cur_indices)
        print('Label, num ex: ', lbl, cur_examples)
        model.eval()

        ### b. get the activation as representation for each data
        for iex in trange(cur_examples):
            cur_im = cur_indices[iex]
            x_batch = dataset[cur_im][0].unsqueeze(0).to(device)
            y_batch = dataset[cur_im][1]
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
        print('Length Scores:'+str(len(scores)) )
        p_score = np.percentile(scores, p)
        top_scores = np.where(scores>p_score)[0]
        print(top_scores)
    
        removed_inds = np.copy(top_scores)
        print('removed:'+str(len(removed_inds)))
        re = [cur_indices[v] for i,v in enumerate(removed_inds)]
        left_inds = np.delete(range(len(dataset)), re)
        print('left:'+str(len(left_inds)))       
        
        return removed_inds, left_inds, cur_indices

    # def mymetric(self,benign_id,poisoned_id):
    #     # expected_label = np.zeros(cur_examples)
    #     # for i in expected_label:
    #     #     expected_label[i] = lbl

    #     # predicted_label = 
    #     # for  i,v in enumerate(dataset_y):


    #     # tp, fp, tn, fn = compute_confusion_matrix(newlabel, cur_indices)

    #     # accuracy, precision, recall, F1 = compute_indexes(tp, fp, tn, fn)
    #     # print(accuracy, precision, recall, F1)
    #     poisoned_id = removed_inds
    #     benign_id = left_inds

    #     precited = np.zeros(len(cur_indices))
    #     expected = np.zeros(len(cur_indices))

    #     for i in range(len(cur_indices)-1):
    #         expected[i] == 1

    #     for i,v in enumerate(cur_indices):
    #         if v in poisoned_id:
    #             precited[i] =1
    #         elif v  in benign_id:
    #             precited[i] = 0

    #     tp, fp, tn, fn = compute_confusion_matrix(precited, expected)

    #     accuracy, precision, recall, F1 = compute_indexes(tp, fp, tn, fn)
        
    #     print(accuracy, precision, recall, F1)


    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def save_ckpt(self, ckpt_name):
        ckpt_model_path = os.path.join(self.work_dir, ckpt_name)
        torch.save(self.model.cpu().state_dict(), ckpt_model_path)
        return 

    def _test(self, model, dataset, device, batch_size=16, num_workers=8):
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

    def test(self, model, dataset, schedule=None):
        if schedule is None:
            schedule = self.test_schedule
        # Use GPU
        if 'device' in schedule and schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert schedule['GPU_num'] >0, 'GPU_num should be a positive integer'
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {schedule['GPU_num']} of them to test.")

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

        predict_digits, labels = self._test(model, dataset, device, schedule['batch_size'], schedule['num_workers'])
        total_num = labels.size(0)
        prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
        loss = self.loss(predict_digits, labels)
        return loss, prec1, prec5, total_num

    def _train(self, dataset, schedule):
        log = self.log
        
        if schedule is None:
            raise AttributeError("Reparing Training schedule is None, please check your schedule setting.")
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

        train_loader = DataLoader(
            dataset,
            batch_size=self.current_schedule['batch_size'],
            shuffle=True,
            num_workers=self.current_schedule['num_workers'],
            drop_last=False,
            pin_memory=True,
            worker_init_fn=self._seed_worker
        )

        self.model = self.model.to(device)
        model = self.model.to()
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.current_schedule['lr'], momentum=self.current_schedule['momentum'], weight_decay=self.current_schedule['weight_decay'])
        
        iteration = 0
        last_time = time.time()

        msg = f"Total train samples: {len(dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
        log(msg)

        for i in range(self.current_schedule['epochs']):
            for batch_id, batch in enumerate(train_loader):
                batch_img = batch[0]
                batch_label = batch[1]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                optimizer.zero_grad()
                predict_digits = model(batch_img)
                loss = self.loss(predict_digits, batch_label)
                loss.backward()
                optimizer.step()

                iteration += 1

                if iteration % self.current_schedule['log_iteration_interval'] == 0:
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{self.current_schedule['epochs']}, iteration:{batch_id + 1}/{len(dataset)//self.current_schedule['batch_size']}, lr: {self.current_schedule['lr']}, time: {time.time()-last_time}\n"
                    last_time = time.time()
                    log(msg)
                    model.train()

            if (i + 1) % self.current_schedule['test_epoch_interval'] == 0:
                poison_loss, asr, _, _ = self.test(model, self.poisoned_testset, self.test_schedule)
                clean_loss, acc, _, _ = self.test(model, self.clean_testset, self.test_schedule)
                model.train()
                msg = "==========Test results ==========\n" + \
                            time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{self.current_schedule['epochs']}" +\
                            " ASR: %.2f Acc: %.2f poison_loss: %.3f clean_loss: %.3f\n"%(asr, acc, poison_loss, clean_loss)
                log(msg)

        
