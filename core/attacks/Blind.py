'''
This is the implement of Blind Attack [1].
This code is developed based on its official codes (https://github.com/ebagdasa/backdoors101).

Reference:
[1] Blind Backdoors in Deep Learning Models. USENIX Security, 2021.
'''

import copy
import random
from typing import Pattern

import numpy as np
import PIL
from PIL import Image
from torchvision.datasets.folder import make_dataset
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
from .base import *
import warnings

# 1. Dynamic Trigger
# 2. Static Trigger
# Implement Static ones first

def th(vector):
    return torch.tanh(vector) / 2 + 0.5
def thp(vector):
    return torch.tanh(vector) * 2.2

class NCModel(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.pattern = torch.zeros([self.size , self.size ], requires_grad=True)\
                             + torch.normal(0, 0.5, [self.size , self.size ])
        self.mask = torch.zeros([self.size , self.size ], requires_grad=True)
                   # + torch.normal(0, 2, [self.size , self.size ])
        self.mask = nn.Parameter(self.mask)
        self.pattern = nn.Parameter(self.pattern)

    def forward(self, x, latent=None):
        maskh = th(self.mask)
        patternh = thp(self.pattern)
        x = (1 - maskh) * x + maskh * patternh
        return x

    def re_init(self, device):
        p = torch.zeros([self.size , self.size ], requires_grad=True)\
                             + torch.normal(0, 0.5, [self.size , self.size ])
        self.pattern.data = p.to(device)
        m = torch.zeros([self.size , self.size ], requires_grad=True)
        self.mask.data = m.to(device)

def get_inference_result(model, input):
    model.eval()
    with torch.no_grad():
        result = model(input)
    model.train()
    return result

def switch_grad(model, requires_grad=True):
    for n, p in model.named_parameters():
        p.requires_grad_(requires_grad)

def compute_all_losses_and_grads(loss_tasks, model, nc_model, nc_p_norm,
                                 criterion, batch, batch_back,
                                 compute_grad=None):
    grads = {}
    loss_values = {}
    normal_outputs = None
    for t in loss_tasks:
        if t == 'normal':
            loss_values[t], grads[t], normal_outputs = compute_normal_loss(model,
                                                           criterion,
                                                           batch[0],
                                                           batch[1],
                                                           grads=compute_grad)
        elif t == 'backdoor':
            loss_values[t], grads[t] = compute_backdoor_loss(model,
                                                             criterion,
                                                             batch_back[0],
                                                             batch_back[1],
                                                             grads=compute_grad)
        elif t == 'neural_cleanse':
            loss_values[t], grads[t] = compute_nc_evasion_loss(
                nc_model,
                model,
                batch[0],
                batch[1],
                grads=compute_grad)

        elif t == 'mask_norm':
            loss_values[t], grads[t] = norm_loss(nc_p_norm, nc_model,
                                                 grads=compute_grad)
        elif t == 'neural_cleanse_part1':
            loss_values[t], grads[t], _ = compute_normal_loss(model,
                                                           criterion,
                                                           batch[0],
                                                           batch_back[1],
                                                           grads=compute_grad,
                                                           )
    return loss_values, grads, normal_outputs

def compute_normal_loss(model, criterion, inputs,
                        labels, grads):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    if grads:
        grads = list(torch.autograd.grad(loss,
                                         [x for x in model.parameters() if
                                          x.requires_grad],
                                         retain_graph=True))

    return loss, grads, outputs

def compute_nc_evasion_loss(nc_model, model, inputs,
                            labels, grads=None):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    switch_grad(nc_model, False)
    outputs = model(nc_model(inputs))
    loss = criterion(outputs, labels).mean()

    if grads:
        grads = get_grads(model, loss)
    return loss, grads

def compute_backdoor_loss(model, criterion, inputs_back,
                          labels_back, grads=None):
    outputs = model(inputs_back)
    loss = criterion(outputs, labels_back)

    if grads:
        grads = get_grads(model, loss)

    return loss, grads

def get_latent_grads(target_label, model, inputs, labels):
    model.eval()
    model.zero_grad()
    pred = model(inputs)

    z = torch.zeros_like(pred)
    z[list(range(labels.shape[0])), labels] = 1

    pred = pred * z
    pred.sum().backward(retain_graph=True)
    gradients = model.get_gradient()[labels == target_label]
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]).detach()
    model.zero_grad()
    return pooled_gradients

def norm_loss(mask_p_norm, model, grads=None):
    if mask_p_norm == 1:
        norm = torch.sum(th(model.mask))
    elif mask_p_norm == 2:
        norm = torch.norm(th(model.mask))
    else:
        raise ValueError('Not support mask norm.')

    if grads:
        grads = get_grads(model, norm)
        model.zero_grad()
    return norm, grads

def get_grads(model, loss):
    grads = list(torch.autograd.grad(loss,
                                     [x for x in model.parameters() if
                                      x.requires_grad],
                                     retain_graph=True))
    return grads

# Credits to Ozan Sener
# https://github.com/intel-isl/MultiObjectiveOptimization

class MGDASolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    @staticmethod
    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    @staticmethod
    def _min_norm_2d(vecs: list, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0
        for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        sol = 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.dot(vecs[i][k].view(-1),
                                                 vecs[j][k].view(-1)).detach()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.dot(vecs[i][k].view(-1),
                                                 vecs[i][k].view(-1)).detach()
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.dot(vecs[j][k].view(-1),
                                                 vecs[j][k].view(-1)).detach()
                c, d = MGDASolver._min_norm_element_from2(dps[(i, i)],
                                                          dps[(i, j)],
                                                          dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    @staticmethod
    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    @staticmethod
    def _next_point(cur_val, grad, n):
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

        skippers = np.sum(tm1 < 1e-7) + np.sum(tm2 < 1e-7)
        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = MGDASolver._projection2simplex(next_point)
        return next_point

    @staticmethod
    def find_min_norm_element(vecs: list):
        """
        Given a list of vectors (vecs), this method finds the minimum norm
        element in the convex hull as min |u|_2 st. u = \sum c_i vecs[i]
        and \sum c_i = 1. It is quite geometric, and the main idea is the
        fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution
        lies in (0, d_{i,j})Hence, we find the best 2-task solution , and
        then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MGDASolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MGDASolver.MAX_ITER:
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = MGDASolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MGDASolver._min_norm_element_from2(v1v1.item(),
                                                        v1v2.item(),
                                                        v2v2.item())
            # try:
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            # except AttributeError:
            #     print(sol_vec)
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MGDASolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    @staticmethod
    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm
        element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if
        d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies
        in (0, d_{i,j})Hence, we find the best 2-task solution, and then
        run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MGDASolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MGDASolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MGDASolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MGDASolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    @classmethod
    def get_scales(cls, grads, losses, normalization_type, tasks):
        scale = {}
        gn = gradient_normalizers(grads, losses, normalization_type)
        for t in tasks:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / (gn[t] + 1e-5)
        sol, min_norm = cls.find_min_norm_element([grads[t] for t in tasks])
        for zi, t in enumerate(tasks):
            scale[t] = float(sol[zi])

        return scale

def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = torch.sqrt(
                torch.stack([gr.pow(2).sum().data for gr in grads[t]]).sum())
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = min(losses[t].mean(), 10.0)
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = min(losses[t].mean() * torch.sqrt(
                torch.stack([gr.pow(2).sum().data for gr in grads[t]]).sum()),
                        10)

    elif normalization_type == 'none' or normalization_type == 'eq':
        for t in grads:
            gn[t] = 1.0
    else:
        raise ValueError('ERROR: Invalid Normalization Type')
    return gn

class AddTrigger(nn.Module):
    def __init__(self, pattern, alpha):
        super(AddTrigger, self).__init__()
        self.pattern = nn.Parameter(pattern, requires_grad=False)
        self.alpha = nn.Parameter(alpha, requires_grad=False)

    def forward(self, img, batch=False):
        """Add trigger to image.
        if batch==False, add trigger to single image of shape (C,H,W)
        else , add trigger to a batch of images of shape (N, C, H, W)

        Args:
            img (torch.Tensor): shape (C, H, W) if batch==False else (N, C, H, W)

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W) if batch==False else (N, C, H, W)
        """
        if batch:
            return (1-self.alpha).unsqueeze(0) * img + (self.alpha*self.pattern).unsqueeze(0)
        return (1-self.alpha)*img + self.alpha * self.pattern

class Blind(Base):
    """class for Blind backdoor training and testing.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        pattern (None | torch.Tensor): Trigger pattern, shape (C, H, W) or (H, W).
        alpha (torch.Tensor): Transparency of trigger pattern, shape (C, H, W).
        y_target (int): N-to-1 attack target label.
        schedule (dict): Training or testing global schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
        use_neural_cleanse: ?
        nc_mask_p_norm: ?
        loss_balance: ?
        mgda_normalize: ?
        fixed_scales: ?
    """

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 model,
                 loss,
                 pattern,
                 alpha,
                 y_target,
                 schedule=None,
                 seed=0,
                 deterministic=False,
                 use_neural_cleanse=True,
                 nc_mask_p_norm=1,
                 loss_balance='MGDA',
                 mgda_normalize='loss+',
                 fixed_scales=[]):
        super(Blind, self).__init__(
            train_dataset,
            test_dataset,
            model,
            loss,
            schedule,
            seed,
            deterministic)

        self.loss_balance=loss_balance
        self.mgda_normalize = mgda_normalize
        self.fixed_scales = fixed_scales
        self.NC = use_neural_cleanse
        self.nc_model = NCModel(pattern.shape[-1])
        self.nc_optim = torch.optim.Adam(self.nc_model.parameters(), 0.01)
        self.nc_mask_p_norm=nc_mask_p_norm
        self.add_trigger = AddTrigger(pattern, alpha)
        self.y_target = y_target
        self.crafted = False
    
    def get_model(self, return_NC=False):
        if self.crafted is False:
            warnings.warn("Models haven't complete training yet! Will get incompetent models!")
            print("Models haven't complete training yet! Will get incompetent models!")
        if return_NC:
            return self.model, self.nc_model
        else:
            return self.model

        
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
            ckpt = torch.load(self.current_schedule['pretrain'], strict=False)
            self.model.load_state_dict(ckpt['model'])
            self.nc_model.load_state_dict(ckpt['nc_model'])

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
                self.nc_model = nn.DataParallel(self.nc_model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.current_schedule['batch_size'],
            shuffle=True,
            num_workers=self.current_schedule['num_workers'],
            drop_last=True,
            worker_init_fn=self._seed_worker
        )
        # # test
        # n=0
        # for batch_id, batch in enumerate(train_loader):
        #     n+=1
        # print("pass %d batches of data"%n)
        # return 

        
        self.model = self.model.to(device)
        self.model.train()
        self.nc_model = self.nc_model.to(device)
        self.nc_model.train()
        self.add_trigger = self.add_trigger.to(device)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_schedule['lr'], momentum=self.current_schedule['momentum'], weight_decay=self.current_schedule['weight_decay'])

        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint

        iteration = 0
        last_time = time.time()

        msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.test_dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(self.train_dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
        log(msg)
        
        for i in range(self.current_schedule['epochs']):
            self.adjust_learning_rate(optimizer, i)            
            for batch_id, batch in enumerate(train_loader):
                # print(batch_id)
                batch_img = batch[0]
                batch_label = batch[1]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                optimizer.zero_grad()
                # predict_digits = self.model(batch_img)
                loss, predict_digits = self.compute_blind_loss(batch_img, batch_label)
                loss.backward()
                optimizer.step()

                iteration += 1

                if iteration % self.current_schedule['log_iteration_interval'] == 0:
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch: {i+1}/{self.current_schedule['epochs']}, iteration: {batch_id + 1}/{len(self.train_dataset)//self.current_schedule['batch_size']}, lr: {self.current_schedule['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                    last_time = time.time()
                    log(msg)
            if (i + 1) % self.current_schedule['test_epoch_interval'] == 0:
                # test result on benign test dataset
                predict_digits, labels = self._test(self.test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], backdoor=False)
                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on benign test dataset==========\n" + \
                        time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                        f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num} time: {time.time()-last_time}\n"
                log(msg)

                # test result on poisoned test dataset
                # if self.current_schedule['benign_training'] is False:
                predict_digits, labels = self._test(self.test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on poisoned test dataset==========\n" + \
                        time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                        f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
                log(msg)

                self.model = self.model.to(device)
                self.model.train()
                self.nc_model = self.nc_model.to(device)
                self.nc_model.train()

            if (i + 1) % self.current_schedule['save_epoch_interval'] == 0:
                self.model.eval()
                self.model = self.model.cpu()
                self.nc_model.eval()
                self.nc_model = self.nc_model.cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(i+1) + ".pth"
                ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                ckpt = {'model':self.model.state_dict(), 'nc_model':self.nc_model.state_dict()}
                torch.save(ckpt, ckpt_model_path)
                self.model = self.model.to(device)
                self.model.train()
                self.nc_model = self.nc_model.to(device)
                self.nc_model.train()
                
        self.crafted=True
   
    def _test(self, dataset, device, batch_size=16, num_workers=8, backdoor=True, model=None):
        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                worker_init_fn=self._seed_worker
            )
            if model is None:
                model = self.model
            model = model.to(device)
            model.eval()


            predict_digits = []
            labels = []
            for batch in test_loader:
                batch_img, batch_label = batch
                batch_img = batch_img.to(device)
                if backdoor:
                    batch_img, batch_label = self.make_backdoor_batches(batch_img, batch_label)
                batch_predict_digits = model(batch_img)
                batch_predict_digits = batch_predict_digits.cpu()
                predict_digits.append(batch_predict_digits)
                labels.append(batch_label)

            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            return predict_digits, labels
    
    def test(self, schedule=None, model=None, nc_model=None, test_dataset=None):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Test schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.schedule is not None:
            self.current_schedule = deepcopy(schedule)

        if model is None:
            model = self.model
        if nc_model is None:
            nc_model = self.nc_model

        if 'test_model' in self.current_schedule:
            ckpt = torch.load(self.current_schedule['test_model'])
            model.load_state_dict(ckpt['model'], strict=False)
            nc_model.load_state_dict(ckpt['nc_model'], strict=False)

        if test_dataset is None:
            test_dataset = self.test_dataset

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
        self.add_trigger = self.add_trigger.to(device)
        self.nc_model = self.nc_model.to(device)

        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        if test_dataset is not None:
            last_time = time.time()
            # test result on benign test dataset
            predict_digits, labels = self._test(test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], backdoor=False, model=model)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num} time: {time.time()-last_time}\n"
            log(msg)

        if test_dataset is not None:
            last_time = time.time()
            # test result on poisoned test dataset
            predict_digits, labels = self._test(test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], model=model)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
            log(msg)

    
    def make_backdoor_batches(self, imgs, labels):
        with torch.no_grad():
            # bd_imgs = torch.stack([self.add_trigger(img) for img in imgs],dim=0)
            bd_imgs = self.add_trigger(imgs, batch=True)
            bd_labels = torch.zeros_like(labels).fill_(self.y_target)
        return (bd_imgs, bd_labels)

    def compute_blind_loss(self, x, y, attack=True):
        # assign tasks
        tasks = ['normal']
        if attack: 
            tasks += ['backdoor']
            if self.NC:
                tasks += ['neural_cleanse']
        scale = dict()
        batch = (x, y)
        batch_back = self.make_backdoor_batches(x, y)
        logits = None
        if 'neural_cleanse' in tasks:
            self.neural_cleanse_part1(batch, batch_back)

        if len(tasks) == 1:
            loss_values, grads, logits = compute_all_losses_and_grads(
                tasks, self.model, self.nc_model, self.nc_mask_p_norm, 
                    self.loss, batch, batch_back, compute_grad=False
            )
            scale = {tasks[0]: 1.0}
        elif self.loss_balance == 'MGDA':
            loss_values, grads, _ = compute_all_losses_and_grads(
                tasks, self.model, self.nc_model, self.nc_mask_p_norm, 
                    self.loss, batch, batch_back, compute_grad=True
            )
            if len(tasks) > 1:
                with torch.no_grad():
                    scale = MGDASolver.get_scales(grads, loss_values,
                                                self.mgda_normalize,
                                                tasks)
        elif self.loss_balance == 'fixed':
            loss_values, grads, _ = compute_all_losses_and_grads(
                tasks, self.model, self.nc_model, self.nc_mask_p_norm, 
                    self.loss, batch, batch_back, compute_grad=False
            )
            for t in tasks:
                scale[t] = self.fixed_scales[t]
        else:
            raise ValueError(f'Please choose between `MGDA` and `fixed`.')
            
        blind_loss = self.scale_losses(tasks, loss_values, scale)
        return blind_loss, logits
        
    def scale_losses(self, loss_tasks, loss_values, scale):
        blind_loss = 0
        for t in loss_tasks:
            blind_loss += scale[t] * loss_values[t]
        return blind_loss

    def neural_cleanse_part1(self, batch, batch_back):
        self.nc_model.zero_grad()
        self.model.zero_grad()

        switch_grad(self.nc_model, True)
        switch_grad(self.model, False)
        nc_tasks = ['neural_cleanse_part1', 'mask_norm']

        # criterion = torch.nn.CrossEntropyLoss(reduction='none')
        criterion = torch.nn.CrossEntropyLoss()

        loss_values, _, _ = compute_all_losses_and_grads(nc_tasks,
            self.model, self.nc_model, self.nc_mask_p_norm, criterion, 
            batch, batch_back, compute_grad=False)
        # Using NC paper params
        self.nc_optim.zero_grad()
        loss = 0.999 * loss_values['neural_cleanse_part1'] + 0.001 * loss_values['mask_norm']
        loss.backward()
        self.nc_optim.step()

        switch_grad(self.nc_model, False)
        switch_grad(self.model, True)
    
    def get_poisoned_dataset(self, NC=False):
        """ Train or Test must be called before you call this function """
        if self.current_schedule is None:
            if self.global_schedule:
                self.current_schedule = self.global_schedule
            else:
                raise ValueError("Train or Test must be called before you call this function")
        
        batch_size = self.current_schedule['batch_size']
        num_workers = self.current_schedule['num_workers']
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
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")
        self.poisoned_train_dataset = self.construct_poisoned_dataset(self.train_dataset, batch_size, num_workers, device, NC)
        self.poisoned_test_dataset = self.construct_poisoned_dataset(self.test_dataset, batch_size, num_workers, device, NC)
        return self.poisoned_train_dataset, self.poisoned_test_dataset

    def construct_poisoned_dataset(self, dataset, batch_size, num_workers, device, NC=False):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            worker_init_fn=self._seed_worker
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.nc_model = self.nc_model.to(device)
        self.nc_model.eval()
        self.add_trigger.to(device)
        with torch.no_grad():
            imgs, labels = [], []
            for batch in dataloader:
                batch_img, batch_label = batch
                batch_img = batch_img.to(device)
                batch_img, batch_label = self.make_backdoor_batches(batch_img, batch_label)
                if NC:
                    batch_img = self.nc_model(batch_img)
                imgs.append(batch_img.cpu())
                labels.append(batch_label.cpu())
            imgs = torch.cat(imgs,dim=0)
            labels = torch.cat(labels,dim=0)
            return torch.utils.data.TensorDataset(imgs, labels)
        
