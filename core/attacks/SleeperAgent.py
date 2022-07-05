'''
This is the implement of Sleeper Agent Attack [1].
This code is developed based on its official codes (https://github.com/hsouri/Sleeper-Agent).

Reference:
[1] Sleeper Agent: Scalable Hidden Trigger Backdoors for Neural Networks Trained from Scratch.arXiv, 2021.
'''

from cv2 import compare
from .base import *
from copy import deepcopy
import torch.nn.functional as F
from math import ceil
# from copy import deepcopy

class Deltaset(torch.utils.data.Dataset):
    """Dataset that poison original dataset by adding small perturbation (delta) to original dataset, and changing label to target label (t_lable)
       This Datasets acts like torch.utils.data.Dataset.
    
    Args: 
        dataset: dataset to poison
        delta: small perturbation to add on original image
        t_label: target label for modified image   
    """
    def __init__(self, dataset, delta, t_label):
        self.dataset = dataset
        self.delta = delta
        self.t_label = t_label

    def __getitem__(self, idx):
        (img, target) = self.dataset[idx]
        return (img + self.delta[idx], self.t_label)
    def __len__(self):
        return len(self.dataset)

class RandomTransform(torch.nn.Module):
    """ Differentiable Data Augmentation, intergrated resizing, shifting(ie, padding + cropping) and flipping. Input batch must be square images.

    Args:
        source_size(int): height of input images.
        target_size(int): height of output images.
        shift(int): maximum of allowd shifting size. 
        fliplr(bool): if flip horizonally
        flipud(bool): if flip vertically
        mode(string): the interpolation mode used in data augmentation. Default: bilinear.
        align: the align mode used in data augmentation. Default: True.
    
    For more details, refers to https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5
    """

    def __init__(self, source_size, target_size, shift=8, fliplr=True, flipud=False, mode='bilinear', align=True):
        """Args: source and target size."""
        super().__init__()
        self.grid = self.build_grid(source_size, target_size)
        self.delta = torch.linspace(0, 1, source_size)[shift]
        self.fliplr = fliplr
        self.flipud = flipud
        self.mode = mode
        self.align = True

    @staticmethod
    def build_grid(source_size, target_size):
        """https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5."""
        k = float(target_size) / float(source_size)
        direct = torch.linspace(-1, k, target_size).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)
        full = torch.cat([direct, direct.transpose(1, 0)], dim=2).unsqueeze(0)
        return full

    def random_crop_grid(self, x, randgen=None):
        """https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5."""
        grid = self.grid.repeat(x.size(0), 1, 1, 1).clone().detach()
        grid = grid.to(device=x.device, dtype=x.dtype)
        if randgen is None:
            randgen = torch.rand(x.shape[0], 4, device=x.device, dtype=x.dtype)

        # Add random shifts by x
        x_shift = (randgen[:, 0] - 0.5) * 2 * self.delta
        grid[:, :, :, 0] = grid[:, :, :, 0] + x_shift.unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))
        # Add random shifts by y
        y_shift = (randgen[:, 1] - 0.5) * 2 * self.delta
        grid[:, :, :, 1] = grid[:, :, :, 1] + y_shift.unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))

        if self.fliplr:
            grid[randgen[:, 2] > 0.5, :, :, 0] *= -1
        if self.flipud:
            grid[randgen[:, 3] > 0.5, :, :, 1] *= -1
        return grid


    def forward(self, x, randgen=None):
        # Make a random shift grid for each batch
        grid_shifted = self.random_crop_grid(x, randgen)
        # Sample using grid sample
        return F.grid_sample(x, grid_shifted, align_corners=self.align, mode=self.mode)


def patch_source(trainset, patch, target_label, random_patch=True):
    "Add patch to images, and change label to target label, set random_path to True if patch in random localtion"
    source_delta = []
    for idx, (source_img, label) in enumerate(trainset):
        if random_patch:
            patch_x = random.randrange(0,source_img.shape[1] - patch.shape[1] + 1)
            patch_y = random.randrange(0,source_img.shape[2] - patch.shape[2] + 1)
        else:
            patch_x = source_img.shape[1] - patch.shape[1]
            patch_y = source_img.shape[2] - patch.shape[2]
        delta_slice = torch.zeros_like(source_img)
        diff_patch = patch - source_img[:, patch_x: patch_x + patch.shape[1], patch_y: patch_y + patch.shape[2]]
        delta_slice[:, patch_x: patch_x + patch.shape[1], patch_y: patch_y + patch.shape[2]] = diff_patch
        source_delta.append(delta_slice.cpu())
    trainset = Deltaset(trainset, source_delta, target_label)  
    return trainset

def select_poison_ids(model, trainset, target_class, poison_num, device):
    "select samples from target class with large gradients "
    model.eval()    
    grad_norms = []
    differentiable_params = [p for p in model.parameters() if p.requires_grad]
    for image, label in trainset:
        if label != target_class: # ignore non-target-class
            grad_norms.append(0)
            continue
        if len(image.shape)==3:
            image.unsqueeze_(0)
        if isinstance(label, int):
            label = torch.tensor(label)
        if len(label.shape)==0:
            label.unsqueeze_(0) 
        image, label = image.to(device), label.to(device)
        loss = F.cross_entropy(model(image),label)
        gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norms.append(grad_norm.sqrt().item())
    grad_norms = np.array(grad_norms)
    # print('len(grad_norms):',len(grad_norms))
    poison_ids = np.argsort(grad_norms)[-poison_num:]
    print("Select %d samples, first 10 samples' grads are"%poison_num, grad_norms[poison_ids[-10:]])
    return poison_ids

def initialize_poison_deltas(num_poison_deltas, input_shape, eps, device):
    "uniformly initialize perturbation that will add to selected target image"
    torch.manual_seed(0)
    poison_deltas = (torch.randn(num_poison_deltas, *input_shape)).to(device)
    poison_deltas = (poison_deltas * eps).to(device)
    poison_deltas = torch.clamp(poison_deltas, min=-eps, max=eps)
    poison_deltas.grad = torch.zeros_like(poison_deltas)
    return poison_deltas

def clip_deltas(poison_deltas, imgs, eps):
    "clip delta, to make sure perturbation is bounded by [-eps, eps] and perturbed image is bounded by [0,1] "
    poison_deltas.data = torch.clamp(poison_deltas, min=-eps, max=eps)
    poison_deltas.data = torch.clamp(poison_deltas, min=0-imgs, max=1-imgs)
    return poison_deltas

def get_gradient(model, train_loader, criterion, device):
    """Compute the gradient of criterion(model) w.r.t to given data."""
    model.eval()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        loss = criterion(model(images), labels)
        if batch_idx == 0:
            gradients = torch.autograd.grad(loss, model.parameters(), only_inputs=True)
        else:
            gradients = tuple(map(lambda i, j: i + j, gradients, torch.autograd.grad(loss, model.parameters(), only_inputs=True)))
    gradients = tuple(map(lambda i: i / len(train_loader.dataset), gradients))
    grad_norm = 0
    for grad_ in gradients:
        grad_norm += grad_.detach().pow(2).sum()
    grad_norm = grad_norm.sqrt()
    return gradients, grad_norm

def get_passenger_loss(poison_grad, target_grad, target_gnorm):
    """Compute the blind passenger loss term."""
    # default self.args.loss is 'similarity', self.args.repel is 0, self.args.normreg from the gradient matching repo
    passenger_loss = 0
    poison_norm = 0
    indices = torch.arange(len(target_grad))
    for i in indices:
        passenger_loss -= (target_grad[i] * poison_grad[i]).sum()
        poison_norm += poison_grad[i].pow(2).sum()

    passenger_loss = passenger_loss / target_gnorm  # this is a constant
    passenger_loss = 1 + passenger_loss / poison_norm.sqrt()
    return passenger_loss

def define_objective(inputs, labels):
    """Implement the closure here."""
    def closure(model, criterion, target_grad, target_gnorm):
        """This function will be evaluated on all GPUs."""  # noqa: D401
        # default self.args.centreg is 0, self.retain is False from the gradient matching repo
        outputs = model(inputs)
        poison_loss = criterion(outputs, labels)
        prediction = (outputs.data.argmax(dim=1) == labels).sum()
        poison_grad = torch.autograd.grad(poison_loss, model.parameters(), retain_graph=True, create_graph=True)
        # print(poison_grad[0].abs().sum().item(),poison_grad[1].abs().sum().item())
        passenger_loss = get_passenger_loss(poison_grad, target_grad, target_gnorm)
        passenger_loss.backward(retain_graph=False)
        return passenger_loss.detach(), prediction.detach()
    return closure

def batched_step(model, inputs, labels, poison_delta, poison_slices, criterion, target_grad, target_gnorm, augment):
    """Take a step toward minmizing the current target loss."""
    delta_slice = poison_delta[poison_slices]
    delta_slice.requires_grad_(True)
    poisoned_inputs = inputs.detach() + delta_slice
    closure = define_objective(augment(poisoned_inputs), labels)
    loss, prediction = closure(model, criterion, target_grad, target_gnorm)
    poison_delta.grad[poison_slices] = delta_slice.grad.detach()
    return loss.item(), prediction.item()

def generate_poisoned_trainset(trainset, poison_set, poison_deltas, y_target, poison_ids):
    poisoned_trainset = []

    for i in range(len(trainset)):
        if i not in poison_ids:
            poisoned_trainset.append(trainset[i])
    for psample, pdelta in zip(poison_set, poison_deltas.cpu()):
        poisoned_trainset.append((psample[0]+pdelta, y_target))
    return poisoned_trainset

def prepare_poisonset(model, trainset, target_class, poison_num, device):
    """Add poison_deltas to poisoned_samples"""
    print("selecting poisons...")
    poison_ids = select_poison_ids(model, trainset, target_class, poison_num, device)

    if poison_num <= len(poison_ids): # enough sample in poison set
        poison_set = [deepcopy(trainset[i]) for i in poison_ids]
    else: # not enough sample in poisonset, extend poisonset
        poison_set = []
        while poison_num >= len(poison_ids):
            poison_num-=len(poison_ids)
            poison_set.extend([deepcopy(trainset[i]) for i in poison_ids]) 
        poison_set.extend([deepcopy(trainset[i]) for i in poison_ids[-poison_num:]]) 
    return poison_set, poison_ids

def extend_source(source_set, source_num):
    "Extend source_set to #source_num samples, allowing more samples to poison"
    if source_num==0: # if source num is set to 0, means doesn't need extend source set
        return source_set
    else:
        new_source_set = source_set
        while len(new_source_set)<source_num:
            new_source_set.extend(source_set)
        return new_source_set[:source_num]

def prepare_dataset(source_num, trainset, testset, y_target, y_source, patch, random_patch):
    """ prepare benign datasets and source datasets and patched(poisoned) datasets"""
    if random_patch:
        print("Adding patch randomly...")
    else:
        print("Adding patch to bottom right...")
    base_source_trainset = [data for data in trainset if data[1]==y_source]
    source_testset = [data for data in testset if data[1]==y_source]
    source_trainset = extend_source(base_source_trainset, source_num)
    patch_source_trainset = patch_source(source_trainset, patch, y_target, random_patch)
    patch_source_testset = patch_source(source_testset, patch, y_target, random_patch)
    full_patch_testset = patch_source(testset, patch, y_target, random_patch)
    return source_trainset, source_testset, patch_source_trainset, patch_source_testset, full_patch_testset

class SleeperAgent(Base):
    """class for SleeperAgent backdoor training and testing.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.        
        patch (torch.Tensor): shape (C, H_patch, W_patch). In Sleeper Agent, poison samples mainly refers to patched sample.
        random_patch (bool): whether to patch in random location
        eps: (float) threshold of perturbation
        y_target: (int): target label
        y_source: (int): source label
        poisoned_rate: (float) poison rate,
        source_num: (int) number of source samples
        schedule (dict): Training or testing global schedule. Default: None.
     """
    def __init__(self, 
                 train_dataset, 
                 test_dataset, 
                 model, 
                 loss, 
                 patch, 
                 random_patch, 
                 eps, 
                 y_target, 
                 y_source, 
                 poisoned_rate, 
                 source_num=0, 
                 schedule=None, 
                 seed=0, 
                 deterministic=False):
        super(SleeperAgent, self).__init__(train_dataset, test_dataset, 
            model, loss, schedule, seed, deterministic)
        self.patch = patch
        self.random_patch = random_patch
        self.eps = eps
        self.y_target = y_target
        self.y_source = y_source
        self.poisoned_rate = poisoned_rate
        self.crafted=False
        self.source_num=source_num
    
    def get_poisoned_dataset(self):
        """ must call train to craft poisoned dataset before call this function """
        if self.crafted is False:
            raise ValueError("Poisoned trainset has not been crafted yet, please call SleeperAgent.train() first")
        else:
            return self.poisoned_train_dataset, self.poisoned_test_dataset

    def _train_model(self, model, log, trainset, testset, poison_sourceset, poison_testset, augment, device, schedule):
        "train model using given schedule and test with given datasets"
        epochs, lr, weight_decay, gamma, momentum, miletones, batch_size, num_workers = schedule['epochs'], schedule['lr'], schedule['weight_decay'], schedule['gamma'], schedule['momentum'], schedule['milestones'], schedule['batch_size'], schedule['num_workers'], 
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers, worker_init_fn=self._seed_worker)
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=miletones, gamma=gamma)
        
        for epoch in range(1,epochs+1,1):
            train_correct = 0
            train_loss = 0
            model.train()
            for i, (img, y) in enumerate(trainloader):
                with torch.no_grad():
                    img, y = augment(img.to(device)), y.to(device)
                opt.zero_grad()
                outputs = model(img)
                loss = self.loss(outputs, y)
                loss.backward()
                opt.step()
                train_loss += loss.item()*len(y)
                train_correct += (outputs.max(1)[1]==y).sum().item()
                
            scheduler.step()
            train_loss, train_acc = train_loss/len(trainset), train_correct*100./len(trainset)

            # test
            model.eval()
            with torch.no_grad():
                # on benign testset
                if testset is not None:
                    predict_digits, labels = self._test(testset, device, batch_size, num_workers, model)
                    test_acc = (predict_digits.max(1)[1]==labels).sum().item() * 100. / len(labels)
                else:
                    test_acc = 0
                # on poisoned testset (source class only)
                if poison_sourceset is not None:
                    predict_digits, labels = self._test(poison_sourceset, device, batch_size, num_workers, model)
                    source_asr = (predict_digits.max(1)[1]==labels).sum().item() * 100. / len(labels)
                else:
                    source_asr = 0
                
                # on poisoned testset (all classes)
                if poison_testset is not None:
                    predict_digits, labels = self._test(poison_testset, device, batch_size, num_workers, model)
                    full_asr = (predict_digits.max(1)[1]==labels).sum().item() * 100. / len(labels)
                else:
                    full_asr = 0

            msg =  "Epoch %d"%epoch + time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    "train_acc: %.2f, test_acc: %.2f, source_asr: %.2f, full_asr: %.2f\n"%(train_acc, test_acc, source_asr, full_asr)
            log(msg)
        
    def craft_poison_dataset(self, model, log, init_model, trainset, testset, craft_iters, retrain_iter_interval, y_target, y_source, poison_num, augment, batch_size, num_workers, retrain_schedule, eps, device, test_schedule):
        """ craft poison dataset """
        # prepare dataset and patched dataset
        _, source_testset, patch_source_trainset, patch_source_testset, full_patch_testset = \
            prepare_dataset(self.source_num, trainset, testset, y_target, y_source, self.patch, self.random_patch)
        print(patch_source_trainset[0][0].mean().item())
        # prepare poison dataset
        poison_set, poison_ids = prepare_poisonset(model, trainset, y_target, poison_num, device)
        # Compute source_gradiet
        # round up batch size
        source_batch_size = len(patch_source_trainset) // ceil(len(patch_source_trainset) / batch_size)
        source_grad_loader = torch.utils.data.DataLoader(patch_source_trainset, batch_size=source_batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
                                                    
        source_grad, source_grad_norm = get_gradient(model=model,
                                                    train_loader=source_grad_loader,
                                                    criterion=nn.CrossEntropyLoss(reduction='sum'),
                                                    device=device)
        print("Source grad norm is", source_grad_norm.item())
        
        # Initialize poison deltas
        poison_deltas = initialize_poison_deltas(poison_num, trainset[0][0].shape, eps, device)
        
        print(f"len(patch_source_trainset):{len(patch_source_trainset)}, y_source:{y_source}, y_target:{y_target}")
        
        att_optimizer = torch.optim.Adam([poison_deltas], lr=0.025*batch_size/128, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[craft_iters // 2.667,
                                                                                    craft_iters // 1.6,
                                                                                    craft_iters // 1.142],)
        dataloader = torch.utils.data.DataLoader(poison_set, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=False)
        
        # Craft poison samples
        for t in range(1, craft_iters+1):
            # print("===========iter %d==========="%t)   
            base = 0
            target_losses,  poison_correct = 0., 0.
            poison_imgs = []
            model.eval()
            for imgs, targets in dataloader:
                imgs, targets = imgs.to(device), targets.to(device)
                # align gradients with source_gradient
                loss, prediction = batched_step(model=model, 
                                                inputs=imgs, 
                                                labels=targets, 
                                                poison_delta=poison_deltas,
                                                poison_slices=list(range(base, base+len(imgs))),
                                                criterion=nn.CrossEntropyLoss(), 
                                                target_grad=source_grad, 
                                                target_gnorm=source_grad_norm, 
                                                augment=augment)
                target_losses += loss
                poison_correct += prediction
                base += len(imgs)
                poison_imgs.append(imgs)

            poison_deltas.grad.sign_()
            att_optimizer.step()
            scheduler.step()
            att_optimizer.zero_grad()
            with torch.no_grad():
                # Projection Step
                poison_imgs = torch.cat(poison_imgs)              
                poison_deltas.data = torch.clamp(poison_deltas, min=-eps, max=eps)
                poison_deltas.data = torch.max(torch.min(poison_deltas,1-poison_imgs), -poison_imgs)
                
            
            target_losses = target_losses / (len(dataloader))
            poison_acc = poison_correct / len(dataloader.dataset)
            log("-----craft_iter: %d target_loss: %.3f benign acc of poisoned samples %.3f\n"%(t, target_losses, poison_acc*100.))
            
            # Retrain modeld
            if t % retrain_iter_interval == 0 and t!= craft_iters: 
                temp_poison_trainset = generate_poisoned_trainset(trainset, poison_set, poison_deltas, y_target, poison_ids)
                model = init_model(model)
                log("****retraining******\n")
                self._train_model(model=model, 
                    log=log, 
                    trainset=temp_poison_trainset, 
                    testset=testset, 
                    poison_sourceset=patch_source_testset, 
                    poison_testset=full_patch_testset, 
                    augment=augment, 
                    device=device, 
                    schedule=retrain_schedule)
                log("****retrain complete******\n")
                
                # Compute source gradient when retraining model 
                source_grad, source_grad_norm = get_gradient(model=model,
                                                    train_loader=source_grad_loader,
                                                    criterion=nn.CrossEntropyLoss(reduction='sum'),
                                                    device=device)
                print("Source grad norm is", source_grad_norm.item())
                # test on benign testset(source class only)
                predict_digits, labels = self._test(source_testset, device, batch_size, num_workers, model)
                source_test_acc = (predict_digits.max(1)[1]==labels).sum().item() * 100. / len(labels)
                
                # test on poisoned testset(source class only)
                predict_digits, labels = self._test(patch_source_testset, device, batch_size, num_workers, model)
                source_test_asr = (predict_digits.max(1)[1]==labels).sum().item() * 100. / len(labels)
                
                # test on benign testset(all classes)
                predict_digits, labels = self._test(self.test_dataset, device, batch_size, num_workers, model)
                test_acc = (predict_digits.max(1)[1]==labels).sum().item() * 100. / len(labels)

                # test on poisoned testset(all classes)
                predict_digits, labels = self._test(full_patch_testset, device, batch_size, num_workers, model)
                test_asr = (predict_digits.max(1)[1]==labels).sum().item() * 100. / len(labels)

                msg =  "Iter %d"%t + time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    "source_acc: %.2f, source_asr: %.2f, test_acc: %.2f, test_asr: %.2f\n"%(source_test_acc, source_test_asr, test_acc, test_asr)
                log(msg)
        poison_trainset = generate_poisoned_trainset(trainset, poison_set, poison_deltas, y_target, poison_ids)
        return poison_trainset, patch_source_testset, full_patch_testset

    def train(self, init_model, schedule=None):
        """first pretrain/load a mdoel, then use to craft poisoned dataset under the priciple of gradient alignment, then used the poisoned dataset to train a new model and use the poisoned new model to craft better poison dataset"""
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        if 'pretrain' in self.current_schedule and os.path.exists(self.current_schedule['pretrain']):
            self.model.load_state_dict(torch.load(self.current_schedule['pretrain']), strict=False)

        # Select Device
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
        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))
        
        self.model = self.model.to(device)
        self.model.train()

        h = self.train_dataset[0][0].shape[1]
        augment = RandomTransform(source_size=h, target_size=h, shift=h//4)

              
        if self.current_schedule['benign_training'] is True:
            self.current_schedule['milestones']=self.current_schedule['schedule']  
            self._train_model(self.model, log, self.train_dataset, self.test_dataset, None, None ,augment, device, self.current_schedule)
        elif self.current_schedule['benign_training'] is False:
            _, _, _, patch_source_testset, patch_testset = prepare_dataset(self.source_num, self.train_dataset, self.test_dataset, self.y_target, self.y_source, self.patch, self.random_patch)
            log("******pretraining*********\n")
            if ('pretrain' not in self.current_schedule) or ('pretrain' in self.current_schedule and not os.path.exists(self.current_schedule['pretrain'])):
                self._train_model(model=self.model, 
                                log=log, 
                                trainset=self.train_dataset, 
                                testset=self.test_dataset, 
                                poison_sourceset=patch_source_testset, 
                                poison_testset=patch_testset, 
                                augment=augment, 
                                device=device, 
                                schedule=self.current_schedule['pretrain_schedule'])

                if 'pretrain' in self.current_schedule:
                    torch.save(self.model.state_dict(), self.current_schedule['pretrain'])
                    
            log("******pretrain complete*********\n")

            # craft poison dataset 
            self.poisoned_train_dataset, self.poisoned_source_dataset, self.poisoned_test_dataset = \
                self.craft_poison_dataset(model=self.model, 
                log=log,
                init_model=init_model, 
                trainset=self.train_dataset, 
                testset=self.test_dataset, 
                craft_iters=self.current_schedule['craft_iters'], 
                retrain_iter_interval=self.current_schedule['retrain_iter_interval'],
                y_target=self.y_target,
                y_source=self.y_source,
                poison_num=int(self.poisoned_rate*len(self.train_dataset)),
                batch_size=self.current_schedule['batch_size'],
                num_workers=self.current_schedule['num_workers'],
                retrain_schedule=self.current_schedule['retrain_schedule'],
                eps=self.eps,
                device=device,
                augment=augment,
                test_schedule=schedule)
            self.crafted = True
            log("******poisoning*******\n")
            self.current_schedule['milestones']=self.current_schedule['schedule']  
            self.model = init_model(self.model)
            self._train_model(model=self.model, 
                log=log, 
                trainset=self.poisoned_train_dataset, 
                testset=self.test_dataset, 
                poison_sourceset=self.poisoned_source_dataset, 
                poison_testset=self.poisoned_test_dataset,
                augment=augment, 
                device=device, 
                schedule=self.current_schedule)
            log("******poisoning complete*******\n")
        else:
            raise AttributeError("self.current_schedule['benign_training'] should be True or False.")
            
        self.model.eval()
        self.model = self.model.cpu()
        ckpt_model_filename = "ckpt_epoch_" + str(self.current_schedule['epochs']) + ".pth"
        ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
        torch.save(self.model.state_dict(), ckpt_model_path)
             
