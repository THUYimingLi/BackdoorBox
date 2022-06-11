'''
This is the implement of TUAP [1].

[1] Clean-Label Backdoor Attacks on Video Recognition Models. CVPR, 2020.
'''

import copy
import random
from tqdm import tqdm
import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
import torchvision.transforms as transforms
import numpy as np
import copy
from torch.autograd.gradcheck import zero_gradients
import torchvision

import torch
import random
from torch.autograd import Variable

from .base import *


class AddTrigger:
    # blend 
    def __init__(self):
        pass

    def add_trigger(self, img):
        """Add watermarked trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        return (img.type(torch.float) + self.res).type(torch.uint8)

try:
    import accimage
except ImportError:
    accimage = None


def is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def pil_to_tensor(pic):
    """Convert a ``PIL Image`` to a tensor of the same type.
    This function does not support torchscript.

    See :class:`~torchvision.transforms.PILToTensor` for more details.

    Args:
        pic (PIL Image): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    # print("print pil_to_tensor")

    if not is_pil_image(pic):
        raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

    if accimage is not None and isinstance(pic, accimage.Image):
        # accimage format is always uint8 internally, so always return uint8 here
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.uint8)
        pic.copyto(nppic)
        return torch.as_tensor(nppic)

    # handle PIL Image
    img = torch.as_tensor(np.asarray(pic))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1))
    return img


class AddDatasetFolderTrigger(AddTrigger):
    """Add watermarked trigger to DatasetFolder images.

    Args:
        pattern (torch.Tensor): shape (C, H, W) or (H, W).
        weight (torch.Tensor): shape (C, H, W) or (H, W).
    """

    def __init__(self, pattern, mask):
        super(AddDatasetFolderTrigger, self).__init__()
        self.pattern = pattern * torch.tensor(255)  # the range of pattern lies in [-1,1]
        self.mask = mask
        self.res = self.mask * self.pattern

    def __call__(self, img):
        """Get the poisoned image.

        Args:
            img (PIL.Image.Image | numpy.ndarray | torch.Tensor): If img is numpy.ndarray or torch.Tensor, the shape should be (H, W, C) or (H, W).

        Returns:
            torch.Tensor: The poisoned image.
        """

        def add_trigger(img):
            if img.dim() == 2:
                img = img.unsqueeze(0)
                img = self.add_trigger(img)
                img = img.squeeze()
            else:
                img = self.add_trigger(img)
            return img

        if type(img) == PIL.Image.Image:
            img = pil_to_tensor(img)
            img = add_trigger(img)
            # 1 x H x W
            if img.size(0) == 1:
                img = Image.fromarray(img.squeeze().numpy(), mode='L')
            # 3 x H x W
            elif img.size(0) == 3:
                img = Image.fromarray(img.permute(1, 2, 0).numpy())
            else:
                raise ValueError("Unsupportable image shape.")
            return img
        elif type(img) == np.ndarray:
            # H x W
            if len(img.shape) == 2:
                img = torch.from_numpy(img)
                img = add_trigger(img)
                img = img.numpy()
            # H x W x C
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)
                img = add_trigger(img)
                img = img.permute(1, 2, 0).numpy()
            return img
        elif type(img) == torch.Tensor:
            # H x W
            if img.dim() == 2:
                img = add_trigger(img)
            # H x W x C
            else:
                img = img.permute(2, 0, 1)
                img = add_trigger(img)
                img = img.permute(1, 2, 0)
            return img
        else:
            raise TypeError('img should be PIL.Image.Image or numpy.ndarray or torch.Tensor. Got {}'.format(type(img)))


class AddMNISTTrigger(AddTrigger):
    """Add watermarked trigger to MNIST image.

    Args:
        pattern (None | torch.Tensor): shape (1, 28, 28) or (28, 28).
        weight (None | torch.Tensor): shape (1, 28, 28) or (28, 28).
    """

    def __init__(self, pattern, mask):
        super(AddMNISTTrigger, self).__init__()
        self.pattern = pattern * torch.tensor(255)  # the range of pattern lies in [-1,1]
        self.mask = mask
        self.res = self.mask * self.pattern

    def __call__(self, img):
        img = pil_to_tensor(img)

        img = self.add_trigger(img)
        img = img.squeeze()
        img = Image.fromarray(img.numpy(), mode='L')
        return img


class AddCIFAR10Trigger(AddTrigger):
    """Add watermarked trigger to CIFAR10 image.

    Args:
        pattern (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
        weight (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
    """
    #
    def __init__(self, pattern, mask):
        super(AddCIFAR10Trigger, self).__init__()
        self.pattern = pattern * torch.tensor(255)   # the range of pattern lies in [-1,1]
        self.mask = mask
        self.res = self.mask * self.pattern

    def __call__(self, img):
        img = pil_to_tensor(img)
        img = self.add_trigger(img)
        img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img


class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target


class PoisonedDatasetFolder(DatasetFolder):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 is_train_set,
                 poisoned_rate,
                 pattern,
                 mask,
                 poisoned_transform_index,
                 poisoned_target_transform_index,
                 seed):
        super(PoisonedDatasetFolder, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None)
        random.seed(seed)
     
        self.is_train_set = is_train_set
        self.benign_dataset = benign_dataset
        self.y_target_ori = int(y_target)
        self.poisoned_rate = poisoned_rate
        # indicating poisoned index
        if self.is_train_set:      # for training 
            self.poisoned_set = self.gen_poisoned_index()
        else:                  # for testing
            total_num = len(self.benign_dataset)
            poisoned_num = int(total_num * poisoned_rate)
            assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
            tmp_list = list(range(total_num))
            random.shuffle(tmp_list)
            self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(pattern, mask))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def gen_poisoned_index(self):
        target_label_list = []
        for (index, t) in enumerate(self.benign_dataset.targets):
            if t == self.y_target_ori:
                target_label_list.append(index)

        num_target_sample = len(target_label_list)
        np.random.shuffle(np.array(target_label_list))

        poisoned_num = int(num_target_sample * self.poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        poisoned_set = frozenset(target_label_list[:poisoned_num])
        return poisoned_set

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if index in self.poisoned_set:
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target


class PoisonedMNIST(MNIST):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 is_train_set,
                 poisoned_rate,
                 pattern,
                 mask,
                 poisoned_transform_index,
                 poisoned_target_transform_index,
                 seed):
        super(PoisonedMNIST, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)

        # self.y_target_ori: original y_target
        self.is_train_set = is_train_set
        self.benign_dataset = benign_dataset
        self.y_target_ori = int(y_target)
        self.poisoned_rate = poisoned_rate
        # indicating poisoned index
        if self.is_train_set:      # for training 
            self.poisoned_set = self.gen_poisoned_index()
        else:                  # for testing
            total_num = len(self.benign_dataset)
            poisoned_num = int(total_num * poisoned_rate)
            assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
            tmp_list = list(range(total_num))
            random.shuffle(tmp_list)
            self.poisoned_set = frozenset(tmp_list[:poisoned_num])


        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)

        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddMNISTTrigger(pattern, mask))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def gen_poisoned_index(self):
        target_label_list = []
        for (index, t) in enumerate(self.benign_dataset.targets):
            if t == self.y_target_ori:
                target_label_list.append(index)

        num_target_sample = len(target_label_list)
        print("num_target_sample", num_target_sample)
        np.random.shuffle(np.array(target_label_list))

        poisoned_num = int(num_target_sample * self.poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        poisoned_set = frozenset(target_label_list[:poisoned_num])
        return poisoned_set

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


class PoisonedCIFAR10(CIFAR10):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 is_train_set,
                 poisoned_rate,
                 pattern,
                 mask,
                 poisoned_transform_index,
                 poisoned_target_transform_index,
                 seed):
        super(PoisonedCIFAR10, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)
        # whether or not
        # random.seed(seed)

        # To Do: clean-label
        self.is_train_set = is_train_set
        self.benign_dataset = benign_dataset
        self.y_target_ori = int(y_target)
        self.poisoned_rate = poisoned_rate
        # indicating poisoned index
        if self.is_train_set:      # for training 
            self.poisoned_set = self.gen_poisoned_index()
        else:                  # for testing
            total_num = len(self.benign_dataset)
            poisoned_num = int(total_num * poisoned_rate)
            assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
            tmp_list = list(range(total_num))
            random.shuffle(tmp_list)
            self.poisoned_set = frozenset(tmp_list[:poisoned_num])


        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddCIFAR10Trigger(pattern, mask))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def gen_poisoned_index(self):
        target_label_list = []
        for (index, t) in enumerate(self.benign_dataset.targets):
            if t == self.y_target_ori:
                target_label_list.append(index)

        num_target_sample = len(target_label_list)
        print("num_target_sample", num_target_sample)
        np.random.shuffle(np.array(target_label_list))

        poisoned_num = int(num_target_sample * self.poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        poisoned_set = frozenset(target_label_list[:poisoned_num])
        return poisoned_set

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


def CreatePoisonedDataset(benign_dataset, y_target, poisoned_rate, pattern, mask, poisoned_transform_index,
                          poisoned_target_transform_index, seed, is_train_set=True):
    class_name = type(benign_dataset)
    if class_name == DatasetFolder:
        return PoisonedDatasetFolder(benign_dataset, y_target,is_train_set, poisoned_rate, pattern, mask,
                                     poisoned_transform_index, poisoned_target_transform_index, seed)
    elif class_name == MNIST:
        return PoisonedMNIST(benign_dataset, y_target,is_train_set, poisoned_rate, pattern, mask,
                             poisoned_transform_index, poisoned_target_transform_index, seed)
    elif class_name == CIFAR10:
        return PoisonedCIFAR10(benign_dataset, y_target,is_train_set, poisoned_rate, pattern, mask,
                               poisoned_transform_index, poisoned_target_transform_index, seed)
    else:
        raise NotImplementedError

class UAP:
    def __init__(self, model,  train_dataset, test_dataset, class_name, use_cuda, target_class=0,
                 mask=None, p_samples=0.01, loader=None):
        """
           This class is used to generating UAP given a benign dataset and a benign model.
           :param model: Benign model.
           :param train_dataset : Benign training dataset.
           :param test_dataset: Benign testing dataset.
           :param class_name: The class name of the benign dataset ("MNIST", "CIFAR10", "DatasetFolder")
           :param use_cuda: Whether or not use cuda
           :param target_class: N-to-1 attack target label.
           :param mask: Mask for generating perturbation "v"
           :param p_samples: ratio of samples used for generating UAP
           :param loader: Used to load image when class_name==DatasetFolder
        """
        #self.datasets_root_dir = datasets_root_dir
        self.model = model
        self.use_cuda = use_cuda
        self.mask = mask
        self.target_class = target_class
        self.trainset =  train_dataset
        print("trainset",len(self.trainset))
        self.testset = test_dataset
        print("testset",len(self.testset))
        self.p_samples = p_samples
        assert 0 < self.p_samples <=1,"The ratio can should be in range (0,1]"
        
        self.num_samples = int(self.p_samples * len(self.trainset))+1
        print("self.num_samples",self.num_samples)

        if loader is None:
            self.loader = self.default_loader
        else:
            self.loader = loader

       
        if class_name == DatasetFolder:
            self.testloader = torch.utils.data.DataLoader(dataset=self.testset, batch_size=200, pin_memory=True,
                                                          num_workers=0, shuffle=False)

        elif class_name == MNIST:
            print("self.testloader",len(self.trainset),len(self.testset))
            self.testloader = torch.utils.data.DataLoader(dataset=self.testset, batch_size=200, pin_memory=True,
                                                          num_workers=0, shuffle=False)
            print("self.testloader",self.testloader)
         
        elif class_name == CIFAR10:
            self.testloader = torch.utils.data.DataLoader(dataset=self.testset, batch_size=200, pin_memory=True,
                                                          num_workers=0, shuffle=False)

        else:
            raise NotImplementedError
            

    def default_loader(self, img):
        return Image.open(img).convert('RGB')

    def deepfool_target(self, image, num_classes, overshoot, max_iter):
        """
           :param image: Image of size CxHxW
           :param num_classes: number of classes (limits the number of classes to test against, by default = 10)
           :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
           :param max_iter: maximum number of iterations for deepfool (default = 50)
           :return: minimal perturbation that fools the classifier, number of iterations that it required,
           new estimated_label and perturbed image
        """
        f_image = self.model(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()  # [10,]
        I = f_image.argsort()[::-1]

        I = I[0:num_classes]
        clean_label = I[0]

        input_shape = image.cpu().numpy().shape
        pert_image = copy.deepcopy(image)

        r_tot = np.zeros(input_shape)

        loop_i = 0
        wrapped = tqdm(total=max_iter)

        x = Variable(pert_image[None, :], requires_grad=True)
        fs = self.model(x)
        k_i = clean_label
        while k_i != self.target_class and loop_i < max_iter:
            fs[0, self.target_class].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy()

            zero_gradients(x)

            fs[0, clean_label].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            # add mask
            w_k = (grad_orig - cur_grad) * self.mask.data.numpy()
            f_k = (fs[0, self.target_class] - fs[0, clean_label]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            pert = pert_k
            # update description and progress bar

            wrapped.set_description(f"perturbation: {pert:.5f}")
            wrapped.update(1)
            w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = (pert+1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)

            if self.use_cuda:
                pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
            else:
                pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

            x = Variable(pert_image, requires_grad=True)
            fs = self.model(x)
            k_i = np.argmax(fs.data.cpu().numpy().flatten())

            loop_i += 1

        return (1+overshoot)*r_tot, loop_i, k_i, pert_image


    def proj_lp(self, perturbation, epsilon, p_norm):
        """
            Project on the lp ball centered at 0 and of radius epsilon, SUPPORTS only p = 2 and p = Inf for now
            :param perturbation: Perturbation of size CxHxW
            :param epsilon: Controls the l_p magnitude of the perturbation (default = 10/255.0)
            :param p_norm: Norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
            :return:
        """

        if p_norm == 2:
            perturbation = perturbation * min(1, epsilon/np.linalg.norm(perturbation.flatten(1)))
        elif p_norm == np.inf:
            perturbation = np.sign(perturbation) * np.minimum(abs(perturbation), epsilon)
        else:
             raise ValueError('Values of p different from 2 and Inf are currently not supported...')
        return perturbation

    def universal_perturbation(self, delta=0.2, max_iter_uni=40, epsilon=10.0/255,
                               p_norm=np.inf, num_classes=10, overshoot=0.02, max_iter_df=50):
        """
        :param delta: controls the desired fooling rate (default = 80% fooling rate)
        :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
        :param epsilon: controls the l_p magnitude of the perturbation (default = 10/255.0)
        :param p_norm: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
        :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
        :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
        :param max_iter_df: maximum number of iterations for deepfool (default = 10)
        :return: the universal perturbation.
        """

        if self.use_cuda:
            self.model.cuda()
        device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model.eval()

        v = torch.tensor(0)
        fooling_rate = 0.0
        # epsilon = epsilon/255.0
        # random.seed(seed)
        total_num = len(self.trainset)
        # Using #num_images data for generating UAP
        num_images = min(total_num, self.num_samples)
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        order = np.array(tmp_list[:num_images])
        
        itr = 0
        while fooling_rate < 1-delta and itr < max_iter_uni:
            # Shuffle the self.trainset
            np.random.shuffle(order)
            print('Starting pass number ', itr)
            # Go through the data set and compute the perturbation increments sequentially
            for k in order:
                cur_img, _ = self.trainset[k]  # (3,32,32)
                perturb_img = cur_img + v

                cur_img, perturb_img = cur_img.to(device), perturb_img.to(device)
                if int(self.model(cur_img.unsqueeze(0)).max(1)[1]) == \
                        int(self.model((perturb_img.unsqueeze(0)).type(torch.cuda.FloatTensor)).max(1)[1]):

                    print('>> k = ', np.where(k == order)[0][0], ', pass #', itr)

                    # Compute adversarial perturbation
                    dr, iterr, _, _ = self.deepfool_target(perturb_img, num_classes=num_classes,
                                                           overshoot=overshoot, max_iter=max_iter_df)


                    dr = torch.from_numpy(dr).squeeze(0).type(torch.float32)
                    # Make sure it converged...
                    if iterr < max_iter_df-1:
                        v = v + dr
                        # Project on l_p ball
                        v = self.proj_lp(v, epsilon, p_norm)

            itr = itr + 1
            # Perturb the self.testset with computed perturbation and test the fooling rate on the testset
            with torch.no_grad():
                print("Testing")
                test_num_images = 0
                est_labels_orig = torch.tensor(np.zeros(0, dtype=np.int64))
                est_labels_pert = torch.tensor(np.zeros(0, dtype=np.int64))
                for batch_idx, (inputs, _) in enumerate(self.testloader):
                    test_num_images += inputs.shape[0]
                    inputs_pert = inputs + v
                    inputs = inputs.to(device)
                    outputs = self.model(inputs)
                    inputs_pert = inputs_pert.to(device)
                    outputs_perturb = self.model(inputs_pert)

                    _, predicted = outputs.max(1)
                    _, predicted_pert = outputs_perturb.max(1)
                    est_labels_orig = torch.cat((est_labels_orig, predicted.cpu()))
                    est_labels_pert = torch.cat((est_labels_pert, predicted_pert.cpu()))
                torch.cuda.empty_cache()

                fooling_rate = float(torch.sum(est_labels_orig != est_labels_pert)) / float(test_num_images)

                # Compute the fooling rate
                print('FOOLING RATE = ', fooling_rate)
                # np.save('target_mask_16_50000/targetinner_v' + str(iterr) + '_' + str(round(fooling_rate, 4)), v)
        # np.save('target-v' + str(itr) + '_' + str(round(fooling_rate, 4)), v)
        print('Final FOOLING RATE = ', fooling_rate)
        return v




class TUAP(Base):
    """Construct poisoned datasets with TUAP method.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.

        benign_model (torch.nn.Module): Benign model to generate UAP
        y_target (int): N-to-1 attack target label.
        poisoned_rate (float): Ratio of poisoned samples.
        epsilon (float): The l_p magnitude of the perturbation.  Default: 10.0/255 for CIFAR10 and 76.0/255 for MNIST.
        delta (float): The desired fooling rate. Default: 0.2  (80% fooling rate)
        max_iter_uni (int):  Optional other termination criterion (maximum number of iteration). Default: np.inf
        p_norm (int): Norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!). Default: np.inf
        num_classes (int): Number of classes (limits the number of classes to test against). Default: 10.
        overshoot (float): A termination criterion to prevent vanishing updates. Default: 0.02.
        max_iter_df (int): Maximum number of iterations for deepfool. Default: 50.
        p_samples (float): ratio of samples to be used to generate UAP. Default: 0.01
        mask (None | torch.Tensor): Mask for UAP, shape (C, H, W) or (H, W).
        pattern (None | torch.Tensor): Trigger pattern, shape (C, H, W) or (H, W).

        poisoned_transform_train_index (int): The position index that poisoned transform will be inserted in train dataset. Default: 0.
        poisoned_transform_test_index (int): The position index that poisoned transform will be inserted in test dataset. Default: 0.
        poisoned_target_transform_index (int): The position that poisoned target transform will be inserted. Default: 0.
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.

    """
    # pattern here or later generated

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 model,
                 loss,

                 benign_model,
                 y_target,
                 poisoned_rate,
                 epsilon,
                 delta=0.2,
                 max_iter_uni=np.inf,
                 p_norm=np.inf,
                 num_classes=10,
                 overshoot=0.02,
                 max_iter_df=50,
                 p_samples=0.01,
                 mask=None,  # can be none
                 pattern=None,

                 poisoned_transform_train_index=0,
                 poisoned_transform_test_index=0,
                 poisoned_target_transform_index=0,
                 schedule=None,
                 seed=0,
                 deterministic=False):


        super(TUAP, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)

        class_name = type(train_dataset)
        self.y_target = int(y_target)

        if mask is None:
            assert class_name != DatasetFolder, "Self-defined dataset should define mask"
            if class_name == MNIST:
                self.mask = torch.ones((1, 28, 28), dtype=torch.float32)
            elif class_name == CIFAR10:
                self.mask = torch.ones((3, 32, 32), dtype=torch.float32)
        else:
            if isinstance(mask, np.ndarray):
                mask = torch.tensor(mask, dtype=torch.float32)
            self.mask = mask  # should be consistent with pattern,
        # the values of the pixels where pattern lies should be 1, and otherwise 0

        if pattern is None:

            use_cuda = torch.cuda.is_available()
            UAP_ins = UAP(benign_model, train_dataset,test_dataset,class_name, use_cuda, self.y_target, self.mask, p_samples)
            self.pattern = UAP_ins.universal_perturbation(delta=delta, max_iter_uni=max_iter_uni, epsilon=epsilon, p_norm=p_norm, num_classes=num_classes,
                                                          overshoot=overshoot, max_iter_df=max_iter_df)
        else:
            if isinstance(pattern, np.ndarray):
                pattern = torch.tensor(pattern, dtype=torch.float32)
            self.pattern = pattern

        self.poisoned_train_dataset = CreatePoisonedDataset(
            train_dataset,
            self.y_target,
            poisoned_rate,
            self.pattern,
            self.mask,
            poisoned_transform_train_index,
            poisoned_target_transform_index,
            seed,
            is_train_set=True)

        self.poisoned_test_dataset = CreatePoisonedDataset(
            test_dataset,
            self.y_target,
            1.0,
            self.pattern,
            self.mask,
            poisoned_transform_test_index,
            poisoned_target_transform_index,
            seed,
            is_train_set=False)
