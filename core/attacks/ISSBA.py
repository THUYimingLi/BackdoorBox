'''
This is the implement of invisible sample-specific backdoor attack (ISSBA) [1].

Reference:
[1] Invisible Backdoor Attack with Sample-Specific Triggers. ICCV, 2021.
'''


import collections
from itertools import repeat
import torch
from torch import nn
import torch.nn.functional as F
from operator import __add__
from .base import *
from collections import namedtuple
from torchvision import models as tv
import lpips
from torch.utils.data import DataLoader
import imageio
from torchvision import transforms
from torchvision.datasets import CIFAR10


class Normalize:
    """Normalization of images.

    Args:
        dataset_name (str): the name of the dataset to be normalized.
        expected_values (float): the normalization expected values.
        variance (float): the normalization variance.
    """
    def __init__(self, dataset_name, expected_values, variance):
        if dataset_name == "cifar10" or dataset_name == "gtsrb":
            self.n_channels = 3
        elif dataset_name == "mnist":
            self.n_channels = 1
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class GetPoisonedDataset(CIFAR10):
    """Construct a dataset.

    Args:
        data_list (list): the list of data.
        labels (list): the list of label.
    """
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img = torch.tensor(self.data[index])
        label = torch.tensor(self.targets[index])
        return img, label


def _ntuple(n):
    """Copy from PyTorch since internal function is not importable

    See ``nn/modules/utils.py:6``

    Args:
        n (int): Number of repetitions x.
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


class Conv2dSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword
    argument, this does not export to CoreML as of coremltools 5.1.0,
    so we need to implement the internal torch logic manually.

    Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/6

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            **kwargs):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs)

        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0]*len(kernel_size_)

        # Follow the logic from ``nn/modules/conv.py:_ConvNd``
        for d, k, i in zip(dilation_, kernel_size_,
                                range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = (
                    total_padding - left_pad)

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)


class StegaStampEncoder(nn.Module):
    """The image steganography encoder to implant the backdoor trigger.

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        secret_size (int): Size of the steganography secret.
        height (int): Height of the input image.
        width (int): Width of the input image.
        in_channel (int): Channel of the input image.
    """
    def __init__(self, secret_size=20, height=32, width=32, in_channel=3):
        super(StegaStampEncoder, self).__init__()
        self.height, self.width, self.in_channel = height, width, in_channel

        self.secret_dense = nn.Sequential(nn.Linear(in_features=secret_size, out_features=height * width * in_channel), nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(Conv2dSame(in_channels=in_channel*2, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=256, kernel_size=3, stride=2), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up6 = nn.Sequential(Conv2dSame(in_channels=256, out_channels=128, kernel_size=2), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(Conv2dSame(in_channels=256, out_channels=128, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up7 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=64, kernel_size=2), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=64, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up8 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=32, kernel_size=2), nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up9 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=32, kernel_size=2), nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(Conv2dSame(in_channels=64+in_channel*2, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))

        self.residual = nn.Sequential(Conv2dSame(in_channels=32, out_channels=in_channel, kernel_size=1))

    def forward(self, inputs):
        secret, image = inputs
        secret = secret - .5
        image = image - .5

        secret = self.secret_dense(secret)
        secret = secret.reshape((-1, self.in_channel, self.height, self.width))
        inputs = torch.cat([secret, image], axis=1)

        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        up6 = self.up6(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv5))
        merge6 = torch.cat([conv4,up6], axis=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv6))
        merge7 = torch.cat([conv3,up7], axis=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv7))
        merge8 = torch.cat([conv2,up8], axis=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv8))
        merge9 = torch.cat([conv1,up9,inputs], axis=1)

        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)

        return residual


class StegaStampDecoder(nn.Module):
    """The image steganography decoder to assist the training of the image steganography encoder.

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        secret_size (int): Size of the steganography secret.
        height (int): Height of the input image.
        width (int): Width of the input image.
        in_channel (int): Channel of the input image.
    """
    def __init__(self, secret_size, height, width, in_channel):
        super(StegaStampDecoder, self).__init__()
        self.height = height
        self.width = width
        self.in_channel = in_channel

        # Spatial transformer
        self.stn_params_former = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True),
        )

        self.stn_params_later = nn.Sequential(
            nn.Linear(in_features=128*(height//2//2//2)*(width//2//2//2), out_features=128), nn.ReLU(inplace=True)
        )

        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = torch.FloatTensor(initial.astype('float32').flatten())

        self.W_fc1 = nn.Parameter(torch.zeros([128, 6]))
        self.b_fc1 = nn.Parameter(initial)

        self.decoder = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=32, kernel_size=3), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=64, kernel_size=3), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=128, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True),
        )

        self.decoder_later = nn.Sequential(
            nn.Linear(in_features=128*(height//2//2//2//2//2)*(width//2//2//2//2//2), out_features=512), nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=secret_size)
        )

    def forward(self, image):
        image = image - .5
        stn_params = self.stn_params_former(image)
        stn_params = stn_params.view(stn_params.size(0), -1)
        stn_params = self.stn_params_later(stn_params)

        x = torch.mm(stn_params, self.W_fc1) + self.b_fc1
        x = x.view(-1, 2, 3) # change it to the 2x3 matrix
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self.in_channel, self.height, self.width)))
        transformed_image = F.grid_sample(image, affine_grid_points)

        secret = self.decoder(transformed_image)
        secret = secret.view(secret.size(0), -1)
        secret = self.decoder_later(secret)
        return secret


class Discriminator(nn.Module):
    """The image steganography discriminator to assist the training of the image steganography encoder and decoder.

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        in_channel (int): Channel of the input image.
    """
    def __init__(self, in_channel=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=8, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=8, out_channels=16, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=16, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=1, kernel_size=3), nn.ReLU(inplace=True),
        )

    def forward(self, image):
        x = image - .5
        x = self.model(x)
        output = torch.mean(x)
        return output


class MNISTStegaStampEncoder(nn.Module):
    """The image steganography encoder to implant the backdoor trigger (Customized for MNIST dataset).

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        secret_size (int): Size of the steganography secret.
        height (int): Height of the input image.
        width (int): Width of the input image.
        in_channel (int): Channel of the input image.
    """
    def __init__(self, secret_size=20, height=28, width=28, in_channel=1):
        super(MNISTStegaStampEncoder, self).__init__()
        self.height, self.width, self.in_channel = height, width, in_channel

        self.secret_dense = nn.Sequential(nn.Linear(in_features=secret_size, out_features=height * width * in_channel), nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(Conv2dSame(in_channels=in_channel*2, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up5 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=64, kernel_size=2), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=64, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up6 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=32, kernel_size=2), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up7 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=32, kernel_size=2), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(Conv2dSame(in_channels=66, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))

        self.residual = nn.Sequential(Conv2dSame(in_channels=32, out_channels=1, kernel_size=1))

    def forward(self, inputs):
        secret, image = inputs
        secret = secret - .5
        image = image - .5

        secret = self.secret_dense(secret)
        secret = secret.reshape((-1, self.in_channel, self.height, self.width))

        inputs = torch.cat([secret, image], axis=1)

        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        up5 = self.up5(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv4))
        merge5 = torch.cat([conv3,up5], axis=1)
        conv5 = self.conv5(merge5)

        up6 = self.up6(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv5))
        merge6 = torch.cat([conv2,up6], axis=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(conv6)
        merge7 = torch.cat([conv1,up7,inputs], axis=1)
        conv7 = self.conv7(merge7)

        residual = self.residual(conv7)

        return residual


class MNISTStegaStampDecoder(nn.Module):
    """The image steganography decoder to assist the training of the image steganography encoder (Customized for MNIST dataset).

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        secret_size (int): Size of the steganography secret.
        height (int): Height of the input image.
        width (int): Width of the input image.
        in_channel (int): Channel of the input image.
    """
    def __init__(self, secret_size, height, width, in_channel):
        super(MNISTStegaStampDecoder, self).__init__()
        self.height = height
        self.width = width
        self.in_channel = in_channel

        # Spatial transformer
        self.stn_params_former = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
        )

        self.stn_params_later = nn.Sequential(
            nn.Linear(in_features=64*(height//2//2)*(width//2//2), out_features=64), nn.ReLU(inplace=True)
        )

        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = torch.FloatTensor(initial.astype('float32').flatten())

        self.W_fc1 = nn.Parameter(torch.zeros([64, 6]))
        self.b_fc1 = nn.Parameter(initial)

        self.decoder = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=32, kernel_size=3), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=64, kernel_size=3), nn.ReLU(inplace=True),
        )

        self.decoder_later = nn.Sequential(
            nn.Linear(in_features=64*(height//2//2)*(width//2//2), out_features=256), nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=secret_size)
        )

    def forward(self, image):
        image = image - .5
        stn_params = self.stn_params_former(image)
        stn_params = stn_params.view(stn_params.size(0), -1)
        stn_params = self.stn_params_later(stn_params)

        x = torch.mm(stn_params, self.W_fc1) + self.b_fc1
        x = x.view(-1, 2, 3) # change it to the 2x3 matrix
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self.in_channel, self.height, self.width)))
        transformed_image = F.grid_sample(image, affine_grid_points)


        secret = self.decoder(transformed_image)
        secret = secret.view(secret.size(0), -1)
        secret = self.decoder_later(secret)

        return secret


class MNISTDiscriminator(nn.Module):
    """The image steganography discriminator to assist the training of the image steganography encoder and decoder (Customized for MNIST dataset).

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        in_channel (int): Channel of the input image.
    """
    def __init__(self, in_channel=1):
        super(MNISTDiscriminator, self).__init__()
        self.model = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=4, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=4, out_channels=8, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=8, out_channels=1, kernel_size=3), nn.ReLU(inplace=True),
        )

    def forward(self, image):
        x = image - .5
        x = self.model(x)
        output = torch.mean(x)
        return output



def get_secret_acc(secret_true, secret_pred):
    """The accurate for the steganography secret.

    Args:
        secret_true (torch.Tensor): Label of the steganography secret.
        secret_pred (torch.Tensor): Prediction of the steganography secret.
    """
    secret_pred = torch.round(torch.sigmoid(secret_pred))
    correct_pred = (secret_pred.shape[0] * secret_pred.shape[1]) - torch.count_nonzero(secret_pred - secret_true)
    bit_acc = torch.sum(correct_pred) / (secret_pred.shape[0] * secret_pred.shape[1])

    return bit_acc


class ProbTransform(torch.nn.Module):
    """The data augmentation transform by the probability.

    Args:
        f (nn.Module): the data augmentation transform operation.
        p (float): the probability of the data augmentation transform.
    """
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTensorTransform(torch.nn.Module):
    """The data augmentation transform.

    Args:
        dataset_name (str): the name of the dataset.
    """
    def __init__(self, dataset_name):
        super(PostTensorTransform, self).__init__()
        if dataset_name == 'mnist':
            input_height, input_width = 28, 28
        elif dataset_name == 'cifar10':
            input_height, input_width = 32, 32
        elif dataset_name == 'gtsrb':
            input_height, input_width = 32, 32
        self.random_crop = ProbTransform(transforms.RandomCrop((input_height, input_width), padding=5), p=0.8) # ProbTransform(A.RandomCrop((input_height, input_width), padding=5), p=0.8)
        self.random_rotation = ProbTransform(transforms.RandomRotation(10), p=0.5) # ProbTransform(A.RandomRotation(10), p=0.5)
        if dataset_name == "cifar10":
            self.random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5) # A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class ISSBA(Base):
    """Construct the backdoored model with ISSBA method.

    Args:
        dataset_name (str): the name of the dataset.
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        train_steg_set (types in support_list): Training dataset for the image steganography encoder and decoder.
        model (torch.nn.Module): Victim model.
        loss (torch.nn.Module): Loss.
        y_target (int): N-to-1 attack target label.
        poisoned_rate (float): Ratio of poisoned samples.
        secret_size (int): Size of the steganography secret.
        enc_height (int): Height of the input image into the image steganography encoder.
        enc_width (int): Width of the input image into the image steganography encoder.
        enc_in_channel (int): Channel of the input image into the image steganography encoder.
        enc_total_epoch (int): Training epoch of the image steganography encoder.
        enc_secret_only_epoch (int): The final epoch to train the image steganography encoder with only secret loss function.
        enc_use_dis (bool): Whether to use discriminator during the training of the image steganography encoder. Default: False.
        encoder (torch.nn.Module): The pretrained image steganography encoder. Default: None.
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """
    def __init__(self,
                 dataset_name,
                 train_dataset,
                 test_dataset,
                 train_steg_set,
                 model,
                 loss,
                 y_target,
                 poisoned_rate,
                 encoder_schedule,
                 encoder=None,
                 schedule=None,
                 seed=0,
                 deterministic=False,
                 ):
        super(ISSBA, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)
        self.dataset_name = dataset_name
        self.train_steg_set = train_steg_set
        self.encoder = encoder
        self.encoder_schedule = encoder_schedule

        total_num = len(train_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])
        self.poisoned_rate = poisoned_rate
        self.y_target = y_target
        self.train_poisoned_data, self.train_poisoned_label = [], []
        self.test_poisoned_data, self.test_poisoned_label = [], []

        if dataset_name == "cifar10":
            self.normalizer = Normalize(dataset_name, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif dataset_name == "mnist":
            self.normalizer = Normalize(dataset_name, [0.5], [0.5])
        elif dataset_name == "gtsrb":
            self.normalizer = None
        else:
            self.normalizer = None

    def get_model(self):
        return self.model

    def get_encoder(self):
        return self.encoder

    def get_poisoned_dataset(self):
        """
            Return the poisoned dataset.
        """
        if len(self.train_poisoned_data) == 0 and len(self.test_poisoned_data) == 0:
            return None, None
        elif len(self.train_poisoned_data) == 0 and len(self.test_poisoned_data) != 0:
            poisoned_test_dataset = GetPoisonedDataset(self.test_poisoned_data, self.test_poisoned_label)
            return None, poisoned_test_dataset
        elif len(self.train_poisoned_data) != 0 and len(self.test_poisoned_data) == 0:
            poisoned_train_dataset = GetPoisonedDataset(self.train_poisoned_data, self.train_poisoned_label)
            return poisoned_train_dataset, None
        else:
            poisoned_train_dataset = GetPoisonedDataset(self.train_poisoned_data, self.train_poisoned_label)
            poisoned_test_dataset = GetPoisonedDataset(self.test_poisoned_data, self.test_poisoned_label)
            return poisoned_train_dataset, poisoned_test_dataset

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch in self.current_schedule['schedule']:
            self.current_schedule['lr'] *= self.current_schedule['gamma']
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.current_schedule['lr']

    def reset_grad(self, optimizer, d_optimizer):
        optimizer.zero_grad()
        d_optimizer.zero_grad()

    def train_encoder_decoder(self, train_only=False):
        """Train the image steganography encoder and decoder.

        Args:
            train_only (bool): Whether to only train the image steganography encoder and decoder.
        """
        if train_only:
            device = torch.device("cuda:0")
        else:
            device = self.device if self.device else torch.device("cuda:0")
        if self.dataset_name == 'mnist':
            self.encoder = MNISTStegaStampEncoder(
                secret_size=self.encoder_schedule['secret_size'], 
                height=self.encoder_schedule['enc_height'], 
                width=self.encoder_schedule['enc_width'],
                in_channel=self.encoder_schedule['enc_in_channel']).to(self.device)
            self.decoder = MNISTStegaStampDecoder(
                secret_size=self.encoder_schedule['secret_size'], 
                height=self.encoder_schedule['enc_height'], 
                width=self.encoder_schedule['enc_width'],
                in_channel=self.encoder_schedule['enc_in_channel']).to(self.device)
            self.discriminator = MNISTDiscriminator(in_channel=self.encoder_schedule['enc_in_channel']).to(device)
        else:
            self.encoder = StegaStampEncoder(
                secret_size=self.encoder_schedule['secret_size'], 
                height=self.encoder_schedule['enc_height'], 
                width=self.encoder_schedule['enc_width'],
                in_channel=self.encoder_schedule['enc_in_channel']).to(self.device)
            self.decoder = StegaStampDecoder(
                secret_size=self.encoder_schedule['secret_size'], 
                height=self.encoder_schedule['enc_height'], 
                width=self.encoder_schedule['enc_width'],
                in_channel=self.encoder_schedule['enc_in_channel']).to(self.device)
            self.discriminator = Discriminator(in_channel=self.encoder_schedule['enc_in_channel']).to(device)
        train_dl = DataLoader(
            self.train_steg_set,
            batch_size=32,
            shuffle=True,
            num_workers=8,
            worker_init_fn=self._seed_worker)

        enc_total_epoch = self.encoder_schedule['enc_total_epoch']
        enc_secret_only_epoch = self.encoder_schedule['enc_secret_only_epoch']
        optimizer = torch.optim.Adam([{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}], lr=0.0001)
        d_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=0.00001)
        loss_fn_alex = lpips.LPIPS(net='alex').cuda()
        for epoch in range(enc_total_epoch):
            loss_list, bit_acc_list = [], []
            for idx, (image_input, secret_input) in enumerate(train_dl):
                image_input, secret_input = image_input.to(device), secret_input.to(device)

                residual = self.encoder([secret_input, image_input])
                encoded_image = image_input + residual
                encoded_image = encoded_image.clamp(0, 1)
                decoded_secret = self.decoder(encoded_image)
                D_output_fake = self.discriminator(encoded_image)

                # cross entropy loss for the steganography secret
                secret_loss_op = F.binary_cross_entropy_with_logits(decoded_secret, secret_input, reduction='mean')
                # the LPIPS perceptual loss
                if self.dataset_name == 'mnist':
                    # 28 * 28 is too small to calculate the lpips loss
                    lpips_loss_op = loss_fn_alex(nn.Upsample(scale_factor=(2, 2), mode='nearest')(image_input), nn.Upsample(scale_factor=(2, 2), mode='nearest')(encoded_image))
                else:
                    lpips_loss_op = loss_fn_alex(image_input,encoded_image)
                # L2 residual regularization loss
                l2_loss = torch.square(residual).mean()
                # the critic loss calculated between the encoded image and the original image
                G_loss = D_output_fake

                if epoch < enc_secret_only_epoch:
                    total_loss = secret_loss_op
                else:
                    total_loss = 2.0 * l2_loss + 1.5 * lpips_loss_op.mean() + 1.5 * secret_loss_op + 0.5 * G_loss
                loss_list.append(total_loss.item())

                bit_acc = get_secret_acc(secret_input, decoded_secret)
                bit_acc_list.append(bit_acc.item())

                total_loss.backward()
                optimizer.step()
                self.reset_grad(optimizer, d_optimizer)

                if epoch >= enc_secret_only_epoch and self.encoder_schedule['enc_use_dis']:
                    residual = self.encoder([secret_input, image_input])
                    encoded_image = image_input + residual
                    encoded_image = encoded_image.clamp(0, 1)
                    decoded_secret = self.decoder(encoded_image)
                    D_output_fake = self.discriminator(encoded_image)
                    D_output_real = self.discriminator(image_input)
                    D_loss = D_output_real - D_output_fake
                    D_loss.backward()
                    for p in self.discriminator.parameters():
                        p.grad.data = torch.clamp(p.grad.data, min=-0.01, max=0.01)
                    d_optimizer.step()
                    self.reset_grad(optimizer, d_optimizer)
            if train_only:
                msg = f'Epoch [{epoch + 1}] total loss: {np.mean(loss_list)}, bit acc: {np.mean(bit_acc_list)}\n'
                print(msg)
                exit()
            else:
                msg = f'Epoch [{epoch + 1}] total loss: {np.mean(loss_list)}, bit acc: {np.mean(bit_acc_list)}\n'
                self.log(msg)

        savepath = os.path.join(self.work_dir, 'encoder_decoder.pth')
        state = {
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
        }
        torch.save(state, savepath)

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
            self.model.load_state_dict(torch.load(self.current_schedule['pretrain']), strict=False)

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
        self.device = device

        self.post_transforms = None
        if self.dataset_name == 'cifar10':
            self.post_transforms = PostTensorTransform(self.dataset_name).to(self.device)

        self.work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(self.work_dir, exist_ok=True)
        self.log = Log(osp.join(self.work_dir, 'log.txt'))

        if self.encoder is None:
            assert self.encoder_schedule is not None
            self.train_encoder_decoder(train_only=False)
        self.get_img()
        del self.train_steg_set

        trainset, testset = self.train_dataset, self.test_dataset
        train_dl = DataLoader(
            trainset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            worker_init_fn=self._seed_worker)
        test_dl = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            worker_init_fn=self._seed_worker)

        encoder = self.encoder
        encoder = encoder.eval()

        secret = torch.FloatTensor(np.random.binomial(1, .5, self.encoder_schedule['secret_size']).tolist()).to(self.device)

        cln_train_dataset, cln_train_labset, bd_train_dataset, bd_train_labset = [], [], [], []
        for idx, (img, lab) in enumerate(train_dl):
            if idx in self.poisoned_set:
                img = img.to(self.device)
                residual = encoder([secret, img])
                encoded_image = img + residual
                encoded_image = encoded_image.clamp(0, 1)
                bd_train_dataset.append(encoded_image.cpu().detach().tolist()[0])
                bd_train_labset.append(self.y_target)
            else:
                cln_train_dataset.append(img.tolist()[0])
                cln_train_labset.append(lab.tolist()[0])

        cln_train_dl = GetPoisonedDataset(cln_train_dataset, cln_train_labset)
        bd_train_dl = GetPoisonedDataset(bd_train_dataset, bd_train_labset)

        # self.train_poisoned_data = cln_train_dataset + bd_train_dataset
        # self.train_poisoned_label = cln_train_labset + bd_train_labset

        bd_test_dataset, bd_test_labset = [], []
        for idx, (img, lab) in enumerate(test_dl):
            img = img.to(self.device)
            residual = encoder([secret, img])
            encoded_image = img + residual
            encoded_image = encoded_image.clamp(0, 1)
            bd_test_dataset.append(encoded_image.cpu().detach().tolist()[0])
            bd_test_labset.append(self.y_target)

        cln_test_dl = testset
        bd_test_dl = GetPoisonedDataset(bd_test_dataset, bd_test_labset)

        # self.test_poisoned_data, self.test_poisoned_label = bd_test_dataset, bd_test_labset

        bd_bs = int(self.current_schedule['batch_size'] * self.poisoned_rate)
        cln_bs = int(self.current_schedule['batch_size'] - bd_bs)
        cln_train_dl = DataLoader(
            cln_train_dl,
            batch_size=cln_bs,
            shuffle=True,
            num_workers=self.current_schedule['num_workers'],
            worker_init_fn=self._seed_worker)

        bd_train_dl = DataLoader(
            bd_train_dl,
            batch_size=bd_bs,
            shuffle=True,
            num_workers=self.current_schedule['num_workers'],
            worker_init_fn=self._seed_worker)


        self.model = self.model.to(device)
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_schedule['lr'], momentum=self.current_schedule['momentum'], weight_decay=self.current_schedule['weight_decay'])

        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint

        last_time = time.time()

        msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.test_dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(self.train_dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
        self.log(msg)

        for i in range(self.current_schedule['epochs']):
            self.adjust_learning_rate(optimizer, i)
            loss_list = []
            for (inputs, targets), (inputs_trigger, targets_trigger) in zip(cln_train_dl, bd_train_dl):
                inputs = torch.cat((inputs, inputs_trigger), 0)
                targets = torch.cat((targets, targets_trigger), 0)
                if self.normalizer:
                    inputs = self.normalizer(inputs)
                if self.post_transforms:
                    inputs = self.post_transforms(inputs)

                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                predict_digits = self.model(inputs)
                loss = self.loss(predict_digits, targets)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + 'Train [{}] Loss: {:.4f}\n'.format(i, np.mean(loss_list))
            self.log(msg)

            if (i + 1) % self.current_schedule['test_epoch_interval'] == 0:
                # test result on benign test dataset
                predict_digits, labels = self._test(cln_test_dl, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on benign test dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num} time: {time.time()-last_time}\n"
                self.log(msg)

                # test result on poisoned test dataset
                # if self.current_schedule['benign_training'] is False:
                predict_digits, labels = self._test(bd_test_dl, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on poisoned test dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
                self.log(msg)

                self.model = self.model.to(device)
                self.model.train()

            if (i + 1) % self.current_schedule['save_epoch_interval'] == 0:
                self.model.eval()
                self.model = self.model.cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(i+1) + ".pth"
                ckpt_model_path = os.path.join(self.work_dir, ckpt_model_filename)
                torch.save(self.model.state_dict(), ckpt_model_path)
                self.model = self.model.to(device)
                self.model.train()

        self.train_poisoned_data, self.train_poisoned_label = [], []
        for (inputs, targets), (inputs_trigger, targets_trigger) in zip(cln_train_dl, bd_train_dl):
            inputs = torch.cat((inputs, inputs_trigger), 0)
            targets = torch.cat((targets, targets_trigger), 0)
            if self.normalizer:
                inputs = self.normalizer(inputs)
            if self.post_transforms:
                inputs = self.post_transforms(inputs)

            inputs = inputs.to(device)
            targets = targets.to(device)

            self.train_poisoned_data += inputs.cpu().detach().numpy().tolist()
            self.train_poisoned_label += targets.cpu().detach().numpy().tolist()

        self.test_poisoned_data, self.test_poisoned_label = [], []
        bd_test_dl = DataLoader(
            bd_test_dl,
            batch_size=16,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            pin_memory=True
        )
        for batch in bd_test_dl:
            batch_img, batch_label = batch
            if self.normalizer:
                batch_img = self.normalizer(batch_img)

            self.test_poisoned_data += batch_img.cpu().detach().numpy().tolist()
            self.test_poisoned_label += batch_label.cpu().detach().numpy().tolist()

    def _test(self, dataset, device, batch_size=16, num_workers=8, model=None):
        if model is None:
            model = self.model
        else:
            model = model

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
                if self.normalizer:
                    batch_img = self.normalizer(batch_img)
                batch_img = batch_img.to(device)
                batch_img = model(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)
                labels.append(batch_label)

            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            return predict_digits, labels

    def test(self, schedule=None, model=None, test_dataset=None, poisoned_test_dataset=None):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Test schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        if model is None:
            model = self.model

        if 'test_model' in self.current_schedule:
            model.load_state_dict(torch.load(self.current_schedule['test_model']), strict=False)

        if test_dataset is None and poisoned_test_dataset is None:
            test_dataset = self.test_dataset
            poisoned_test_dataset = self.poisoned_test_dataset

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

        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        if test_dataset is not None:
            last_time = time.time()
            # test result on benign test dataset
            predict_digits, labels = self._test(test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], model)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num} time: {time.time()-last_time}\n"
            log(msg)

        if poisoned_test_dataset is not None:
            last_time = time.time()
            # test result on poisoned test dataset
            predict_digits, labels = self._test(poisoned_test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], model)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
            log(msg)

    def get_img(self, path=None):
        """Get the encoded images with the trigger pattern.

        Args:
            path (str): The path of the saved image steganography encoder.
        """
        
    
        if path is not None:
            device = torch.device("cuda:0")
            if self.device is None:
                self.device = device
            encoder = StegaStampEncoder(
                secret_size=self.encoder_schedule['secret_size'], 
                height=self.encoder_schedule['enc_height'], 
                width=self.encoder_schedule['enc_width'], 
                in_channel=self.encoder_schedule['enc_in_channel']).to(self.device)
            decoder = StegaStampDecoder(
                secret_size=self.encoder_schedule['secret_size'], 
                height=self.encoder_schedule['enc_height'], 
                width=self.encoder_schedule['enc_width'],
                in_channel=self.encoder_schedule['enc_in_channel']).to(self.device)
            encoder.load_state_dict(torch.load(os.path.join(path, 'encoder_decoder.pth'))['encoder_state_dict'])
            decoder.load_state_dict(torch.load(os.path.join(path, 'encoder_decoder.pth'))['decoder_state_dict'])
        else:
            encoder = self.encoder
            decoder = self.decoder
        encoder = encoder.eval()
        decoder = decoder.eval()
        train_dl = DataLoader(
            self.train_steg_set,
            batch_size=1,
            shuffle=True,
            num_workers=8,
            worker_init_fn=self._seed_worker)

        for _, (image_input, secret_input) in enumerate(train_dl):
            image_input, secret_input = image_input.cuda(), secret_input.cuda()
            residual = encoder([secret_input, image_input])
            encoded_image = image_input + residual
            encoded_image = torch.clamp(encoded_image, min=0, max=1)
            decoded_secret = decoder(encoded_image)
            bit_acc = get_secret_acc(secret_input, decoded_secret)
            print('bit_acc: ', bit_acc)
            image_input = image_input.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
            encoded_image = encoded_image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
            residual = residual.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
            imageio.imwrite(os.path.join(self.work_dir, 'image_input.jpg'), image_input)
            imageio.imwrite(os.path.join(self.work_dir, 'encoded_image.jpg'), encoded_image)
            imageio.imwrite(os.path.join(self.work_dir, 'residual.jpg'), residual)
            break
