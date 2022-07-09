import torch
import torch.nn as nn


class AutoEncoder1x28x28(nn.Module):
    """Autoencoder for 1x28x28 input image.

    This is a reimplementation of the blog post 'Building Autoencoders in Keras', from blog
    `Building Autoencoders in Keras <https://blog.keras.io/building-autoencoders-in-keras.html>`_.
    """
    def __init__(self):
        super(AutoEncoder1x28x28, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 1, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(2)
        self.upsampler = nn.UpsamplingNearest2d(scale_factor=2.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input size: [batch, 1, 28, 28]
        # Output size: [batch, 1, 28, 28]
        x = self.conv1(x) # [batch, 32, 28, 28]
        x = self.relu(x) # [batch, 32, 28, 28]
        x = self.max_pool(x) # [batch, 32, 14, 14]

        x = self.conv2(x) # [batch, 32, 14, 14]
        x = self.relu(x) # [batch, 32, 14, 14]
        x = self.max_pool(x) # [batch, 32, 7, 7]

        x = self.conv3(x) # [batch, 32, 7, 7]
        x = self.relu(x) # [batch, 32, 7, 7]
        x = self.upsampler(x) # [batch, 32, 14, 14]

        x = self.conv4(x) # [batch, 32, 14, 14]
        x = self.relu(x) # [batch, 32, 14, 14]
        x = self.upsampler(x) # [batch, 32, 28, 28]

        x = self.conv5(x) # [batch, 1, 28, 28]
        x = self.sigmoid(x) # [batch, 1, 28, 28]

        return x


class AutoEncoder3x32x32(nn.Module):
    """Autoencoder for 3x32x32 input image.

    This is modified from 'PyTorch-CIFAR-10-autoencoder', from github
    `PyTorch-CIFAR-10-autoencoder <https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder>`_.
    """
    def __init__(self):
        super(AutoEncoder3x32x32, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
            # nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            # nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def AutoEncoder(img_size):
    assert isinstance(img_size, tuple), f"img_size should be tuple, but got {type(img_size)}"
    if img_size == (1, 28, 28):
        return AutoEncoder1x28x28()
    elif img_size == (3, 32, 32):
        return AutoEncoder3x32x32()
    else:
        raise NotImplementedError("Unsupported img size!")


if __name__=='__main__':
    x = torch.randn((128, 1, 28, 28))
    # model = AutoEncoder([1, 28, 28])
    model = AutoEncoder((1, 28, 28))
    x = model(x)

    x = torch.randn((128, 3, 32, 32))
    model = AutoEncoder((3, 32, 32))
    x = model(x)
