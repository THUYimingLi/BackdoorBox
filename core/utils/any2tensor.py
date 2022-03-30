import cv2
import PIL
import numpy
import torch
from torchvision.transforms import functional as F


def _any2tensor(x):
    """Convert a strpath, PIL.Image.Image, numpy.ndarray, torch.Tensor object to a torch.Tensor object.

    Args:
        x (strpath | PIL.Image.Image | numpy.ndarray | torch.Tensor): numpy.ndarray and torch.Tensor can have any shape.
        Hint: For strpath, x is converted to a torch.Tensor with shape (C, H, W), the channel order is decided by opencv.
        For PIL.Image.Image, x is converted to a torch.Tensor with shape (C, H, W), the channel order is decided by x itself.
        The channel order between opencv and PIL is different.

    Returns:
        torch.Tensor: The converted object.
    """
    if type(x) == str:
        tmp = cv2.imread(x, cv2.IMREAD_UNCHANGED)
        if tmp.ndim == 2:
            return torch.from_numpy(tmp.reshape(1, tmp.shape[0], tmp.shape[1]))
        else:
            return torch.from_numpy(tmp.transpose((2, 0, 1)))
    elif type(x) == PIL.Image.Image:
        return F.pil_to_tensor(x)
    elif type(x) == numpy.ndarray:
        return torch.from_numpy(x)
    elif type(x) == torch.Tensor:
        return x.clone().detach()
    else:
        raise TypeError('x is an unsupported type, x should be strpath or PIL.Image.Image or numpy.ndarray or torch.Tensor. But got {}'.format(type(x)))


def any2tensor(imgs):
    """Convert strpath, PIL.Image.Image, numpy.ndarray, torch.Tensor image(s) to a torch.Tensor.

    Args:
        imgs (list[strpath] | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] | strpath | PIL.Image.Image | numpy.ndarray | torch.Tensor): The input images.

    Returns:
        torch.Tensor: The converted image(s).
    """
    if isinstance(imgs, list):
        return torch.stack([_any2tensor(img) for img in imgs], dim=0)
    elif isinstance(imgs, (str, PIL.Image.Image, numpy.ndarray, torch.Tensor)):
        return _any2tensor(imgs)
    else:
        raise TypeError('imgs is an unsupported type, imgs should be list[strpath] | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] | strpath | PIL.Image.Image | numpy.ndarray | torch.Tensor. But got {}'.format(type(imgs)))
