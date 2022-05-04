'''
This is the part of the implement of model-repairing-based backdoor defense with MCR proposed in [1].

Reference:
[1] Bridging Mode Connectivity in Loss Landscapes and Adversarial Robustness. ICLR, 2020.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import curves

class VGGCurve(nn.Module):

    def __init__(self, features, num_classes=43, fix_points=None, initialize=False):
        super(VGGCurve, self).__init__()
        self.features = features
        self.classifier = curves.Linear(512, num_classes, fix_points)
        # self._initialize_weights()
        if initialize:
            for m in self.modules():
                if isinstance(m, curves.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    for i in range(m.num_bends):
                        getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2. / n))
                        getattr(m, 'bias_%d' % i).data.zero_()


    def forward(self, x, coeffs_t):
        # x = self.features(x)
        for module in self.features:
            if isinstance(module, curves.CurveModule):
                x = module(x, coeffs_t)
            else:
                x = module(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x, coeffs_t)
        return x
 
    #TODO: check if this is needed
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, curves.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, curves.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, curves.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, fix_points=None, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = curves.Conv2d(in_channels, v,  kernel_size=3, fix_points=fix_points, padding=1)
            if batch_norm:
                layers += [conv2d, curves.BatchNorm2d(v, fix_points), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(num_classes=10, fix_points=None, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGGCurve(make_layers(cfg['A'], fix_points), num_classes=num_classes, fix_points=fix_points, **kwargs)
    return model


def vgg11_bn(num_classes=10, fix_points=None, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGGCurve(make_layers(cfg['A'], fix_points, batch_norm=True), num_classes=num_classes, fix_points=fix_points, **kwargs)
    return model


def vgg13(num_classes=10, fix_points=None, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGGCurve(make_layers(cfg['B'], fix_points), num_classes=num_classes, fix_points=fix_points, **kwargs)
    return model


def vgg13_bn(num_classes=10, fix_points=None, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGGCurve(make_layers(cfg['B'], fix_points, batch_norm=True), num_classes=num_classes, fix_points=fix_points, **kwargs)
    return model


def vgg16(num_classes=10, fix_points=None, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGGCurve(make_layers(cfg['D'], fix_points), num_classes=num_classes, fix_points=fix_points, **kwargs)
    return model


def vgg16_bn(num_classes=10, fix_points=None, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGGCurve(make_layers(cfg['D'], fix_points, batch_norm=True), num_classes=num_classes, fix_points=fix_points, **kwargs)
    return model


def vgg19(num_classes=10, fix_points=None, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGGCurve(make_layers(cfg['E'], fix_points), num_classes=num_classes, fix_points=fix_points, **kwargs)
    return model


def vgg19_bn(num_classes=10, fix_points=None, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGGCurve(make_layers(cfg['E'], fix_points, batch_norm=True), num_classes=num_classes, fix_points=fix_points, **kwargs)
    return model