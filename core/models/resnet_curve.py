'''
This is the part of the implement of model-repairing-based backdoor defense with MCR proposed in [1].

Reference:
[1] Bridging Mode Connectivity in Loss Landscapes and Adversarial Robustness. ICLR, 2020.
'''
from calendar import c
import sys, os
sys.path.append(os.path.join(os.getcwd(), 'core/models'))
import torch.nn as nn
import torch.nn.functional as F
import curves
import math

class BasicBlockCurves(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, fix_points, stride=1):
        super(BasicBlockCurves, self).__init__()
        self.conv1 = curves.Conv2d(in_planes, planes, kernel_size=3, fix_points=fix_points, stride=stride, padding=1, bias=False)
        self.bn1 = curves.BatchNorm2d(planes, fix_points)
        self.conv2 = curves.Conv2d(planes, planes, kernel_size=3, fix_points=fix_points, stride=1, padding=1, bias=False)
        self.bn2 = curves.BatchNorm2d(planes, fix_points)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                curves.Conv2d(in_planes, self.expansion*planes, kernel_size=1, fix_points=fix_points, stride=stride, bias=False),
                curves.BatchNorm2d(self.expansion*planes, fix_points)
            )

    def forward(self, x, coeffs_t):
        out = F.relu(self.bn1(self.conv1(x, coeffs_t), coeffs_t))
        out = self.bn2(self.conv2(out, coeffs_t), coeffs_t)
        # out += self.shortcut(x)
        for module in self.shortcut:
            if isinstance(module, curves.CurveModule):
                x = module(x, coeffs_t)
            else:
                x = module(x)
        out += x
        out = F.relu(out)
        return out


class BottleneckCurve(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, fix_points, stride=1):
        super(BottleneckCurve, self).__init__()
        self.conv1 = curves.Conv2d(in_planes, planes, kernel_size=1, fix_points=fix_points, bias=False)
        self.bn1 = curves.BatchNorm2d(planes, fix_points)
        self.conv2 = curves.Conv2d(planes, planes, kernel_size=3, fix_points=fix_points, stride=stride, padding=1, bias=False)
        self.bn2 = curves.BatchNorm2d(planes, fix_points)
        self.conv3 = curves.Conv2d(planes, self.expansion*planes, fix_points=fix_points, kernel_size=1, bias=False)
        self.bn3 = curves.BatchNorm2d(self.expansion*planes, fix_points)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                curves.Conv2d(in_planes, self.expansion*planes, kernel_size=1, fix_points=fix_points, stride=stride, bias=False),
                curves.BatchNorm2d(self.expansion*planes, fix_points)
            )

    def forward(self, x, coeffs_t):
        out = F.relu(self.bn1(self.conv1(x, coeffs_t), coeffs_t))
        out = F.relu(self.bn2(self.conv2(out, coeffs_t), coeffs_t))
        out = self.bn3(self.conv3(out, coeffs_t), coeffs_t)
        # out += self.shortcut(x)
        for module in self.shortcut:
            if isinstance(module, curves.CurveModule):
                x = module(x, coeffs_t)
            else:
                x = module(x)
        out += x
        out = F.relu(out)
        return out


class _ResNetCurve(nn.Module):
    def __init__(self, block, num_blocks, fix_points, num_classes=10, initialize=False):
        super(_ResNetCurve, self).__init__()
        self.in_planes = 64

        self.conv1 = curves.Conv2d(3, 64, kernel_size=3, fix_points=fix_points, stride=1, padding=1, bias=False)
        self.bn1 = curves.BatchNorm2d(64, fix_points)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], fix_points, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], fix_points, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], fix_points, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], fix_points, stride=2)
        self.linear = curves.Linear(512*block.expansion, num_classes, fix_points)

        if initialize:
            for m in self.modules():
                if isinstance(m, curves.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    for i in range(m.num_bends):
                        getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, curves.BatchNorm2d):
                    for i in range(m.num_bends):
                        getattr(m, 'weight_%d' % i).data.fill_(1)
                        getattr(m, 'bias_%d' % i).data.zero_()

    def _make_layer(self, block, planes, num_blocks, fix_points, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, fix_points, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, coeffs_t):
        out = F.relu(self.bn1(self.conv1(x, coeffs_t), coeffs_t))
        # out = self.layer1(out, coeffs_t)
        # out = self.layer2(out, coeffs_t)
        # out = self.layer3(out, coeffs_t)
        # out = self.layer4(out, coeffs_t)
        for module in self.layer1:
            out = module(out,coeffs_t)
        for module in self.layer2:
            out = module(out,coeffs_t)
        for module in self.layer3:
            out = module(out,coeffs_t)
        for module in self.layer4:
            out = module(out,coeffs_t)       

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out, coeffs_t)
        return out


def ResNetCurve(num, fix_points, num_classes=10, initialize=False):
    if num == 18:
        return _ResNetCurve(BasicBlockCurves, [2,2,2,2], fix_points, num_classes, initialize)
    elif num == 34:
        return _ResNetCurve(BasicBlockCurves, [3,4,6,3], fix_points, num_classes, initialize)
    elif num == 50:
        return _ResNetCurve(BottleneckCurve, [3,4,6,3], fix_points, num_classes, initialize)
    elif num == 101:
        return _ResNetCurve(BottleneckCurve, [3,4,23,3], fix_points, num_classes, initialize)
    elif num == 152:
        return _ResNetCurve(BottleneckCurve, [3,8,36,3], fix_points, num_classes, initialize)
    else:
        raise NotImplementedError