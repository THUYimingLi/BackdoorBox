# Welcome to BackdoorBox (Under Development)
![Python 3.8](https://img.shields.io/badge/python-3.8-DodgerBlue.svg?style=plastic)
![Pytorch 1.8.0](https://img.shields.io/badge/pytorch-1.8.0-DodgerBlue.svg?style=plastic)
![torchvision 0.9.0](https://img.shields.io/badge/torchvision-0.9.0-DodgerBlue.svg?style=plastic)
![CUDA 11.1](https://img.shields.io/badge/cuda-11.1-DodgerBlue.svg?style=plastic)
![License GPL](https://img.shields.io/badge/license-GPL-DodgerBlue.svg?style=plastic)

BackdoorBox is a Python toolbox for backdoor learning research. Specifically, BackdoorBox contains modules for conducting backdoor attacks and backdoor defenses.  

This project is still under development and therefore there is no user manual yet. Please refer to the 'tests' sub-folder to get more insights about how to use our implemented methods.


# Current Status

## Developed Methods

- [BadNets](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/BadNets.py) (**Key Properties**: poison-only, visible, poison-label, non-optimized, non-semantic, sample-agnostic, digital)
- [Blended Attack](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/Blended.py) (**Key Properties**: poison-only, invisible, poison-label, non-optimized, non-semantic, sample-agnostic, digital)
- [Refool (simplified version)](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/Refool.py) (**Key Properties**: poison-only, visible, poison-label, non-optimized, non-semantic, sample-specific, physical)
- [WaNet](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/WaNet.py) (**Key Properties**: poison-only, invisible, poison-label, non-optimized, non-semantic, sample-specific, digital)
- [Label-consistent Attack](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/LabelConsistent.py) (**Key Properties**: poison-only, invisible, clean-label, non-optimized, non-semantic, sample-agnostic, digital)
- [Blind Backdoor (blended-based)](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/Blind.py) (**Key Properties**: training-controlled, invisible, poison-label, non-optimized, non-semantic, sample-agnostic, digital)
- [Input-aware Dynamic Attack](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/IAD.py) (**Key Properties**: training-controlled, visible, poison-label, optimized, non-semantic, sample-specific, digital)
- [LIRA](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/LIRA.py) (**Key Properties**: training-controlled, invisible, poison-label, optimized, non-semantic, sample-specific, digital)


## Methods Under Development
- TUAP (basic version)
- Physical Attack
- ISSBA
- SleeperAgent



# Contributors

| Organization        | Contributors                                                 |
| ------------------- | ------------------------------------------------------------ |
| Tsinghua University | [Yiming Li](http://liyiming.tech/), [Mengxi Ya](https://github.com/yamengxi), [Guanhao Gan](https://github.com/GuanhaoGan), [Kuofeng Gao](https://github.com/KuofengGao), [Xin Yan](https://scholar.google.com/citations?hl=zh-CN&user=08WTTPMAAAAJ), [Jia Xu](https://www.researchgate.net/profile/Xu-Jia-10), [Yang Bai](https://scholar.google.com/citations?user=wBH_Q1gAAAAJ&hl=zh-CN), [Linghui Zhu](https://github.com/zlh-thu) |
