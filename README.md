# Welcome to BackdoorBox (Under Development)
![Python 3.8](https://img.shields.io/badge/python-3.8-DodgerBlue.svg?style=plastic)
![Pytorch 1.8.0](https://img.shields.io/badge/pytorch-1.8.0-DodgerBlue.svg?style=plastic)
![torchvision 0.9.0](https://img.shields.io/badge/torchvision-0.9.0-DodgerBlue.svg?style=plastic)
![CUDA 11.1](https://img.shields.io/badge/cuda-11.1-DodgerBlue.svg?style=plastic)
![License GPL](https://img.shields.io/badge/license-GPL-DodgerBlue.svg?style=plastic)

BackdoorBox is a Python toolbox for backdoor attacks and defenses.  

This project is still under development and therefore there is no user manual yet. Please refer to the 'tests' sub-folder to get more insights about how to use our implemented methods.


# Current Status

## Developed Methods

- [BadNets](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/BadNets.py) (**Key Properties**: poison-only)
- [Blended Attack](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/Blended.py) (**Key Properties**: poison-only, invisible)
- [Refool (simplified version)](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/Refool.py) (**Key Properties**: poison-only, sample-specific, physical)
- [WaNet](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/WaNet.py) (**Key Properties**: poison-only, invisible, sample-specific)
- [Label-consistent Attack](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/LabelConsistent.py) (**Key Properties**: poison-only, invisible, clean-label)
- [Blind Backdoor (blended-based)](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/Blind.py) (**Key Properties**: training-controlled)
- [Input-aware Dynamic Attack](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/IAD.py) (**Key Properties**: training-controlled, optimized, sample-specific)
- [LIRA](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/LIRA.py) (**Key Properties**: training-controlled, invisible, optimized, sample-specific)
- [Physical Attack](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/PhysicalBA.py) (**Key Properties**: training-controlled, physical)

## Methods Under Development
- TUAP (basic version)
- ISSBA
- SleeperAgent



# Contributors

| Organization        | Contributors                                                 |
| ------------------- | ------------------------------------------------------------ |
| Tsinghua University | [Yiming Li](http://liyiming.tech/), [Mengxi Ya](https://github.com/yamengxi), [Guanhao Gan](https://github.com/GuanhaoGan), [Kuofeng Gao](https://github.com/KuofengGao), [Xin Yan](https://scholar.google.com/citations?hl=zh-CN&user=08WTTPMAAAAJ), [Jia Xu](https://www.researchgate.net/profile/Xu-Jia-10), [Tong Xu](https://github.com/spicy1007), [Yang Bai](https://scholar.google.com/citations?user=wBH_Q1gAAAAJ&hl=zh-CN), [Linghui Zhu](https://github.com/zlh-thu) |


# Citation
If our toolbox is useful for your research, please cite our paper(s) as follows:
```
@article{li2022backdoorbox,
  title={{BackdoorBox}: A Python Toolbox for Backdoor Learning},
  author={Li, Yiming and Ya, Mengxi and Bai, Yang and Jiang, Yong and Xia, Shu-Tao},
  year={2022}
}
```

```
@article{li2020backdoor,
  title={Backdoor Learning: A Survey},
  author={Li, Yiming and Jiang, Yong and Li, Zhifeng and Xia, Shu-Tao},
  journal={arXiv preprint arXiv:2007.08745},
  year={2020}
}
```
