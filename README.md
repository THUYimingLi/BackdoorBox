# BackdoorBox: A Python Toolbox for Backdoor Attacks and Defenses
![Python 3.8](https://img.shields.io/badge/python-3.8-DodgerBlue.svg?style=plastic)
![Pytorch 1.8.0](https://img.shields.io/badge/pytorch-1.8.0-DodgerBlue.svg?style=plastic)
![torchvision 0.9.0](https://img.shields.io/badge/torchvision-0.9.0-DodgerBlue.svg?style=plastic)
![CUDA 11.1](https://img.shields.io/badge/cuda-11.1-DodgerBlue.svg?style=plastic)
![License GPL](https://img.shields.io/badge/license-GPL-DodgerBlue.svg?style=plastic)

Backdoor attacks are emerging yet critical threats in the training process of deep neural networks (DNNs), where the adversary intends to embed specific hidden backdoor into the models. The attacked DNNs will behave normally in predicting benign samples, whereas the predictions will be maliciously changed whenever the adversary-specified trigger patterns appear. Currently, there were many existing backdoor attacks and defenses. Although most of them were open-sourced, there is still no toolbox that can easily and flexibly implement and compare them simultaneously.

[BackdoorBox](https://www.researchgate.net/publication/359439455_BackdoorBox_A_Python_Toolbox_for_Backdoor_Learning) is an open-sourced Python toolbox, aiming to implement representative and advanced backdoor attacks and defenses under a unified framework that can be used in a flexible manner. We will keep updating this toolbox to track the latest backdoor attacks and defenses. 

Currently, this toolbox is still under development (but the attack parts are almost done) and there is no user manual yet. However, you can easily implement our provided methods by referring to the `tests` sub-folder to see the example codes of each implemented method. In particular, you are always welcome to contribute your backdoor attacks or defenses by pull requests!


## Toolbox Characteristics
- **Consistency**: Instead of implementing each method separately, we develop all methods in a unified manner. Specifically, variables having the same function have a consistent name. Similar methods inherit the same base class for further development, have a unified workflow, and have the same core sub-functions (*e.g.*, `get_model()`.
- **Simplicity**: We provide code examples for each implemented backdoor attack and defense to explain how to use them, the definitions and default settings of all required attributes, and the necessary code comments. Users can easily implement and develop our toolbox.
- **Flexibility**: We allow users to easily obtain important intermediate outputs and components of each method (*e.g.*, poisoned dataset and attacked/repaired model), use their local samples and model structure for attacks and defenses, and interact with their local codes. The attack and defense modules can be used jointly or separately.
- **Co-development**: All codes and developments are hosted on Github to facilitate collaboration. Currently, there are more than seven contributors have helped develop the code base and others have contributed to the code test. This developing paradigm facilitates rapid and comprehensive development and bug finding.

## Current Status

### Developed Methods
#### Backdoor Attacks
|                                                    **Method**                                                   |       **Source**      | **Key Properties**                                         | **Note**                                                    |
|:---------------------------------------------------------------------------------------------------------------:|:---------------------:|------------------------------------------------------------|-------------------------------------------------------------|
|             [BadNets](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/BadNets.py)             |   IEEE ACCESS, 2019   | poison-only                                                | first backdoor attack                                       |
|          [Blended Attack](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/Blended.py)         |      arXiv, 2017      | poison-only, invisible                                     | first invisible attack                                      |
|    [Refool (simplified version)](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/Refool.py)   |       ECCV, 2020      | poison-only, sample-specific                               | first stealthy attack with visible yet natural trigger      |
| [Label-consistent Attack](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/LabelConsistent.py) |      arXiv, 2019      | poison-only, invisible, clean-label                        | first clean-label backdoor attack                           |
|               [ISSBA](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/ISSBA.py)               |       ICCV, 2021      | poison-only, sample-specific, physical                     | first poison-only sample-specific attack                    |
|               [WaNet](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/WaNet.py)               |       ICLR, 2021      | poison-only, invisible, sample-specific                    |                                                             |
|   [Blind Backdoor (blended-based)](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/Blind.py)  | USENIX Security, 2021 | training-controlled                                        | first training-controlled attack targeting loss computation |
|      [Input-aware Dynamic Attack](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/IAD.py)     |     NeurIPS, 2020     | training-controlled, optimized, sample-specific            | first training-controlled sample-specific attack            |
|        [Physical Attack](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/PhysicalBA.py)       |  ICLR Workshop, 2021  | training-controlled, physical                              | first physical backdoor attack                              |
|                [LIRA](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/LIRA.py)                |       ICCV, 2021      | training-controlled, invisible, optimized, sample-specific |                                                             |

**Note**: For the convenience of users, all our implemented attacks support obtaining poisoned dataset (via `.get_poisoned_dataset()`), obtaining infected model (via `.get_model()`), and training with your own local samples (loaded via `torchvision.datasets.DatasetFolder`). Please refer to [base.py](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/base.py) and the attack's codes for more details.

#### Backdoor Defenses
- [ShrinkPad](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/ShrinkPad.py) (**Key Properties**: Pre-processing-based Defense)
- [FineTuning](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/FineTuning.py) (**Key Properties**: Model Repairing)
- [MCR](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/MCR.py) (**Key Properties**: Model Repairing)

### Methods Under Development
- TUAP (basic version)
- Sleeper Agent
- NAD
- Pruning
- DBD
- SS
- ABL



## Contributors

| Organization        | Contributors                                                 |
| ------------------- | ------------------------------------------------------------ |
| Tsinghua University | [Yiming Li](http://liyiming.tech/), [Mengxi Ya](https://github.com/yamengxi), [Guanhao Gan](https://github.com/GuanhaoGan), [Kuofeng Gao](https://github.com/KuofengGao), [Xin Yan](https://scholar.google.com/citations?hl=zh-CN&user=08WTTPMAAAAJ), [Jia Xu](https://www.researchgate.net/profile/Xu-Jia-10), [Tong Xu](https://github.com/spicy1007), [Sheng Yang](https://github.com/20000yshust), [Haoxiang Zhong](https://scholar.google.com/citations?user=VOw9qmYAAAAJ&hl=zh-CN&oi=ao), [Linghui Zhu](https://github.com/zlh-thu), [Yang Bai](https://scholar.google.com/citations?user=wBH_Q1gAAAAJ&hl=zh-CN) |


## Citation
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
