# BackdoorBox: An Open-sourced Python Toolbox for Backdoor Attacks and Defenses
![Python 3.8](https://img.shields.io/badge/python-3.8-DodgerBlue.svg?style=plastic)
![Pytorch 1.8.0](https://img.shields.io/badge/pytorch-1.8.0-DodgerBlue.svg?style=plastic)
![torchvision 0.9.0](https://img.shields.io/badge/torchvision-0.9.0-DodgerBlue.svg?style=plastic)
![CUDA 11.1](https://img.shields.io/badge/cuda-11.1-DodgerBlue.svg?style=plastic)
![License GPL](https://img.shields.io/badge/license-GPL-DodgerBlue.svg?style=plastic)

Backdoor attacks are emerging yet critical threats in the training process of deep neural networks (DNNs), where the adversary intends to embed specific hidden backdoor into the models. The attacked DNNs will behave normally in predicting benign samples, whereas the predictions will be maliciously changed whenever the adversary-specified trigger patterns appear. Currently, there were many existing backdoor attacks and defenses. Although most of them were open-sourced, there is still no toolbox that can easily and flexibly implement and compare them simultaneously.

[BackdoorBox](https://www.researchgate.net/publication/359439455_BackdoorBox_A_Python_Toolbox_for_Backdoor_Learning) is an open-sourced Python toolbox, aiming to implement representative and advanced backdoor attacks and defenses under a unified framework that can be used in a flexible manner. We will keep updating this toolbox to track the latest backdoor attacks and defenses. 

Currently, this toolbox is still under development (but the attack parts are almost done) and there is no user manual yet. However, **users can easily implement our provided methods by referring to the `tests` sub-folder to see the example codes of each implemented method**. Please refer to [our paper](https://www.researchgate.net/publication/359439455_BackdoorBox_A_Python_Toolbox_for_Backdoor_Learning) for more details! In particular, you are always welcome to contribute your backdoor attacks or defenses by pull requests!


## Toolbox Characteristics
- **Consistency**: Instead of implementing each method separately, we develop all methods in a unified manner. Specifically, variables having the same function have a consistent name. Similar methods inherit the same base class for further development, have a unified workflow, and have the same core sub-functions (*e.g.*, `get_model()`).
- **Simplicity**: We provide code examples for each implemented backdoor attack and defense to explain how to use them, the definitions and default settings of all required attributes, and the necessary code comments. Users can easily implement and develop our toolbox.
- **Flexibility**: We allow users to easily obtain important intermediate outputs and components of each method (*e.g.*, poisoned dataset and attacked/repaired model), use their local samples and model structure for attacks and defenses, and interact with their local codes. The attack and defense modules can be used jointly or separately.
- **Co-development**: All codes and developments are hosted on Github to facilitate collaboration. Currently, there are more than seven contributors have helped develop the code base and others have contributed to the code test. This developing paradigm facilitates rapid and comprehensive development and bug finding.

## Backdoor Attacks
|                                                    **Method**                                                   |       **Source**      | **Key Properties**                                         | **Additional Notes**                                                    |
|:---------------------------------------------------------------------------------------------------------------:|:---------------------:|------------------------------------------------------------|-------------------------------------------------------------|
|             [BadNets](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/BadNets.py)             |   Badnets: Evaluating Backdooring Attacks on Deep Neural Networks. [IEEE Access, 2019](https://ieeexplore.ieee.org/abstract/document/8685687).   | poison-only                                                | first backdoor attack                                       |
|          [Blended](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/Blended.py)         |      Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning. [arXiv, 2017](https://arxiv.org/pdf/1712.05526.pdf).      | poison-only, invisible                                     | first invisible attack                                      |
|    [Refool (simplified version)](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/Refool.py)   |       Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks. [ECCV, 2020](https://arxiv.org/pdf/2007.02343.pdf).      | poison-only, sample-specific                               | first stealthy attack with visible yet natural trigger      |
| [LabelConsistent](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/LabelConsistent.py) |     Label-Consistent Backdoor Attacks. [arXiv, 2019](https://arxiv.org/pdf/1912.02771.pdf).      | poison-only, invisible, clean-label                        | first clean-label backdoor attack                           |
| [TUAP](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/TUAP.py) |      Clean-Label Backdoor Attacks on Video Recognition Models. [CVPR, 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_Clean-Label_Backdoor_Attacks_on_Video_Recognition_Models_CVPR_2020_paper.pdf).      | poison-only, invisible, clean-label                        | first clean-label backdoor attack with optimized trigger pattern                          |
| [SleeperAgent](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/SleeperAgent.py) |      Sleeper Agent: Scalable Hidden Trigger Backdoors for Neural Networks Trained from Scratch. [NeurIPS, 2022](https://arxiv.org/pdf/2106.08970.pdf).      | poison-only, invisible, clean-label                        | effective clean-label backdoor attack                         |
|               [ISSBA](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/ISSBA.py)               |       Invisible Backdoor Attack with Sample-Specific Triggers. [ICCV, 2021](https://arxiv.org/pdf/2012.03816.pdf).      | poison-only, sample-specific, physical                     | first poison-only sample-specific attack                    |
|               [WaNet](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/WaNet.py)               |       WaNet - Imperceptible Warping-based Backdoor Attack. [ICLR, 2021](https://openreview.net/pdf?id=eEn8KTtJOx).      | poison-only, invisible, sample-specific                    |                                                             |
|   [Blind (blended-based)](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/Blind.py)  | Blind Backdoors in Deep Learning Models. [USENIX Security, 2021](https://arxiv.org/pdf/2005.03823.pdf). | training-controlled                                        | first training-controlled attack targeting loss computation |
|      [IAD](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/IAD.py)     |     Input-Aware Dynamic Backdoor Attack. [NeurIPS, 2020](https://arxiv.org/pdf/2010.08138.pdf).     | training-controlled, optimized, sample-specific            | first training-controlled sample-specific attack            |
|        [PhysicalBA](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/PhysicalBA.py)       |  Backdoor Attack in the Physical World. [ICLR Workshop, 2021](https://arxiv.org/pdf/2104.02361.pdf).  | training-controlled, physical                              | first physical backdoor attack                              |
|                [LIRA](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/LIRA.py)                |       LIRA: Learnable, Imperceptible and Robust Backdoor Attacks. [ICCV, 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Doan_LIRA_Learnable_Imperceptible_and_Robust_Backdoor_Attacks_ICCV_2021_paper.pdf).      | training-controlled, invisible, optimized, sample-specific |                                                             |

**Note**: For the convenience of users, all our implemented attacks support obtaining poisoned dataset (via `.get_poisoned_dataset()`), obtaining infected model (via `.get_model()`), and training with your own local samples (loaded via `torchvision.datasets.DatasetFolder`). Please refer to [base.py](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/base.py) and the attack's codes for more details.

## Backdoor Defenses
|                                                    **Method**                                                   |       **Source**      | **Defense Type**                                         | **Additional Notes**                                                    |
|:---------------------------------------------------------------------------------------------------------------:|:---------------------:|------------------------------------------------------------|-------------------------------------------------------------|
|             [AutoEncoderDefense](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/AutoEncoderDefense.py)             |   Neural Trojans. [ICCD, 2017](https://arxiv.org/pdf/1710.00942.pdf).    | Sample Pre-processing                                                | first pre-processing-based defense                                     |
|             [ShrinkPad](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/ShrinkPad.py)             |   Backdoor Attack in the Physical World. [ICLR Workshop, 2021](https://arxiv.org/pdf/2104.02361.pdf).    | Sample Pre-processing                                                | efficient defense                                     |
|          [FineTuning](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/FineTuning.py)         |      Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks. [RAID, 2018](https://arxiv.org/pdf/1805.12185.pdf).     | Model Repairing                                     | first defense based on model repairing                                      |
|          [Pruning](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/Pruning.py)         |      Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks. [RAID, 2018](https://arxiv.org/pdf/1805.12185.pdf).     | Model Repairing                                     |                                     |
|          [MCR](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/MCR.py)         |      Bridging Mode Connectivity in Loss Landscapes and Adversarial Robustness. [ICLR, 2020](https://arxiv.org/pdf/2005.00060.pdf).     | Model Repairing                                     |                                     |
|          [NAD](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/NAD.py)         |      Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks. [ICLR, 2021](https://openreview.net/pdf?id=9l0K4OM-oXE).     | Model Repairing                                     |  first distillation-based defense                                   |
|          [ABL](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/ABL.py)         |      Anti-Backdoor Learning: Training Clean Models on Poisoned Data. [NeurIPS, 2021](https://arxiv.org/pdf/2110.11571.pdf).     | Poison Suppression                                     |                                     |

## Methods Under Development
- DBD
- SS
- Neural Cleanse
- DP
- CutMix


## Attack & Defense Benchmark
The benchmark is coming soon.


## Contributors

| Organization        | Contributors                                                 |
| ------------------- | ------------------------------------------------------------ |
| Tsinghua University | [Yiming Li](http://liyiming.tech/), [Mengxi Ya](https://github.com/yamengxi), [Guanhao Gan](https://github.com/GuanhaoGan), [Kuofeng Gao](https://github.com/KuofengGao), [Xin Yan](https://scholar.google.com/citations?hl=zh-CN&user=08WTTPMAAAAJ), [Jia Xu](https://www.researchgate.net/profile/Xu-Jia-10), [Tong Xu](https://github.com/spicy1007), [Sheng Yang](https://github.com/20000yshust), [Haoxiang Zhong](https://scholar.google.com/citations?user=VOw9qmYAAAAJ&hl=zh-CN&oi=ao), [Linghui Zhu](https://github.com/zlh-thu)
| Tencent Security Zhuque Lab | [Yang Bai](https://scholar.google.com/citations?user=wBH_Q1gAAAAJ&hl=zh-CN) |
| ShanghaiTech University | [Zhe Zhao](https://s3l.shanghaitech.edu.cn/people/zhezhao/) |

## Citation
If our toolbox is useful for your research, please cite our paper(s) as follows:
```
@inproceedings{li2023backdoorbox,
  title={{BackdoorBox}: A Python Toolbox for Backdoor Learning},
  author={Li, Yiming and Ya, Mengxi and Bai, Yang and Jiang, Yong and Xia, Shu-Tao},
  booktitle={ICLR Workshop},
  year={2023}
}
```

```
@article{li2022backdoor,
  title={Backdoor learning: A survey},
  author={Li, Yiming and Jiang, Yong and Li, Zhifeng and Xia, Shu-Tao},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022}
}
```
