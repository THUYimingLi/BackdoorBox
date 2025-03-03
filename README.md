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
- **Consistency**: Instead of directly collecting and combining the original codes from each method, we re-implement all methods in a unified manner. Specifically, variables having the same function have a consistent name. Similar methods inherit the same base class for further development, have a unified workflow, and have the same core sub-functions (*e.g.*, `get_model()`).
- **Simplicity**: We provide code examples for each implemented backdoor attack and defense to explain how to use them, the definitions and default settings of all required attributes, and the necessary code comments. Users can easily implement and develop our toolbox.
- **Flexibility**: We allow users to easily obtain important intermediate outputs and components of each method (*e.g.*, poisoned dataset and attacked/repaired model), use their local samples and model structure for attacks and defenses, and interact with their local codes. The attack and defense modules can be used jointly or separately. You can also use your local dataset via `torchvision.datasets.DatasetFolder`. (See examples of using the GTSRB dataset)
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
| [BATT](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/BATT.py)                |       BATT: Backdoor Attack with Transformation-based Triggers. [ICASSP, 2023](https://arxiv.org/pdf/2211.01806).      | poison-only, invisible, physical |                                                                 |
| [AdaptivePatch](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/AdaptivePatch.py)                |       Revisiting the Assumption of Latent Separability for Backdoor Defenses. [ICLR, 2023](https://openreview.net/pdf?id=_wSHsgrVali).      | poison-only | adaptive attack                                                 |

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
|          [SCALE-UP](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/SCALE-UP.py) | SCALE-UP: An Efficient Black-box Input-level Backdoor Detection via Analyzing Scaled Prediction Consistency. |[ICLR, 2023](https://arxiv.org/abs/2302.03251). |  Input-level Backdoor Detection|  black-box online detection|
|          [IBD-PSC](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/IBD-PSC.py) | IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling Consistency. [ICML, 2024](https://arxiv.org/abs/2405.09786). | Input-level Backdoor Detection |  simple yet effective, safeguarded by theoretical analysis|
|          [REFINE](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/REFINE.py) | REFINE: Inversion-Free Backdoor Defense via Model Reprogramming. [ICLR, 2025](https://openreview.net/pdf?id=4IYdCws9fc). | Sample Pre-processing |  SOTA pre-processing-based defense|

## Methods Under Development
- DBD
- SS
- Neural Cleanse
- DP
- CutMix
- AEVA
- STRIP


## Attack & Defense Benchmark
The benchmark is coming soon.

## Contributors

| Organization        | Contributors                                                 |
| ------------------- | ------------------------------------------------------------ |
| Tsinghua University | [Yiming Li](http://liyiming.tech/), [Mengxi Ya](https://github.com/yamengxi), [Guanhao Gan](https://github.com/GuanhaoGan), [Kuofeng Gao](https://github.com/KuofengGao), [Xin Yan](https://scholar.google.com/citations?hl=zh-CN&user=08WTTPMAAAAJ), [Jia Xu](https://www.researchgate.net/profile/Xu-Jia-10), [Tong Xu](https://github.com/spicy1007), [Sheng Yang](https://github.com/20000yshust), [Haoxiang Zhong](https://scholar.google.com/citations?user=VOw9qmYAAAAJ&hl=zh-CN&oi=ao), [Linghui Zhu](https://github.com/zlh-thu)|
| Tencent Security Zhuque Lab | [Yang Bai](https://scholar.google.com/citations?user=wBH_Q1gAAAAJ&hl=zh-CN) |
| ShanghaiTech University | [Zhe Zhao](https://s3l.shanghaitech.edu.cn/people/zhezhao/) |
| Harbin Institute of Technology, Shenzhen| [Linshan Hou](https://scholar.google.com/citations?user=uHVNhf8AAAAJ&hl=zh-CN&oi=ao) |
| Zhejiang University | [Yukun Chen](https://github.com/WhitolfChen) |

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

```S
@article{li2022backdoor,
  title={Backdoor learning: A survey},
  author={Li, Yiming and Jiang, Yong and Li, Zhifeng and Xia, Shu-Tao},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022}
}
```

## Repository Structure

```
.
|-- core
|   |-- attacks           # include Classes of different attack methods     
|   |-- defenses          # include Classes of different defense methods
|   |-- experiments       # the log and .pth files will be saved in this folder
|   `-- utils             # include some utility tools or helper functions
|       `-- torchattacks  # include PGD attack borrowed from torchattacks     
`-- tests                 # include example codes of each implemented method
```


## Environment configuration

You can run the following script to configure the necessary environment.

```
git clone https://github.com/THUYimingLi/BackdoorBox.git
cd BackdoorBox
conda create -n backdoorbox python=3.8
conda activate backdoorbox
pip install -r requirements.txt
```

## Quick Start

### Attack

#### BadNets

This is a example for BadNets.

Before getting started, you need to obtain the corresponding dataset. We use the [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) as an example here.You should download the CIFAR-10 dataset and place it in the appropriate directory:

```
./data/cifar10
```

We provided an example programme `example.py`, you can run the following script to use our example.
```
python example.py
```
The log of the programme will be stored in `./experiment/Train_benign_DatasetFolder_CIFAR10_*/log.txt` whlie * refers to the date today. This folder also contains the model data(Distinguish between different epochs) we saved during training.

##### Change Triggers

If you want to change the trigger for BadNets, you should do some modifications in `example.py`

```python
pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
weight = torch.zeros((1, 32, 32), dtype=torch.float32)
weight[0, -3:, -3:] = 1.0
```
The above codes are our triggers generator. Since the CIFAR-10 dataset uses 32x32 images, we use a 32x32 matrix for trigger. The above codes show that we use a 3x3 white matrix as a trigger, placed in the lower right corner of the image.

You could change the place, color and size for our trigger.

##### Change Model

By default, the project uses the ResNet-18 model. You can easily replace it with other models as needed. To do this, modify the following line in your code:

```
model = core.models.ResNet(18)
```

You could change `ResNet(18)` to any other model supported by the framework.

##### Change Hyperparameterizations

According to our example, you could change almost every hyperparameterization in our programm.For example:

- **Poisoned Rate**: Adjust the percentage of poisoned samples in the dataset.

	```python
	poisoned_rate=0.05  # 5% of the dataset will be poisoned
	```

- **Target Label**: Change the target label for the poisoned samples.

	```python
	y_target=1  # Poisoned samples will be misclassified as label 1
	```

- **Training Schedule**: Modify the learning rate, batch size, number of epochs, etc.

	```python
	schedule = {
	    'lr': 0.1,
	    'batch_size': 128,
	    'epochs': 200,
	    ...
	}
	```

#### Blended

This is a example for Blended.

Before getting started, you need to obtain the corresponding dataset. We use the [ImageNet50 Dataset](https://www.image-net.org/) as an example here.You should download the ImageNet50 dataset and place it in the appropriate directory:

```
./data/ImageNet50 
```

We provided an example programm `example2.py`, you can run the following script to use our example.

```
python example2.py
```

The log of the programm will be stored in `./experiment/Train_benign_DatasetFolder_ImageNet50_*/log.txt`. Whlie * refers to the date today. This folder also contains the model data(Distinguish between different epochs) we saved during training.

##### Change Triggers

If you want to change the trigger for Blended attack, you should do some modifications in `example.py`

```python
pattern = torch.zeros((3, 224, 224), dtype=torch.uint8)
pattern[:, -3:, -3:] = 255
weight = torch.zeros((3, 224, 224), dtype=torch.float32)
weight[:, -3:, -3:] = 0.2
```

The above code generates the trigger. Since the ImageNet50 dataset uses 224x224 images, we use a 224x224 matrix for the trigger. The above codes show that we use a 3x3 white matrix as a trigger, placed in the lower right corner of the image.

You could change the place, color and size for our trigger.

##### Change Model

By default, the project uses the ResNet-18 model. You can easily replace it with other models as needed. To do this, modify the following line in your code:

```
blended.model = core.models.ResNet(18, num_classes=50)
```

You could change `ResNet(18)` to any other model supported by the framework.

##### Change Hyperparameterizations

According to our example, you could change almost every hyperparameterization in our programm.For example:

- **Poisoned Rate**: Adjust the percentage of poisoned samples in the dataset.

	```python
	poisoned_rate=0.1  # 10% of the dataset will be poisoned
	```

- **Target Label**: Change the target label for the poisoned samples.

	```python
	y_target=1  # Poisoned samples will be misclassified as label 1
	```

- **Training Schedule**: Modify the learning rate, batch size, number of epochs, etc.

	```python
	schedule = {
	    'lr': 0.1,
	    'batch_size': 128,
	    'epochs': 100,
	    ...
	}
	```

### Defense

#### ShrinkPad

This is an example for ShrinkPad defense.

ShrinkPad is a sample pre-processing defense that works by shrinking the image and then padding it back to its original size. This effectively removes backdoor triggers that are typically placed at the corners or edges of images.

Before getting started, you need to obtain the corresponding dataset and a poisoned model. We use the [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) as an example here. You should download the CIFAR-10 dataset and place it in the appropriate directory:

```python
./data/cifar10
```

We provided an example program `defense1.py`, you can run the following script to use our example.

```python
python defense1.py
```

The log of the program will be stored in `./experiments/CIFAR10_test_*/log.txt` while * refers to the date today. This folder also contains the evaluation results of the defense.

##### Defense Parameters

If you want to configure the ShrinkPad defense, you can modify the following parameters in the `defense1.py` file:

```python
shrinkpad = core.ShrinkPad(size_map=32, pad=4)
```

- `size_map`: The target size to which the image will be resized.
- `pad`: The padding size to be applied after shrinking.

These parameters determine how much of the image edges (where triggers are often placed) will be removed or modified during the defense process.

##### Test Against BadNet Attack

Our example specifically targets a model that was poisoned using BadNet attack with a 3x3 white square trigger in the bottom-right corner of the image:

```python
pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
```

The ShrinkPad defense is particularly effective against such attacks because it modifies the area where the trigger is located.

##### Evaluation Model

By default, the project uses the ResNet-18 model. The poisoned model is loaded from a specified checkpoint:

```python
model = core.models.ResNet(18)
```

You can replace this with other model architectures supported by the framework.

##### Test Configuration

You can modify the testing configuration by changing the following schedule dictionary:

```python
schedule = {
    'test_model': './experiments/train_poisoned_DatasetFolder-CIFAR10_2025-02-24_18:20:13/ckpt_epoch_50.pth',  # Path to your model file
    'save_dir': './experiments',  # Directory to save results
    'CUDA_VISIBLE_DEVICES': '0',  # Which GPU to use
    'GPU_num': 1,
    'experiment_name': 'CIFAR10_test',  # Experiment name
    'device': 'GPU',  # Use GPU for testing
    'metric': 'ASR_NoTarget',  # Testing metric method
    'y_target': 0,  # Target class (if applicable)
    'batch_size': 64,  # Test batch size
    'num_workers': 4,  # Number of data loading worker threads
}
```

##### Device Settings

You can specify whether to use CPU or GPU for testing, and which GPU to use if multiple GPUs are available:

```python
'device': 'GPU',
'CUDA_VISIBLE_DEVICES': '0',  # Use all available GPUs
'GPU_num': 1  # Number of GPUs to use
```

**Important Note on GPU Indexing**:

- When setting `'CUDA_VISIBLE_DEVICES': '0'`, the system will use **all available GPUs**.
- To use only the first GPU, set `'CUDA_VISIBLE_DEVICES': '1'`.
- For subsequent GPUs, the numbering continues sequentially (e.g., '2' for the second GPU, '3' for the third, and so on).

From the logs, you can view the following GPU-related information:

- Visible GPUs:
  - The log shows which GPUs are visible to the program.
- Selected GPUs:
  - The log also shows which GPUs are actually being used.
- Device Configuration:
  - The log displays the device configuration used for testing.






