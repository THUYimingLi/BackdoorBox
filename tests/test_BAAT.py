"""
This is the test code of poisoned training on VGGFace2 and ImageNet, using dataset class of torchvision.datasets.DatasetFolder.
The attack method is BAAT.

Before using this code, please download the pretrained models.
- For face dataset, please download the [HairCLIP](https://github.com/wtybest/HairCLIP) and [e4e](https://github.com/omertov/encoder4editing) models. Please put the `encoder4editing` folder and `HairCLIP/mapper, HairCLIP/criteria` folders in the root directory of BackdoorBox.
  - Please follow the instructions in [HairCLIP](https://github.com/wtybest/HairCLIP) to download the pretrained HairCLIP model and put it in the `pretrained_models` folder.
- For nature dataset, please download the [ArtFLow](https://github.com/pkuanjie/ArtFlow) and put `ArtFlow/glow_adain.py` in the root directory of BackdoorBox. And we use the official [pretrained model](https://drive.google.com/file/d/1xusus0d8ONO-j5mMQXhXl5Gt9OOMOO0H/view?usp=drive_link) and put it in the `ArtFlow` folder. We also use the official [style image](https://github.com/pkuanjie/ArtFlow/blob/main/data/style/654d10cd803dcdc4469f6fccd236b8c9.jpg) as the trigger image, rename it as `style.jpg` and put it in the `ArtFlow` folder.

We also provide the datasets we used in the experiments, please download from the [Google Drive](https://drive.google.com/drive/folders/1p612Pn1IBiIHBulKbke9o2kuLWDr-8rL?usp=sharing) and put them in the `datasets` folder.
The final directory structure should be like this:

```text
BackdoorBox
├── core/
├── tests/
├── Attack_BadNets.py
├── Attack_Blended.py
├── attack.py
├── Defense_ShrinkPad.py
├── LICENSE
├── README.md
├── requirements.txt
├── datasets/
├── ArtFlow/
│   ├── glow_adain.py
│   ├── glow.pth
│   └── style.jpg
├── encoder4editing/
├── criteria/
├── mapper/
└── pretrained_models/
    ├── e4e_ffhq_encode.pt
    ├── hairclip.pt
    ├── model_ir_se50.pth
    ├── parsenet.pth
    ├── shape_predictor_68_face_landmarks.dat
    └── stylegan2-ffhq-config-f.pt
```

"""

import os
import random
import sys

import dlib
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize, ToTensor
from tqdm import tqdm

import core

sys.path.insert(0, "encoder4editing")
from encoder4editing.utils.alignment import align_face

# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_SELECTED_DEVICES = "0"
datasets_root_dir = "./datasets"


class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target


# ========== ResNet-18_ImageNet-100_BAAT ==========

transform_train = Compose([Resize((128, 128)), RandomHorizontalFlip(0.5), ToTensor()])

transform_test = Compose([Resize((128, 128)), ToTensor()])

benign_trainset = ImageFolder(root=os.path.join(datasets_root_dir, "benign_100", "train"), transform=transform_train)

benign_testset = ImageFolder(root=os.path.join(datasets_root_dir, "benign_100", "val"), transform=transform_test)

model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 100)

y_target = 1

schedule = {
    "device": "GPU",
    "CUDA_SELECTED_DEVICES": CUDA_SELECTED_DEVICES,
    "benign_training": False,
    "batch_size": 128,
    "num_workers": 4,
    "lr": 0.001,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "gamma": 0.1,
    "schedule": [15, 20],
    "epochs": 30,
    "log_iteration_interval": 100,
    "test_epoch_interval": 5,
    "save_epoch_interval": 5,
    "save_dir": "experiments",
    "experiment_name": "ResNet-18_ImageNet_BAAT",
}

baat = core.BAAT(
    train_dataset=benign_trainset,
    test_dataset=benign_testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=y_target,
    poisoned_rate=0.8,
    image_type="nature",
    transform_train=transform_train,
    transform_test=transform_test,
    other_args=None,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic,
)

poisoned_train_dataset, poisoned_test_dataset = baat.get_poisoned_dataset()

baat.train(schedule)
baat.test(schedule)

# ========== ResNet-18_VGGFace2_BAAT ==========


class VGGFace2:
    """
    A more efficient VGGFace2 dataset processing class.
    """

    def __init__(self, *data_roots):
        super().__init__()
        self.class_to_paths = {}

        for root in data_roots:
            if not os.path.isdir(root):
                print(f"Warning: Directory not found, skipping: {root}")
                continue

            for class_name in os.listdir(root):
                class_dir = os.path.join(root, class_name)
                if not os.path.isdir(class_dir):
                    continue

                img_list = [os.path.join(class_dir, img_name) for img_name in os.listdir(class_dir)]

                if class_name in self.class_to_paths:
                    self.class_to_paths[class_name].extend(img_list)
                else:
                    self.class_to_paths[class_name] = img_list

        self.classes = sorted(list(self.class_to_paths.keys()))
        print(f"Found {len(self.classes)} unique classes across all directories.")

    def get_top_k(self, k):
        if not self.classes:
            return []

        class_nums = {name: len(paths) for name, paths in self.class_to_paths.items()}
        df = pd.DataFrame(list(class_nums.items()), columns=["class_name", "class_num"])
        df = df.sort_values(by=["class_num", "class_name"], ascending=[False, True])
        df = df.reset_index(drop=True)

        print(f"\n--- Top {k} Classes by Image Count (for verification) ---")
        print(df.head(k).to_string())
        print("----------------------------------------------------------\n")
        top_k_df = df.head(k)
        selected_classes = top_k_df["class_name"].tolist()

        print(
            f"Selected top {k} classes. The class with the most images has {top_k_df['class_num'].max()} images, the least has {top_k_df['class_num'].min()}."
        )
        return selected_classes


def preprocess_vggface2(
    train_dir="data/vggface2/train",
    test_dir="data/vggface2/test",
    target_dir="data/vggface2/benign",
    predictor_path="pretrained_models/shape_predictor_68_face_landmarks.dat",
    num_per_class=500,
    selected_class_num=20,
    resize_size=(256, 256),
    test_ratio=0.2,
):
    """
    Efficiently preprocess the VGGFace2 dataset and split it into training and validation sets.
    Workflow:
    1. Scan the train and test directories.
    2. Select the `selected_class_num` classes with the most images.
    3. According to `test_ratio`, save the processed images to `target_dir/train` and `target_dir/val`.
    """
    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"Dlib shape predictor not found at: {predictor_path}")

    train_base_dir = os.path.join(target_dir, "train")
    val_base_dir = os.path.join(target_dir, "val")
    os.makedirs(train_base_dir, exist_ok=True)
    os.makedirs(val_base_dir, exist_ok=True)
    print(f"Data will be saved to:\n  - Train: {train_base_dir}\n  - Val:   {val_base_dir}")

    print("\nScanning train and test directories...")
    dataset = VGGFace2(train_dir, test_dir)

    selected_classes = dataset.get_top_k(selected_class_num)
    selected_classes.sort()

    print("\nStarting face alignment and preprocessing...")
    print(f"Target classes ({len(selected_classes)}): {selected_classes}")

    predictor = dlib.shape_predictor(predictor_path)

    for class_name in tqdm(selected_classes, desc="Total Classes Progress"):
        count = 0

        dst_train_class_dir = os.path.join(train_base_dir, class_name)
        dst_val_class_dir = os.path.join(val_base_dir, class_name)
        os.makedirs(dst_train_class_dir, exist_ok=True)
        os.makedirs(dst_val_class_dir, exist_ok=True)

        image_paths = sorted(dataset.class_to_paths[class_name])

        num_val = int(num_per_class * test_ratio)
        num_train = num_per_class - num_val

        inner_pbar = tqdm(
            image_paths, desc=f"Processing '{class_name}' (Train:{num_train}, Val:{num_val})", leave=False
        )
        for image_path in inner_pbar:
            if count >= num_per_class:
                inner_pbar.close()
                break

            try:
                aligned_image = align_face(filepath=image_path, predictor=predictor)

                if aligned_image:
                    resized_image = aligned_image.resize(resize_size)
                    img_name = os.path.basename(image_path)

                    if count < num_train:
                        dst_pic_path = os.path.join(dst_train_class_dir, img_name)
                    else:
                        dst_pic_path = os.path.join(dst_val_class_dir, img_name)

                    resized_image.save(dst_pic_path)
                    count += 1

            except Exception as e:
                tqdm.write(f"Warning: Failed to process '{image_path}'. Reason: {e}.")

    print("\nAll selected classes have been processed and split into train/val sets.")


""" You can uncomment the following code to preprocess the VGGFace2 dataset and run BAAT on it."""
# preprocess_vggface2()

""" Make sure that vggface2 dataset is preprocessed before running the following code. """

transform_train = Compose([Resize((128, 128)), RandomHorizontalFlip(0.5), ToTensor()])

transform_test = Compose([Resize((128, 128)), ToTensor()])

benign_trainset = ImageFolder(root=os.path.join(datasets_root_dir, "vggface2", "train"), transform=transform_train)

benign_testset = ImageFolder(root=os.path.join(datasets_root_dir, "vggface2", "val"), transform=transform_test)

model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 20)

schedule = {
    "device": "GPU",
    "CUDA_SELECTED_DEVICES": CUDA_SELECTED_DEVICES,
    "benign_training": False,
    "batch_size": 64,
    "num_workers": 4,
    "lr": 0.001,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "gamma": 0.1,
    "schedule": [15, 20],
    "epochs": 30,
    "log_iteration_interval": 100,
    "test_epoch_interval": 5,
    "save_epoch_interval": 5,
    "save_dir": "experiments",
    "experiment_name": "ResNet-18_VGGFace2_BAAT",
}

y_target = 1

baat = core.BAAT(
    train_dataset=benign_trainset,
    test_dataset=benign_testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=y_target,
    poisoned_rate=0.8,
    image_type="face",
    transform_train=transform_train,
    transform_test=transform_test,
    other_args=None,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic,
)

poisoned_train_dataset, poisoned_test_dataset = baat.get_poisoned_dataset()

baat.train(schedule)
baat.test(schedule)
