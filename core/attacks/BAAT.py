"""
This is the implementation of BAAT [1].
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

Reference:
[1] Towards Sample-Specific Backdoor Attack With Clean Labels via Attribute Trigger. TDSC, 2025.
"""

import abc
import os
import random
import re
import shutil
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from ArtFlow.glow_adain import Glow
from criteria.parse_related_loss import average_lab_color_loss
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm

from .base import *

sys.path.insert(0, "encoder4editing")
from models.psp import pSp

sys.path.insert(0, "mapper")
from datasets.latents_dataset_inference import LatentsDatasetInference
from hairclip_mapper import HairCLIPMapper


class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target


class AttributeEditor(abc.ABC):
    """
    An abstract base class for image attribute editing.
    This class provides a standardized workflow for loading models, editing single images,
    and processing entire directories of images. Subclasses must implement the specific
    logic for model loading, attribute preparation, and the core image editing function.
    """

    def __init__(self, checkpoint_dir: str, attribute_description: Any, size: int = 256):
        self.checkpoint_dir = Path(checkpoint_dir)
        if not self.checkpoint_dir.is_dir():
            raise ValueError(f"Checkpoint directory {self.checkpoint_dir} is invalid.")

        self.attribute_description = attribute_description
        self.size = size

        self._load_checkpoint()
        self.precomputed_latents = self._prepare_attribute_latents(self.attribute_description)

    @abc.abstractmethod
    def _load_checkpoint(self) -> None:
        """Load all necessary model checkpoints from the checkpoint directory."""
        pass

    @abc.abstractmethod
    def _prepare_attribute_latents(self, attribute_description: Any) -> Any:
        """
        Pre-process the attribute description (e.g., text or image) into a latent representation.
        This is pre-computed for efficiency when processing a directory.
        """
        pass

    @abc.abstractmethod
    def edit_image(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        attribute_description: Optional[Any] = None,
        save_ext: str = ".jpg",
    ) -> Path:
        """
        Edit a single image based on the provided attribute description and save it.
        If `attribute_description` is None, the pre-computed latents from initialization are used.
        Returns the path to the saved image.
        """
        pass

    @abc.abstractmethod
    def _get_attribute_identifier(self, attribute_description: Any) -> str:
        """Return a sanitized, file-safe string identifier for the given attribute."""
        pass

    def _sanitize_string(self, text: str) -> str:
        """A utility function to create a file-safe string from text."""
        text = text.lower()
        text = re.sub(r"\s+", "-", text)
        text = re.sub(r"[^a-z0-9_-]", "", text)
        return text

    def process_directory(
        self,
        content_dir: Union[str, Path],
        output_dir: Union[str, Path],
        target_label: Optional[Union[str, int]] = None,
        poisoned_rate: float = 1.0,
    ) -> Path:
        """
        Processes a directory of images, applying the attribute edit to a subset of them.
        This is the "Template Method" that uses the abstract methods implemented by subclasses.
        """
        content_dir = Path(content_dir)
        output_dir = Path(output_dir)
        if not content_dir.is_dir():
            raise ValueError(f"Content directory {content_dir} is invalid.")
        attribute_id = self._get_attribute_identifier(self.attribute_description)
        final_output_dir = output_dir / f"{content_dir.stem}_BAAT_{attribute_id}_{target_label}_{poisoned_rate}"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output will be saved to: {final_output_dir}")
        all_labels = sorted([d.name for d in content_dir.iterdir() if d.is_dir()])

        target_label_list = []
        if target_label is None:
            target_label_list = all_labels
        elif isinstance(target_label, str):
            if target_label.lower() == "all":
                target_label_list = all_labels
            elif target_label in all_labels:
                target_label_list = [target_label]
            else:
                raise ValueError(f"Target label '{target_label}' not found: {all_labels}")
        elif isinstance(target_label, int) and 0 <= target_label < len(all_labels):
            target_label_list = [all_labels[target_label]]
        else:
            raise ValueError(f"Invalid target_label: {target_label}.")
        with tqdm(all_labels, desc="Processing Labels") as pbar:
            for label in pbar:
                label_input_dir = content_dir / label
                label_output_dir = final_output_dir / label
                label_output_dir.mkdir(exist_ok=True)

                content_images = [f for f in label_input_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
                if label in target_label_list:
                    num_to_poison = int(poisoned_rate * len(content_images))
                    random.shuffle(content_images)
                    images_to_edit = content_images[:num_to_poison]
                    images_to_copy = content_images[num_to_poison:]

                    for image_path in images_to_edit:
                        self.edit_image(image_path, label_output_dir)
                    for image_path in images_to_copy:
                        shutil.copy2(image_path, label_output_dir)

                    pbar.set_description(
                        f"Processing {label} ({len(images_to_edit)} edited, {len(images_to_copy)} copied)"
                    )
                else:
                    for image_path in content_images:
                        shutil.copy2(image_path, label_output_dir)
                    pbar.set_description(f"Processing {label} ({len(content_images)} copied)")

        print(f"Processed dataset saved in {final_output_dir}")
        return final_output_dir


class HairEditor(AttributeEditor):
    def __init__(
        self,
        checkpoint_dir: str,
        attribute_description: Dict[str, str],
        size: int = 256,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.editing_type = attribute_description.get("editing_type", "both")

        self.e4e_model_path = Path(checkpoint_dir) / "e4e_ffhq_encode.pt"
        self.hairclip_model_path = Path(checkpoint_dir) / "hairclip.pt"
        self.parsenet_path = Path(checkpoint_dir) / "parsenet.pth"
        self.stylegan_path = Path(checkpoint_dir) / "stylegan2-ffhq-config-f.pt"
        self.ir_se50_path = Path(checkpoint_dir) / "model_ir_se50.pth"
        super().__init__(checkpoint_dir, attribute_description, size)

        self.img_transforms = transforms.Compose(
            [
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def _load_checkpoint(self) -> None:
        """Load all necessary model checkpoints from the checkpoint directory."""
        if not self.e4e_model_path.is_file():
            raise FileNotFoundError(f"e4e model not found at {self.e4e_model_path}")
        e4e_ckpt = torch.load(self.e4e_model_path)

        e4e_opts_dict = e4e_ckpt["opts"]
        e4e_opts_dict["checkpoint_path"] = self.e4e_model_path  # Update the path
        self.e4e_opts = Namespace(**e4e_opts_dict)

        self.e4e_net = pSp(self.e4e_opts).to(self.device).eval()
        print(f"=> Loaded e4e model '{self.e4e_model_path}'")

        if not self.hairclip_model_path.is_file():
            raise FileNotFoundError(f"HairCLIP model not found at {self.hairclip_model_path}")
        hairclip_ckpt = torch.load(self.hairclip_model_path)
        hairclip_opts_dict = hairclip_ckpt["opts"]
        hairclip_opts_dict.update(
            {
                "input_type": "text",
                "editing_type": self.editing_type,
                "checkpoint_path": self.hairclip_model_path,
                "parsenet_weights": self.parsenet_path,
                "stylegan_weights": self.stylegan_path,
                "ir_se50_weights": self.ir_se50_path,
            }
        )
        self.opts = Namespace(**hairclip_opts_dict)

        required_paths = [self.parsenet_path, self.stylegan_path, self.ir_se50_path]
        if not all(p.is_file() for p in required_paths):
            raise FileNotFoundError(f"One or more required weight files are missing from {required_paths}")

        self.hairclip_net = HairCLIPMapper(self.opts).to(self.device).eval()
        self.average_color_loss = average_lab_color_loss.AvgLabLoss(self.opts).to(self.device).eval()
        print(f"=> Loaded HairCLIP model '{self.hairclip_model_path}'")

    def _prepare_attribute_latents(self, attribute_description: Dict[str, str]) -> Dict[str, torch.Tensor]:
        hairstyle_desc = attribute_description["hairstyle_description"]
        color_desc = attribute_description["color_description"]
        print(f"Preparing latents for hairstyle: '{hairstyle_desc}', color: '{color_desc}'")
        tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8")
        try:
            tmp_file.write(hairstyle_desc)
            tmp_file.close()
            # This part is highly specific to HairCLIP's internal dataset logic
            self.opts.hairstyle_description = tmp_file.name
            self.opts.color_description = color_desc
            temp_dataset = LatentsDatasetInference(latents=torch.zeros(1, 18, 512), opts=self.opts)
            (_, h_list, c_list, _, h_tensor_list, c_tensor_list) = temp_dataset[0]
            latents = {
                "hairstyle_text": h_list[0].to(self.device),
                "color_text": c_list[0].to(self.device),
                "hairstyle_tensor": h_tensor_list[0].to(self.device),
                "color_tensor": c_tensor_list[0].to(self.device),
            }
        finally:
            os.remove(tmp_file.name)
        return latents

    def _get_attribute_identifier(self, attribute_description: Dict[str, str]) -> str:
        safe_style = self._sanitize_string(attribute_description["hairstyle_description"])
        safe_color = self._sanitize_string(attribute_description["color_description"])
        return f"{safe_style}_{safe_color}"

    def edit_image(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        attribute_description: Optional[Dict[str, str]] = None,
        save_ext: str = ".jpg",
    ) -> Path:
        input_path, output_dir = Path(input_path), Path(output_dir)

        if attribute_description:
            latents = self._prepare_attribute_latents(attribute_description)
        else:
            latents = self.precomputed_latents

        input_image = Image.open(input_path).convert("RGB")
        transformed_image = self.img_transforms(input_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, w = self.e4e_net(transformed_image, randomize_noise=False, return_latents=True)

            h_tensor = latents["hairstyle_tensor"].unsqueeze(0)
            c_tensor = latents["color_tensor"].unsqueeze(0)
            h_text = latents["hairstyle_text"].unsqueeze(0)
            c_text = latents["color_text"].unsqueeze(0)

            h_tensor_masked = (
                h_tensor * self.average_color_loss.gen_hair_mask(h_tensor)
                if h_tensor.shape[1] > 1
                else torch.zeros(1, 1, device=self.device)
            )
            c_tensor_masked = (
                c_tensor * self.average_color_loss.gen_hair_mask(c_tensor)
                if c_tensor.shape[1] > 1
                else torch.zeros(1, 1, device=self.device)
            )

            w_hat = w + 0.1 * self.hairclip_net.mapper(w, h_text, c_text, h_tensor_masked, c_tensor_masked)
            x_hat, _ = self.hairclip_net.decoder(
                [w_hat], input_is_latent=True, return_latents=False, randomize_noise=False, truncation=1
            )

        output_path = output_dir / f"{input_path.stem}_edited{save_ext}"
        save_image(x_hat, output_path, normalize=True, range=(-1, 1))
        return output_path


class StyleEditor(AttributeEditor):
    def __init__(
        self,
        checkpoint_dir: str,
        attribute_description: Union[str, Path],
        decoder_filename: str = "glow.pth",
        n_flow: int = 8,
        n_block: int = 2,
        affine: bool = False,
        no_lu: bool = False,
        size: int = 224,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.decoder_path = Path(checkpoint_dir) / decoder_filename
        self.glow = Glow(3, n_flow, n_block, affine=affine, conv_lu=not no_lu).to(self.device)

        super().__init__(checkpoint_dir, attribute_description, size)

    def _load_checkpoint(self) -> None:
        if not self.decoder_path.is_file():
            raise FileNotFoundError(f"Decoder checkpoint not found at {self.decoder_path}")
        checkpoint = torch.load(self.decoder_path)
        self.glow.load_state_dict(checkpoint["state_dict"])
        self.glow.eval()
        print(f"=> Loaded Glow model '{self.decoder_path}'")

    def _prepare_attribute_latents(self, attribute_description: Union[str, Path]) -> torch.Tensor:
        style_image_path = Path(attribute_description)
        if not style_image_path.is_file():
            raise FileNotFoundError(f"Style image not found at {style_image_path}")
        print(f"Preparing latents for style image: '{style_image_path.name}'")

        with torch.no_grad():
            style_image = Image.open(style_image_path).convert("RGB")
            transform = self._get_image_transform(np.array(style_image), self.size)
            style_tensor = transform(style_image).unsqueeze(0).to(self.device)
            return self.glow(style_tensor, forward=True)  # returns z_s

    def _get_attribute_identifier(self, attribute_description: Union[str, Path]) -> str:
        return Path(attribute_description).stem

    def _get_image_transform(self, img: np.ndarray, size: int) -> transforms.Compose:
        h, w, _ = img.shape
        if h < w:
            new_h, new_w = size, int(w / h * size)
        else:
            new_h, new_w = int(h / w * size), size
        return transforms.Compose(
            [transforms.Resize((int(new_h // 4 * 4), int(new_w // 4 * 4))), transforms.ToTensor()]
        )

    def edit_image(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        attribute_description: Optional[Union[str, Path]] = None,
        save_ext: str = ".jpg",
    ) -> Path:
        input_path, output_dir = Path(input_path), Path(output_dir)

        with torch.no_grad():
            if attribute_description:
                z_s = self._prepare_attribute_latents(attribute_description)
                style_id = Path(attribute_description).stem
            else:
                z_s = self.precomputed_latents
                style_id = self._get_attribute_identifier(self.attribute_description)

            content_image = Image.open(input_path).convert("RGB")
            transform = self._get_image_transform(np.array(content_image), self.size)
            content_tensor = transform(content_image).unsqueeze(0).to(self.device)

            z_c = self.glow(content_tensor, forward=True)
            output = self.glow(z_c, forward=False, style=z_s).cpu()

        output_path = output_dir / f"{input_path.stem}_stylized_{style_id}{save_ext}"
        save_image(output, output_path)
        return output_path


def _create_poisoned_dataset_from_folder(
    benign_dataset: DatasetFolder,
    editor: AttributeEditor,
    target_label: Union[str, int],
    poisoned_rate: float,
    transform: Optional[Any] = None,
    target_transform: Optional[Any] = None,
) -> ImageFolder:
    """A generic function to create a poisoned dataset using any AttributeEditor."""
    dataset_path = Path(benign_dataset.root)
    output_dir = dataset_path.parent / "poisoned"
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Creating poisoned dataset at: {output_dir}")
    final_save_dir = editor.process_directory(
        content_dir=dataset_path,
        output_dir=output_dir,
        target_label=target_label,
        poisoned_rate=poisoned_rate,
    )
    print(f"Poisoned dataset generation complete. Location: {final_save_dir}")
    return ImageFolder(
        root=final_save_dir,
        transform=transform or transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()]),
        target_transform=target_transform,
    )


def GetPoisonedVggFace2Dataset(
    benign_dataset: DatasetFolder,
    target_label: Union[str, int],
    poisoned_rate: float,
    transform: Any = None,
    target_transform: Any = None,
    other_args: Optional[Dict] = None,
) -> ImageFolder:
    args = other_args or {}
    editor = HairEditor(
        checkpoint_dir=args.get("checkpoint_dir", "pretrained_models"),
        attribute_description={
            "hairstyle_description": args.get("hairstyle_description", "hi-top fade hairstyle"),
            "color_description": args.get("color_description", "purple"),
            "editing_type": args.get("editing_type", "both"),
        },
    )
    return _create_poisoned_dataset_from_folder(
        benign_dataset, editor, target_label, poisoned_rate, transform, target_transform
    )


def GetPoisonedImageNetDataset(
    benign_dataset: DatasetFolder,
    target_label: Union[str, int],
    poisoned_rate: float,
    transform: Any = None,
    target_transform: Any = None,
    other_args: Optional[Dict] = None,
) -> ImageFolder:
    args = other_args or {}
    editor = StyleEditor(
        checkpoint_dir=args.get("checkpoint_dir", "ArtFlow"),
        attribute_description=args.get("style_image_path", "ArtFlow/style.jpg"),
        decoder_filename=args.get("decoder_filename", "glow.pth"),
    )
    return _create_poisoned_dataset_from_folder(
        benign_dataset, editor, target_label, poisoned_rate, transform, target_transform
    )


def CreatePoisonedDataset(
    benign_dataset,
    target_label,
    poisoned_rate,
    image_type="nature",
    transform=None,
    target_transform=None,
    other_args=None,
):
    class_name = type(benign_dataset)
    if issubclass(class_name, DatasetFolder):
        if image_type == "nature":
            return GetPoisonedImageNetDataset(
                benign_dataset, target_label, poisoned_rate, transform, target_transform, other_args
            )
        elif image_type == "face":
            return GetPoisonedVggFace2Dataset(
                benign_dataset, target_label, poisoned_rate, transform, target_transform, other_args
            )
        else:
            raise ValueError(f"Unsupported image type: {image_type}. Supported types are 'nature' and 'face'.")
    else:
        raise NotImplementedError("Poisoned dataset creation for this dataset type is not implemented.")


class BAAT(Base):
    """
    Construct poisoned datasets with BAAT method.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        y_target (int): N-to-1 attack target label.
        poisoned_rate (float): The rate of poisoned samples in the training dataset, should be in (0, 1].
        transform_train (callable, optional): A function/transform of training dataset that takes in an PIL image
            and returns a transformed version.
        transform_test (callable, optional): A function/transform of test dataset that takes in an PIL image
            and returns a transformed version.
        other_args (dict, optional): Other arguments for poisoned dataset generation.
            For face dataset, it may include:
                - checkpoint_dir (str): Directory containing pretrained models for HairCLIP and e4e.
                - hairstyle_description (str): Description of the hairstyle to be used as trigger.
                - color_description (str): Description of the hair color to be used as trigger.
                - editing_type (str): Type of editing to perform. Default is 'both'.
            For nature dataset, it may include:
                - checkpoint_dir (str): Directory containing pretrained models for Glow.
                - style_image_path (str): File path to the style image to be used as trigger
        image_type (str): Type of images in the dataset, either 'nature' or 'face'.
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(
        self,
        train_dataset,
        test_dataset,
        model,
        loss,
        y_target,
        poisoned_rate,
        image_type="nature",
        transform_train=None,
        transform_test=None,
        other_args=None,
        schedule=None,
        seed=0,
        deterministic=False,
    ):

        super(BAAT, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic,
        )

        self.poisoned_train_dataset = CreatePoisonedDataset(
            train_dataset, y_target, poisoned_rate, image_type, transform=transform_train, other_args=other_args
        )

        self.poisoned_test_dataset = CreatePoisonedDataset(
            test_dataset,
            "ALL",
            1.0,
            image_type,
            transform=transform_test,
            target_transform=ModifyTarget(y_target),
            other_args=other_args,
        )
