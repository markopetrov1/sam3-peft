"""
Remote Sensing Segmentation Datasets for SAM LoRA Fine-tuning.

Supports ISPRS Potsdam and Vaihingen in MMSeg layout.
Returns (image, label) tensors ready for SAM input.

Usage:
    from datasets import PotsdamDataset, VaihingenDataset, create_dataset

    train_ds = PotsdamDataset(root="/data/potsdam_mmseg", split="train", augment=True)
    val_ds   = VaihingenDataset(root="/data/vaihingen_mmseg", split="val")
    ds       = create_dataset("potsdam", root="/data/potsdam_mmseg", split="train")
"""

import os
import glob
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# =============================================================================
# Base Dataset
# =============================================================================

class SegmentationDataset(Dataset):
    """
    Base dataset for remote sensing segmentation with SAM LoRA.

    Reads MMSeg-format datasets:
        root/
          img_dir/{split}/*.png
          ann_dir/{split}/*.png

    Annotation masks are single-channel PNGs where pixel values = class IDs.
    Subclasses define the class mapping via ID2LABEL, IGNORE_INDEX, NUM_CLASSES.

    Returns:
        dict with:
          "image": [3, H, W] float32 tensor in [0, 255] range (SAM expected input)
          "label": [H, W]   int64   tensor with class IDs
    """

    DATASET_NAME: str = "base"
    ID2LABEL: Dict[int, str] = {}
    IGNORE_INDEX: int = 0
    NUM_CLASSES: int = 0

    IMG_EXTS = (
        "*.png", "*.PNG",
        "*.tif", "*.TIF", "*.tiff", "*.TIFF",
        "*.jpg", "*.JPG", "*.jpeg", "*.JPEG",
    )

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 1024,
        img_subdir: str = "img_dir",
        ann_subdir: str = "ann_dir",
        augment: bool = False,
        exclude_classes: Optional[List[int]] = None,
    ):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.augment = augment and ("train" in split)
        self.exclude_classes = set(exclude_classes) if exclude_classes else set()

        self.img_dir = os.path.join(root, img_subdir, split)
        self.ann_dir = os.path.join(root, ann_subdir, split)

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.isdir(self.ann_dir):
            raise FileNotFoundError(f"Annotation directory not found: {self.ann_dir}")

        img_paths: List[str] = []
        for ext in self.IMG_EXTS:
            img_paths.extend(glob.glob(os.path.join(self.img_dir, ext)))
        img_paths = sorted(img_paths)

        if not img_paths:
            raise RuntimeError(f"No images found in {self.img_dir}")

        self.pairs: List[Tuple[str, str]] = []
        missing = 0
        for img_path in img_paths:
            basename = os.path.basename(img_path)
            ann_path = os.path.join(self.ann_dir, basename)
            if os.path.exists(ann_path):
                self.pairs.append((img_path, ann_path))
            else:
                missing += 1

        if not self.pairs:
            raise RuntimeError(
                f"No matching image-annotation pairs.\n"
                f"  img_dir: {self.img_dir}\n"
                f"  ann_dir: {self.ann_dir}\n"
                f"  images found: {len(img_paths)}"
            )

        self.active_classes: Dict[int, str] = {
            k: v for k, v in self.ID2LABEL.items()
            if k not in self.exclude_classes
        }

        print(f"[{self.DATASET_NAME}] {split}: {len(self.pairs)} samples, "
              f"image_size={image_size}, augment={self.augment}")
        if missing > 0:
            print(f"[{self.DATASET_NAME}] WARNING: {missing} images had no "
                  f"matching annotation (skipped)")

    def __len__(self) -> int:
        return len(self.pairs)

    # -----------------------------------------------------------------
    # Augmentation (applied jointly to image + mask)
    # -----------------------------------------------------------------

    def _augment(
        self, image: PILImage.Image, mask: np.ndarray
    ) -> Tuple[PILImage.Image, np.ndarray]:
        mask_pil = PILImage.fromarray(mask)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask_pil = TF.hflip(mask_pil)

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask_pil = TF.vflip(mask_pil)

        k = random.choice([0, 1, 2, 3])
        if k > 0:
            angle = k * 90
            image = TF.rotate(image, angle, expand=False,
                              interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            mask_pil = TF.rotate(mask_pil, angle, expand=False,
                                 interpolation=TF.InterpolationMode.NEAREST,
                                 fill=self.IGNORE_INDEX)

        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))
        if random.random() > 0.5:
            image = TF.adjust_saturation(image, random.uniform(0.85, 1.15))

        return image, np.array(mask_pil)

    # -----------------------------------------------------------------
    # __getitem__
    # -----------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, ann_path = self.pairs[idx]

        pil_image = PILImage.open(img_path).convert("RGB")
        mask_np = np.array(PILImage.open(ann_path))

        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

        if self.augment:
            pil_image, mask_np = self._augment(pil_image, mask_np)

        pil_image = pil_image.resize(
            (self.image_size, self.image_size), PILImage.BILINEAR
        )
        mask_np = np.array(
            PILImage.fromarray(mask_np).resize(
                (self.image_size, self.image_size), PILImage.NEAREST
            )
        )

        # [3, H, W] float32 in [0, 255] — SAM normalises internally
        image_tensor = torch.from_numpy(
            np.array(pil_image, dtype=np.float32)
        ).permute(2, 0, 1)

        label_tensor = torch.from_numpy(mask_np.astype(np.int64))

        return {"image": image_tensor, "label": label_tensor}


# =============================================================================
# Potsdam
# =============================================================================

class PotsdamDataset(SegmentationDataset):
    """
    ISPRS Potsdam — 6-class aerial semantic segmentation.

    RGB aerial imagery at ~5 cm GSD, tiled to 512x512 patches.

    Annotation encoding: pixel value = class ID
        0 = unlabeled (ignore), 1-6 = classes

    Dataset structure (MMSeg format):
        root/
          img_dir/train/*.png
          img_dir/val/*.png
          ann_dir/train/*.png
          ann_dir/val/*.png
    """

    DATASET_NAME = "Potsdam"
    IGNORE_INDEX = 0
    NUM_CLASSES = 6

    ID2LABEL: Dict[int, str] = {
        1: "impervious surface",
        2: "building",
        3: "low vegetation",
        4: "tree",
        5: "car",
        6: "clutter",
    }


# =============================================================================
# Vaihingen
# =============================================================================

class VaihingenDataset(SegmentationDataset):
    """
    ISPRS Vaihingen — 6-class aerial semantic segmentation.

    IRRG aerial imagery at ~9 cm GSD, tiled to 512x512 patches.

    Annotation encoding: pixel value = class ID
        0 = unlabeled (ignore), 1-6 = classes

    Dataset structure (MMSeg format):
        root/
          img_dir/train/*.png
          img_dir/val/*.png
          ann_dir/train/*.png
          ann_dir/val/*.png
    """

    DATASET_NAME = "Vaihingen"
    IGNORE_INDEX = 0
    NUM_CLASSES = 6

    ID2LABEL: Dict[int, str] = {
        1: "impervious surface",
        2: "building",
        3: "low vegetation",
        4: "tree",
        5: "car",
        6: "clutter",
    }


# =============================================================================
# Registry & Factory
# =============================================================================

DATASET_REGISTRY: Dict[str, type] = {
    "potsdam": PotsdamDataset,
    "vaihingen": VaihingenDataset,
}


def create_dataset(
    dataset_type: str,
    root: str,
    split: str = "train",
    image_size: int = 1024,
    augment: bool = False,
    exclude_classes: Optional[List[int]] = None,
) -> SegmentationDataset:
    """
    Factory function to instantiate a dataset by name.

    Args:
        dataset_type: "potsdam" or "vaihingen"
        root: Path to MMSeg-format dataset root.
        split: "train" or "val".
        image_size: Target resolution (default 1024 for SAM).
        augment: Enable augmentation (auto-disabled for val).
        exclude_classes: Class IDs to exclude.
    """
    dtype = dataset_type.lower()
    if dtype not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: '{dtype}'. Available: {available}")

    return DATASET_REGISTRY[dtype](
        root=root,
        split=split,
        image_size=image_size,
        augment=augment,
        exclude_classes=exclude_classes,
    )
