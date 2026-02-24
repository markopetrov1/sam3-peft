"""
Test-time augmentation (TTA) for semantic segmentation.

Provides geometric and color augmentations plus inverse transforms so that
predictions from augmented views are merged in the original image space.
Merge is done by averaging softmax logits (mean of probabilities) then argmax.

Usage:
    from utils.tta import run_tta, tta_d4_transform, tta_d4_plus_color

    preds = run_tta(model, images, image_size, multimask_output, tta_d4_transform(), ...)
"""

from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F


def _identity_aug(images: torch.Tensor) -> torch.Tensor:
    return images


def _identity_inv(masks: torch.Tensor) -> torch.Tensor:
    return masks


def _hflip_aug(images: torch.Tensor) -> torch.Tensor:
    return torch.flip(images, dims=(-1,))


def _hflip_inv(masks: torch.Tensor) -> torch.Tensor:
    return torch.flip(masks, dims=(-1,))


def _vflip_aug(images: torch.Tensor) -> torch.Tensor:
    return torch.flip(images, dims=(-2,))


def _vflip_inv(masks: torch.Tensor) -> torch.Tensor:
    return torch.flip(masks, dims=(-2,))


def _rot90_aug(images: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(images, k=k, dims=(-2, -1))


def _rot90_inv(masks: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(masks, k=(4 - k) % 4, dims=(-2, -1))


def _multiply_aug(images: torch.Tensor, factor: float) -> torch.Tensor:
    out = images * factor
    return out.clamp(0.0, 255.0)


# No inverse for color: we just add the prediction to the ensemble as-is.


def tta_d4_transform() -> List[Tuple[Callable, Callable]]:
    """
    D4: identity + 3 rotations (90, 180, 270) + horizontal flip + (hflip + 3 rotations).
    Total 8 views. Good balance of robustness vs compute.
    """
    return [
        (_identity_aug, _identity_inv),
        (lambda x: _rot90_aug(x, 1), lambda x: _rot90_inv(x, 1)),
        (lambda x: _rot90_aug(x, 2), lambda x: _rot90_inv(x, 2)),
        (lambda x: _rot90_aug(x, 3), lambda x: _rot90_inv(x, 3)),
        (_hflip_aug, _hflip_inv),
        (lambda x: _hflip_aug(_rot90_aug(x, 1)), lambda x: _rot90_inv(_hflip_inv(x), 1)),
        (lambda x: _hflip_aug(_rot90_aug(x, 2)), lambda x: _rot90_inv(_hflip_inv(x), 2)),
        (lambda x: _hflip_aug(_rot90_aug(x, 3)), lambda x: _rot90_inv(_hflip_inv(x), 3)),
    ]


def tta_d2_transform() -> List[Tuple[Callable, Callable]]:
    """D2: identity + horizontal flip. Fast, often helps."""
    return [
        (_identity_aug, _identity_inv),
        (_hflip_aug, _hflip_inv),
    ]


def tta_d4_plus_color(
    factors: Tuple[float, ...] = (0.9, 1.0, 1.1),
) -> List[Tuple[Callable, Callable]]:
    """
    D4 (8 views) × 3 color factors = 24 views. Strong but slow.
    Color has no inverse; we just add each view's logits (then softmax once at the end).
    """
    base = tta_d4_transform()
    out = []
    for aug, inv in base:
        for f in factors:
            out.append((lambda x, a=aug, fac=f: _multiply_aug(a(x), fac), inv))
    return out


def run_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    image_size: int,
    multimask_output: bool,
    transforms: List[Tuple[Callable, Callable]],
    merge_mode: str = "mean",
    use_amp: bool = True,
) -> torch.Tensor:
    """
    Run test-time augmentation and merge predictions.

    Args:
        model: Segmentation model with forward(images, multimask_output, image_size) -> dict with "masks" logits.
        images: [B, 3, H, W] float tensor in [0, 255].
        image_size: Target size (model output is resized to this).
        multimask_output: Passed to model.forward.
        transforms: List of (augment_fn, inverse_fn). augment_fn(images) -> augmented; inverse_fn(masks) -> original geometry.
        merge_mode: "mean" = average softmax logits then argmax (recommended).
        use_amp: Use autocast for model forward.

    Returns:
        preds: [B, image_size, image_size] long tensor, class indices.
    """
    device = images.device
    n_transforms = len(transforms)
    accumulated = None

    for aug_fn, inv_fn in transforms:
        aug_images = aug_fn(images)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(aug_images, multimask_output, image_size)
        logits = out["masks"]  # [B, C, image_size, image_size]
        logits = inv_fn(logits)
        probs = F.softmax(logits, dim=1)
        if accumulated is None:
            accumulated = probs
        else:
            accumulated = accumulated + probs

    accumulated = accumulated / n_transforms
    preds = accumulated.argmax(dim=1)
    return preds
