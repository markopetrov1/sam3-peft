"""
SAM3 + LoRA wrapper for semantic segmentation on Potsdam/Vaihingen.

Trainable:
  - LoRA adapters injected into the vision encoder.
  - SAM3 segmentation_head.
  - 1×1 conv probe head on backbone features.

Memory optimisations:
  1. Gradient checkpointing on the vision backbone.
  2. Unused SAM3 modules (transformer, geometry encoder, …) deleted after init.
  3. Text encoder deleted — we call forward_image (vision-only).
  4. Mixed precision handled in train.py via AMP.
"""

import gc
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model_builder import build_sam3_image_model

from .lora import LoRAConfig, apply_lora_to_model


class SAM3LoRAForSegmentation(nn.Module):
    """
    Trainable params:
      - LoRA adapters inside SAM3 vision encoder
      - SAM3 segmentation_head (~2.3M)
      - 1×1 conv probe head
    """

    def __init__(
        self,
        num_classes: int,
        image_size: int,
        sam3_checkpoint: Optional[str] = None,
        bpe_path: Optional[str] = None,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_output_channels = num_classes + 1
        self.image_size = image_size

        self.sam3 = build_sam3_image_model(
            bpe_path=bpe_path,
            eval_mode=True,
            checkpoint_path=sam3_checkpoint,
            load_from_HF=(sam3_checkpoint is None),
            enable_segmentation=True,
            compile=False,
            device="cpu",
        )

        # Freeze everything, then selectively unfreeze
        for p in self.sam3.parameters():
            p.requires_grad = False

        # Unfreeze segmentation_head (~2.3M params)
        for name, p in self.sam3.named_parameters():
            if name.startswith("segmentation_head"):
                p.requires_grad = True

        # Inject LoRA adapters into the vision encoder (creates new trainable params)
        lora_cfg = LoRAConfig(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            apply_to_vision_encoder=True,
            apply_to_text_encoder=False,
            apply_to_geometry_encoder=False,
            apply_to_detr_encoder=False,
            apply_to_detr_decoder=False,
            apply_to_mask_decoder=False,
        )
        self.sam3 = apply_lora_to_model(self.sam3, lora_cfg)

        self._cleanup_unused_modules()
        self.sam3.backbone.act_ckpt_whole_vision_backbone = True

        self.head = nn.Conv2d(256, self.num_output_channels, kernel_size=1)


    def _cleanup_unused_modules(self):
        """Delete SAM3 sub-modules not used in forward. Keep segmentation_head (trainable ~2.3M)."""
        for attr in (
            "transformer",
            "geometry_encoder",
            "dot_prod_scoring",
            "instance_dot_prod_scoring",
            "matcher",
            "inst_interactive_predictor",
            "class_embed",
            "instance_class_embed",
        ):
            if hasattr(self.sam3, attr):
                setattr(self.sam3, attr, None)
        if hasattr(self.sam3.backbone, "language_backbone"):
            self.sam3.backbone.language_backbone = None
        gc.collect()


    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Run vision backbone → [B, 256, H', W'] feature map."""
        x = images / 255.0
        x = (x - 0.5) / 0.5
        # forward_image runs only the ViT + neck, skipping the text encoder.
        backbone_out = self.sam3.backbone.forward_image(x)
        features = backbone_out["vision_features"]

        if features.ndim == 3:
            b, n, c = features.shape
            s = int(math.sqrt(n))
            if s * s != n:
                raise RuntimeError(f"SAM3 features [B,N,C] have non-square N={n}.")
            features = features.permute(0, 2, 1).reshape(b, c, s, s)

        if features.ndim != 4:
            raise RuntimeError(f"Expected 4D SAM3 feature map, got {tuple(features.shape)}")
        return features


    def forward(self, batched_input, multimask_output, image_size) -> Dict[str, torch.Tensor]:
        del multimask_output, image_size
        feats = self._extract_features(batched_input)
        logits = self.head(feats)
        masks = F.interpolate(
            logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return {
            "masks": masks,
            "iou_predictions": None,
            "low_res_logits": logits,
        }

    def save_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith(".pth")
        trainable_names = {n for n, p in self.sam3.named_parameters() if p.requires_grad}
        full_state = self.sam3.state_dict()
        trainable_state = {
            k: full_state[k].detach().cpu() for k in trainable_names if k in full_state
        }
        state = {"sam3_trainable": trainable_state, "head": self.head.state_dict()}
        if getattr(self.sam3, "segmentation_head", None) is not None:
            state["segmentation_head"] = self.sam3.segmentation_head.state_dict()
        torch.save(state, filename)

    def load_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith(".pth")
        payload = torch.load(filename, map_location="cpu")
        trainable_state = payload.get("sam3_trainable", {})
        if trainable_state:
            self.sam3.load_state_dict(trainable_state, strict=False)
        self.head.load_state_dict(payload["head"])
        if "segmentation_head" in payload and getattr(self.sam3, "segmentation_head", None) is not None:
            self.sam3.segmentation_head.load_state_dict(payload["segmentation_head"])
