"""
SAM3 linear probing for semantic segmentation (frozen backbone + trainable heads).

Matches the mentor's approach in 15_SAM3_linearn_probing.ipynb:
- SAM3 backbone frozen.
- SAM3 segmentation_head trainable (~2.3M params), same as notebook.
- Optional small Conv2d(256, num_classes+1, 1) probe head on backbone features.
- Same forward / save / load interface as SAM3LoRAForSegmentation.

Freeze all SAM3 params, then unfreeze only parameters whose
name starts with "segmentation_head" → 2,298,881 trainable.
Forward uses frozen backbone + probe head; segmentation_head is kept for
checkpoint compatibility and parameter count.
"""

import gc
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model_builder import build_sam3_image_model


class SAM3LinearProbingForSegmentation(nn.Module):
    """
    Frozen SAM3 backbone + trainable segmentation_head (~2.3M) + probe head.
    """

    def __init__(
        self,
        num_classes: int,
        image_size: int,
        sam3_checkpoint: Optional[str] = None,
        bpe_path: Optional[str] = None,
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
            enable_segmentation=True,  # keep segmentation_head for ~2.3M trainable (notebook match)
            compile=False,
            device="cpu",
        )
        for p in self.sam3.parameters():
            p.requires_grad = False
        for name, p in self.sam3.named_parameters():
            if name.startswith("segmentation_head"):
                p.requires_grad = True

        self._cleanup_unused_modules()

        self.head = nn.Conv2d(256, self.num_output_channels, kernel_size=1)


    def _cleanup_unused_modules(self):
        """Delete SAM3 sub-modules we never use in forward. Keep segmentation_head (trainable ~2.3M)."""
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


    def train(self, mode=True):
        """Probe head and segmentation_head train; frozen backbone stays in eval."""
        self.head.train(mode)
        if getattr(self.sam3, "segmentation_head", None) is not None:
            self.sam3.segmentation_head.train(mode)
        return self


    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Run frozen vision backbone → [B, 256, H', W'] feature map."""
        x = images / 255.0
        x = (x - 0.5) / 0.5
        with torch.no_grad():
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
        state = {"head": self.head.state_dict()}
        if getattr(self.sam3, "segmentation_head", None) is not None:
            state["segmentation_head"] = self.sam3.segmentation_head.state_dict()
        torch.save(state, filename)

    def load_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith(".pth")
        payload = torch.load(filename, map_location="cpu")
        self.head.load_state_dict(payload["head"])
        if "segmentation_head" in payload and getattr(self.sam3, "segmentation_head", None) is not None:
            self.sam3.segmentation_head.load_state_dict(payload["segmentation_head"])
