"""
SAM3 + Adapter wrapper for semantic segmentation on Potsdam/Vaihingen.

Trainable:
  - PromptGenerator (FFT handcrafted features + embedding projections + adapter MLPs)
  - 1×1 conv probe head on backbone features

Frozen:
  - SAM3 ViT backbone (weights untouched)
  - SAM3 segmentation_head (kept for checkpoint compatibility)

Memory optimisations (same as LoRA wrapper):
  1. Gradient checkpointing on the vision backbone.
  2. Unused SAM3 modules (transformer, geometry encoder, …) deleted after init.
  3. Text encoder deleted — we call forward_image (vision-only).
  4. Mixed precision handled in train.py via AMP.

Reference: Chen et al., "SAM-Adapter: Adapting Segment Anything in Underperformed Scenes", ICLR 2024.
"""

import gc
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model_builder import build_sam3_image_model

from .adapter import PromptGenerator, inject_adapter_into_vit


class SAM3AdapterForSegmentation(nn.Module):
    """
    Trainable params:
      - PromptGenerator (adapter) inside SAM3 vision encoder
      - 1×1 conv probe head
    """

    def __init__(
        self,
        num_classes: int,
        image_size: int,
        sam3_checkpoint: Optional[str] = None,
        bpe_path: Optional[str] = None,
        # Adapter hyper-parameters
        scale_factor: int = 32,
        input_type: str = "fft",
        freq_nums: float = 0.25,
        prompt_type: str = "highpass",
        tuning_stage: str = "1234",
        handcrafted_tune: bool = True,
        embedding_tune: bool = True,
        adaptor: str = "adaptor",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_output_channels = num_classes + 1
        self.image_size = image_size

        # ── Build SAM3 (frozen) ──────────────────────────────────────────
        self.sam3 = build_sam3_image_model(
            bpe_path=bpe_path,
            eval_mode=True,
            checkpoint_path=sam3_checkpoint,
            load_from_HF=(sam3_checkpoint is None),
            enable_segmentation=True,
            compile=False,
            device="cpu",
        )
        for p in self.sam3.parameters():
            p.requires_grad = False

        # ── Create PromptGenerator (trainable) ───────────────────────────
        vit = self.sam3.backbone.visual.trunk
        embed_dim = vit.blocks[0].attn.qkv.in_features  # infer from ViT
        depth = len(vit.blocks)
        depth_per_stage = depth // 4
        remainder = depth % 4
        depths_list = [depth_per_stage] * 4
        if remainder > 0:
            depths_list[-1] += remainder

        self.prompt_generator = PromptGenerator(
            scale_factor=scale_factor,
            prompt_type=prompt_type,
            embed_dims=[embed_dim] * 4,
            tuning_stage=str(tuning_stage),
            depths=depths_list,
            input_type=input_type,
            freq_nums=freq_nums,
            handcrafted_tune=handcrafted_tune,
            embedding_tune=embedding_tune,
            adaptor=adaptor,
            img_size=image_size,
        )

        # ── Inject adapter into ViT forward ──────────────────────────────
        inject_adapter_into_vit(vit, self.prompt_generator, depth_per_stage)

        # ── Cleanup + gradient checkpointing ─────────────────────────────
        self._cleanup_unused_modules()
        self.sam3.backbone.act_ckpt_whole_vision_backbone = True

        # ── Probe head ───────────────────────────────────────────────────
        self.head = nn.Conv2d(256, self.num_output_channels, kernel_size=1)

    def _cleanup_unused_modules(self):
        """Delete SAM3 sub-modules not used in forward to save memory."""
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
        backbone_out = self.sam3.backbone.forward_image(x)
        features = backbone_out["vision_features"]

        if features.ndim == 3:
            b, n, c = features.shape
            s = int(math.sqrt(n))
            if s * s != n:
                raise RuntimeError(f"SAM3 features [B,N,C] have non-square N={n}.")
            features = features.permute(0, 2, 1).reshape(b, c, s, s)

        if features.ndim != 4:
            raise RuntimeError(
                f"Expected 4D SAM3 feature map, got {tuple(features.shape)}"
            )
        return features

    def forward(
        self, batched_input, multimask_output, image_size
    ) -> Dict[str, torch.Tensor]:
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

    # ── Save / Load ─────────────────────────────────────────────────────
    def save_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith(".pth")
        state = {
            "prompt_generator": self.prompt_generator.state_dict(),
            "head": self.head.state_dict(),
        }
        torch.save(state, filename)

    def load_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith(".pth")
        payload = torch.load(filename, map_location="cpu")
        self.prompt_generator.load_state_dict(payload["prompt_generator"])
        self.head.load_state_dict(payload["head"])
