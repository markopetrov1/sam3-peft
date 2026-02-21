"""
Linear probing for SAM: freeze the entire model, train only a 1×1 conv head.

The segmentation head is placed directly on the image-encoder feature maps
(256 channels at H/16 × W/16 spatial resolution) and bilinearly upsampled to
the original input size.  This is the simplest possible probe — a per-pixel
linear classifier on frozen representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling import Sam


class LinearProbingSam(nn.Module):
    """Frozen SAM encoder + trainable 1×1 conv segmentation head.

    Trainable parameters:
        - A single Conv2d(256, num_classes + 1, kernel_size=1)

    Everything else (image encoder, prompt encoder, mask decoder) is frozen
    and the prompt / mask decoder is bypassed entirely during forward.

    Args:
        sam_model:   Pre-built Sam model whose weights will be frozen.
        num_classes: Number of *active* segmentation classes.  Background
                     (class 0) is added automatically, so the head outputs
                     num_classes + 1 channels.
    """

    ENCODER_OUT_CHANNELS = 256  # SAM encoder neck always produces 256-ch maps

    def __init__(self, sam_model: Sam, num_classes: int):
        super().__init__()
        self.sam = sam_model
        self.num_classes = num_classes
        num_output_channels = num_classes + 1

        # Freeze the entire SAM model
        for param in self.sam.parameters():
            param.requires_grad = False

        # Per-pixel linear classifier on encoder features
        self.head = nn.Conv2d(
            self.ENCODER_OUT_CHANNELS, num_output_channels, kernel_size=1
        )


    def train(self, mode=True):
        super().train(mode)
        self.sam.eval()
        return self


    def forward(self, batched_input, multimask_output, image_size):
        input_images = self.sam.preprocess(batched_input)

        with torch.no_grad():
            features = self.sam.image_encoder(input_images)  # [B, 256, H', W']

        logits = self.head(features)  # [B, C, H', W']

        masks = F.interpolate(
            logits,
            size=(image_size, image_size),
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
        torch.save({"head": self.head.state_dict()}, filename)

    def load_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith(".pth")
        state_dict = torch.load(filename)
        self.head.load_state_dict(state_dict["head"])
