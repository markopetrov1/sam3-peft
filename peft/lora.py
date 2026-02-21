"""
LoRA (Low-Rank Adaptation) for SAM's image encoder.

Injects low-rank matrices into the QKV projections of each ViT attention
block.  Only the LoRA weights, prompt encoder, and mask decoder are trainable;
the image encoder backbone is frozen.
"""

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from segment_anything.modeling import Sam


class _LoRA_qkv(nn.Module):
    """Drop-in replacement for the combined QKV linear layer that adds
    low-rank branches for Q and V.

    Original SAM attention computes:
        qkv = self.qkv(x)          # Linear(dim, dim * 3)

    With LoRA the output becomes:
        qkv[:, :, :, :dim]  += linear_b_q(linear_a_q(x))   # Q
        qkv[:, :, :, -dim:] += linear_b_v(linear_a_v(x))   # V
    """

    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim :] += new_v
        return qkv


class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Trainable parameters:
        - LoRA weight matrices (w_A, w_B) in each selected attention block
        - Prompt encoder
        - Mask decoder

    Args:
        sam_model: Pre-built Sam model.
        r:         LoRA rank (default 4).
        lora_layer: Optional list of block indices to inject LoRA into.
                    Defaults to all encoder blocks.
    """

    def __init__(self, sam_model: Sam, r: int = 4, lora_layer=None):
        super().__init__()

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))

        self.w_As = []
        self.w_Bs = []

        # Freeze image encoder
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Inject LoRA into selected attention blocks
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

        self.reset_parameters()
        self.sam = sam_model

    def forward(self, batched_input, multimask_output, image_size):
        return self.sam(batched_input, multimask_output, image_size)

    def save_parameters(self, filename: str) -> None:
        """Save LoRA weights + prompt encoder + mask decoder."""
        self.save_lora_parameters(filename)

    def load_parameters(self, filename: str) -> None:
        """Load LoRA weights + prompt encoder + mask decoder."""
        self.load_lora_parameters(filename)

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith(".pth")

        num_layer = len(self.w_As)
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        if isinstance(self.sam, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()

        for key, value in state_dict.items():
            if "prompt_encoder" in key:
                prompt_encoder_tensors[key] = value
            if "mask_decoder" in key:
                mask_decoder_tensors[key] = value

        merged_dict = {
            **a_tensors,
            **b_tensors,
            **prompt_encoder_tensors,
            **mask_decoder_tensors,
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith(".pth")

        state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        prompt_encoder_keys = [k for k in sam_keys if "prompt_encoder" in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = dict(zip(prompt_encoder_keys, prompt_encoder_values))
        sam_dict.update(prompt_encoder_new_state_dict)

        mask_decoder_keys = [k for k in sam_keys if "mask_decoder" in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = dict(zip(mask_decoder_keys, mask_decoder_values))
        sam_dict.update(mask_decoder_new_state_dict)
        self.sam.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
