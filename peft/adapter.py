"""
Adapter modules for SAM3 vision encoder fine-tuning.

Implements the PromptGenerator from SAM-Adapter (ICLR 2024) adapted for SAM3's
ViT backbone (depth=32, patch_size=14, embed_dim=1024, RoPE).

The adapter injects lightweight learned prompts (FFT-based handcrafted features +
embedding projections + per-block MLPs) before each ViT transformer block.
Only adapter parameters are trained; the backbone stays frozen.
"""

import math
from itertools import repeat
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return tuple(repeat(x, 2))


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding used in the handcrafted feature pyramid."""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, H*W, C]
        x = self.norm(x)
        return x, H, W


class PromptGenerator(nn.Module):
    """
    Generates adapter prompts that are added to ViT features before each block.

    Combines three signal sources:
      1. Handcrafted features: FFT high/low-pass filtered input → multi-scale pyramid
      2. Embedding projections: linear down-projection of current ViT features
      3. Adapter MLPs: lightweight per-block + shared per-stage projections

    Args:
        scale_factor: channel reduction ratio (embed_dim // scale_factor = reduced_dim)
        prompt_type: 'highpass' or 'lowpass' for FFT filtering
        embed_dims: [embed_dim]*4, one per stage
        tuning_stage: string like '1234' indicating which stages get adapters
        depths: list of 4 ints, blocks per stage
        input_type: 'fft', 'gaussian', 'srm', or 'all'
        freq_nums: FFT mask radius fraction (0.0–1.0)
        handcrafted_tune: whether to use handcrafted features
        embedding_tune: whether to use embedding projections
        adaptor: 'adaptor' (shared+lightweight), 'fully_shared', or 'fully_unshared'
        img_size: input image resolution
    """

    def __init__(
        self,
        scale_factor: int,
        prompt_type: str,
        embed_dims: List[int],
        tuning_stage: str,
        depths: List[int],
        input_type: str,
        freq_nums: float,
        handcrafted_tune: bool,
        embedding_tune: bool,
        adaptor: str,
        img_size: int,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.prompt_type = prompt_type
        self.embed_dims = embed_dims
        self.input_type = input_type
        self.freq_nums = freq_nums
        self.tuning_stage = tuning_stage
        self.depths = depths
        self.handcrafted_tune = handcrafted_tune
        self.embedding_tune = embedding_tune
        self.adaptor = adaptor

        if self.input_type == 'all':
            self.prompt = nn.Parameter(
                torch.zeros(3, img_size, img_size), requires_grad=False
            )

        # Handcrafted feature pyramid (OverlapPatchEmbed cascade)
        if self.handcrafted_tune:
            if '1' in self.tuning_stage:
                self.handcrafted_generator1 = OverlapPatchEmbed(
                    img_size=img_size, patch_size=7, stride=4,
                    in_chans=3, embed_dim=self.embed_dims[0] // self.scale_factor,
                )
            if '2' in self.tuning_stage:
                self.handcrafted_generator2 = OverlapPatchEmbed(
                    img_size=img_size // 4, patch_size=3, stride=2,
                    in_chans=self.embed_dims[0] // self.scale_factor,
                    embed_dim=self.embed_dims[1] // self.scale_factor,
                )
            if '3' in self.tuning_stage:
                self.handcrafted_generator3 = OverlapPatchEmbed(
                    img_size=img_size // 8, patch_size=3, stride=2,
                    in_chans=self.embed_dims[1] // self.scale_factor,
                    embed_dim=self.embed_dims[2] // self.scale_factor,
                )
            if '4' in self.tuning_stage:
                self.handcrafted_generator4 = OverlapPatchEmbed(
                    img_size=img_size // 16, patch_size=3, stride=2,
                    in_chans=self.embed_dims[2] // self.scale_factor,
                    embed_dim=self.embed_dims[3] // self.scale_factor,
                )

        # Embedding generators (down-project current ViT features)
        if self.embedding_tune:
            for stage in range(1, 5):
                if str(stage) in self.tuning_stage:
                    setattr(
                        self,
                        f'embedding_generator{stage}',
                        nn.Linear(
                            self.embed_dims[stage - 1],
                            self.embed_dims[stage - 1] // self.scale_factor,
                        ),
                    )

        # Adapter MLPs
        if self.adaptor == 'adaptor':
            for stage in range(1, 5):
                if str(stage) not in self.tuning_stage:
                    continue
                dim_reduced = self.embed_dims[stage - 1] // self.scale_factor
                for d in range(self.depths[stage - 1] + 1):
                    setattr(
                        self,
                        f'lightweight_mlp{stage}_{d}',
                        nn.Sequential(
                            nn.Linear(dim_reduced, dim_reduced),
                            nn.GELU(),
                        ),
                    )
                setattr(
                    self,
                    f'shared_mlp{stage}',
                    nn.Linear(dim_reduced, self.embed_dims[stage - 1]),
                )

        elif self.adaptor == 'fully_shared':
            for stage in range(1, 5):
                if str(stage) not in self.tuning_stage:
                    continue
                dim_reduced = self.embed_dims[stage - 1] // self.scale_factor
                setattr(
                    self,
                    f'fully_shared_mlp{stage}',
                    nn.Sequential(
                        nn.Linear(dim_reduced, dim_reduced),
                        nn.GELU(),
                        nn.Linear(dim_reduced, self.embed_dims[stage - 1]),
                    ),
                )

        elif self.adaptor == 'fully_unshared':
            for stage in range(1, 5):
                if str(stage) not in self.tuning_stage:
                    continue
                dim_reduced = self.embed_dims[stage - 1] // self.scale_factor
                for d in range(self.depths[stage - 1]):
                    setattr(
                        self,
                        f'fully_unshared_mlp{stage}_{d}',
                        nn.Sequential(
                            nn.Linear(dim_reduced, dim_reduced),
                            nn.GELU(),
                            nn.Linear(dim_reduced, self.embed_dims[stage - 1]),
                        ),
                    )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.amp.autocast("cuda", enabled=False)
    def fft(self, x: torch.Tensor, rate: float, prompt_type: str) -> torch.Tensor:
        """Apply FFT high/low-pass filter to extract frequency-domain features."""
        x = x.float()
        mask = torch.zeros_like(x)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** 0.5 // 2)
        mask[:, :, w // 2 - line:w // 2 + line, h // 2 - line:h // 2 + line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        if prompt_type == 'highpass':
            fft = fft * (1 - mask)
        elif prompt_type == 'lowpass':
            fft = fft * mask

        fft_hires = torch.fft.ifftshift(torch.complex(fft.real, fft.imag))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real
        return torch.abs(inv)

    def init_handcrafted(
        self, x: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Build 4-level handcrafted feature pyramid from the raw image.

        Returns features in [B, C, H, W] format for correct spatial resizing.
        """
        if self.input_type == 'fft':
            x = self.fft(x, self.freq_nums, self.prompt_type)
        elif self.input_type == 'all':
            x = self.prompt.unsqueeze(0).expand(x.shape[0], -1, -1, -1)

        B = x.shape[0]
        prev_4d = None
        results: list = [None, None, None, None]

        if '1' in self.tuning_stage:
            flat, H, W = self.handcrafted_generator1(x)
            prev_4d = flat.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            results[0] = prev_4d

        if '2' in self.tuning_stage:
            flat, H, W = self.handcrafted_generator2(prev_4d)
            prev_4d = flat.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            results[1] = prev_4d

        if '3' in self.tuning_stage:
            flat, H, W = self.handcrafted_generator3(prev_4d)
            prev_4d = flat.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            results[2] = prev_4d

        if '4' in self.tuning_stage:
            flat, H, W = self.handcrafted_generator4(prev_4d)
            prev_4d = flat.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            results[3] = prev_4d

        return tuple(results)

    def init_prompt(
        self,
        embedding_feature: torch.Tensor,
        handcrafted_feature: Optional[torch.Tensor],
        block_num: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare (handcrafted, embedding) prompt pair for a given stage."""
        emb = None
        if self.embedding_tune:
            gen = getattr(self, f'embedding_generator{block_num}')
            emb = gen(embedding_feature)
        hc = handcrafted_feature if self.handcrafted_tune else None
        return hc, emb

    def get_prompt(
        self,
        x: torch.Tensor,
        prompt: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
        block_num: int,
        depth_num: int,
    ) -> torch.Tensor:
        """Combine handcrafted + embedding features via adapter MLP and add to x."""
        handcrafted, embedding = prompt
        if embedding is None and handcrafted is None:
            return x

        # Use embedding shape as spatial reference
        ref = embedding if embedding is not None else handcrafted
        B, H, W = ref.shape[0], ref.shape[1], ref.shape[2]

        feat = torch.zeros(B, H, W, ref.shape[-1], device=x.device, dtype=x.dtype)
        if handcrafted is not None:
            feat = feat + handcrafted.reshape(B, H, W, -1)
        if embedding is not None:
            feat = feat + embedding

        if self.adaptor == 'adaptor':
            lw_mlp = getattr(self, f'lightweight_mlp{block_num}_{depth_num}')
            shared = getattr(self, f'shared_mlp{block_num}')
            feat = shared(lw_mlp(feat))
        elif self.adaptor == 'fully_shared':
            mlp = getattr(self, f'fully_shared_mlp{block_num}')
            feat = mlp(feat)
        elif self.adaptor == 'fully_unshared':
            mlp = getattr(self, f'fully_unshared_mlp{block_num}_{depth_num}')
            feat = mlp(feat)

        return x + feat


def resize_handcrafted(
    feature: Optional[torch.Tensor], target_h: int, target_w: int
) -> Optional[torch.Tensor]:
    """
    Resize a handcrafted feature map [B, C, H, W] to [B, target_h, target_w, C].
    """
    if feature is None:
        return None
    if feature.shape[2] != target_h or feature.shape[3] != target_w:
        feature = F.interpolate(
            feature.float(), size=(target_h, target_w),
            mode='bilinear', align_corners=False,
        ).to(feature.dtype)
    return feature.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]


def inject_adapter_into_vit(
    vit: nn.Module, prompt_generator: nn.Module, depth_per_stage: int
) -> None:
    """
    Monkey-patch a SAM3 ViT's forward to inject adapter prompts before each block.

    The PromptGenerator must be a sub-module of the *caller* (not of the ViT itself)
    so that its parameters are properly tracked for optimizer / save / load.
    """
    from sam3.model.vitdet import get_abs_pos
    import torch.utils.checkpoint as ckpt_utils

    _pg = prompt_generator
    _dps = depth_per_stage

    def adapter_forward(x: torch.Tensor) -> list:
        inp = x

        x = vit.patch_embed(x)
        h, w = x.shape[1], x.shape[2]

        s = 0
        if vit.retain_cls_token:
            x = torch.cat([vit.class_embedding, x.flatten(1, 2)], dim=1)
            s = 1

        if vit.pos_embed is not None:
            x = x + get_abs_pos(
                vit.pos_embed,
                vit.pretrain_use_cls_token,
                (h, w),
                vit.retain_cls_token,
                tiling=vit.tile_abs_pos,
            )

        x = vit.ln_pre(x)

        handcrafted_list = _pg.init_handcrafted(inp)

        outputs = []
        for i, blk in enumerate(vit.blocks):
            if i < _dps:
                stage_idx, rel_idx = 1, i
            elif i < _dps * 2:
                stage_idx, rel_idx = 2, i - _dps
            elif i < _dps * 3:
                stage_idx, rel_idx = 3, i - _dps * 2
            else:
                stage_idx, rel_idx = 4, i - _dps * 3

            if str(stage_idx) in _pg.tuning_stage:
                hc = handcrafted_list[stage_idx - 1]
                resized_hc = resize_handcrafted(hc, h, w)
                prompt_tuple = _pg.init_prompt(x, resized_hc, stage_idx)
                x = _pg.get_prompt(x, prompt_tuple, stage_idx, rel_idx)

            if vit.use_act_checkpoint and vit.training:
                x = ckpt_utils.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

            if (i == vit.full_attn_ids[-1]) or (
                vit.return_interm_layers and i in vit.full_attn_ids
            ):
                if i == vit.full_attn_ids[-1]:
                    x = vit.ln_post(x)

                feats = x[:, s:]
                if feats.ndim == 4:
                    feats = feats.permute(0, 3, 1, 2)
                else:
                    assert feats.ndim == 3
                    feats = feats.reshape(
                        feats.shape[0], h, w, feats.shape[-1]
                    ).permute(0, 3, 1, 2)

                outputs.append(feats)

        return outputs

    vit.forward = adapter_forward
