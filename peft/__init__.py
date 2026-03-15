"""
PEFT (parameter-efficient fine-tuning) package.

Structure:
  - lora.py              — Generic LoRA layers and apply_lora_to_model (model-agnostic).
  - sam3_lora.py         — SAM3 + LoRA + segmentation head (uses .lora).
  - sam3_linear_probing.py — Frozen SAM3 + linear head.
  - adapter.py           — PromptGenerator and injection logic (model-agnostic).
  - sam3_adapter.py      — SAM3 + Adapter + probe head (uses .adapter).

Train/test use build_peft_model(method="sam3_lora" | "sam3_linear_probing" | "sam3_adapter" | ...).
"""

from .sam3_lora import SAM3LoRAForSegmentation
from .sam3_linear_probing import SAM3LinearProbingForSegmentation
from .sam3_adapter import SAM3AdapterForSegmentation
from .lora import LoRAConfig, apply_lora_to_model, save_lora_weights, load_lora_weights, count_parameters

PEFT_REGISTRY = {
    "sam3_lora": SAM3LoRAForSegmentation,
    "sam3_linear_probing": SAM3LinearProbingForSegmentation,
    "sam3_adapter": SAM3AdapterForSegmentation,
}


def build_peft_model(sam_model, method, num_classes, **kwargs):
    """
    Factory that returns a PEFT-wrapped model with a unified interface:
        - forward(images, multimask_output, image_size) -> dict(masks=...)
        - save_parameters(path)
        - load_parameters(path)
    """
    del sam_model  # SAM3 path builds its own model

    if method not in PEFT_REGISTRY:
        available = ", ".join(sorted(PEFT_REGISTRY.keys()))
        raise ValueError(f"Unknown PEFT method: '{method}'. Available: {available}")

    if method == "sam3_lora":
        return SAM3LoRAForSegmentation(
            num_classes=num_classes,
            image_size=kwargs["image_size"],
            sam3_checkpoint=kwargs.get("sam3_checkpoint"),
            bpe_path=kwargs.get("bpe_path"),
            rank=kwargs.get("rank", 8),
            alpha=kwargs.get("alpha", 16),
            dropout=kwargs.get("dropout", 0.0),
        )
    if method == "sam3_linear_probing":
        return SAM3LinearProbingForSegmentation(
            num_classes=num_classes,
            image_size=kwargs["image_size"],
            sam3_checkpoint=kwargs.get("sam3_checkpoint"),
            bpe_path=kwargs.get("bpe_path"),
        )
    if method == "sam3_adapter":
        return SAM3AdapterForSegmentation(
            num_classes=num_classes,
            image_size=kwargs["image_size"],
            sam3_checkpoint=kwargs.get("sam3_checkpoint"),
            bpe_path=kwargs.get("bpe_path"),
            scale_factor=kwargs.get("scale_factor", 32),
            input_type=kwargs.get("input_type", "fft"),
            freq_nums=kwargs.get("freq_nums", 0.25),
            prompt_type=kwargs.get("prompt_type", "highpass"),
            tuning_stage=kwargs.get("tuning_stage", "1234"),
            handcrafted_tune=kwargs.get("handcrafted_tune", True),
            embedding_tune=kwargs.get("embedding_tune", True),
            adaptor=kwargs.get("adaptor", "adaptor"),
        )

    return PEFT_REGISTRY[method](num_classes=num_classes, **kwargs)

