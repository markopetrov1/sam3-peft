"""
Parameter-Efficient Fine-Tuning (PEFT) methods for SAM.

Supported methods:
    - lora:            Low-Rank Adaptation of the image encoder
    - linear_probing:  Frozen encoder + 1×1 conv segmentation head

Usage:
    from peft import build_peft_model

    model = build_peft_model(sam, method="lora", num_classes=5, rank=4)
    model = build_peft_model(sam, method="linear_probing", num_classes=5)

    model.save_parameters("checkpoint.pth")
    model.load_parameters("checkpoint.pth")
"""

from .lora import LoRA_Sam
from .linear_probing import LinearProbingSam

PEFT_REGISTRY = {
    "lora": LoRA_Sam,
    "linear_probing": LinearProbingSam,
}


def build_peft_model(sam_model, method, num_classes, **kwargs):
    """
    Factory: wrap a base SAM model with the requested PEFT method.

    Every returned model exposes:
        - forward(batched_input, multimask_output, image_size) -> dict
        - save_parameters(filename)
        - load_parameters(filename)

    Args:
        sam_model: A Sam instance from segment_anything.
        method:    PEFT method name (key in PEFT_REGISTRY).
        num_classes: Number of active segmentation classes (excl. background).
        **kwargs:  Method-specific args forwarded to the constructor
                   (e.g. rank, lora_layer for LoRA).

    Returns:
        nn.Module with the unified interface described above.
    """
    if method not in PEFT_REGISTRY:
        available = ", ".join(sorted(PEFT_REGISTRY.keys()))
        raise ValueError(
            f"Unknown PEFT method: '{method}'. Available: {available}"
        )

    if method == "lora":
        rank = kwargs.get("rank", 4)
        lora_layer = kwargs.get("lora_layer", None)
        return LoRA_Sam(sam_model, r=rank, lora_layer=lora_layer)

    if method == "linear_probing":
        return LinearProbingSam(sam_model, num_classes=num_classes)

    # Generic fallback for future methods
    return PEFT_REGISTRY[method](sam_model, num_classes=num_classes, **kwargs)
