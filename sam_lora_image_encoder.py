"""Backward-compatible import shim — canonical code now lives in peft.lora."""

from peft.lora import LoRA_Sam, _LoRA_qkv  # noqa: F401

__all__ = ["LoRA_Sam", "_LoRA_qkv"]
