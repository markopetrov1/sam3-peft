"""
YAML config loader with attribute-style access.

Usage:
    from utils.config import load_config
    cfg = load_config("configs/sam3_lora_potsdam.yaml")
    print(cfg.training.epochs)
    print(cfg.dataset.type)
"""

import os

import yaml
from typing import Any


class Config:
    """Recursive namespace — allows cfg.section.key access."""

    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

    def __repr__(self):
        return _format(self.__dict__)

    def to_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out


def _format(d, indent=0):
    lines = []
    for k, v in d.items():
        prefix = "  " * indent
        if isinstance(v, Config):
            lines.append(f"{prefix}{k}:")
            lines.append(_format(v.__dict__, indent + 1))
        else:
            lines.append(f"{prefix}{k}: {v}")
    return "\n".join(lines)


def load_config(path: str, overrides: dict = None) -> Config:
    """
    Load a YAML config file and return a Config object.

    Args:
        path: Path to .yaml file.
        overrides: Optional dict of dotted-key overrides, e.g.
                   {"training.epochs": 100, "dataset.root": "/data/new"}
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if overrides:
        for dotted_key, value in overrides.items():
            keys = dotted_key.split(".")
            d = raw
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

    # The configs record dataset paths as they existed on the machine the experiments were run on.
    # Rather than editing 21 YAML files after a move, set SAM3_DATA_ROOT to the directory that
    # holds the datasets and the basename of the configured path is resolved underneath it.
    data_root = os.environ.get("SAM3_DATA_ROOT")
    if data_root and isinstance(raw.get("dataset"), dict) and raw["dataset"].get("root"):
        raw["dataset"]["root"] = os.path.join(data_root,
                                              os.path.basename(str(raw["dataset"]["root"]).rstrip("/")))

    return Config(raw)
