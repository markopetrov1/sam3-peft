"""
Resolve SAM pretrained checkpoint path: use local file or download from Meta.
"""

import os
import sys
import urllib.request
from typing import Optional

# Official Meta Segment Anything checkpoints
SAM_CHECKPOINT_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

DEFAULT_FILENAMES = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}


def get_sam_checkpoint(
    path: Optional[str] = None,
    model_type: str = "vit_b",
    download: bool = True,
) -> Optional[str]:
    """
    Return path to SAM checkpoint. If path is given and exists, return it.
    If path is None or empty, use default under pre_weight/.
    If the file does not exist and download=True, download from Meta and return path.
    If download=False or download fails, return None (model will train from scratch).

    Args:
        path: Explicit checkpoint path, or None to use pre_weight/<default_filename>.
        model_type: One of vit_b, vit_l, vit_h (used for default filename and URL).
        download: Whether to download the checkpoint if the file is missing.

    Returns:
        Resolved path string, or None if file missing and not downloaded.
    """
    if path:
        resolved = path
    else:
        os.makedirs("pre_weight", exist_ok=True)
        resolved = os.path.join("pre_weight", DEFAULT_FILENAMES.get(model_type, DEFAULT_FILENAMES["vit_b"]))

    if os.path.isfile(resolved):
        return resolved

    if not download:
        return None

    url = SAM_CHECKPOINT_URLS.get(model_type)
    if not url:
        print(f"[SAM checkpoint] Unknown model_type '{model_type}', no download URL.", file=sys.stderr)
        return None

    print(f"[SAM checkpoint] Not found at {resolved}")
    print(f"[SAM checkpoint] Downloading from {url} ...")
    try:
        os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
        urllib.request.urlretrieve(url, resolved)
        print(f"[SAM checkpoint] Saved to {resolved}")
        return resolved
    except Exception as e:
        print(f"[SAM checkpoint] Download failed: {e}", file=sys.stderr)
        print("[SAM checkpoint] Train with --sam_checkpoint <path> after downloading manually.", file=sys.stderr)
        return None
