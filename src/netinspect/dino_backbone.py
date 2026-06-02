"""Self-supervised (DINOv2) feature extractor for the PatchCore pipeline.

Why
---
`patchcore.py` describes each patch with features from an **ImageNet-supervised**
ResNet. A long-standing question for this project (see RESEARCH_SYNTHESIS) is
whether **self-supervised** features — learned without labels — transfer better
to underwater net imagery, which looks nothing like ImageNet. DINOv2 (Oquab et
al., 2023) is a vision transformer trained self-supervised on 142M unlabelled
images; its patch tokens are a strong, label-free descriptor.

This module swaps the backbone, nothing else: the same one-class memory-bank +
nearest-neighbour scoring in `patchcore.py` runs on DINOv2 patch tokens instead
of ResNet feature maps. That makes it a clean **ablation** — supervised-ImageNet
vs self-supervised-DINOv2 features for the *same* anomaly detector.

Honesty
-------
These are **off-the-shelf** DINOv2 weights (self-supervised on natural images),
**not** pretrained on SOLAQUA. So this tests "do published SSL features transfer
better than supervised ones here?", which is a real and useful question — but it
is **not** the deferred experiment of pretraining DINO/MAE *on* the unlabelled
SOLAQUA frames (that needs a GPU and remains the documented next step). As with
every detector here, it flags *deviation from normal net*, not confirmed damage.

CPU notes
---------
ViT-S/14 (~21M params) runs on CPU for the few dozen frames we have. The weights
download once via ``torch.hub`` and are then cached locally; with no network and
no cache, loading raises a clear, actionable error instead of hanging.
"""
from __future__ import annotations

import numpy as np

from .utils import get_logger, optional_import

LOGGER = get_logger()

# DINOv2 was trained with ImageNet normalisation.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

_PATCH = 14  # DINOv2 ViT patch size; input side must be a multiple of this.

# Short names -> torch.hub entrypoints.
_DINO_MODELS = {
    "dinov2_vits14": "dinov2_vits14",
    "dinov2_vitb14": "dinov2_vitb14",
    "dino_vits14": "dinov2_vits14",
}


def is_dino_backbone(name: str) -> bool:
    return name.startswith("dino")


class DinoFeatureExtractor:
    """Frozen DINOv2 patch-token extractor with the PatchCore extractor interface.

    ``extract(image_rgb) -> (features [H*W, C], (H, W))`` — identical contract to
    ``patchcore._FeatureExtractor`` so the rest of the pipeline is unchanged.
    """

    def __init__(self, cfg):
        torch = optional_import("torch")
        if torch is None:
            raise RuntimeError("DINOv2 backbone needs torch. Install `.[ml]`.")
        self.torch = torch
        self.cfg = cfg
        entry = _DINO_MODELS.get(cfg.backbone, cfg.backbone)
        try:
            model = torch.hub.load("facebookresearch/dinov2", entry, verbose=False)
        except Exception as exc:  # offline + uncached, or hub failure
            raise RuntimeError(
                f"Could not load DINOv2 '{entry}' via torch.hub ({exc}). "
                "It downloads once with internet access and is then cached; "
                "run it online first, or use a torchvision backbone instead."
            ) from exc
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        # Round the configured input size to a multiple of the patch size.
        s = max(_PATCH, int(round(cfg.input_size / _PATCH)) * _PATCH)
        self.input_size = s
        self.grid = s // _PATCH

    def _preprocess(self, image_rgb: np.ndarray):
        cv2 = optional_import("cv2")
        s = self.input_size
        if cv2 is not None:
            img = cv2.resize(image_rgb, (s, s), interpolation=cv2.INTER_AREA)
        else:
            from .preprocess import resize_keep_aspect
            img = resize_keep_aspect(image_rgb, s)[:s, :s]
        x = (img.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))[None]
        return self.torch.from_numpy(np.ascontiguousarray(x))

    def extract(self, image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        torch = self.torch
        with torch.no_grad():
            out = self.model.forward_features(self._preprocess(image_rgb))
        tokens = out["x_norm_patchtokens"]          # [1, N, C], N = grid*grid
        g = self.grid
        c = tokens.shape[-1]
        grid = tokens.reshape(1, g, g, c).permute(0, 3, 1, 2)   # [1, C, g, g]
        # Optional local aggregation, mirroring PatchCore's "locally aware" trick.
        n = getattr(self.cfg, "neighbourhood", 1)
        if n and n > 1:
            grid = torch.nn.functional.avg_pool2d(grid, kernel_size=n, stride=1,
                                                  padding=n // 2)
        feat = grid[0].permute(1, 2, 0).reshape(g * g, -1).numpy()
        return feat.astype(np.float32), (g, g)
