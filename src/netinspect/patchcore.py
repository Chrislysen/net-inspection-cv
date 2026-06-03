"""PatchCore-style deep-feature anomaly detector (foundation-model approach).

Why this is better than ``anomaly.py``
--------------------------------------
The hand-crafted Mahalanobis model (``anomaly.py``) uses ~6 colour/texture
features per patch and localises damage poorly (F1 ~0.12 on the composite set).
PatchCore instead describes each patch with features from a **pretrained CNN**
(ImageNet), which encode far richer texture/structure. The recipe (Roth et al.,
2022):

1. Extract mid-level feature maps (here ResNet ``layer2`` + ``layer3``) from
   *normal* frames; each spatial location is a patch embedding.
2. Build a **memory bank** of normal patch embeddings (coreset-subsampled).
3. Score a test patch by its **nearest-neighbour distance** to the memory bank;
   far from all normal patches => anomalous. Upsample to a heatmap.

It needs only *normal* images (no damage labels) — exactly SOLAQUA's situation —
yet is a genuine, modern, much stronger anomaly localiser. Honesty unchanged: it
flags *deviation from normal net* (fouling/lighting too), not confirmed damage.

CPU-friendly: ResNet18 backbone, small input, subsampled bank.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import BBox, ensure_dir, get_logger, optional_import

LOGGER = get_logger()

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class PatchCoreConfig:
    backbone: str = "resnet18"        # torchvision model with layer2/layer3
    backbone_weights: str | None = None  # path to SELF-SUPERVISED weights (else ImageNet)
    input_size: int = 224
    layers: tuple[str, ...] = ("layer2", "layer3")
    coreset_size: int = 4000          # memory-bank size after subsampling
    coreset_method: str = "random"    # "random" (fast) or "greedy" (k-center)
    neighbourhood: int = 3            # local feature aggregation (avg pool)
    # Anomaly threshold = threshold_factor x median training NN distance.
    # (2.0 was the F1-optimal operating point on the composite eval.)
    threshold_factor: float = 2.0
    seed: int = 0


def _torch():
    torch = optional_import("torch")
    tv = optional_import("torchvision")
    if torch is None or tv is None:
        raise RuntimeError("PatchCore needs torch + torchvision. Install `.[ml]`.")
    return torch, tv


class _FeatureExtractor:
    """Wraps a pretrained backbone to return concatenated layer2+layer3 maps."""

    def __init__(self, cfg: PatchCoreConfig):
        torch, tv = _torch()
        from torchvision.models.feature_extraction import create_feature_extractor
        # SELF-SUPERVISED weights (e.g. SOLAQUA SimCLR) if given, else ImageNet-supervised.
        model = getattr(tv.models, cfg.backbone)(weights=None if cfg.backbone_weights else "DEFAULT")
        if cfg.backbone_weights:
            sd = torch.load(cfg.backbone_weights, map_location="cpu")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            LOGGER.info("Loaded SSL backbone %s (missing=%d unexpected=%d)",
                        cfg.backbone_weights, len(missing), len(unexpected))
        model.eval()
        self.body = create_feature_extractor(model, return_nodes={ly: ly for ly in cfg.layers})
        self.cfg = cfg
        self.torch = torch
        for p in self.body.parameters():
            p.requires_grad_(False)

    def _preprocess(self, image_rgb: np.ndarray):
        cv2 = optional_import("cv2")
        s = self.cfg.input_size
        if cv2 is not None:
            img = cv2.resize(image_rgb, (s, s), interpolation=cv2.INTER_AREA)
        else:
            from .preprocess import resize_keep_aspect
            img = resize_keep_aspect(image_rgb, s)[:s, :s]
        x = (img.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))[None]
        return self.torch.from_numpy(np.ascontiguousarray(x))

    def extract(self, image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Return patch features [H*W, C] and the (H, W) feature-grid shape."""
        torch = self.torch
        with torch.no_grad():
            feats = self.body(self._preprocess(image_rgb))
        maps = list(feats.values())
        ref_hw = maps[0].shape[-2:]
        pooled = []
        for m in maps:
            if m.shape[-2:] != ref_hw:
                m = torch.nn.functional.interpolate(m, size=ref_hw, mode="bilinear",
                                                    align_corners=False)
            # Local neighbourhood aggregation (PatchCore "locally aware" features).
            if self.cfg.neighbourhood > 1:
                m = torch.nn.functional.avg_pool2d(
                    m, kernel_size=self.cfg.neighbourhood, stride=1,
                    padding=self.cfg.neighbourhood // 2)
            pooled.append(m)
        cat = torch.cat(pooled, dim=1)[0]            # [C, H, W]
        c, h, w = cat.shape
        feat = cat.permute(1, 2, 0).reshape(h * w, c).numpy()
        return feat, (h, w)


@dataclass
class PatchCoreModel:
    bank: np.ndarray                  # [M, C] normal patch embeddings
    grid: tuple[int, int]
    threshold: float
    cfg: PatchCoreConfig
    train_stats: dict

    def save(self, path: str | Path) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        np.savez(path, bank=self.bank, grid=np.array(self.grid),
                 threshold=np.array(self.threshold),
                 cfg=np.array([self.cfg.backbone, self.cfg.input_size,
                               ",".join(self.cfg.layers), self.cfg.coreset_size,
                               self.cfg.coreset_method, self.cfg.neighbourhood,
                               self.cfg.threshold_factor,
                               self.cfg.backbone_weights or ""], dtype=object))

    @staticmethod
    def load(path: str | Path) -> "PatchCoreModel":
        d = np.load(Path(path).with_suffix(".npz"), allow_pickle=True)
        c = d["cfg"]
        bw = str(c[7]) if len(c) > 7 and str(c[7]) else None
        cfg = PatchCoreConfig(str(c[0]), bw, int(c[1]), tuple(str(c[2]).split(",")),
                              int(c[3]), str(c[4]), int(c[5]), float(c[6]))
        return PatchCoreModel(d["bank"], tuple(int(x) for x in d["grid"]),
                              float(d["threshold"]), cfg, {})


def _coreset_subsample(feats: np.ndarray, size: int, seed: int,
                       method: str = "random") -> np.ndarray:
    """Subsample the patch bank. ``random`` (fast) or greedy k-center (diverse)."""
    n = len(feats)
    if n <= size:
        return feats
    rng = np.random.default_rng(seed)
    if method == "random":
        idx = rng.choice(n, size=size, replace=False)
        return feats[idx]
    selected = [int(rng.integers(n))]
    min_d = np.linalg.norm(feats - feats[selected[0]], axis=1)
    for _ in range(size - 1):
        idx = int(np.argmax(min_d))
        selected.append(idx)
        d = np.linalg.norm(feats - feats[idx], axis=1)
        min_d = np.minimum(min_d, d)
    return feats[selected]


_EXTRACTOR_CACHE: dict = {}


def _get_extractor(cfg: PatchCoreConfig):
    """Cache the (heavy) backbone so repeated scoring doesn't reload it.

    A ``dino*`` backbone selects the self-supervised DINOv2 extractor
    (``dino_backbone.py``); anything else uses the torchvision CNN. Both expose
    the same ``extract`` contract, so ``fit``/``score_image`` are backbone-agnostic.
    """
    key = (cfg.backbone, cfg.backbone_weights, cfg.input_size, cfg.layers, cfg.neighbourhood)
    if key not in _EXTRACTOR_CACHE:
        from .dino_backbone import DinoFeatureExtractor, is_dino_backbone
        _EXTRACTOR_CACHE[key] = (DinoFeatureExtractor(cfg) if is_dino_backbone(cfg.backbone)
                                 else _FeatureExtractor(cfg))
    return _EXTRACTOR_CACHE[key]


def fit(normal_images: list[np.ndarray], cfg: PatchCoreConfig | None = None) -> PatchCoreModel:
    cfg = cfg or PatchCoreConfig()
    extractor = _get_extractor(cfg)
    all_feats, grid = [], None
    for im in normal_images:
        f, grid = extractor.extract(im)
        all_feats.append(f)
    feats = np.vstack(all_feats).astype(np.float32)
    LOGGER.info("PatchCore: %d patch features from %d frames; coreset -> %d",
                len(feats), len(normal_images), min(cfg.coreset_size, len(feats)))
    bank = _coreset_subsample(feats, cfg.coreset_size, cfg.seed, cfg.coreset_method)

    # Calibrate threshold on training nearest-neighbour distances.
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(bank)
    d, _ = nn.kneighbors(feats)
    median = float(np.median(d[:, 0]))
    threshold = cfg.threshold_factor * median
    stats = {"train_patches": int(len(feats)), "bank": int(len(bank)),
             "dist_median": median, "dist_p99": float(np.percentile(d, 99))}
    return PatchCoreModel(bank, grid, threshold, cfg, stats)


@dataclass
class PatchCoreResult:
    score_map: np.ndarray
    grid: tuple[int, int]
    boxes: list[BBox]
    max_score: float


def score_image(image_rgb: np.ndarray, model: PatchCoreModel) -> PatchCoreResult:
    from sklearn.neighbors import NearestNeighbors
    cv2 = optional_import("cv2")
    extractor = _get_extractor(model.cfg)
    feats, (h, w) = extractor.extract(image_rgb)
    nn = NearestNeighbors(n_neighbors=1).fit(model.bank)
    d, _ = nn.kneighbors(feats)
    score_map = d[:, 0].reshape(h, w)

    orig_h, orig_w = image_rgb.shape[:2]
    mask = (score_map >= model.threshold).astype(np.uint8)
    boxes: list[BBox] = []
    if mask.any() and cv2 is not None:
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, _ = stats[i]
            x1, y1 = x / w * orig_w, y / h * orig_h
            x2, y2 = (x + bw) / w * orig_w, (y + bh) / h * orig_h
            region = float(score_map[y:y + bh, x:x + bw].max())
            norm = float(np.clip(region / (model.threshold * 2), 0, 1))
            boxes.append(BBox(x1, y1, x2, y2, 0, "anomaly", norm))
    boxes.sort(key=lambda b: b.score, reverse=True)
    return PatchCoreResult(score_map, (h, w), boxes, float(score_map.max()))


def heatmap(image_rgb: np.ndarray, result: PatchCoreResult, model: PatchCoreModel,
            alpha: float = 0.5) -> np.ndarray:
    cv2 = optional_import("cv2")
    if cv2 is None:
        return image_rgb
    h, w = image_rgb.shape[:2]
    norm = np.clip(result.score_map / (model.threshold * 2), 0, 1)
    heat = cv2.resize((norm * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    hc = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - alpha, hc, alpha, 0)
