"""A detector with no AGPL in the inference path.

The shipped YOLO weights derive from Ultralytics, which is **AGPL-3.0**. AGPL is
viral over a network, and serving a model over HTTP is exactly the use that
triggers it — so a corporate legal review stops there, whatever the code quality.
That is a licence problem, not an engineering one, and no amount of hardening
fixes it.

This is the way out. torchvision is **BSD-3-Clause**, is already a dependency of
this project, and ships detection architectures good enough for the job. Train
one of those on your own footage and the resulting artifact carries no
Ultralytics obligation at all.

    netinspect onboard ./my_footage --out data/mysite
    python scripts/train_permissive.py --data data/mysite --out models/permissive_v1.pt
    netinspect gate --data data/mysite --weights models/permissive_v1.pt --method permissive

What you give up, stated plainly: Ultralytics is a better-engineered training
stack with stronger augmentation defaults, and on equal data a YOLO will
generally beat an equivalently-sized SSDlite. This trades some accuracy for a
licence you can actually deploy. Measure both with ``netinspect gate`` and decide
with numbers rather than with this paragraph.

Default architecture is ``ssdlite320_mobilenet_v3_large``: 3.7M parameters and
~110 ms per frame on a CPU, which matters because inspection boats do not have
GPUs. ``fasterrcnn_mobilenet_v3_large_320_fpn`` is available when accuracy
matters more than latency.

One licence caveat this module cannot remove: weights fitted on SOLAQUA frames
are derived from CC BY-SA 4.0 data regardless of the framework. For a fully
unencumbered artifact, train on footage you own — or on the purely synthetic
generator in :mod:`netinspect.synthetic`, which contains no SOLAQUA content.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .utils import BBox, get_logger, optional_import

LOGGER = get_logger()

ARCHITECTURES = {
    # name: (constructor, needs_num_classes_kwarg)
    "ssdlite320_mobilenet_v3_large": "fast, 3.7M params, CPU-friendly",
    "fasterrcnn_mobilenet_v3_large_320_fpn": "more accurate, slower",
    "fasterrcnn_mobilenet_v3_large_fpn": "most accurate of the mobile family",
    "retinanet_resnet50_fpn_v2": "large; only worth it on a GPU",
}
DEFAULT_ARCH = "ssdlite320_mobilenet_v3_large"

# Background is class 0 in torchvision's detection API, so a single "damage"
# class means num_classes=2. Getting this wrong trains a model that predicts
# only background and reports a suspiciously clean loss.
NUM_CLASSES = 2
DAMAGE_LABEL = 1


@dataclass
class PermissiveConfig:
    arch: str = DEFAULT_ARCH
    conf: float = 0.25
    iou: float = 0.5
    epochs: int = 20
    batch_size: int = 4
    lr: float = 5e-4
    weight_decay: float = 1e-4
    pretrained_backbone: bool = True
    max_detections: int = 100
    seed: int = 0
    # Fraction of training frames degraded through the Jerlov water model
    # (netinspect.water). 0 disables it. Targets the documented weakness: the
    # between-clip false-alarm spread is a water/scene sensitivity.
    water_augment: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _torch():
    torch = optional_import("torch")
    tv = optional_import("torchvision")
    if torch is None or tv is None:
        raise RuntimeError("This path needs torch + torchvision: pip install -e '.[ml]'")
    return torch, tv


def build_model(arch: str = DEFAULT_ARCH, num_classes: int = NUM_CLASSES,
                pretrained_backbone: bool = True):
    """Construct an untrained detector with a head sized for our classes."""
    torch, tv = _torch()
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}. "
                         f"Choose from: {', '.join(ARCHITECTURES)}")
    ctor = getattr(tv.models.detection, arch)
    # weights=None keeps the *detector* head untrained; the backbone may still
    # start from ImageNet, which is what pretrained_backbone controls.
    return ctor(weights=None, num_classes=num_classes,
                weights_backbone="DEFAULT" if pretrained_backbone else None)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _to_target(sample, torch):
    """A dataset Sample -> torchvision target, in pixels."""
    w, h = sample.width or 1, sample.height or 1
    boxes, labels = [], []
    for b in sample.boxes:
        x1, y1, x2, y2 = b.x1 * w, b.y1 * h, b.x2 * w, b.y2 * h
        if x2 <= x1 or y2 <= y1:
            continue                        # the audit rejects these upstream
        boxes.append([x1, y1, x2, y2])
        labels.append(DAMAGE_LABEL)
    if boxes:
        return {"boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64)}
    # A clean frame is a real training signal, not a missing label. torchvision
    # wants correctly-shaped empty tensors rather than None.
    return {"boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64)}


def _from_array(arr, torch):
    return torch.from_numpy(np.asarray(arr, dtype=np.float32) / 255.0).permute(2, 0, 1)


def _to_image(path, torch):
    from PIL import Image
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(samples: Sequence[Any], cfg: PermissiveConfig | None = None,
          out_path: str | Path = "models/permissive_v1.pt",
          device: str | None = None, progress=None) -> dict[str, Any]:
    """Fine-tune a torchvision detector on prepared samples.

    Returns a summary that is saved alongside the weights, so a deployed model
    can always say what it was trained on and with which settings.
    """
    torch, _ = _torch()
    cfg = cfg or PermissiveConfig()
    torch.manual_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    labelled = [s for s in samples if s.boxes]
    if not labelled:
        raise ValueError(
            "No labelled samples. This trains a supervised detector; for "
            "label-free work use the PatchCore anomaly path instead.")

    model = build_model(cfg.arch, NUM_CLASSES, cfg.pretrained_backbone).to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    order = list(range(len(samples)))
    rng = np.random.default_rng(cfg.seed)
    history = []

    LOGGER.info("Training %s on %d frames (%d labelled) for %d epochs on %s",
                cfg.arch, len(samples), len(labelled), cfg.epochs, device)

    for epoch in range(1, cfg.epochs + 1):
        rng.shuffle(order)
        running, batches = 0.0, 0
        for start in range(0, len(order), cfg.batch_size):
            chunk = order[start:start + cfg.batch_size]
            images = []
            for i in chunk:
                arr = None
                if cfg.water_augment > 0:
                    from .utils import read_image
                    from .water import augment as water_augment
                    arr = water_augment(read_image(samples[i].image), rng,
                                        probability=cfg.water_augment)
                images.append((_from_array(arr, torch) if arr is not None
                               else _to_image(samples[i].image, torch)).to(device))
            targets = [{k: v.to(device) for k, v in _to_target(samples[i], torch).items()}
                       for i in chunk]
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                LOGGER.warning("Non-finite loss at epoch %d; skipping batch", epoch)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            running += float(loss.detach())
            batches += 1
        mean = running / max(1, batches)
        history.append({"epoch": epoch, "loss": round(mean, 4)})
        LOGGER.info("  epoch %2d/%d  loss %.4f", epoch, cfg.epochs, mean)
        if progress:
            progress(epoch, cfg.epochs, mean)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(),
                "num_classes": NUM_CLASSES}, out_path)

    summary = {
        "weights": str(out_path),
        "arch": cfg.arch,
        "frames": len(samples),
        "labelled_frames": len(labelled),
        "clean_frames": len(samples) - len(labelled),
        "epochs": cfg.epochs,
        "final_loss": history[-1]["loss"] if history else None,
        "history": history,
        "device": device,
        "licence": ("torchvision (BSD-3-Clause) — no Ultralytics/AGPL code in "
                    "the training or inference path. Note that weights fitted "
                    "on CC BY-SA data inherit that data's terms."),
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2),
                                             encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def load_model(path: str | Path, device: str | None = None):
    torch, _ = _torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=True: a checkpoint is data, and unpickling arbitrary objects
    # from one is remote code execution wearing a .pt extension.
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = PermissiveConfig(**{k: v for k, v in ckpt.get("config", {}).items()
                              if k in PermissiveConfig.__dataclass_fields__})
    # The flag must match TRAINING, not be "False because we are loading".
    # torchvision's ssdlite builds a reduced-tail backbone when no pretrained
    # weights are requested (480 channels instead of 960), so loading with
    # pretrained_backbone=False into a model trained with True is a shape
    # mismatch across most of the network. The download it triggers is cached
    # and the values are overwritten by the state dict immediately; only the
    # architecture it selects matters here.
    model = build_model(cfg.arch, ckpt.get("num_classes", NUM_CLASSES),
                        pretrained_backbone=cfg.pretrained_backbone)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    model._netinspect_cfg = cfg          # noqa: SLF001 - carried for predict()
    return model


def predict_image(model, image_rgb: np.ndarray,
                  cfg: PermissiveConfig | None = None) -> list[BBox]:
    """Detections for one RGB frame, in the project's shared BBox form."""
    torch, _ = _torch()
    cfg = cfg or getattr(model, "_netinspect_cfg", None) or PermissiveConfig()
    device = next(model.parameters()).device

    arr = np.asarray(image_rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model([tensor])[0]
    finally:
        if was_training:
            model.train()

    boxes = []
    for box, score, label in zip(out["boxes"].cpu().numpy(),
                                 out["scores"].cpu().numpy(),
                                 out["labels"].cpu().numpy()):
        if float(score) < cfg.conf:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        boxes.append(BBox(x1=x1, y1=y1, x2=x2, y2=y2, score=float(score),
                          class_name="damage" if int(label) == DAMAGE_LABEL else "object"))
        if len(boxes) >= cfg.max_detections:
            break
    return boxes


__all__ = ["PermissiveConfig", "ARCHITECTURES", "DEFAULT_ARCH", "NUM_CLASSES",
           "build_model", "train", "load_model", "predict_image"]
