"""SimCLR self-supervised pretraining, implemented from scratch (torch only).

Learns a ResNet18 backbone from *unlabelled* frames by contrastive learning:
two random augmentations of the same frame are pulled together in embedding space
while all other frames in the batch are pushed apart (the NT-Xent loss). The
trained backbone state-dict drops straight into ``patchcore`` via its
``backbone_weights`` option.

Kept deliberately small and dependency-free (no SSL library) so the method is
visible, not hidden behind a framework. CPU-runnable for a quick proof; a GPU
makes a full schedule trivial.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import ensure_dir, get_logger, optional_import

LOGGER = get_logger()

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class SimCLRConfig:
    img_size: int = 224
    proj_dim: int = 128
    epochs: int = 200
    batch: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-6
    temperature: float = 0.5
    device: str | None = None
    workers: int = 4
    seed: int = 0


def _require_torch():
    torch = optional_import("torch")
    tv = optional_import("torchvision")
    if torch is None or tv is None:
        raise RuntimeError("SSL pretraining needs torch + torchvision. Install `.[ml]`.")
    return torch, tv


def _two_view_transform(tv, size: int):
    T = tv.transforms
    # The standard SimCLR augmentation stack (crop, colour, grey, blur, flip).
    blur_k = max(3, (int(0.1 * size) // 2) * 2 + 1)
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.4, 1.0), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(blur_k, sigma=(0.1, 2.0))], p=0.5),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def _build_dataset(torch, tv, paths, size):
    from PIL import Image
    transform = _two_view_transform(tv, size)

    class _TwoView(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            img = Image.open(paths[i]).convert("RGB")
            return transform(img), transform(img)

    return _TwoView()


def _build_model(torch, tv, cfg: SimCLRConfig):
    nn = torch.nn
    backbone = tv.models.resnet18(weights=None)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()

    class _SimCLR(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True),
                nn.Linear(feat_dim, cfg.proj_dim))

        def forward(self, x):
            z = self.proj(self.backbone(x))
            return nn.functional.normalize(z, dim=1)

    return _SimCLR()


def _nt_xent(torch, z1, z2, temperature: float):
    """Normalised temperature-scaled cross-entropy (SimCLR loss)."""
    n = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)               # [2N, D]
    sim = (z @ z.t()) / temperature              # [2N, 2N]
    sim.fill_diagonal_(float("-inf"))            # no self-similarity
    # positive for row i is its other view: i <-> i+N
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return torch.nn.functional.cross_entropy(sim, targets)


def pretrain(frame_paths, out_path: str | Path, cfg: SimCLRConfig | None = None) -> Path:
    """SimCLR-pretrain a ResNet18 on the given frames; save the backbone state-dict."""
    torch, tv = _require_torch()
    cfg = cfg or SimCLRConfig()
    torch.manual_seed(cfg.seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = _build_dataset(torch, tv, list(frame_paths), cfg.img_size)
    batch = min(cfg.batch, len(ds))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=True, drop_last=len(ds) > batch,
        num_workers=cfg.workers, pin_memory=(device == "cuda"))
    model = _build_model(torch, tv, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    LOGGER.info("SimCLR: %d frames, batch %d, %d epochs on %s", len(ds), batch, cfg.epochs, device)
    model.train()
    for epoch in range(cfg.epochs):
        losses = []
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            z1, z2 = model(v1), model(v2)
            loss = _nt_xent(torch, z1, z2, cfg.temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        if epoch == 0 or (epoch + 1) % max(1, cfg.epochs // 20) == 0:
            LOGGER.info("  epoch %3d/%d  loss %.4f", epoch + 1, cfg.epochs, float(np.mean(losses)))

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    # Save the *backbone* only (ResNet18 weights, fc removed) — patchcore loads this.
    torch.save(model.backbone.state_dict(), out_path)
    LOGGER.info("Saved SOLAQUA-pretrained ResNet18 backbone -> %s", out_path)
    return out_path
