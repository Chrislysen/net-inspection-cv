"""Synthesise underwater degradation from the physics, as a training augmentation.

The idea is taken from UWCNN (Li, Anwar & Porikli, *Pattern Recognition* 2019),
whose real contribution is not its network but how it built training data: apply
the Jerlov water-type attenuation model to clean images and learn from the
result. That is a physical model, not code — nothing is copied from that
repository, which carries no licence at all.

The model is the standard simplified underwater image formation equation:

    I_c(x) = J_c(x) · t_c(x)  +  B_c · (1 - t_c(x)),      t_c = exp(-beta_c · d)

for each colour channel c: the scene radiance ``J`` attenuated by transmission
``t``, plus backscattered ambient light ``B`` filling in the difference. Red is
absorbed fastest, which is why everything underwater turns blue-green with depth.

Why this project wants it
------------------------
The documented weakness here is not the day-to-day gap — it is the **between-clip
spread on a single day: 0%, 0% and 31% false alarms across three clips**, roughly
thirty times the day effect. That is a *scene and water* sensitivity, and this is
the augmentation that targets it directly, by showing the detector the same net
under water it has never been filmed in.

An earlier attempt at robustness through stronger *photometric* augmentation
(HSV jitter, rotation, perspective) made things worse — 18% to 22%. This is a
different hypothesis rather than more of the same one: random colour jitter
explores directions the physics never takes, while this walks the one-parameter
family that real water actually produces. It was tested, and **it failed too**: trained with 60% of frames degraded through
this model, the permissive detector held 10% false alarms on the worst clip
(unchanged) and went from 0% to 2% on the held-out day, with recall flat at 88%.
Two independent augmentation strategies have now failed to close the between-clip
gap, which is itself informative — the gap is unlikely to be a data-diversity
problem. The module is kept because the model is correct and reusable, not
because it helped.

On the coefficients
-------------------
``WATER_TYPES`` holds per-metre transmission per RGB channel for Jerlov's oceanic
(I–III) and coastal (1C–9C) classifications. They are approximate values in the
range used throughout this literature, kept configurable rather than presented as
precise measurements. Norwegian coastal water at a farm site is turbid and green,
around type **3C–5C**; open Atlantic is nearer type I–II.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import get_logger

LOGGER = get_logger()

# Per-metre transmission per (R, G, B). Lower = absorbed faster.
#
# The ordering CROSSES OVER along the series, which is easy to get wrong. While
# absorption dominates (I through 5C) red dies first and the scene reads blue,
# then green. In the murkiest coastal water (7C, 9C) scattering strips the short
# wavelengths faster than absorption strips the long ones, the order inverts, and
# red is left the best-transmitted channel — which is why harbour water looks
# brown rather than blue. tests/test_water.py asserts both regimes.
WATER_TYPES: dict[str, tuple[float, float, float]] = {
    "I":   (0.85, 0.961, 0.982),      # clearest open ocean
    "IA":  (0.84, 0.955, 0.975),
    "IB":  (0.83, 0.950, 0.968),
    "II":  (0.80, 0.925, 0.940),
    "III": (0.75, 0.885, 0.890),      # turbid open ocean
    "1C":  (0.75, 0.885, 0.875),      # clearest coastal
    "3C":  (0.71, 0.820, 0.800),
    "5C":  (0.67, 0.730, 0.670),      # ~ Norwegian farm site
    "7C":  (0.62, 0.610, 0.500),
    "9C":  (0.55, 0.460, 0.290),      # very turbid coastal
}

# Types plausible for a Norwegian aquaculture site. Sampling the full range would
# train for water the equipment will never see, at the cost of capacity spent on
# it.
COASTAL_TYPES = ("1C", "3C", "5C", "7C")

DEFAULT_BACKGROUND = (0.28, 0.55, 0.62)     # green-blue veiling light, [0, 1]


@dataclass
class WaterConfig:
    """One synthetic water condition."""
    water_type: str = "5C"
    distance_m: float = 1.0          # camera-to-scene distance; ROV standoff here
    background: tuple[float, float, float] = DEFAULT_BACKGROUND
    strength: float = 1.0            # 0 = untouched, 1 = full physical model

    def transmission(self) -> np.ndarray:
        if self.water_type not in WATER_TYPES:
            raise ValueError(f"Unknown water type {self.water_type!r}. "
                             f"Choose from: {', '.join(WATER_TYPES)}")
        per_metre = np.asarray(WATER_TYPES[self.water_type], dtype=np.float32)
        # t = k^d, i.e. exp(-beta*d) with beta = -ln(k).
        t = per_metre ** float(max(0.0, self.distance_m))
        # strength interpolates towards "no water at all" (t -> 1).
        return 1.0 - self.strength * (1.0 - t)


def apply_water(image: np.ndarray, cfg: WaterConfig | None = None) -> np.ndarray:
    """Degrade a frame as though it were shot through `distance_m` of that water.

    Uniform transmission across the frame: a per-pixel depth map would be more
    faithful, but the ROV holds a roughly constant standoff from a flat net
    panel, so the extra parameter would add noise rather than realism.
    """
    cfg = cfg or WaterConfig()
    t = cfg.transmission()
    b = np.asarray(cfg.background, dtype=np.float32)

    img = image.astype(np.float32) / 255.0
    out = img * t + b * (1.0 - t)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def random_water(rng: np.random.Generator,
                 types: tuple[str, ...] = COASTAL_TYPES,
                 distance_range: tuple[float, float] = (0.3, 2.5),
                 strength_range: tuple[float, float] = (0.0, 1.0)) -> WaterConfig:
    """Sample a plausible water condition for augmentation.

    ``strength`` reaching 0 matters: some fraction of the augmented set must stay
    close to the original footage, or the detector is trained for water it will
    never be deployed in and loses the condition it actually ships against.
    """
    return WaterConfig(
        water_type=str(rng.choice(types)),
        distance_m=float(rng.uniform(*distance_range)),
        strength=float(rng.uniform(*strength_range)),
    )


def augment(image: np.ndarray, rng: np.random.Generator,
            probability: float = 0.5, **kw) -> np.ndarray:
    """Apply a random water condition to a fraction of frames."""
    if rng.random() >= probability:
        return image
    return apply_water(image, random_water(rng, **kw))


__all__ = ["WATER_TYPES", "COASTAL_TYPES", "DEFAULT_BACKGROUND", "WaterConfig",
           "apply_water", "random_water", "augment"]
