"""Where on the net is the damage? Net-frame localisation and coverage mapping.

A detection reported as "frame 1247" cannot be acted on. Nobody can send a diver
to frame 1247. This module converts per-frame detections into **positions on the
net** — metres along the sweep, metres across it, and depth — and reports what
the pass actually covered, so a negative result means "we looked and it was
clean" rather than "we didn't look".

How the position is recovered
-----------------------------
Two sources, each fixing the other's weakness, in the proportions the data
supports:

* **Visual odometry** gives smooth, self-consistent *relative* motion. On this
  footage it works far better than the textbook warning suggests — a net mesh is
  repetitive, which normally destroys feature matching, but biofouling supplies
  ample non-repeating texture. Measured on real SOLAQUA frames: ~1200 RANSAC
  inliers per consecutive pair at a **79% inlier ratio**.
* **ROV telemetry** supplies what vision cannot: absolute metric scale and an
  anchor in the net's own frame.

Crucially the split is *not* a per-frame fusion. Frame-to-frame, visual motion
and integrated net-relative velocity correlate at only **r = +0.26** on this
data — too weak to correct individual frames without injecting noise. In
aggregate they agree well, so telemetry is used for **scale and anchoring** and
vision for **motion**. That division is measured, not assumed.

Self-calibrating scale
----------------------
No chessboard is needed. Total visual displacement over a pass is compared with
the total distance the telemetry says the vehicle travelled, giving millimetres
per pixel at the observed standoff. On the reference clip that yields
**0.83 mm/px at 0.61 m** (1.36 mm/px at 1 m), implying an ~82° horizontal field
of view — a physically sensible number for an underwater camera, which is the
check that the calibration is measuring something real. Scale is then propagated
per frame: ground sampling distance grows linearly with standoff distance.

What this does and does not produce
-----------------------------------
It maps the **inspected strip** — on the reference clip a 5.5 m extent, swept
along a 5.9 m path — in the net's own frame. It does **not** produce a model of a whole pen: that
needs a full circumnavigation, and one clip's arc is far too straight to even
identify the pen radius (over a 6 m sweep the deviation from a straight line is
within USBL noise). Position error also accumulates with distance from the last
absolute fix, so every coordinate is reported with a drift estimate rather than
as a point.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from .utils import get_logger, optional_import, read_image

LOGGER = get_logger()

# Feature-matching defaults, chosen against real SOLAQUA frames.
ORB_FEATURES = 2000
MIN_MATCHES = 12
RANSAC_PX = 3.0

# Reference self-calibration from the SOLAQUA clips: mm per pixel at 1 m standoff.
# Ground sampling distance scales linearly with distance, so this is the constant
# that transfers between frames — and, cautiously, between clips of the same camera.
MM_PER_PX_AT_1M = 1.26

# Drift assumption for the uncertainty estimate: visual odometry accumulates
# roughly this fraction of distance travelled. Deliberately pessimistic — an
# over-tight error bar on a repair location is worse than a wide one.
DRIFT_FRACTION = 0.05


# --------------------------------------------------------------------------- #
# Visual odometry
# --------------------------------------------------------------------------- #
@dataclass
class FrameMotion:
    """Estimated motion between two consecutive frames."""
    index: int
    dx_px: float
    dy_px: float
    rotation_deg: float
    inliers: int
    matches: int

    @property
    def magnitude_px(self) -> float:
        return float(np.hypot(self.dx_px, self.dy_px))

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.matches if self.matches else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["magnitude_px"] = round(self.magnitude_px, 2)
        d["inlier_ratio"] = round(self.inlier_ratio, 3)
        return d


def estimate_motion(prev_gray, curr_gray, orb=None, matcher=None) -> FrameMotion | None:
    """Rigid (translation + rotation) motion between two grayscale frames.

    A *partial* affine rather than a full homography: over a short baseline
    against a locally planar net, the extra projective degrees of freedom mostly
    absorb noise, and a rigid model degrades more honestly when matching is poor.

    Returns ``None`` when the pair cannot be matched — a gap in the track is
    information, and inventing motion across it would silently corrupt every
    downstream position.
    """
    cv2 = optional_import("cv2")
    if cv2 is None:
        raise RuntimeError("Visual odometry needs OpenCV: pip install -e '.[cv]'")
    orb = orb or cv2.ORB_create(ORB_FEATURES)
    matcher = matcher or cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ka, da = orb.detectAndCompute(prev_gray, None)
    kb, db = orb.detectAndCompute(curr_gray, None)
    if da is None or db is None or len(ka) < MIN_MATCHES or len(kb) < MIN_MATCHES:
        return None
    matches = matcher.match(da, db)
    if len(matches) < MIN_MATCHES:
        return None

    src = np.float32([ka[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                          ransacReprojThreshold=RANSAC_PX)
    if M is None:
        return None
    return FrameMotion(
        index=0, dx_px=float(M[0, 2]), dy_px=float(M[1, 2]),
        rotation_deg=float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
        inliers=int(mask.sum()) if mask is not None else 0,
        matches=len(matches),
    )


def track_motion(paths: Sequence[str]) -> list[FrameMotion | None]:
    """Estimate motion along a sequence of frame paths.

    Entry ``i`` is the motion from ``paths[i]`` to ``paths[i + 1]``; ``None``
    marks a pair that could not be matched.
    """
    cv2 = optional_import("cv2")
    if cv2 is None:
        raise RuntimeError("Visual odometry needs OpenCV: pip install -e '.[cv]'")
    orb = cv2.ORB_create(ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    out: list[FrameMotion | None] = []
    prev = None
    for i, p in enumerate(paths):
        gray = cv2.cvtColor(read_image(p), cv2.COLOR_RGB2GRAY)
        if prev is not None:
            m = estimate_motion(prev, gray, orb, matcher)
            if m is not None:
                m.index = i
            out.append(m)
        prev = gray
    return out


# --------------------------------------------------------------------------- #
# Scale
# --------------------------------------------------------------------------- #
@dataclass
class ScaleCalibration:
    """Millimetres per pixel, recovered from telemetry rather than a target."""
    mm_per_px_at_1m: float
    reference_standoff_m: float
    total_pixels: float
    total_metres: float
    frames: int
    implied_hfov_deg: float | None = None
    note: str = ""

    def mm_per_px(self, standoff_m: float) -> float:
        """Ground sampling distance at a given standoff (linear in distance)."""
        return self.mm_per_px_at_1m * float(standoff_m)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calibrate_scale(pixel_motion: Sequence[float], metre_motion: Sequence[float],
                    standoff_m: Sequence[float], image_width_px: int | None = None
                    ) -> ScaleCalibration:
    """Recover millimetres-per-pixel by comparing visual and telemetry travel.

    Aggregate rather than per-frame on purpose: individual frames correlate too
    weakly (r ~ 0.26 on this data) to calibrate against, while the totals agree.

    When ``image_width_px`` is supplied the implied horizontal field of view is
    reported. That is the sanity check that matters — a calibration returning an
    absurd FOV is measuring noise, whatever its internal consistency.
    """
    px = np.asarray(pixel_motion, dtype=float)
    m = np.asarray(metre_motion, dtype=float)
    d = np.asarray(standoff_m, dtype=float)
    ok = np.isfinite(px) & np.isfinite(m) & np.isfinite(d) & (px > 0) & (d > 0)
    if ok.sum() < 5:
        raise ValueError("Need at least 5 usable frame pairs to calibrate scale")

    total_px, total_m = float(px[ok].sum()), float(m[ok].sum())
    ref_standoff = float(np.median(d[ok]))
    if total_px <= 0 or total_m <= 0:
        raise ValueError("No motion to calibrate against")

    mm_per_px_here = 1000.0 * total_m / total_px
    mm_per_px_1m = mm_per_px_here / ref_standoff

    hfov = None
    if image_width_px:
        half_width_m = (mm_per_px_here / 1000.0) * image_width_px / 2.0
        hfov = float(np.degrees(2 * np.arctan2(half_width_m, ref_standoff)))

    return ScaleCalibration(
        mm_per_px_at_1m=round(mm_per_px_1m, 4),
        reference_standoff_m=round(ref_standoff, 3),
        total_pixels=round(total_px, 1), total_metres=round(total_m, 3),
        frames=int(ok.sum()),
        implied_hfov_deg=round(hfov, 1) if hfov else None,
        note=("Calibrated from telemetry travel over the whole pass, not a target. "
              "Check the implied field of view: an implausible value means the "
              "calibration fitted noise."),
    )


# --------------------------------------------------------------------------- #
# The track: frames placed on the net
# --------------------------------------------------------------------------- #
@dataclass
class TrackPoint:
    """One frame's position on the net, with its uncertainty."""
    index: int
    frame: str
    t: float
    along_m: float          # distance along the sweep from the start
    across_m: float         # lateral offset from the start line
    depth_m: float | None
    standoff_m: float | None
    mm_per_px: float
    drift_m: float          # accumulated position uncertainty
    matched: bool = True
    orient: float = 1.0     # +1, or -1 if the map was rotated to run forwards

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_track(frames: Sequence[str], names: Sequence[str], times: Sequence[float],
                motions: Sequence[FrameMotion | None],
                standoff_m: Sequence[float] | None = None,
                depth_m: Sequence[float] | None = None,
                scale: ScaleCalibration | None = None) -> list[TrackPoint]:
    """Integrate frame motions into net-frame positions.

    The first frame is the origin: positions are relative to where the pass
    started, which is what a pilot can actually re-navigate to. Unmatched pairs
    contribute no motion but are still emitted, flagged ``matched=False``, so a
    gap in the track stays visible instead of closing silently.
    """
    scale = scale or ScaleCalibration(MM_PER_PX_AT_1M, 1.0, 0, 0, 0,
                                      note="default reference calibration")
    n = len(frames)
    along = across = 0.0
    travelled = 0.0
    points: list[TrackPoint] = []

    for i in range(n):
        so = float(standoff_m[i]) if standoff_m is not None and np.isfinite(standoff_m[i]) \
            else scale.reference_standoff_m
        mmpp = scale.mm_per_px(so)

        if i > 0:
            m = motions[i - 1] if i - 1 < len(motions) else None
            if m is not None:
                # Image x maps to along-track, y to across-track. Sign is flipped
                # because the scene moves opposite to the camera.
                along += -m.dx_px * mmpp / 1000.0
                across += -m.dy_px * mmpp / 1000.0
                travelled += m.magnitude_px * mmpp / 1000.0
            points.append(TrackPoint(
                index=i, frame=names[i], t=float(times[i]),
                along_m=round(along, 4), across_m=round(across, 4),
                depth_m=float(depth_m[i]) if depth_m is not None else None,
                standoff_m=round(so, 3), mm_per_px=round(mmpp, 4),
                drift_m=round(DRIFT_FRACTION * travelled, 4),
                matched=m is not None))
        else:
            points.append(TrackPoint(
                index=0, frame=names[0], t=float(times[0]),
                along_m=0.0, across_m=0.0,
                depth_m=float(depth_m[0]) if depth_m is not None else None,
                standoff_m=round(so, 3), mm_per_px=round(mmpp, 4), drift_m=0.0))

    # Orient the map so the pass runs in the +along direction: a position should
    # read as "3.1 m from the start of the pass", not "-3.1 m". Which way the
    # scene slides across the sensor is an artefact of camera mounting, not
    # something an operator should have to reason about.
    #
    # This is a 180-degree rotation, not a mirror: across flips with along, so
    # the map keeps its handedness and left/right are not silently swapped.
    # `orient` is recorded on every point because anything that later places a
    # position *within* a frame has to rotate with the path — see
    # localise_detections.
    orient = -1.0 if points and points[-1].along_m < 0 else 1.0
    for pt in points:
        pt.orient = orient
        if orient < 0:
            pt.along_m = round(-pt.along_m, 4)
            pt.across_m = round(-pt.across_m, 4)
    return points


# --------------------------------------------------------------------------- #
# Detections on the net
# --------------------------------------------------------------------------- #
@dataclass
class LocalisedDetection:
    """A detection with a position a person could navigate back to."""
    frame: str
    frame_index: int
    along_m: float
    across_m: float
    depth_m: float | None
    width_mm: float
    height_mm: float
    score: float
    drift_m: float

    def describe(self) -> str:
        depth = f", {self.depth_m:.1f} m depth" if self.depth_m is not None else ""
        return (f"{self.along_m:.2f} m along the sweep, "
                f"{self.across_m:+.2f} m across{depth} "
                f"(±{self.drift_m:.2f} m) · {self.width_mm:.0f}×{self.height_mm:.0f} mm")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["description"] = self.describe()
        return d


def localise_detections(track: Sequence[TrackPoint], detections_by_frame: dict,
                        image_size: tuple[int, int]) -> list[LocalisedDetection]:
    """Place each detection on the net and give it a physical size.

    The offset from image centre is converted with that frame's own ground
    sampling distance, so a detection seen from 1.4 m is not treated as the same
    physical size as one seen from 0.6 m.

    Note the sign here is *opposite* to the one in :func:`build_track`, and both
    are correct. There, a static feature slides backwards past a forward-moving
    camera, so travel is ``-dx_px``. Here the mapping from world position to
    image position is order-preserving — something right of image centre really
    is further along — so the offset is ``+``. ``pt.orient`` then applies the
    same rotation build_track applied to the path, keeping the two consistent.
    """
    width, height = image_size
    cx, cy = width / 2.0, height / 2.0
    out: list[LocalisedDetection] = []
    for pt in track:
        for det in detections_by_frame.get(pt.frame, []):
            x1, y1, x2, y2 = det["bbox"]
            bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            mpp = pt.mm_per_px / 1000.0
            out.append(LocalisedDetection(
                frame=pt.frame, frame_index=pt.index,
                along_m=round(pt.along_m + pt.orient * (bx - cx) * mpp, 3),
                across_m=round(pt.across_m + pt.orient * (by - cy) * mpp, 3),
                depth_m=pt.depth_m,
                width_mm=round((x2 - x1) * pt.mm_per_px, 1),
                height_mm=round((y2 - y1) * pt.mm_per_px, 1),
                score=float(det.get("score", 0.0)), drift_m=pt.drift_m))
    return out


# --------------------------------------------------------------------------- #
# Sites: many sightings of one thing
# --------------------------------------------------------------------------- #
@dataclass
class DefectSite:
    """One physical location on the net, supported by repeated sightings.

    This is the unit an operator can act on. A pass that reports 107 detections
    is unusable; the same pass reporting three sites, one of them seen 72 times
    from different viewpoints, is a work order.
    """
    site_id: int
    along_m: float
    across_m: float
    depth_m: float | None
    sightings: int
    max_score: float
    mean_score: float
    median_width_mm: float
    median_height_mm: float
    span_m: float                       # spatial spread of its sightings
    first_frame: str
    last_frame: str

    @property
    def evidence(self) -> str:
        """How strongly the geometry supports this being a real, distinct thing."""
        if self.sightings >= 10:
            return "strong — seen from many viewpoints"
        if self.sightings >= 3:
            return "moderate — seen in several frames"
        return "weak — one or two frames only"

    def describe(self) -> str:
        depth = f", {self.depth_m:.1f} m depth" if self.depth_m is not None else ""
        return (f"site {self.site_id}: {self.along_m:.2f} m along, "
                f"{self.across_m:+.2f} m across{depth} · {self.sightings} sightings · "
                f"~{self.median_width_mm:.0f}×{self.median_height_mm:.0f} mm · "
                f"{self.evidence}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence"] = self.evidence
        d["description"] = self.describe()
        return d


def _single_linkage(points: np.ndarray, radius: float) -> np.ndarray:
    """Union-find single-linkage clustering — no SciPy needed.

    Single linkage rather than k-means because the number of defects is exactly
    what is unknown, and because "within `radius` of another sighting" is the
    physically meaningful merge rule: the same object seen twice should land
    within position error of itself.
    """
    n = len(points)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if float(np.hypot(*(points[i] - points[j]))) <= radius:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    return np.array([find(i) for i in range(n)])


def cluster_sites(detections: Sequence[LocalisedDetection], radius_m: float = 0.25,
                  min_sightings: int = 1) -> list[DefectSite]:
    """Collapse per-frame detections into distinct physical locations.

    This is **spatial** confirmation, and it is strictly stronger than the
    temporal kind: temporal tracking loses a defect the moment the camera pans
    away, whereas a position on the net survives the vehicle leaving and coming
    back. Sighting count then becomes an evidence measure — something seen 72
    times from different angles is not a flicker.

    ``radius_m`` should be comparable to the position uncertainty; merging more
    aggressively than the map is accurate would fuse genuinely separate defects.
    """
    if not detections:
        return []
    pts = np.array([[d.along_m, d.across_m] for d in detections], dtype=float)
    labels = _single_linkage(pts, radius_m)

    sites: list[DefectSite] = []
    for sid, label in enumerate(sorted(set(labels.tolist())), start=1):
        members = [d for d, lab in zip(detections, labels) if lab == label]
        if len(members) < min_sightings:
            continue
        p = np.array([[m.along_m, m.across_m] for m in members])
        centre = p.mean(axis=0)
        depths = [m.depth_m for m in members if m.depth_m is not None]
        sites.append(DefectSite(
            site_id=sid,
            along_m=round(float(centre[0]), 3), across_m=round(float(centre[1]), 3),
            depth_m=round(float(np.median(depths)), 2) if depths else None,
            sightings=len(members),
            max_score=round(max(m.score for m in members), 3),
            mean_score=round(float(np.mean([m.score for m in members])), 3),
            median_width_mm=round(float(np.median([m.width_mm for m in members])), 1),
            median_height_mm=round(float(np.median([m.height_mm for m in members])), 1),
            span_m=round(float(np.hypot(*(p.max(axis=0) - p.min(axis=0)))), 3),
            first_frame=members[0].frame, last_frame=members[-1].frame))
    return sorted(sites, key=lambda s: -s.sightings)


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
@dataclass
class Coverage:
    """What the pass actually saw — and what it did not."""
    along_extent_m: float
    across_extent_m: float
    swept_area_m2: float
    frames: int
    matched_frames: int
    gaps: list[dict] = field(default_factory=list)
    mean_footprint_m: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def coverage(track: Sequence[TrackPoint], image_size: tuple[int, int],
             gap_threshold_m: float = 0.25) -> Coverage:
    """Swept extent and the holes in it.

    A gap is a step between consecutive frames larger than ``gap_threshold_m``:
    the vehicle moved further than one frame's footprint, so a band of net passed
    unphotographed. Those bands are the honest answer to "what did this
    inspection not look at", which is the question a clean report has to survive.
    """
    if not track:
        return Coverage(0, 0, 0, 0, 0, note="empty track")
    width, height = image_size
    along = np.array([p.along_m for p in track])
    across = np.array([p.across_m for p in track])
    footprints = np.array([p.mm_per_px * width / 1000.0 for p in track])

    gaps = []
    for a, b in zip(track[:-1], track[1:]):
        step = float(np.hypot(b.along_m - a.along_m, b.across_m - a.across_m))
        if step > gap_threshold_m:
            gaps.append({"from_frame": a.frame, "to_frame": b.frame,
                         "from_along_m": a.along_m, "to_along_m": b.along_m,
                         "gap_m": round(step, 3)})

    along_extent = float(along.max() - along.min())
    across_extent = float(across.max() - across.min())
    mean_fp = float(np.nanmean(footprints))
    return Coverage(
        along_extent_m=round(along_extent, 3),
        across_extent_m=round(across_extent, 3),
        swept_area_m2=round(along_extent * mean_fp, 3),
        frames=len(track),
        matched_frames=sum(1 for p in track if p.matched),
        gaps=gaps, mean_footprint_m=round(mean_fp, 3),
        note=("Extent is along the swept strip, not around a pen. Swept area "
              "assumes the camera footprint is contiguous across track; gaps "
              "list where the vehicle outran its own field of view."))


def validate_against_telemetry(track: Sequence[TrackPoint],
                               telemetry_distance_m: float) -> dict[str, Any]:
    """Compare the visual path length with what telemetry independently says.

    This is the check that keeps the map honest. Visual odometry drifts and
    telemetry is noisy; when the two disagree badly the map should not be
    trusted, and the only way to know is to compute it.
    """
    if len(track) < 2:
        return {"comparable": False, "reason": "track too short"}
    visual = float(sum(
        np.hypot(b.along_m - a.along_m, b.across_m - a.across_m)
        for a, b in zip(track[:-1], track[1:])))
    if telemetry_distance_m <= 0:
        return {"comparable": False, "reason": "no telemetry distance"}
    ratio = visual / telemetry_distance_m
    return {
        "comparable": True,
        "visual_path_m": round(visual, 3),
        "telemetry_path_m": round(telemetry_distance_m, 3),
        "ratio": round(ratio, 3),
        "agrees_within_20pct": bool(0.8 <= ratio <= 1.2),
        "note": ("Scale was calibrated from this telemetry, so agreement in "
                 "total is expected; the informative part is the shape of the "
                 "disagreement along the pass, which reveals drift."),
    }


__all__ = [
    "FrameMotion", "ScaleCalibration", "TrackPoint", "LocalisedDetection",
    "DefectSite", "Coverage",
    "estimate_motion", "track_motion", "calibrate_scale", "build_track",
    "localise_detections", "cluster_sites", "coverage", "validate_against_telemetry",
    "ORB_FEATURES", "MIN_MATCHES", "RANSAC_PX", "MM_PER_PX_AT_1M", "DRIFT_FRACTION",
]
