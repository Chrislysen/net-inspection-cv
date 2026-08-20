"""Tests for the synthetic placeholder generator — the module that writes the
ground truth every reported number is measured against.

The images this module produces are admittedly crude, and the README says so
loudly. That is not what these tests are about. This module also *writes the
labels*: `_box_to_yolo_line` is the single conversion from pixel xyxy to
normalised YOLO cxcywh, and `generate_dataset` is what `make_report_assets.py`
calls to build `data/sample` before precision and recall are computed. A swapped
axis, a width normalised by the height, or a box that drifts a few pixels off
the damage it is supposed to describe would not crash, would not look wrong in
an overlay, and would silently move every metric in the reports.

So the tests are weighted accordingly:

* the label line is checked against hand-computed values and round-tripped
  through `netinspect.data.parse_yolo_label` — the production consumer, not a
  reimplementation of the writer;
* the boxes are checked against the *pixels*: with confounders disabled, every
  dark pixel in the frame must fall inside a labelled box and every labelled box
  must contain dark pixels, which is what catches a box that is merely offset;
* determinism is checked at the byte level, because a demo that is not
  reproducible cannot be audited.

Two genuine defects are pinned as strict xfails at the bottom of the geometry
section. See the reasons on those tests.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from netinspect import synthetic as S
from netinspect.data import DEFAULT_CLASSES, load_dataset, parse_yolo_label
from netinspect.utils import BBox

# The tear generator derives its length from the image WIDTH and needs
# int(w * 0.03) > max(3, int(w * 0.01)), which is only true from 134 px up (see
# the xfail at the end of the geometry section). Everything below is kept just
# above that floor so the tests stay fast without tripping over it.
H, W = 160, 200

# The damage colour is a flat (10, 25, 30) -> mean 21.7, while the darkest
# background pixel across seeds measures ~45. 30 sits cleanly between them.
DARK = 30


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _gen(seed: int = 0, num_damage: int = 2, distractors: bool = False, **kw):
    return S.make_synthetic_image(height=H, width=W, num_damage=num_damage,
                                  seed=seed, distractors=distractors, **kw)


def _luma(img: np.ndarray) -> np.ndarray:
    return img.mean(axis=2)


def _covered(boxes, shape) -> np.ndarray:
    """Boolean mask of everything the ground-truth boxes claim."""
    mask = np.zeros(shape, bool)
    for b in boxes:
        mask[max(0, int(b.y1)):int(b.y2) + 1, max(0, int(b.x1)):int(b.x2) + 1] = True
    return mask


def _write_label(tmp_path: Path, boxes, name="lbl.txt") -> Path:
    p = tmp_path / name
    p.write_text("\n".join(S._box_to_yolo_line(b, W, H) for b in boxes), encoding="utf-8")
    return p


def _tree_digest(root) -> dict[str, str]:
    """sha256 of every file under root, keyed by posix-style relative path."""
    root = Path(root)
    return {
        str(p.relative_to(root)).replace(os.sep, "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


# --------------------------------------------------------------------------- #
# the label line — every reported metric is measured against this conversion
# --------------------------------------------------------------------------- #
def test_a_label_line_is_a_class_id_and_exactly_four_normalised_numbers():
    """Five fields, or `parse_yolo_label` reads the line as a polygon instead.

    A six-field line (e.g. someone appending the score) is silently rejected;
    an eight-field one is parsed as a segmentation mask. Either way the box
    disappears from the ground truth and recall goes up.
    """
    parts = S._box_to_yolo_line(BBox(10, 20, 50, 80, 1, "hole"), 200, 100).split()
    assert len(parts) == 5, f"expected 'cls cx cy w h', got {parts}"
    assert parts[0] == "1" and float(parts[0]) == int(parts[0])
    assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


def test_the_label_line_matches_a_hand_computed_conversion():
    """Non-square image, asymmetric box, four distinct expected values.

    cx=0.15, cy=0.5, w=0.2, h=0.6 are pairwise different on purpose: any
    permutation of the four fields, and any normalisation of x by the height (or
    y by the width), lands on a different string.
    """
    line = S._box_to_yolo_line(BBox(10, 20, 50, 80), w=200, h=100)
    assert line == "0 0.150000 0.500000 0.200000 0.600000"


def test_a_box_survives_the_round_trip_through_the_production_label_parser(tmp_path):
    """Writer -> file -> `netinspect.data.parse_yolo_label` must recover the box.

    Round-tripping through the real consumer rather than an inverse written here
    means the two halves cannot drift apart: if the writer and the parser ever
    disagree about the coordinate convention, this fails.
    """
    original = BBox(13.0, 27.0, 91.0, 58.0, 2, "tear")
    boxes, _, kind, errors = parse_yolo_label(_write_label(tmp_path, [original]), W, H)
    assert kind == "detection" and errors == []
    got = boxes[0]
    assert (got.x1, got.y1, got.x2, got.y2) == pytest.approx(
        (original.x1, original.y1, original.x2, original.y2), abs=1e-2)


def test_every_box_of_a_generated_image_round_trips(tmp_path):
    """The same round trip on real generator output, across several seeds.

    The hand-written box above is symmetric-ish and small; the generator emits
    boxes near the frame edges and boxes whose corners come from a rotated
    rectangle, which is where an off-by-one in the min/max would show up.
    """
    for seed in range(6):
        img, gts = _gen(seed=seed, num_damage=3)
        h, w = img.shape[:2]
        path = tmp_path / f"s{seed}.txt"
        path.write_text("\n".join(S._box_to_yolo_line(b, w, h) for b in gts), encoding="utf-8")
        back, _, _, errors = parse_yolo_label(path, w, h)
        assert errors == [], f"seed {seed}: {errors}"
        assert len(back) == len(gts)
        for a, b in zip(gts, back):
            assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx(
                (a.x1, a.y1, a.x2, a.y2), abs=1e-2), f"seed {seed}"


def test_the_class_id_written_is_the_class_the_parser_names(tmp_path):
    """holes must come back as holes.

    `synthetic` hard-codes 1 for a hole and 2 for a tear; `data.DEFAULT_CLASSES`
    is the list those indices are looked up in. Nothing links the two, so an
    edit to either would relabel every defect in the dataset without erroring.
    """
    assert DEFAULT_CLASSES[1] == "hole" and DEFAULT_CLASSES[2] == "tear"
    written = [BBox(10, 10, 30, 30, 1, "hole"), BBox(40, 40, 90, 60, 2, "tear")]
    back, _, _, _ = parse_yolo_label(_write_label(tmp_path, written), W, H)
    assert [b.class_id for b in back] == [1, 2]
    assert [b.class_name for b in back] == ["hole", "tear"]


def test_a_one_pixel_box_does_not_round_away_to_zero_area():
    """Six decimals must still resolve a single pixel, including on 4K frames.

    A box whose width rounds to 0.000000 is a degenerate label: `dataset.audit`
    grades it as an error, and IoU against it is 0 for every prediction, so the
    defect turns into a permanent false negative.
    """
    for width in (W, 720, 3840):
        parts = S._box_to_yolo_line(BBox(37, 41, 38, 42), width, width).split()
        assert float(parts[3]) > 0.0 and float(parts[4]) > 0.0, f"{width}px frame lost a 1px box"
        assert float(parts[3]) * width == pytest.approx(1.0, abs=1e-2)


def test_a_full_frame_box_lands_exactly_on_the_unit_square():
    """The corner case an off-by-one (w vs w-1) breaks by a hair.

    0.500001 or 1.000001 is out of range for a YOLO label and `parse_yolo_label`
    records it as an error, so this must be exact, not merely close.
    """
    assert S._box_to_yolo_line(BBox(0, 0, W, H), W, H) == \
        "0 0.500000 0.500000 1.000000 1.000000"


def test_a_box_pinned_to_the_top_left_corner_stays_non_negative():
    parts = S._box_to_yolo_line(BBox(0, 0, 4, 4), W, H).split()
    assert all(float(v) >= 0.0 for v in parts[1:]), f"corner box went negative: {parts}"


def test_an_inverted_box_is_clamped_to_zero_extent_never_negative():
    """A negative w/h in a label file is accepted by most parsers as garbage.

    `BBox.width` clamps at 0, so the writer emits 0.000000 rather than a
    negative number. Worth pinning: the clamp lives in `utils`, not here, and
    dropping it would push nonsense downstream instead of a flagged degenerate.
    """
    parts = S._box_to_yolo_line(BBox(90, 80, 10, 20), W, H).split()
    assert float(parts[3]) == 0.0 and float(parts[4]) == 0.0


def test_a_box_overhanging_both_edges_is_written_unclipped(tmp_path):
    """The writer has no clipping guard — the geometry is the only safeguard.

    Documented deliberately: `_box_to_yolo_line` will happily emit 1.5 for a box
    taller than the image. Here the parser does catch it, which is the lucky
    case; see the next test for the one it misses.
    """
    line = S._box_to_yolo_line(BBox(10, -40, 60, 200), W, H)
    assert float(line.split()[4]) > 1.0, line
    p = tmp_path / "bad.txt"
    p.write_text(line, encoding="utf-8")
    _, _, _, errors = parse_yolo_label(p, W, H)
    assert any("not normalised" in e for e in errors)


def test_a_box_hanging_off_one_edge_slips_past_the_range_check(tmp_path):
    """The [0, 1] check on cx/cy/w/h does NOT mean the box is inside the image.

    y1=100, y2=200 in a 160px frame encodes as cy=0.9375, h=0.625 — both legal,
    so nothing downstream complains, yet the decoded box reaches 40px past the
    bottom edge. This is why "boxes stay inside the frame" has to be asserted on
    the geometry itself and cannot be delegated to the label validator.
    """
    line = S._box_to_yolo_line(BBox(10, 100, 60, 200), W, H)
    assert all(0.0 <= float(v) <= 1.0 for v in line.split()[1:]), line
    p = tmp_path / "sneaky.txt"
    p.write_text(line, encoding="utf-8")
    boxes, _, _, errors = parse_yolo_label(p, W, H)
    assert errors == []
    assert boxes[0].y2 == pytest.approx(200.0, abs=1e-2)  # 40px outside a 160px frame


# --------------------------------------------------------------------------- #
# the geometry the labels describe — do the boxes point at the actual damage?
# --------------------------------------------------------------------------- #
def test_every_dark_pixel_belongs_to_a_labelled_defect():
    """With confounders off, the only dark thing painted is labelled damage.

    This is the test that catches a box which is the right size but in the wrong
    place: shifting a box by a few pixels leaves part of the damage uncovered,
    and uncovered damage is a false negative that inflates precision.
    """
    for seed in range(12):
        img, boxes = _gen(seed=seed, num_damage=2)
        stray = (_luma(img) < DARK) & ~_covered(boxes, img.shape[:2])
        assert stray.sum() == 0, (
            f"seed {seed}: {int(stray.sum())} damage pixels outside every ground-truth box")


def test_every_label_box_actually_covers_dark_pixels():
    """The converse: a box over clean mesh is a phantom defect.

    A tear is a rotated rectangle, so its axis-aligned box is mostly background
    at 45 degrees — 10% is a floor, not an expectation.
    """
    for seed in range(12):
        img, boxes = _gen(seed=seed, num_damage=2)
        g = _luma(img)
        for b in boxes:
            patch = g[int(b.y1):int(b.y2) + 1, int(b.x1):int(b.x2) + 1]
            assert patch.size > 0
            assert (patch < DARK).mean() > 0.10, (
                f"seed {seed}: {b.class_name} box at {b.to_list()} contains no damage")


def test_generated_boxes_stay_inside_the_frame():
    """The invariant the unclipped writer depends on."""
    for seed in range(40):
        img, boxes = _gen(seed=seed, num_damage=3, distractors=True)
        h, w = img.shape[:2]
        for b in boxes:
            assert 0 <= b.x1 < b.x2 <= w, f"seed {seed}: x out of frame: {b}"
            assert 0 <= b.y1 < b.y2 <= h, f"seed {seed}: y out of frame: {b}"


def test_generated_boxes_have_positive_area():
    for seed in range(40):
        _, boxes = _gen(seed=seed, num_damage=3)
        for b in boxes:
            assert b.area > 0, f"seed {seed}: degenerate ground-truth box {b}"


def test_holes_and_tears_carry_the_documented_class_ids():
    """Both kinds must actually be produced, and each with its own id.

    A coin flip that got stuck (or two kinds collapsed onto one class id) would
    quietly turn this into a single-class dataset while the reports still print
    a per-class breakdown.
    """
    kinds = {(b.class_id, b.class_name) for s in range(20) for b in _gen(seed=s, num_damage=2)[1]}
    assert kinds == {(1, "hole"), (2, "tear")}


def test_a_hole_is_centred_on_the_ellipse_it_paints():
    """Box centre must coincide with the centroid of the dark pixels it covers.

    Catches an asymmetric box (e.g. cx-rx .. cx+2*rx) that still passes the
    coverage tests above because it merely over-claims on one side.
    """
    for seed in range(20):
        img, boxes = _gen(seed=seed, num_damage=2)
        ys, xs = np.nonzero(_luma(img) < DARK)
        for b in [x for x in boxes if x.class_name == "hole"]:
            inside = ((xs >= b.x1) & (xs <= b.x2) & (ys >= b.y1) & (ys <= b.y2))
            assert xs[inside].mean() == pytest.approx((b.x1 + b.x2) / 2, abs=2.0)
            assert ys[inside].mean() == pytest.approx((b.y1 + b.y2) / 2, abs=2.0)


def test_a_tear_box_cannot_escape_the_frame_on_a_wide_short_image():
    """Regression: a tear's extent comes from the image WIDTH while its centre is
    bounded by the HEIGHT, so on a wide, short frame the rotated rectangle's
    corners landed outside the image and nothing clipped them. The box was then
    written to a YOLO label with normalised coordinates outside [0, 1], which
    `data.parse_yolo_label` refuses to read back. 217 of 600 boxes were affected
    at 100x800 before `_clip` was added.
    """
    for seed in range(20):
        img, boxes = S.make_synthetic_image(height=100, width=800, num_damage=3,
                                            seed=seed, distractors=False)
        h, w = img.shape[:2]
        for b in boxes:
            assert 0 <= b.y1 and b.y2 <= h, f"seed {seed}: {b}"
            assert 0 <= b.x1 and b.x2 <= w, f"seed {seed}: {b}"


def test_a_narrow_frame_does_not_crash_the_generator():
    """Regression: every size here is a percentage of a dimension, and on a small
    frame two percentages collapse to the same integer — `rng.integers(3, 3)`
    raises "low >= high". Below 134 px wide the generator crashed on roughly half
    of all seeds, whichever way the hole/tear coin came down.
    """
    for w in (16, 32, 64, 100, 133, 134):
        for seed in range(10):
            img, boxes = S.make_synthetic_image(height=w, width=w, num_damage=2,
                                                seed=seed, distractors=False)
            assert img.shape == (w, w, 3)
            for b in boxes:
                assert b.x2 > b.x1 and b.y2 > b.y1, (
                    f"{w}px seed {seed}: zero-area ground-truth box {b}")


# --------------------------------------------------------------------------- #
# image invariants
# --------------------------------------------------------------------------- #
def test_the_image_has_the_requested_size_and_is_rgb_uint8():
    img, _ = S.make_synthetic_image(height=97, width=181, num_damage=0, seed=0)
    assert img.shape == (97, 181, 3)
    assert img.dtype == np.uint8 and img.min() >= 0 and img.max() <= 255


def test_the_image_is_rgb_not_bgr():
    """Channel order is a repo-wide contract: `write_image` assumes RGB in.

    Underwater, blue survives and red dies, so B > G > R. If this module emitted
    BGR the saved demo images would be red-tinted and every colour-based cue in
    the classical baseline would be tuned against the wrong channel.
    """
    img, _ = S.make_synthetic_image(height=64, width=160, num_damage=0, seed=0,
                                    distractors=False)
    r, g, b = (float(img[..., i].mean()) for i in range(3))
    assert r < g < b, f"expected blue-dominant underwater RGB, got R={r} G={g} B={b}"


def test_the_scene_is_lit_from_above():
    """The gradient parameter claims "brighter near the top"; verify it is real."""
    img, _ = S.make_synthetic_image(height=200, width=64, num_damage=0, seed=0,
                                    distractors=False)
    top = _luma(img[:40, 20:44]).mean()
    bottom = _luma(img[-40:, 20:44]).mean()
    assert top > bottom + 5, f"top {top:.1f} vs bottom {bottom:.1f}"


def test_the_background_carries_per_pixel_noise():
    """A noiseless background makes the demo task trivially easy.

    Measured as the median absolute difference between horizontal neighbours,
    with the mesh spaced off the frame so only the background is left: the
    smooth gradient contributes almost nothing to that statistic, so it isolates
    the noise term rather than re-measuring the overall contrast.
    """
    img, _ = _gen(seed=0, num_damage=0, mesh_spacing=10_000)
    g = _luma(img)
    assert g.std() > 5.0, "the frame collapsed to a flat fill"
    assert float(np.median(np.abs(np.diff(g, axis=1)))) > 1.5, "the grain is gone"


def test_the_requested_number_of_defects_is_the_number_of_labels():
    for n in (0, 1, 2, 5):
        _, boxes = _gen(seed=4, num_damage=n)
        assert len(boxes) == n


def test_mesh_spacing_controls_mesh_density():
    """Halving the spacing must roughly double the painted mesh.

    A parameter that is accepted and ignored is the easiest kind of dead knob to
    ship, and this one changes how hard the detection task is.
    """
    frac = []
    for spacing in (8, 16, 32, 64):
        img, _ = _gen(seed=3, num_damage=0, mesh_spacing=spacing)
        frac.append(float((_luma(img) > 140).mean()))
    assert frac == sorted(frac, reverse=True), f"mesh did not thin out: {frac}"
    assert frac[0] > 4 * frac[-1], f"spacing barely mattered: {frac}"


def test_confounders_are_drawn_but_never_labelled():
    """Distractors exist to generate honest false positives.

    If a future edit appended them to the ground truth, the baseline would score
    against its own confounders and every metric in the reports would improve
    for no real reason.
    """
    for seed in range(10):
        _, boxes = S.make_synthetic_image(height=H, width=W, num_damage=2,
                                          seed=seed, distractors=True)
        assert len(boxes) == 2


def test_a_shadow_distractor_darkens_the_frame_without_raising():
    """Seed 6 trips both confounder branches (shadow and biofouling)."""
    img = np.full((H, W, 3), 120, np.uint8)
    S._add_distractors(img, np.random.default_rng(6))
    assert img.dtype == np.uint8 and img.shape == (H, W, 3)
    assert img.mean() < 119.0, "the shadow branch left the frame untouched"


def test_distractors_are_probabilistic_not_unconditional():
    """Seed 5 skips both branches — proof the 0.7/0.6 gates are live.

    Without this, dropping the probability checks would go unnoticed and every
    single frame would carry a confounder.
    """
    img = np.full((H, W, 3), 120, np.uint8)
    before = img.copy()
    S._add_distractors(img, np.random.default_rng(5))
    assert np.array_equal(img, before)


def test_the_generator_still_works_without_opencv(monkeypatch):
    """The module advertises a numpy fallback; the fallback also emits labels.

    Its geometry is a *different* code path (axis-aligned streaks, mask-based
    ellipses) that had never been executed by the test suite, and it divides by
    the ellipse radius.
    """
    monkeypatch.setattr(S, "optional_import", lambda name: None)
    for seed in range(6):
        img, boxes = _gen(seed=seed, num_damage=2)
        h, w = img.shape[:2]
        assert img.dtype == np.uint8 and img.shape == (H, W, 3)
        assert len(boxes) == 2
        for b in boxes:
            assert b.area > 0, f"seed {seed}: {b}"
            assert 0 <= b.x1 < b.x2 <= w and 0 <= b.y1 < b.y2 <= h, f"seed {seed}: {b}"


# --------------------------------------------------------------------------- #
# determinism — an unreproducible demo cannot be audited
# --------------------------------------------------------------------------- #
def test_the_same_seed_reproduces_the_image_and_the_boxes():
    a_img, a_box = _gen(seed=11, num_damage=3, distractors=True)
    b_img, b_box = _gen(seed=11, num_damage=3, distractors=True)
    assert np.array_equal(a_img, b_img)
    assert [b.to_dict() for b in a_box] == [b.to_dict() for b in b_box]


def test_different_seeds_produce_different_images_and_boxes():
    """Guards the opposite failure: a seed that is accepted and then ignored."""
    a_img, a_box = _gen(seed=11, num_damage=3)
    b_img, b_box = _gen(seed=12, num_damage=3)
    assert not np.array_equal(a_img, b_img)
    assert [b.to_list() for b in a_box] != [b.to_list() for b in b_box]


def test_generate_dataset_is_byte_identical_for_a_seed(tmp_path):
    """Byte level, because JPEG encoding is part of what has to be reproducible."""
    S.generate_dataset(tmp_path / "a", n_images=3, seed=0)
    S.generate_dataset(tmp_path / "b", n_images=3, seed=0)
    assert _tree_digest(tmp_path / "a") == _tree_digest(tmp_path / "b")


def test_generate_dataset_changes_with_the_seed(tmp_path):
    S.generate_dataset(tmp_path / "a", n_images=3, seed=0)
    S.generate_dataset(tmp_path / "c", n_images=3, seed=99)
    a, c = _tree_digest(tmp_path / "a"), _tree_digest(tmp_path / "c")
    assert set(a) == set(c), "the layout must not depend on the seed"
    assert a != c, "the seed was accepted and ignored"


def test_each_image_in_a_dataset_is_a_different_scene(tmp_path):
    """`seed + i + 1` must actually vary per image, not repeat one frame."""
    info = S.generate_dataset(tmp_path / "ds", n_images=5, seed=0)
    digests = {h for k, h in _tree_digest(tmp_path / "ds").items() if k.startswith("images/")}
    assert len(digests) == info["n_images"]


# --------------------------------------------------------------------------- #
# generate_dataset — the artefact make_report_assets.py measures against
# --------------------------------------------------------------------------- #
def test_labels_are_written_only_for_damaged_frames(tmp_path):
    """Clean frames get *no* file, not an empty one.

    The distinction matters downstream: `data.find_label_path` treats a missing
    file as an unlabelled image and an empty file as a labelled negative. Getting
    it backwards removes the negatives, and with them the only measurement of
    false alarms in the whole report.
    """
    S.generate_dataset(tmp_path / "ds", n_images=6, n_damaged=4, seed=0)
    images = sorted((tmp_path / "ds" / "images").glob("*.jpg"))
    labels = sorted((tmp_path / "ds" / "labels").glob("*.txt"))
    assert len(images) == 6
    assert [p.stem for p in labels] == [f"synthetic_{i:03d}" for i in range(4)]
    assert all(p.read_text(encoding="utf-8").strip() for p in labels)


def test_every_written_label_line_is_a_valid_yolo_detection_line(tmp_path):
    """The last gate before the evaluator: nothing out of [0, 1], no odd fields."""
    S.generate_dataset(tmp_path / "ds", n_images=6, seed=1)
    seen = 0
    for path in sorted((tmp_path / "ds" / "labels").glob("*.txt")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            assert len(parts) == 5, f"{path.name}:{lineno}: {line!r}"
            assert int(parts[0]) in range(len(DEFAULT_CLASSES))
            for v in parts[1:]:
                assert 0.0 <= float(v) <= 1.0, f"{path.name}:{lineno}: {v} out of range"
            seen += 1
    assert seen > 0, "the dataset carried no labels at all"


def test_the_written_dataset_reloads_through_the_production_loader(tmp_path):
    """End-to-end: what `make_report_assets.py` does, minus the model.

    `load_dataset` re-derives width and height from the JPEG on disk, so this
    also catches labels normalised against a size the image does not have.
    """
    info = S.generate_dataset(tmp_path / "ds", n_images=6, seed=2)
    samples = load_dataset(tmp_path / "ds" / "images", tmp_path / "ds" / "labels")
    assert len(samples) == info["n_images"]
    assert not [e for s in samples for e in s.label_errors]
    by_name = {s.image_path.name: s for s in samples}
    for entry in info["manifest"]:
        assert len(by_name[entry["image"]].boxes) == entry["num_damage"]
    for s in samples:
        for b in s.boxes:
            assert b.area > 0
            assert 0 <= b.x1 < b.x2 <= s.width and 0 <= b.y1 < b.y2 <= s.height


def test_the_manifest_matches_the_files_on_disk(tmp_path):
    """The manifest is the summary a caller trusts instead of listing the tree."""
    info = S.generate_dataset(tmp_path / "ds", n_images=5, seed=3)
    assert len(info["manifest"]) == info["n_images"] == 5
    for entry in info["manifest"]:
        label = tmp_path / "ds" / "labels" / (Path(entry["image"]).stem + ".txt")
        on_disk = len(label.read_text(encoding="utf-8").splitlines()) if label.exists() else 0
        assert on_disk == entry["num_damage"], entry


def test_a_quarter_of_the_default_dataset_is_left_clean(tmp_path):
    """Negatives are not optional: without them precision is unmeasurable.

    The module docstring promises "a fraction of images are left undamaged"; the
    default is three quarters damaged.
    """
    info = S.generate_dataset(tmp_path / "ds", n_images=8, seed=0)
    assert info["n_damaged"] == 6
    clean = [m for m in info["manifest"] if m["num_damage"] == 0]
    assert len(clean) == 2
    assert not (tmp_path / "ds" / "labels" / "synthetic_007.txt").exists()


def test_every_damaged_frame_carries_at_least_one_defect(tmp_path):
    """A "damaged" frame with zero boxes would be an unlabelled positive."""
    info = S.generate_dataset(tmp_path / "ds", n_images=8, seed=5)
    for entry in info["manifest"][:info["n_damaged"]]:
        assert entry["num_damage"] >= 1, entry


def test_the_synthetic_placeholder_marker_is_written(tmp_path):
    """The honesty guard rail that stops this being mistaken for real data.

    It ships next to the images so anyone who finds the directory later, without
    the README, still sees the warning.
    """
    S.generate_dataset(tmp_path / "ds", n_images=2, seed=0)
    text = (tmp_path / "ds" / "SYNTHETIC_PLACEHOLDER.txt").read_text(encoding="utf-8")
    assert "synthetic" in text.lower()
    assert "not" in text.lower() and "real" in text.lower()


def test_the_summary_reports_where_it_wrote(tmp_path):
    info = S.generate_dataset(tmp_path / "ds", n_images=2, seed=0)
    assert Path(info["out_dir"]) == tmp_path / "ds"
    assert info["n_images"] == 2
