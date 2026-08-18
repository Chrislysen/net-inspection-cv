"""Tests for bring-your-own-data ingestion, auditing and splitting.

The defect that matters in this module never raises. It produces a dataset that
trains fine and reports a number that is wrong in the flattering direction, so
these tests are mostly about leakage, negatives and unit confusion.
"""
from __future__ import annotations

import json

import pytest

from netinspect import dataset as D

pytest.importorskip("PIL")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _img(path, w=64, h=48, shade=128):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (shade, shade, shade)).save(path)
    return path


def _textured(path, seed=0, w=64, h=48):
    """A distinguishable image, so perceptual hashes differ between frames."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype="uint8")).save(path)
    return path


def _yolo_set(root, clips=("clipA", "clipB", "clipC"), per_clip=4, boxes=True):
    for c in clips:
        for i in range(per_clip):
            _textured(root / "images" / f"{c}_{i:04d}.jpg", seed=hash((c, i)) % 9999)
            lbl = root / "labels" / f"{c}_{i:04d}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("0 0.5 0.5 0.2 0.2\n" if boxes else "", encoding="utf-8")
    return root


# --------------------------------------------------------------------------- #
# format detection + loading
# --------------------------------------------------------------------------- #
def test_detects_yolo_and_loads_boxes(tmp_path):
    _yolo_set(tmp_path)
    assert D.detect_format(tmp_path) == "yolo"
    s = D.load_dataset(tmp_path)
    assert len(s) == 12
    assert all(len(x.boxes) == 1 for x in s)
    b = s[0].boxes[0]
    assert b.x1 == pytest.approx(0.4) and b.x2 == pytest.approx(0.6)


def test_images_with_no_labels_are_a_valid_dataset(tmp_path):
    for i in range(5):
        _img(tmp_path / f"clip_{i:04d}.jpg")
    assert D.detect_format(tmp_path) == "images"
    s = D.load_dataset(tmp_path)
    assert len(s) == 5 and all(x.is_negative for x in s)


def test_detects_and_loads_coco_converting_pixels_to_normalised(tmp_path):
    _img(tmp_path / "images" / "clipA_0001.jpg", w=100, h=50)
    (tmp_path / "ann.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "clipA_0001.jpg", "width": 100, "height": 50}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 5, 20, 10]}],
        "categories": [{"id": 0, "name": "damage"}],
    }), encoding="utf-8")
    assert D.detect_format(tmp_path) == "coco"
    s = D.load_dataset(tmp_path)
    assert len(s) == 1
    b = s[0].boxes[0]
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((0.1, 0.1, 0.3, 0.3))


def test_detects_and_loads_voc(tmp_path):
    _img(tmp_path / "clipA_0001.jpg", w=100, h=100)
    (tmp_path / "clipA_0001.xml").write_text(
        "<annotation><object><name>hole</name><bndbox>"
        "<xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax>"
        "</bndbox></object></annotation>", encoding="utf-8")
    assert D.detect_format(tmp_path) == "voc"
    b = D.load_dataset(tmp_path)[0].boxes[0]
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((0.1, 0.2, 0.3, 0.4))


def test_an_empty_directory_is_refused_rather_than_returning_nothing(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        D.load_dataset(tmp_path / "empty")


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #
def test_frames_from_one_clip_share_a_group():
    from pathlib import Path
    assert D.group_of(Path("2024-08-22_14-29-05_video_000123.jpg")) == "2024-08-22_14-29-05_video"
    assert D.group_of(Path("clipA_0007.png")) == "clipA"


def test_a_name_without_a_frame_number_is_its_own_group():
    from pathlib import Path
    assert D.group_of(Path("some_photo.jpg")) == "some_photo"


# --------------------------------------------------------------------------- #
# splitting — the leakage-critical part
# --------------------------------------------------------------------------- #
def test_a_clip_never_straddles_two_splits(tmp_path):
    _yolo_set(tmp_path, clips=[f"clip{i}" for i in range(9)], per_clip=5)
    splits = D.split_samples(D.load_dataset(tmp_path), seed=1)
    seen = {}
    for name, samples in splits.items():
        for s in samples:
            assert seen.setdefault(s.group, name) == name, \
                f"clip {s.group} appears in more than one split"


def test_every_sample_lands_in_exactly_one_split(tmp_path):
    _yolo_set(tmp_path, clips=[f"clip{i}" for i in range(6)], per_clip=3)
    samples = D.load_dataset(tmp_path)
    splits = D.split_samples(samples, seed=3)
    placed = [s.image for v in splits.values() for s in v]
    assert len(placed) == len(samples)
    assert len(set(placed)) == len(samples)


def test_ungrouped_splitting_is_available_but_not_the_default(tmp_path):
    _yolo_set(tmp_path, clips=("only_one",), per_clip=30)
    grouped = D.split_samples(D.load_dataset(tmp_path), seed=0)
    # One clip cannot be split when grouping — it all goes to one side, which is
    # the honest outcome rather than a fake split.
    assert sum(1 for v in grouped.values() if v) == 1
    loose = D.split_samples(D.load_dataset(tmp_path), seed=0, group=False)
    assert sum(1 for v in loose.values() if v) == 3


def test_ratios_must_sum_to_one(tmp_path):
    _yolo_set(tmp_path)
    with pytest.raises(ValueError):
        D.split_samples(D.load_dataset(tmp_path), ratios=(0.5, 0.4, 0.3))


def test_the_split_is_deterministic_for_a_seed(tmp_path):
    _yolo_set(tmp_path, clips=[f"c{i}" for i in range(8)], per_clip=2)
    s = D.load_dataset(tmp_path)
    a = {k: [x.image for x in v] for k, v in D.split_samples(s, seed=7).items()}
    b = {k: [x.image for x in v] for k, v in D.split_samples(s, seed=7).items()}
    assert a == b


# --------------------------------------------------------------------------- #
# perceptual hashing / duplicate detection
# --------------------------------------------------------------------------- #
def test_identical_images_hash_identically(tmp_path):
    a = _textured(tmp_path / "a.jpg", seed=5)
    b = _textured(tmp_path / "b.jpg", seed=5)
    assert D.hamming(D.dhash(a), D.dhash(b)) == 0


def test_different_images_hash_differently(tmp_path):
    a = _textured(tmp_path / "a.jpg", seed=1)
    b = _textured(tmp_path / "b.jpg", seed=999)
    assert D.hamming(D.dhash(a), D.dhash(b)) > 4


def test_the_same_footage_in_two_splits_is_caught(tmp_path):
    """Grouping by clip does not catch footage exported twice under new names."""
    a = _textured(tmp_path / "clipA_0001.jpg", seed=42)
    b = _textured(tmp_path / "renamed_0001.jpg", seed=42)   # same picture
    splits = {"train": [D.Sample(image=a, group="clipA")],
              "test": [D.Sample(image=b, group="renamed")]}
    dupes = D.find_cross_split_duplicates(splits)
    assert dupes, "an identical frame across splits must be reported"


def test_duplicates_within_one_split_are_not_reported(tmp_path):
    a = _textured(tmp_path / "clipA_0001.jpg", seed=42)
    b = _textured(tmp_path / "clipA_0002.jpg", seed=42)
    splits = {"train": [D.Sample(image=a, group="clipA"), D.Sample(image=b, group="clipA")]}
    assert D.find_cross_split_duplicates(splits) == []


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def _sample(boxes, group="c", w=100, h=100):
    from pathlib import Path
    return D.Sample(image=Path("x.jpg"), boxes=boxes, group=group, width=w, height=h)


def test_a_clean_dataset_raises_no_errors(tmp_path):
    _yolo_set(tmp_path, clips=[f"c{i}" for i in range(5)], per_clip=6)
    samples = D.load_dataset(tmp_path)
    # Half the frames clean, so the false-alarm warning does not fire.
    for s in samples[::2]:
        s.boxes = []
    assert [i for i in D.audit(samples) if i.severity == "error"] == []


def test_a_zero_area_box_is_an_error_not_a_warning():
    issues = D.audit([_sample([D.Box(0, 0.5, 0.5, 0.5, 0.5)])])
    assert any(i.kind == "degenerate_box" and i.severity == "error" for i in issues)


def test_pixel_coordinates_left_unnormalised_are_caught():
    """The classic unit mix-up: xyxy in pixels handed to a normalised loader."""
    issues = D.audit([_sample([D.Box(0, 10, 20, 30, 40)])])
    assert any(i.kind == "box_out_of_bounds" and i.severity == "error" for i in issues)


def test_a_dataset_with_no_clean_frames_is_flagged():
    samples = [_sample([D.Box(0, 0.1, 0.1, 0.2, 0.2)], group=f"c{i}") for i in range(10)]
    assert any(i.kind == "few_negatives" for i in D.audit(samples))


def test_a_single_clip_dataset_is_flagged_as_not_really_held_out():
    samples = [_sample([D.Box(0, 0.1, 0.1, 0.2, 0.2)], group="only") for _ in range(10)]
    assert any(i.kind == "too_few_groups" for i in D.audit(samples))


def test_an_unlabelled_dataset_is_a_warning_not_a_failure():
    issues = D.audit([_sample([], group=f"c{i}") for i in range(5)])
    assert any(i.kind == "no_labels" and i.severity == "warning" for i in issues)
    assert not [i for i in issues if i.severity == "error"]


def test_an_empty_dataset_is_an_error():
    assert any(i.severity == "error" for i in D.audit([]))


# --------------------------------------------------------------------------- #
# writing + reporting
# --------------------------------------------------------------------------- #
def test_write_yolo_produces_a_loadable_dataset(tmp_path):
    src = _yolo_set(tmp_path / "src", clips=[f"c{i}" for i in range(6)], per_clip=3)
    splits = D.split_samples(D.load_dataset(src), seed=2)
    yaml_path = D.write_yolo(splits, tmp_path / "out")
    assert yaml_path.exists()
    text = yaml_path.read_text(encoding="utf-8")
    assert "train: images/train" in text and "damage" in text
    for name in ("train", "val", "test"):
        assert (tmp_path / "out" / "images" / name).is_dir()
    # Round-trip from the dataset ROOT, which exercises the images/ -> labels/
    # sibling-tree resolution that the YOLO layout depends on.
    again = D.load_dataset(tmp_path / "out")
    assert len(again) == 18
    assert all(len(s.boxes) == 1 for s in again), "labels must survive the round trip"


def test_health_report_records_which_split_strategy_was_used(tmp_path):
    _yolo_set(tmp_path, clips=[f"c{i}" for i in range(6)], per_clip=3)
    samples = D.load_dataset(tmp_path)
    splits = D.split_samples(samples, seed=0)
    rep = D.health_report(samples, splits, D.audit(samples), [], grouped=True)
    assert "grouped" in rep["split_strategy"]
    assert rep["images"] == len(samples)
    assert set(rep["splits"]) == {"train", "val", "test"}
    assert "grouped" in rep["note"]


def test_a_report_with_blocking_errors_is_marked_unusable():
    rep = D.health_report([], {"train": [], "val": [], "test": []}, D.audit([]), [], True)
    assert rep["usable"] is False and rep["blocking_errors"] >= 1


def test_summary_surfaces_duplicates_loudly(tmp_path):
    _yolo_set(tmp_path, clips=[f"c{i}" for i in range(6)], per_clip=2)
    samples = D.load_dataset(tmp_path)
    splits = D.split_samples(samples, seed=0)
    rep = D.health_report(samples, splits, [], [("a.jpg", "b.jpg")], grouped=True)
    assert "near-duplicate" in D.summarise(rep)
