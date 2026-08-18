"""Tests for the `netinspect` command line.

The CLI is the only part of this toolkit most users will ever touch, so the
behaviour under test is mostly about exit codes: a command that fails must say
so in a way a deployment pipeline can act on, and a command that refuses must
not leave a half-written dataset behind.
"""
from __future__ import annotations

import json

import pytest

from netinspect import cli

pytest.importorskip("PIL")


def _dataset(root, clips=6, per_clip=6, label="0 0.5 0.5 0.2 0.2"):
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)
    for c in range(clips):
        for i in range(per_clip):
            img = root / "images" / f"clip{c}_{i:04d}.jpg"
            img.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype="uint8")).save(img)
            lbl = root / "labels" / f"clip{c}_{i:04d}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            # Half the frames clean, so both rates are measurable.
            lbl.write_text(label if i % 2 == 0 else "", encoding="utf-8")
    return root


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def test_help_lists_the_workflow_commands(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for command in ("doctor", "onboard", "train", "calibrate", "gate", "serve"):
        assert command in out


def test_no_arguments_prints_help_rather_than_an_error(capsys):
    assert cli.main([]) == 0
    assert "usage: netinspect" in capsys.readouterr().out


def test_version(capsys):
    assert cli.main(["version"]) == 0
    assert "netinspect" in capsys.readouterr().out


def test_doctor_reports_the_environment_and_succeeds(capsys):
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "numpy" in out
    # The honesty reminder is part of the contract, not decoration.
    assert "SYNTHETIC" in out


# --------------------------------------------------------------------------- #
# onboard
# --------------------------------------------------------------------------- #
def test_onboard_prepares_a_trainable_dataset(tmp_path, capsys):
    src = _dataset(tmp_path / "src")
    out = tmp_path / "prepared"
    assert cli.main(["onboard", str(src), "--out", str(out)]) == 0
    assert (out / "dataset.yaml").exists()
    assert (out / "data_health.json").exists()
    for split in ("train", "val", "test"):
        assert (out / "images" / split).is_dir()
    assert "next:  netinspect train" in capsys.readouterr().out


def test_onboard_writes_a_health_report_naming_the_split_strategy(tmp_path):
    src = _dataset(tmp_path / "src")
    out = tmp_path / "prepared"
    cli.main(["onboard", str(src), "--out", str(out)])
    rep = json.loads((out / "data_health.json").read_text(encoding="utf-8"))
    assert "grouped" in rep["split_strategy"]
    assert rep["usable"] is True
    assert sum(s["images"] for s in rep["splits"].values()) == rep["images"]


def test_onboard_refuses_broken_labels_and_writes_no_dataset(tmp_path, capsys):
    # Pixel coordinates handed to a normalised format — the classic unit error.
    src = _dataset(tmp_path / "src", label="0 320 240 64 48")
    out = tmp_path / "prepared"
    assert cli.main(["onboard", str(src), "--out", str(out)]) == 1
    assert not (out / "dataset.yaml").exists(), "must not write a dataset it just rejected"
    assert (out / "data_health.json").exists(), "but must still explain why"
    assert "Refusing" in capsys.readouterr().err


def test_onboard_refuses_when_there_are_too_few_clips_to_split(tmp_path, capsys):
    """Two clips cannot make three splits; an empty split is an error."""
    src = _dataset(tmp_path / "src", clips=2, per_clip=10)
    assert cli.main(["onboard", str(src), "--out", str(tmp_path / "prepared")]) == 1
    assert "empty" in json.dumps(
        json.loads((tmp_path / "prepared" / "data_health.json").read_text(encoding="utf-8")))


def test_onboard_reports_the_split_sizes(tmp_path, capsys):
    src = _dataset(tmp_path / "src")
    cli.main(["onboard", str(src), "--out", str(tmp_path / "prepared")])
    out = capsys.readouterr().out
    assert "train" in out and "val" in out and "test" in out
    assert "clip(s)" in out


def test_onboard_on_a_missing_folder_fails_loudly(tmp_path):
    with pytest.raises((FileNotFoundError, SystemExit, ValueError)):
        cli.main(["onboard", str(tmp_path / "nope"), "--out", str(tmp_path / "o")])


# --------------------------------------------------------------------------- #
# delegation
# --------------------------------------------------------------------------- #
def test_delegating_to_a_missing_script_reports_it_instead_of_crashing(capsys):
    assert cli._delegate("definitely_not_a_script.py", []) == 2
    assert "not available" in capsys.readouterr().err


def test_wrapped_commands_forward_their_arguments(monkeypatch):
    """`netinspect train --epochs 5` must reach the script as `--epochs 5`.

    Regression: argparse.REMAINDER still intercepts a leading "--flag" as an
    unknown top-level option, so this failed before ever reaching the script.
    """
    seen = {}
    monkeypatch.setattr(cli, "_delegate",
                        lambda script, args: seen.update(script=script, args=list(args)) or 0)
    assert cli.main(["train", "--epochs", "5", "--data", "x.yaml"]) == 0
    assert seen["script"] == "train_yolo.py"
    assert seen["args"] == ["--epochs", "5", "--data", "x.yaml"]


def test_every_wrapped_command_names_a_script_that_exists():
    for name, (script, _) in cli.WRAPPED.items():
        assert (cli.SCRIPTS / script).exists(), f"{name} -> missing {script}"


# --------------------------------------------------------------------------- #
# gate wiring (the model itself is exercised in test_acceptance)
# --------------------------------------------------------------------------- #
def test_gate_on_a_missing_split_fails_rather_than_passing_vacuously(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["gate", "--data", str(tmp_path / "nothing"), "--split", "test"])


def test_prepared_split_loads_its_labels_from_the_sibling_tree(tmp_path):
    """Regression: sniffing the format inside images/<split>/ found no .txt and
    silently returned unlabelled samples, so the gate saw zero damaged frames."""
    src = _dataset(tmp_path / "src")
    out = tmp_path / "prepared"
    cli.main(["onboard", str(src), "--out", str(out)])
    samples = cli._load_split(out / "images" / "test")
    assert samples, "test split should not be empty"
    assert any(s.boxes for s in samples), "ground-truth boxes must survive loading"
