"""Demonstrate the OOD gate: does it flag different-day frames for human review?

Calibrates the gate on in-distribution (training-clip) frames, then measures the
flag rate as backgrounds drift away from training: same clip -> same site/other
clip -> different DAY. A good gate flags few in-distribution frames and many
different-day ones, so a deployed system defers exactly the frames it is least
qualified to judge.

Example
-------
    python scripts/eval_ood_gate.py --patchcore-model models/patchcore_normal_net \\
        --out reports/results/ood_gate
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401

from netinspect.ood_gate import OODGate
from netinspect.patchcore import PatchCoreModel
from netinspect.utils import ensure_dir, get_logger, list_images, read_image, write_json

LOGGER = get_logger()

SETS = {
    "in-clip (training backgrounds)": "data/processed/solaqua_frames",
    "same site, other clip": "data/processed/solaqua_bag2",
    "different DAY": "data/processed/solaqua_diffday",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patchcore-model", default="models/patchcore_normal_net")
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--max-frames", type=int, default=40, help="Cap per set (speed)")
    ap.add_argument("--out", default="reports/results/ood_gate")
    args = ap.parse_args()

    model = PatchCoreModel.load(args.patchcore_model)
    scores = {}
    for name, d in SETS.items():
        imgs = list_images(d)[: args.max_frames]
        if not imgs:
            continue
        LOGGER.info("Scoring %d frames: %s", len(imgs), name)
        scores[name] = [OODGate.frame_score(read_image(p), model) for p in imgs]

    cal_key = "in-clip (training backgrounds)"
    gate = OODGate.calibrate(scores[cal_key], args.percentile)
    rows = {name: {"frames": len(s), "flag_rate": gate.flag_rate(s),
                   "mean_score": round(sum(s) / len(s), 3)}
            for name, s in scores.items()}

    out = ensure_dir(args.out)
    write_json({"threshold": gate.threshold, "percentile": args.percentile, "sets": rows},
               out / "ood_gate.json")
    md = ["# Out-of-distribution gate — route shifted frames to human review\n",
          f"Gate threshold = p{args.percentile:.0f} of in-distribution scores "
          f"(= {gate.threshold:.3f}). Higher flag rate = more frames deferred to a human.\n",
          "| Frame set | Frames | Mean OOD score | Flagged for review |",
          "|---|---|---|---|"]
    for name, r in rows.items():
        md.append(f"| {name} | {r['frames']} | {r['mean_score']} | {r['flag_rate']:.0%} |")
    md.append("\n> The gate flags *distribution shift*, not damage. It lets a not-yet-certified "
              "detector run safely: auto-handle familiar frames, defer the unfamiliar ones. "
              "Scores come from the label-free PatchCore anomaly model.")
    (out / "ood_gate.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {out / 'ood_gate.md'}")


if __name__ == "__main__":
    main()
