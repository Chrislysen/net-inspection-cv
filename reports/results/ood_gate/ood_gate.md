# Out-of-distribution gate — route shifted frames to human review

Gate threshold = p95 of in-distribution scores (= 2.699). Higher flag rate = more frames deferred to a human.

| Frame set | Frames | Mean OOD score | Flagged for review |
|---|---|---|---|
| in-clip (training backgrounds) | 35 | 2.229 | 6% |
| same site, other clip | 35 | 2.213 | 0% |
| different DAY | 35 | 3.165 | 100% |

> The gate flags *distribution shift*, not damage. It lets a not-yet-certified detector run safely: auto-handle familiar frames, defer the unfamiliar ones. Scores come from the label-free PatchCore anomaly model.