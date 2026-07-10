# Failure cases - S3DIS Area-5

Where the geometry and learned arms break, with real per-room / per-class numbers from the
reproducible runs (outputs/eval_report.json, outputs/hybrid_eval.json; global mIoU, full
resolution). This is the qualitative other half of EXPERIMENTS.md: *why* the remaining error
is where it is, and which of it PointNet++ (P3) is expected to reach.

Every number below is from a real in-repo run - no estimates.

## 1. Occluded ceilings - storage rooms (ceiling IoU = 0.00)

The geometry gate nails ceilings on average (global ceiling IoU **0.79**), but three rooms
score **exactly 0.00**:

| room | ceiling IoU | room mIoU | height |
|---|---|---|---|
| storage_1 | 0.000 | 0.426 | 4.63 m |
| storage_2 | 0.000 | 0.416 | - |
| storage_3 | 0.000 | 0.477 | - |

**Why:** storage rooms are packed with floor-to-ceiling shelving. The scanner never sees a
clean overhead plane, so RANSAC finds no ceiling to label - the topmost horizontal surface is
shelf tops, not the ceiling. This is an *acquisition* limit, not a heuristic bug: there is no
ceiling evidence in the cloud. A learned arm that infers "ceiling" from context (walls meeting
overhead, room extent) rather than a visible plane is the only thing that can help here.

## 2. Tall rooms - sparse far-field ceilings (ceiling IoU 0.09-0.16)

The next-worst ceilings are all unusually tall rooms:

| room | ceiling IoU | height |
|---|---|---|
| office_38 | 0.087 | 4.44 m |
| conferenceRoom_2 | 0.111 | 4.43 m |
| office_40 | 0.158 | 4.53 m |

(For contrast, a normal office is ~3.2 m and scores ceiling ~ 0.8-0.9.)

**Why:** at 4.4-4.6 m the ceiling is far from the scanner, so its points are sparse and noisy.
RANSAC still finds *a* plane but with weak, ragged support, and the normalized-height ceiling
heuristic (anchored on the topmost horizontal band) captures only part of it. The error is
proportional to height - a structured, systematic bias, not random.

## 3. Embedded openings - the naive hybrid cannibalizes doors/windows

Trusting geometry's blanket "wall" label actively *destroys* the classes that live inside walls.
Global per-class IoU, feature_ml vs the naive hybrid:

| class | feature_ml | hybrid (naive) | hybrid_v2 |
|---|---|---|---|
| door | 0.322 | **0.147** | 0.322 |
| window | 0.114 | **0.009** | 0.114 |

**Why:** doors and windows are coplanar with the wall, so geometry labels their points "wall"
and the naive hybrid takes that as ground truth - halving door IoU and erasing window IoU. This
is exactly what motivated **hybrid_v2** (trust geometry for floor/ceiling only, hand every
vertical plane to the model), which restores door/window to feature_ml parity. It is the
clearest evidence that geometry's *value is the scaffold* (segments + features), not its hard
labels: forcing the hard labels backfires precisely on the objects.

## 4. Rare, geometrically ambiguous objects - every current arm fails

Some classes sit near zero for **all four arms** - geometry, feature_ml, hybrid, hybrid_v2:

| class | best of the 4 arms |
|---|---|
| beam | 0.000 |
| sofa | 0.000 |
| board | 0.008 |
| column | 0.006 |

**Why:** these are rare in Area-5 and geometrically indistinguishable from their surroundings -
a beam is a wall-colored box against the ceiling, a column reads as wall, a board as a flat
patch on a wall, a sofa as generic clutter. Neither hand rules nor a RandomForest on segment
geometry has the shape/context to separate them. This is the **specific gap P3 (PointNet++) is
thought to close**: a deep network learning local shape + context is the arm most likely to move
beam/sofa/board/column off zero. If it does, "learning earns its keep on objects" gets real
teeth; if it does not, the scaffold reframe holds all the way up.

## Summary

| failure mode | rooms/classes | cause | who can fix it |
|---|---|---|---|
| Occluded ceiling | storage_1/2/3 (0.00) | shelving hides the ceiling plane | context-inferring learner |
| Sparse tall ceiling | office_38, conferenceRoom_2, office_40 | far-field sparse points at 4.4 m+ | denser sampling / learned prior |
| Embedded openings | door 0.32->0.15, window 0.11->0.01 | coplanar with wall; geometry over-trusted | hybrid_v2 (done) / learning |
| Rare ambiguous objects | beam, sofa, board, column (~0) | rare + shape/context-dependent | PointNet++ (P3) - open question |

The structural error that remains (occluded and far-field ceilings) is largely an acquisition
limit; the object error is the learnable frontier, and P3 is the test of how far a deep net
pushes it past the RandomForest.
