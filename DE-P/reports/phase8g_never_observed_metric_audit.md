# Phase 8G never-observed metric audit

## Historical Phase 8F result

The frozen Phase 8F implementation hash remains
`fc9c5370a1da929cfb8cf267b0c11176ec15e3aeb291b8e5117ec6fe753abf04`.
Its consumed test report is retained and was not rerun.

The 13 historical frame records represent three unique stable
track–actor events:

| Sequence | Actor | Track | Meaning in the old evaluator |
|---|---:|---:|---|
| `phase8c_test_0008` | 10300311 | 15 | proximity match |
| `phase8c_test_0028` | 10320111 | 0 | proximity match |
| `phase8c_test_0032` | 10320311 | 1 | proximity match |

## Old metric semantics

`evaluate_dynamic_perception_v2.py` marked an actor as ever observed from the
Simulator visibility/inside-image/occlusion fields and a projected geometric
proxy. It did not have actual actor pixels. After the primary assignment it
ran a second gated Hungarian assignment between unmatched tracks and
never-before-visible actors. That second assignment used world-centroid
distance only, with the global association threshold; actor radius and
observation pixel origin were not used.

Consequently an unrelated component near an actor could be counted as a
never-observed match. Each frame was appended independently, explaining why
three sustained identities produced 13 records. The old report cannot prove
that any of those tracks contained pixels from the named actor.

## Phase 8G semantics

- **frame record**: one classified event at one sequence frame;
- **unique track–actor event**: a deduplicated track ID, actor ID and
  classification tuple;
- **track birth event**: the first frame of a runtime track, linked to its
  birth observation;
- **attention publication event**: an attention-authorized track at a frame;
- **proximity-only event**: world-space proximity without instance overlap or
  actor identity.

Only actors with current instance pixels, or an identity established by prior
instance overlap within the occlusion window, enter formal assignment.
Actors with zero instance pixels over the complete sequence are audited
separately. Proximity no longer establishes identity. Illegal instance/GT
contradictions, confirmed tracks without direct evidence, and ungrounded
attention are independent hard failures.

The instance image is loaded by the evaluator only. It is never passed to
`DynamicPerception`, foreground extraction, association, confidence, the CNN,
or network inputs.
