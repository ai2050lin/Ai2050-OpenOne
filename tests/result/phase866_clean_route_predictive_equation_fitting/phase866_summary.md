# Phase 866 Clean Route Predictive Equation Fitting

- Source: Phase 865 route purity rows.
- Boundary: simple empirical rule check, not a learned model and not closure.

## Rule Results

| scope | rule | TP | FP | FN | TN | precision | recall | accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full_set | `answer_blocker_object_rule` | 6 | 0 | 0 | 6 | 1.000 | 1.000 | 1.000 |
| full_set | `answer_blocker_only_rule` | 6 | 1 | 0 | 5 | 0.857 | 1.000 | 0.917 |
| full_set | `answer_only_object_rule` | 6 | 1 | 0 | 5 | 0.857 | 1.000 | 0.917 |
| dominant_channel | `answer_blocker_object_rule` | 6 | 0 | 0 | 6 | 1.000 | 1.000 | 1.000 |
| dominant_channel | `answer_blocker_only_rule` | 6 | 1 | 0 | 5 | 0.857 | 1.000 | 0.917 |
| dominant_channel | `answer_only_object_rule` | 6 | 1 | 0 | 5 | 0.857 | 1.000 | 0.917 |
| full_and_dominant | `answer_blocker_object_rule` | 12 | 0 | 0 | 12 | 1.000 | 1.000 | 1.000 |
| full_and_dominant | `answer_blocker_only_rule` | 12 | 2 | 0 | 10 | 0.857 | 1.000 | 0.917 |
| full_and_dominant | `answer_only_object_rule` | 12 | 2 | 0 | 10 | 0.857 | 1.000 | 0.917 |

## Selected Equation

```text
CleanMixedRoute(g,d,m) =
  [answer_delta > 0]
  and [blocker_reduction > 0]
  and [original_blocker_delta < 0]
  and [object_delta <= 0.25]
  and [object_echo_induced = 0]
  and [format_or_other_induced = 0]
```
