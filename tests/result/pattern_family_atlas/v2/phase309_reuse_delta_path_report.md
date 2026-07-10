# Phase309 Shared Backbone and Delta Pair Three-Position Path Atlas

## Summary

- component_rows: 31200
- summary_rows: 900
- pair_path_rows: 150
- pair_matrix_rows: 1350
- missing_rows: 0
- token_match_confidence_mean: 0.8

## Mean Reuse By Pair Group

- delta_control: reuse=0.563324, delta=0.436676
- shared_backbone: reuse=0.702623, delta=0.297377

## Mean Reuse By Position

- last: reuse=0.681211, delta=0.318789
- object: reuse=0.551451, delta=0.448549
- query: reuse=0.666259, delta=0.333741

## Pair Group x Position

- delta_control / last: reuse=0.603535, n=225
- delta_control / object: reuse=0.493614, n=225
- delta_control / query: reuse=0.592822, n=225
- shared_backbone / last: reuse=0.758886, n=225
- shared_backbone / object: reuse=0.609287, n=225
- shared_backbone / query: reuse=0.739696, n=225

## Pair Group x Attribute

- delta_control / category: reuse=0.622357, n=135
- delta_control / color: reuse=0.587033, n=135
- delta_control / subclass: reuse=0.558913, n=135
- delta_control / taste: reuse=0.497986, n=135
- delta_control / use: reuse=0.55033, n=135
- shared_backbone / category: reuse=0.751802, n=135
- shared_backbone / color: reuse=0.590974, n=135
- shared_backbone / subclass: reuse=0.768946, n=135
- shared_backbone / taste: reuse=0.745104, n=135
- shared_backbone / use: reuse=0.65629, n=135

## Component

- attention: reuse=0.707727, n=450
- mlp: reuse=0.688073, n=450
- residual: reuse=0.50312, n=450

## Top Route Types

- attention->attention->attention: 33
- attention->mlp->attention: 22
- attention->attention->mlp: 12
- mlp->mlp->attention: 10
- mlp->attention->attention: 9
- mlp->attention->mlp: 8
- residual->mlp->mlp: 8
- residual->mlp->attention: 6
- residual->attention->attention: 6
- attention->mlp->mlp: 6
- residual->residual->mlp: 5
- residual->attention->mlp: 4

## Caution

This is an observational path-signature atlas. It compares per-layer component margin profiles, not direct causal necessity.
The current linear target-distractor readout direction is a probe, not a final mechanism formula.

