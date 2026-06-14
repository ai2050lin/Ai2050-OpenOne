# Phase 105 Global Category Atlas: qwen3

Generated: 2026-06-14 00:10:06

## Global Layer Shape
- Best top1 layer: L25 with 4 / 4 categories top1.
- Best mean margin layer: L32 margin=24.573.
- Best mean boundary layer: L35 norm=262.978.

## Category Layer Map
- fruit: class=sharp_readout_cohesive, marginL=32, boundaryL=35, cohesionL=0, best_margin=33.16, best_rank=1, neighbors=animal(0.77), vehicle(0.75), tool(0.72), local_release=animal+7.24, tool+6.89, vehicle+4.28
- animal: class=sharp_readout_cohesive, marginL=32, boundaryL=35, cohesionL=0, best_margin=23.23, best_rank=1, neighbors=vehicle(0.84), tool(0.78), fruit(0.77), local_release=tool+9.27, fruit+9.12, vehicle+4.47
- tool: class=sharp_readout_cohesive, marginL=35, boundaryL=35, cohesionL=0, best_margin=30.62, best_rank=1, neighbors=vehicle(0.91), animal(0.87), fruit(0.85), local_release=animal+11.27, vehicle+9.60, fruit+6.34
- vehicle: class=sharp_readout_cohesive, marginL=33, boundaryL=35, cohesionL=0, best_margin=15.79, best_rank=1, neighbors=tool(0.85), animal(0.84), fruit(0.75), local_release=fruit+10.91, animal+3.45

## Caution
- Layer 0 is embedding output; layer k>0 is hidden_states[k], i.e. after transformer block k-1.
- Boundary removal here is local logit-lens removal at the best margin layer, not downstream causal patching.
- Metrics are basic centroid/readout/neighbor measurements, not statistical proof.
