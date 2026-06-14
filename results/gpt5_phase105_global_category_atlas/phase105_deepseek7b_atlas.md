# Phase 105 Global Category Atlas: deepseek7b

Generated: 2026-06-14 00:11:15

## Global Layer Shape
- Best top1 layer: L28 with 8 / 32 categories top1.
- Best mean margin layer: L0 margin=-0.018.
- Best mean boundary layer: L27 norm=238.797.

## Category Layer Map
- fruit: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.03, best_rank=2, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- animal: class=cohesive_boundary_unclear_readout, marginL=28, boundaryL=27, cohesionL=0, best_margin=0.91, best_rank=1, neighbors=plant(0.98), relation(0.98), role(0.97), local_release=machine+0.90, emotion+0.89, communication+0.83
- tool: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.01, best_rank=3, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- vehicle: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=0.00, best_rank=1, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- clothing: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.02, best_rank=4, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- furniture: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.04, best_rank=2, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- food: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.02, best_rank=2, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- plant: class=cohesive_boundary_unclear_readout, marginL=28, boundaryL=27, cohesionL=0, best_margin=0.75, best_rank=1, neighbors=fruit(0.99), color(0.99), food(0.99), local_release=event+0.69, action+0.64, emotion+0.56
- body: class=cohesive_boundary_unclear_readout, marginL=28, boundaryL=27, cohesionL=0, best_margin=0.46, best_rank=1, neighbors=material(0.98), container(0.98), tool(0.98), local_release=substance+1.43, light+1.33, sound+1.27
- place: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.02, best_rank=9, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- building: class=cohesive_boundary_unclear_readout, marginL=9, boundaryL=27, cohesionL=0, best_margin=0.26, best_rank=1, neighbors=place(0.98), furniture(0.97), machine(0.96), local_release=action+0.80, light+0.48, fruit+0.42
- material: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.02, best_rank=4, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- color: class=cohesive_boundary_unclear_readout, marginL=14, boundaryL=27, cohesionL=0, best_margin=1.01, best_rank=1, neighbors=material(0.92), fruit(0.92), substance(0.92), local_release=sound+1.25, weather+1.09, building+1.06
- emotion: class=cohesive_boundary_unclear_readout, marginL=28, boundaryL=27, cohesionL=0, best_margin=0.46, best_rank=1, neighbors=time(0.97), weather(0.97), sound(0.97), local_release=furniture+1.16, material+1.00, role+1.00
- role: class=cohesive_boundary_unclear_readout, marginL=28, boundaryL=27, cohesionL=0, best_margin=0.04, best_rank=1, neighbors=profession(1.00), relation(1.00), animal(0.97), local_release=relation+0.14, tool+0.09, furniture+0.02
- profession: class=sharp_readout_cohesive, marginL=27, boundaryL=27, cohesionL=0, best_margin=26.42, best_rank=1, neighbors=role(1.00), relation(0.99), action(0.97), local_release=communication+4.77, fruit+3.87, container+3.38
- abstract: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.01, best_rank=10, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- action: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.00, best_rank=2, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- event: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.02, best_rank=5, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- time: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.03, best_rank=4, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- number: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.02, best_rank=6, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- shape: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.03, best_rank=2, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- sound: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.02, best_rank=2, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- light: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.02, best_rank=8, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- weather: class=cohesive_boundary_unclear_readout, marginL=28, boundaryL=27, cohesionL=0, best_margin=0.17, best_rank=1, neighbors=sound(0.98), light(0.98), time(0.97), local_release=role+0.41, emotion+0.41, plant+0.39
- container: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.04, best_rank=2, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- instrument: class=cohesive_boundary_unclear_readout, marginL=28, boundaryL=27, cohesionL=0, best_margin=0.38, best_rank=1, neighbors=tool(0.97), container(0.97), furniture(0.97), local_release=none
- machine: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.02, best_rank=2, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- communication: class=cohesive_boundary_unclear_readout, marginL=8, boundaryL=27, cohesionL=0, best_margin=0.08, best_rank=1, neighbors=event(0.97), sound(0.96), action(0.96), local_release=tool+0.27, color+0.27, instrument+0.24
- relation: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.00, best_rank=3, neighbors=substance(1.00), property(1.00), communication(1.00), local_release=none
- property: class=cohesive_boundary_unclear_readout, marginL=12, boundaryL=27, cohesionL=0, best_margin=1.48, best_rank=1, neighbors=abstract(0.97), light(0.95), action(0.94), local_release=role+1.12, color+1.04, sound+1.03
- substance: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=27, cohesionL=0, best_margin=-0.02, best_rank=10, neighbors=property(1.00), relation(1.00), communication(1.00), local_release=none

## Caution
- Layer 0 is embedding output; layer k>0 is hidden_states[k], i.e. after transformer block k-1.
- Boundary removal here is local logit-lens removal at the best margin layer, not downstream causal patching.
- Metrics are basic centroid/readout/neighbor measurements, not statistical proof.
