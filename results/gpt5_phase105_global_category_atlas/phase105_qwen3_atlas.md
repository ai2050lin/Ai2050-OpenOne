# Phase 105 Global Category Atlas: qwen3

Generated: 2026-06-14 00:10:26

## Global Layer Shape
- Best top1 layer: L36 with 23 / 32 categories top1.
- Best mean margin layer: L36 margin=0.677.
- Best mean boundary layer: L35 norm=161.167.

## Category Layer Map
- fruit: class=sharp_readout_cohesive, marginL=32, boundaryL=35, cohesionL=0, best_margin=12.54, best_rank=1, neighbors=plant(0.91), color(0.87), food(0.85), local_release=profession+4.69, sound+3.85, animal+3.14
- animal: class=sharp_readout_cohesive, marginL=32, boundaryL=35, cohesionL=0, best_margin=11.58, best_rank=1, neighbors=relation(0.89), plant(0.88), role(0.88), local_release=profession+3.38, building+2.98, instrument+2.79
- tool: class=readout_clear, marginL=35, boundaryL=35, cohesionL=0, best_margin=6.88, best_rank=1, neighbors=container(0.95), machine(0.95), action(0.94), local_release=color+5.17, light+3.50, food+3.42
- vehicle: class=sharp_readout_cohesive, marginL=33, boundaryL=35, cohesionL=0, best_margin=12.25, best_rank=1, neighbors=machine(0.93), container(0.92), building(0.92), local_release=food+1.61, plant+1.37, relation+1.24
- clothing: class=cohesive_boundary_unclear_readout, marginL=33, boundaryL=35, cohesionL=0, best_margin=2.07, best_rank=1, neighbors=container(0.92), furniture(0.91), action(0.90), local_release=sound+4.78, action+3.75, vehicle+3.43
- furniture: class=cohesive_boundary_unclear_readout, marginL=36, boundaryL=35, cohesionL=0, best_margin=1.07, best_rank=1, neighbors=container(0.97), building(0.95), clothing(0.95), local_release=weather+1.47, fruit+1.19, action+1.16
- food: class=sharp_readout_cohesive, marginL=33, boundaryL=35, cohesionL=0, best_margin=16.44, best_rank=1, neighbors=substance(0.89), material(0.89), container(0.89), local_release=sound+2.01, relation+1.87, place+1.80
- plant: class=sharp_readout_cohesive, marginL=34, boundaryL=35, cohesionL=0, best_margin=15.91, best_rank=1, neighbors=color(0.93), fruit(0.93), relation(0.91), local_release=sound+3.39, communication+3.38, profession+2.99
- body: class=cohesive_boundary_unclear_readout, marginL=36, boundaryL=35, cohesionL=0, best_margin=0.84, best_rank=1, neighbors=light(0.95), action(0.95), substance(0.95), local_release=profession+0.53, light+0.45, animal+0.44
- place: class=cohesive_boundary_unclear_readout, marginL=34, boundaryL=35, cohesionL=0, best_margin=2.96, best_rank=1, neighbors=building(0.98), event(0.96), relation(0.95), local_release=tool+7.19, fruit+6.47, instrument+6.28
- building: class=sharp_readout_cohesive, marginL=35, boundaryL=35, cohesionL=0, best_margin=14.42, best_rank=1, neighbors=place(0.98), container(0.96), action(0.96), local_release=fruit+4.45, action+3.93, food+3.76
- material: class=readout_clear, marginL=35, boundaryL=35, cohesionL=0, best_margin=5.87, best_rank=1, neighbors=substance(0.98), light(0.96), color(0.96), local_release=clothing+6.04, emotion+5.90, abstract+5.88
- color: class=readout_clear, marginL=32, boundaryL=35, cohesionL=0, best_margin=6.23, best_rank=1, neighbors=number(0.96), action(0.96), relation(0.95), local_release=tool+4.07, instrument+3.29, machine+3.06
- emotion: class=cohesive_boundary_unclear_readout, marginL=36, boundaryL=35, cohesionL=0, best_margin=1.83, best_rank=1, neighbors=abstract(0.98), time(0.97), sound(0.97), local_release=building+0.30, tool+0.22, shape+0.14
- role: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=35, cohesionL=0, best_margin=-0.09, best_rank=3, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- profession: class=sharp_readout_cohesive, marginL=35, boundaryL=35, cohesionL=0, best_margin=22.24, best_rank=1, neighbors=role(1.00), relation(0.99), action(0.97), local_release=weather+5.97, time+4.54, furniture+4.17
- abstract: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=35, cohesionL=0, best_margin=-0.06, best_rank=4, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- action: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=35, cohesionL=0, best_margin=-0.09, best_rank=7, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- event: class=cohesive_boundary_unclear_readout, marginL=34, boundaryL=35, cohesionL=0, best_margin=2.48, best_rank=1, neighbors=communication(0.98), action(0.98), abstract(0.97), local_release=plant+6.65, animal+5.67, fruit+5.40
- time: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=35, cohesionL=0, best_margin=-0.09, best_rank=2, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- number: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=35, cohesionL=0, best_margin=-0.09, best_rank=2, neighbors=substance(1.00), property(1.00), relation(1.00), local_release=none
- shape: class=sharp_readout_cohesive, marginL=34, boundaryL=35, cohesionL=0, best_margin=8.63, best_rank=1, neighbors=property(0.95), number(0.95), light(0.95), local_release=plant+1.55, sound+1.24, instrument+1.21
- sound: class=sharp_readout_cohesive, marginL=33, boundaryL=35, cohesionL=0, best_margin=23.19, best_rank=1, neighbors=action(0.97), communication(0.97), light(0.97), local_release=container+9.04, furniture+5.99, body+2.36
- light: class=cohesive_boundary_unclear_readout, marginL=34, boundaryL=35, cohesionL=0, best_margin=3.01, best_rank=1, neighbors=property(0.97), sound(0.97), action(0.97), local_release=container+3.32, building+3.25, food+2.49
- weather: class=readout_clear, marginL=30, boundaryL=35, cohesionL=0, best_margin=6.71, best_rank=1, neighbors=light(0.94), sound(0.93), substance(0.92), local_release=tool+2.49, instrument+2.02, building+1.59
- container: class=cohesive_boundary_unclear_readout, marginL=36, boundaryL=35, cohesionL=0, best_margin=0.88, best_rank=1, neighbors=furniture(0.97), action(0.96), light(0.96), local_release=sound+0.90, role+0.73, communication+0.70
- instrument: class=readout_clear, marginL=32, boundaryL=35, cohesionL=0, best_margin=6.21, best_rank=1, neighbors=tool(0.81), sound(0.81), machine(0.79), local_release=substance+4.60, material+4.12, profession+3.82
- machine: class=cohesive_boundary_unclear_readout, marginL=36, boundaryL=35, cohesionL=0, best_margin=0.69, best_rank=1, neighbors=vehicle(0.96), light(0.96), container(0.95), local_release=plant+1.07, weather+0.98, color+0.98
- communication: class=cohesive_boundary_unclear_readout, marginL=36, boundaryL=35, cohesionL=0, best_margin=0.60, best_rank=1, neighbors=abstract(0.99), action(0.98), event(0.98), local_release=animal+0.24
- relation: class=cohesive_boundary_unclear_readout, marginL=0, boundaryL=35, cohesionL=0, best_margin=-0.13, best_rank=3, neighbors=substance(1.00), property(1.00), communication(1.00), local_release=none
- property: class=cohesive_boundary_unclear_readout, marginL=36, boundaryL=35, cohesionL=0, best_margin=0.38, best_rank=1, neighbors=abstract(0.99), action(0.99), number(0.98), local_release=none
- substance: class=cohesive_boundary_unclear_readout, marginL=36, boundaryL=35, cohesionL=0, best_margin=1.03, best_rank=1, neighbors=material(0.98), light(0.97), weather(0.96), local_release=shape+0.31, clothing+0.31, profession+0.26

## Caution
- Layer 0 is embedding output; layer k>0 is hidden_states[k], i.e. after transformer block k-1.
- Boundary removal here is local logit-lens removal at the best margin layer, not downstream causal patching.
- Metrics are basic centroid/readout/neighbor measurements, not statistical proof.
