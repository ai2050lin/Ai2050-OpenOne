# Phase255 ModelClose Internal Stop Trace

- status: complete
- stop_trace_rows: 5
- generation_step_rows: 245
- prefix_projection_rows: 1225
- stop_type_counts: {"eos_stop": 3, "client_truncation": 2}
- eos_stop_conditions: ["glm4:natural_raw_suppression", "glm4:combined_suppression", "glm4:weighted_combined_c0.25_r1.0"]

## Stop Trace
- glm4 no_intervention: stop=client_truncation, tokens=96, answer_step=7, eos_pos=None, final_closure=1.3125, over_generation=507
- glm4 tokenbank_suppression: stop=client_truncation, tokens=96, answer_step=None, eos_pos=None, final_closure=-1.0625, over_generation=499
- glm4 natural_raw_suppression: stop=eos_stop, tokens=12, answer_step=1, eos_pos=12, final_closure=2.90625, over_generation=56
- glm4 combined_suppression: stop=eos_stop, tokens=12, answer_step=1, eos_pos=12, final_closure=7.382812, over_generation=56
- glm4 weighted_combined_c0.25_r1.0: stop=eos_stop, tokens=29, answer_step=3, eos_pos=29, final_closure=11.03125, over_generation=131

## Final Layer Prefix Projection Averages
- no_intervention: readout_projection_mean=-4.4498, control_projection_mean=-11.366206, closure_proxy_mean=-1.157878
- tokenbank_suppression: readout_projection_mean=-6.094681, control_projection_mean=-14.62358, closure_proxy_mean=-0.183431
- natural_raw_suppression: readout_projection_mean=-3.546949, control_projection_mean=-9.140234, closure_proxy_mean=-0.814779
- combined_suppression: readout_projection_mean=-3.546949, control_projection_mean=-9.140234, closure_proxy_mean=1.399089
- weighted_combined_c0.25_r1.0: readout_projection_mean=-5.060608, control_projection_mean=-10.590529, closure_proxy_mean=-0.123072
