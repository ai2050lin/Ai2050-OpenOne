# Phase256 Done Signature Counterfactual Localization

- status: complete
- done_vector_component_rows: 3
- done_signature_rows: 1225
- counterfactual_rows: 5
- interpretation_counts: {"eos_aligned_done_growth": 3, "no_eos_or_no_growth": 2}

## Counterfactual Summary
- glm4 no_intervention: stop=client_truncation, answer_done=-42.577522, pre_eos_done=0.0, eos_done=0.0, late_done=-4.214076, gain_answer_to_eos=None, interpretation=no_eos_or_no_growth
- glm4 tokenbank_suppression: stop=client_truncation, answer_done=0.0, pre_eos_done=0.0, eos_done=0.0, late_done=-5.510945, gain_answer_to_eos=None, interpretation=no_eos_or_no_growth
- glm4 natural_raw_suppression: stop=eos_stop, answer_done=-37.017658, pre_eos_done=3.829839, eos_done=21.242704, late_done=6.836779, gain_answer_to_eos=58.260362, interpretation=eos_aligned_done_growth
- glm4 combined_suppression: stop=eos_stop, answer_done=-37.017658, pre_eos_done=3.829839, eos_done=21.242704, late_done=6.836779, gain_answer_to_eos=58.260362, interpretation=eos_aligned_done_growth
- glm4 weighted_combined_c0.25_r1.0: stop=eos_stop, answer_done=-50.142101, pre_eos_done=7.158975, eos_done=21.224991, late_done=5.540722, gain_answer_to_eos=71.367092, interpretation=eos_aligned_done_growth
