# Phase 299 Dynamic Naturalness Report

- input_dir: `results/gpt5_phase298_expanded_dynamic_normal`
- z_threshold: 3.0
- success_threshold: 0.8
- over_threshold: 1.05

## Summary

### qwen3
- total_rows: 43920
- total_rows: 43920
- off_manifold_rows: 84
- off_manifold_rate: 0.001913
- off_manifold_high_progress_rows: 10
- off_manifold_high_progress_rate: 0.000228
- off_manifold_negative_progress_rows: 1
- off_manifold_negative_progress_rate: 0.000023
- off_manifold_over_conversion_rows: 2
- off_manifold_over_conversion_rate: 0.000046
- on_manifold_high_progress_rows: 7289
- on_manifold_high_progress_rate: 0.165961
- on_manifold_negative_progress_rows: 2157
- on_manifold_negative_progress_rate: 0.049112
- on_manifold_over_conversion_rows: 1587
- on_manifold_over_conversion_rate: 0.036134

### glm4
- total_rows: 43920
- total_rows: 43920
- off_manifold_rows: 171
- off_manifold_rate: 0.003893
- off_manifold_high_progress_rows: 75
- off_manifold_high_progress_rate: 0.001708
- off_manifold_negative_progress_rows: 2
- off_manifold_negative_progress_rate: 0.000046
- off_manifold_over_conversion_rows: 0
- off_manifold_over_conversion_rate: 0.000000
- on_manifold_high_progress_rows: 10843
- on_manifold_high_progress_rate: 0.246881
- on_manifold_negative_progress_rows: 188
- on_manifold_negative_progress_rate: 0.004281
- on_manifold_over_conversion_rows: 776
- on_manifold_over_conversion_rate: 0.017668

### deepseek7b
- total_rows: 39040
- total_rows: 39040
- off_manifold_rows: 145
- off_manifold_rate: 0.003714
- off_manifold_high_progress_rows: 19
- off_manifold_high_progress_rate: 0.000487
- off_manifold_negative_progress_rows: 28
- off_manifold_negative_progress_rate: 0.000717
- off_manifold_over_conversion_rows: 11
- off_manifold_over_conversion_rate: 0.000282
- on_manifold_high_progress_rows: 3600
- on_manifold_high_progress_rate: 0.092213
- on_manifold_negative_progress_rows: 3012
- on_manifold_negative_progress_rate: 0.077152
- on_manifold_over_conversion_rows: 869
- on_manifold_over_conversion_rate: 0.022259

## Notes

- This is norm/z-score naturalness only. It does not compute PCA, kNN, Mahalanobis, entropy, or loss.
- on_manifold/off_manifold labels are diagnostic filters, not final mechanism proof.
