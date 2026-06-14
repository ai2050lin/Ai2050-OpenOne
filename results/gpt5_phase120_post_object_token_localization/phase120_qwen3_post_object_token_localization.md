# Phase 120 Post-object Token Localization: qwen3

Generated: 2026-06-14 15:46:34
Layers: [32, 33, 34, 35]; sites: ['object_last', 'after_object_first', 'after_object_middle', 'pre_answer_last', 'post_object_excluding_answer', 'answer_last', 'post_object_including_answer']

| category | axis | best pre-answer excluding answer | answer_last | including answer |
|---|---|---|---|---|
| number | local_varimax_best | L33 after_object_middle T-0.29 R+0.08 | L35 answer_last T-1.41 R+2.53 | L35 post_object_including_answer T-4.43 R+1.93 |
| number | local_svd_subspace | L33 after_object_middle T-0.29 R+0.06 | L35 answer_last T-1.82 R+2.76 | L35 post_object_including_answer T-4.41 R+2.30 |
| container | local_varimax_best | L34 post_object_excluding_answer T-0.14 R+0.40 | L35 answer_last T-2.64 R+1.33 | L32 post_object_including_answer T-1.23 R+1.86 |
| container | local_svd_subspace | L33 pre_answer_last T-0.08 R+0.00 | L35 answer_last T-2.53 R+1.90 | L32 post_object_including_answer T-1.73 R+3.61 |
| plant | local_varimax_best | L32 pre_answer_last T-0.36 R+0.00 | L35 answer_last T-0.94 R+1.36 | L35 post_object_including_answer T-5.29 R+1.37 |
| plant | local_svd_subspace | L32 pre_answer_last T-0.47 R+0.00 | L33 answer_last T-1.28 R+1.00 | L35 post_object_including_answer T-4.66 R+1.83 |
