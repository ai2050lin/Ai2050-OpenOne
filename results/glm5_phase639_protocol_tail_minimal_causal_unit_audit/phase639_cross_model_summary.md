# Phase 639 Cross-Model Summary

目标：把 Phase638 的 question-tail protocol state 缩小到最小 token / token-pair 因果单位。

## qwen3

- raw_cases: 256 / target_seen: 17 / cases_written: 256 / mode_rows: 2560
- top_k: 20
- filtered: `{'not_target': 0, 'unit_missing': 256, 'unit_len_mismatch': 256, 'empty_patch': 512}`
- unit_token_lens_sample: `{'qmark': {'original': {'1': 8}, 'inline': {'1': 8}}, 'separator': {'original': {'2': 8}, 'inline': {'2': 8}}, 'answer_word': {'original': {'1': 8}, 'inline': {'0': 8}}, 'colon': {'original': {'1': 8}, 'inline': {'1': 8}}, 'prompt_last': {'original': {'1': 8}, 'inline': {'1': 8}}, 'qmark_separator': {'original': {'3': 8}, 'inline': {'3': 8}}, 'separator_answer': {'original': {'2': 8}, 'inline': {'2': 8}}, 'answer_colon': {'original': {'2': 8}, 'inline': {'1': 8}}, 'tail_all': {'original': {'3': 8}, 'inline': {'3': 8}}}`

### target

| mode | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | margin-top | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 17 | 14/17 | 11/17 | 3/17 | 0/17 | 1.2 | 1.272 | -0.118 | correct_prefix:14, space:3 |  v:14,  :3 |
| inline | 17 | 1/17 | 0/17 | 0/17 | 9/17 | 4.8 | -1.471 | -1.566 | newline:9, space:7, correct_prefix:1 |  ?\n\n:9,  :7,  v:1 |
| final_output_inline_to_original | 17 | 1/17 | 0/17 | 0/17 | 9/17 | 4.8 | -1.471 | -1.566 | newline:9, space:7, correct_prefix:1 |  ?\n\n:9,  :7,  v:1 |
| patch_qmark | 17 | 13/17 | 10/17 | 2/17 | 0/17 | 1.2 | 1.051 | -0.140 | correct_prefix:13, space:4 |  v:13,  :4 |
| patch_separator | 17 | 0/17 | 0/17 | 0/17 | 16/17 | 5.1 | -1.904 | -1.912 | newline:16, space:1 |  ?\n\n:15,  :1,  \n\n:1 |
| patch_colon | 17 | 2/17 | 1/17 | 0/17 | 7/17 | 4.1 | -1.309 | -1.684 | space:8, newline:7, correct_prefix:2 |  :8,  ?\n\n:7,  v:2 |
| patch_prompt_last | 17 | 2/17 | 1/17 | 0/17 | 7/17 | 4.1 | -1.309 | -1.684 | space:8, newline:7, correct_prefix:2 |  :8,  ?\n\n:7,  v:2 |
| patch_qmark_separator | 17 | 1/17 | 0/17 | 0/17 | 9/17 | 4.8 | -1.471 | -1.566 | newline:9, space:7, correct_prefix:1 |  ?\n\n:9,  :7,  v:1 |
| patch_separator_answer | 17 | 0/17 | 0/17 | 0/17 | 16/17 | 5.1 | -1.904 | -1.912 | newline:16, space:1 |  ?\n\n:15,  :1,  \n\n:1 |
| patch_tail_all | 17 | 1/17 | 0/17 | 0/17 | 9/17 | 4.8 | -1.471 | -1.566 | newline:9, space:7, correct_prefix:1 |  ?\n\n:9,  :7,  v:1 |

### non_target

| mode | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | margin-top | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 239 | 144/239 | 140/239 | 3/239 | 2/239 | 2.1 | 1.272 | -0.631 | correct_prefix:144, space:92, newline:2, word:1 |  v:144,  :92,  o:1,  \n\n:1,  ?\n\n:1 |
| inline | 239 | 26/239 | 27/239 | 0/239 | 123/239 | 5.3 | -1.721 | -1.970 | newline:123, space:90, correct_prefix:26 |  ?\n\n:123,  :90,  v:26 |
| final_output_inline_to_original | 239 | 26/239 | 27/239 | 0/239 | 123/239 | 5.3 | -1.721 | -1.970 | newline:123, space:90, correct_prefix:26 |  ?\n\n:123,  :90,  v:26 |
| patch_qmark | 239 | 129/239 | 126/239 | 2/239 | 1/239 | 2.2 | 0.969 | -0.629 | correct_prefix:129, space:107, word:2, newline:1 |  v:129,  :107,  o:2,  ?\n\n:1 |
| patch_separator | 239 | 24/239 | 24/239 | 0/239 | 177/239 | 5.3 | -2.263 | -2.394 | newline:177, space:38, correct_prefix:24 |  ?\n\n:176,  :38,  v:24,  \n\n:1 |
| patch_colon | 239 | 39/239 | 39/239 | 0/239 | 66/239 | 4.7 | -1.623 | -2.280 | space:134, newline:66, correct_prefix:39 |  :134,  ?\n\n:66,  v:39 |
| patch_prompt_last | 239 | 39/239 | 39/239 | 0/239 | 66/239 | 4.7 | -1.623 | -2.280 | space:134, newline:66, correct_prefix:39 |  :134,  ?\n\n:66,  v:39 |
| patch_qmark_separator | 239 | 26/239 | 27/239 | 0/239 | 123/239 | 5.3 | -1.721 | -1.970 | newline:123, space:90, correct_prefix:26 |  ?\n\n:123,  :90,  v:26 |
| patch_separator_answer | 239 | 24/239 | 24/239 | 0/239 | 177/239 | 5.3 | -2.263 | -2.394 | newline:177, space:38, correct_prefix:24 |  ?\n\n:176,  :38,  v:24,  \n\n:1 |
| patch_tail_all | 239 | 26/239 | 27/239 | 0/239 | 123/239 | 5.3 | -1.721 | -1.970 | newline:123, space:90, correct_prefix:26 |  ?\n\n:123,  :90,  v:26 |

### Examples

- sample=0 split=non_target mode=original top0=' '/space rank=2 exact=False text=' 22' ladder=1: [space], 2: v[correct_prefix], 3: \n\n[newline], 4: ?\n\n[newline], 5: ?\n[newline]
- sample=0 split=non_target mode=inline top0=' '/space rank=6 exact=False text=' 22' ladder=1: [space], 2: ?\n\n[newline], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=0 split=non_target mode=final_output_inline_to_original top0=' '/space rank=6 exact=False text=' 22' ladder=1: [space], 2: ?\n\n[newline], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=0 split=non_target mode=patch_qmark top0=' '/space rank=2 exact=False text=' 22' ladder=1: [space], 2: v[correct_prefix], 3: ?\n\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=0 split=non_target mode=patch_separator top0=' ?\\n\\n'/newline rank=6 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=0 split=non_target mode=patch_colon top0=' '/space rank=5 exact=False text=' 22' ladder=1: [space], 2: ?\n\n[newline], 3: ?\n[newline], 4: ?[punctuation], 5: v[correct_prefix]
- sample=0 split=non_target mode=patch_prompt_last top0=' '/space rank=5 exact=False text=' 22' ladder=1: [space], 2: ?\n\n[newline], 3: ?\n[newline], 4: ?[punctuation], 5: v[correct_prefix]
- sample=0 split=non_target mode=patch_qmark_separator top0=' '/space rank=6 exact=False text=' 22' ladder=1: [space], 2: ?\n\n[newline], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=0 split=non_target mode=patch_separator_answer top0=' ?\\n\\n'/newline rank=6 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=0 split=non_target mode=patch_tail_all top0=' '/space rank=6 exact=False text=' 22' ladder=1: [space], 2: ?\n\n[newline], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=1 split=non_target mode=original top0=' v'/correct_prefix rank=1 exact=True text=' v48' ladder=1: v[correct_prefix], 2: \n\n[newline], 3: ?\n\n[newline], 4: [space], 5: ?\n[newline]
- sample=1 split=non_target mode=inline top0=' ?\\n\\n'/newline rank=6 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=1 split=non_target mode=final_output_inline_to_original top0=' ?\\n\\n'/newline rank=6 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=1 split=non_target mode=patch_qmark top0=' v'/correct_prefix rank=1 exact=True text=' v48' ladder=1: v[correct_prefix], 2: [space], 3: ?\n\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=1 split=non_target mode=patch_separator top0=' ?\\n\\n'/newline rank=5 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: ?\n[newline], 3: \n\n[newline], 4: [space], 5: v[correct_prefix]
- sample=1 split=non_target mode=patch_colon top0=' ?\\n\\n'/newline rank=4 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: ?[punctuation], 5: v[correct_prefix]

## glm4

- raw_cases: 256 / target_seen: 31 / cases_written: 256 / mode_rows: 2560
- top_k: 20
- filtered: `{'not_target': 0, 'unit_missing': 256, 'unit_len_mismatch': 256, 'empty_patch': 512}`
- unit_token_lens_sample: `{'qmark': {'original': {'1': 8}, 'inline': {'1': 8}}, 'separator': {'original': {'2': 8}, 'inline': {'2': 8}}, 'answer_word': {'original': {'1': 8}, 'inline': {'0': 8}}, 'colon': {'original': {'1': 8}, 'inline': {'1': 8}}, 'prompt_last': {'original': {'1': 8}, 'inline': {'1': 8}}, 'qmark_separator': {'original': {'3': 8}, 'inline': {'3': 8}}, 'separator_answer': {'original': {'2': 8}, 'inline': {'2': 8}}, 'answer_colon': {'original': {'2': 8}, 'inline': {'1': 8}}, 'tail_all': {'original': {'3': 8}, 'inline': {'3': 8}}}`

### target

| mode | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | margin-top | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 31 | 29/31 | 28/31 | 1/31 | 0/31 | 1.1 | 80.722 | -0.020 | correct_prefix:29, word:2 |  v:29,  c:2 |
| inline | 31 | 27/31 | 26/31 | 1/31 | 0/31 | 1.2 | 71.648 | -0.042 | correct_prefix:27, explanation:3, word:1 |  v:27,  Yes:3,  c:1 |
| final_output_inline_to_original | 31 | 27/31 | 27/31 | 1/31 | 0/31 | 1.2 | 71.648 | -0.042 | correct_prefix:27, explanation:3, word:1 |  v:27,  Yes:3,  c:1 |
| patch_qmark | 31 | 30/31 | 29/31 | 1/31 | 0/31 | 1.1 | 19.429 | -0.010 | correct_prefix:30, word:1 |  v:30,  c:1 |
| patch_separator | 31 | 27/31 | 25/31 | 2/31 | 0/31 | 1.3 | 99.000 | -0.089 | correct_prefix:27, explanation:3, word:1 |  v:27,  Yes:3,  c:1 |
| patch_colon | 31 | 26/31 | 25/31 | 1/31 | 0/31 | 1.3 | 99.000 | -0.081 | correct_prefix:26, explanation:3, word:2 |  v:26,  Yes:3,  c:2 |
| patch_prompt_last | 31 | 26/31 | 25/31 | 1/31 | 0/31 | 1.3 | 99.000 | -0.081 | correct_prefix:26, explanation:3, word:2 |  v:26,  Yes:3,  c:2 |
| patch_qmark_separator | 31 | 27/31 | 27/31 | 1/31 | 0/31 | 1.2 | 71.648 | -0.042 | correct_prefix:27, explanation:3, word:1 |  v:27,  Yes:3,  c:1 |
| patch_separator_answer | 31 | 27/31 | 25/31 | 2/31 | 0/31 | 1.3 | 99.000 | -0.089 | correct_prefix:27, explanation:3, word:1 |  v:27,  Yes:3,  c:1 |
| patch_tail_all | 31 | 27/31 | 27/31 | 1/31 | 0/31 | 1.2 | 71.648 | -0.042 | correct_prefix:27, explanation:3, word:1 |  v:27,  Yes:3,  c:1 |

### non_target

| mode | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | margin-top | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 225 | 209/225 | 183/225 | 23/225 | 0/225 | 1.1 | 78.048 | -0.018 | correct_prefix:209, word:15, explanation:1 |  v:209,  c:15,  No:1 |
| inline | 225 | 203/225 | 177/225 | 23/225 | 0/225 | 1.2 | 65.054 | -0.041 | correct_prefix:203, word:13, explanation:9 |  v:203,  c:13,  Yes:7,  No:2 |
| final_output_inline_to_original | 225 | 203/225 | 177/225 | 23/225 | 0/225 | 1.2 | 65.054 | -0.041 | correct_prefix:203, word:13, explanation:9 |  v:203,  c:13,  Yes:7,  No:2 |
| patch_qmark | 225 | 216/225 | 189/225 | 23/225 | 0/225 | 1.0 | 18.475 | -0.007 | correct_prefix:216, word:8, explanation:1 |  v:216,  c:8,  No:1 |
| patch_separator | 225 | 184/225 | 156/225 | 24/225 | 0/225 | 1.3 | 99.000 | -0.079 | correct_prefix:184, word:27, explanation:14 |  v:184,  c:27,  Yes:11,  No:3 |
| patch_colon | 225 | 176/225 | 152/225 | 22/225 | 0/225 | 1.4 | 99.000 | -0.092 | correct_prefix:176, word:39, explanation:10 |  v:176,  c:39,  Yes:8,  No:2 |
| patch_prompt_last | 225 | 176/225 | 152/225 | 22/225 | 0/225 | 1.4 | 99.000 | -0.092 | correct_prefix:176, word:39, explanation:10 |  v:176,  c:39,  Yes:8,  No:2 |
| patch_qmark_separator | 225 | 203/225 | 177/225 | 23/225 | 0/225 | 1.2 | 65.054 | -0.041 | correct_prefix:203, word:13, explanation:9 |  v:203,  c:13,  Yes:7,  No:2 |
| patch_separator_answer | 225 | 184/225 | 156/225 | 24/225 | 0/225 | 1.3 | 99.000 | -0.079 | correct_prefix:184, word:27, explanation:14 |  v:184,  c:27,  Yes:11,  No:3 |
| patch_tail_all | 225 | 203/225 | 177/225 | 23/225 | 0/225 | 1.2 | 65.054 | -0.041 | correct_prefix:203, word:13, explanation:9 |  v:203,  c:13,  Yes:7,  No:2 |

### Examples

- sample=0 split=non_target mode=original top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: c[word], 3: o[word], 4: [space], 5: Yes[explanation]
- sample=0 split=non_target mode=inline top0=' v'/correct_prefix rank=1 exact=False text=' c59' ladder=1: v[correct_prefix], 2: c[word], 3: Yes[explanation], 4: r[word], 5: [space]
- sample=0 split=non_target mode=final_output_inline_to_original top0=' v'/correct_prefix rank=1 exact=False text=' c59' ladder=1: v[correct_prefix], 2: c[word], 3: Yes[explanation], 4: r[word], 5: [space]
- sample=0 split=non_target mode=patch_qmark top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: c[word], 3: o[word], 4: [space], 5: r[word]
- sample=0 split=non_target mode=patch_separator top0=' c'/word rank=2 exact=False text=' c59' ladder=1: c[word], 2: v[correct_prefix], 3: Yes[explanation], 4: r[word], 5: [space]
- sample=0 split=non_target mode=patch_colon top0=' c'/word rank=2 exact=False text=' c59' ladder=1: c[word], 2: v[correct_prefix], 3: Yes[explanation], 4: r[word], 5: [space]
- sample=0 split=non_target mode=patch_prompt_last top0=' c'/word rank=2 exact=False text=' c59' ladder=1: c[word], 2: v[correct_prefix], 3: Yes[explanation], 4: r[word], 5: [space]
- sample=0 split=non_target mode=patch_qmark_separator top0=' v'/correct_prefix rank=1 exact=False text=' c59' ladder=1: v[correct_prefix], 2: c[word], 3: Yes[explanation], 4: r[word], 5: [space]
- sample=0 split=non_target mode=patch_separator_answer top0=' c'/word rank=2 exact=False text=' c59' ladder=1: c[word], 2: v[correct_prefix], 3: Yes[explanation], 4: r[word], 5: [space]
- sample=0 split=non_target mode=patch_tail_all top0=' v'/correct_prefix rank=1 exact=False text=' c59' ladder=1: v[correct_prefix], 2: c[word], 3: Yes[explanation], 4: r[word], 5: [space]
- sample=1 split=non_target mode=original top0=' v'/correct_prefix rank=1 exact=True text=' v48' ladder=1: v[correct_prefix], 2: c[word], 3: Yes[explanation], 4: o[word], 5: No[explanation]
- sample=1 split=non_target mode=inline top0=' c'/word rank=2 exact=False text=' c59' ladder=1: c[word], 2: v[correct_prefix], 3: Yes[explanation], 4: No[explanation], 5: o[word]
- sample=1 split=non_target mode=final_output_inline_to_original top0=' c'/word rank=2 exact=False text=' c59' ladder=1: c[word], 2: v[correct_prefix], 3: Yes[explanation], 4: No[explanation], 5: o[word]
- sample=1 split=non_target mode=patch_qmark top0=' v'/correct_prefix rank=1 exact=True text=' v48' ladder=1: v[correct_prefix], 2: c[word], 3: Yes[explanation], 4: o[word], 5: No[explanation]
- sample=1 split=non_target mode=patch_separator top0=' c'/word rank=3 exact=False text=' c59' ladder=1: c[word], 2: Yes[explanation], 3: v[correct_prefix], 4: No[explanation], 5: o[word]
- sample=1 split=non_target mode=patch_colon top0=' c'/word rank=3 exact=False text=' c59' ladder=1: c[word], 2: Yes[explanation], 3: v[correct_prefix], 4: No[explanation], 5: o[word]

## deepseek7b

- raw_cases: 256 / target_seen: 82 / cases_written: 256 / mode_rows: 2560
- top_k: 20
- filtered: `{'not_target': 0, 'unit_missing': 256, 'unit_len_mismatch': 256, 'empty_patch': 512}`
- unit_token_lens_sample: `{'qmark': {'original': {'1': 8}, 'inline': {'1': 8}}, 'separator': {'original': {'2': 8}, 'inline': {'2': 8}}, 'answer_word': {'original': {'1': 8}, 'inline': {'0': 8}}, 'colon': {'original': {'1': 8}, 'inline': {'1': 8}}, 'prompt_last': {'original': {'1': 8}, 'inline': {'1': 8}}, 'qmark_separator': {'original': {'3': 8}, 'inline': {'3': 8}}, 'separator_answer': {'original': {'2': 8}, 'inline': {'2': 8}}, 'answer_colon': {'original': {'2': 8}, 'inline': {'1': 8}}, 'tail_all': {'original': {'3': 8}, 'inline': {'3': 8}}}`

### target

| mode | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | margin-top | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 82 | 20/82 | 20/82 | 0/82 | 57/82 | 9.4 | -1.704 | -1.970 | newline:57, correct_prefix:20, word:3, space:1, explanation:1 |  ?\n\n:57,  v:20,  o:2,  c:1,  :1,  yes:1 |
| inline | 82 | 75/82 | 72/82 | 0/82 | 0/82 | 1.1 | 2.236 | -0.012 | correct_prefix:75, space:7 |  v:75,  :7 |
| final_output_inline_to_original | 82 | 75/82 | 72/82 | 0/82 | 0/82 | 1.1 | 2.236 | -0.012 | correct_prefix:75, space:7 |  v:75,  :7 |
| patch_qmark | 82 | 20/82 | 20/82 | 0/82 | 60/82 | 7.0 | -1.572 | -1.752 | newline:60, correct_prefix:20, word:1, explanation:1 |  ?\n\n:60,  v:20,  c:1,  yes:1 |
| patch_separator | 82 | 72/82 | 70/82 | 0/82 | 1/82 | 1.1 | 2.280 | -0.030 | correct_prefix:72, space:8, word:1, newline:1 |  v:72,  :8,  o:1,  ?\n\n:1 |
| patch_colon | 82 | 69/82 | 68/82 | 0/82 | 1/82 | 1.2 | 1.968 | -0.060 | correct_prefix:69, space:11, word:1, newline:1 |  v:69,  :11,  o:1,  ?\n\n:1 |
| patch_prompt_last | 82 | 67/82 | 63/82 | 1/82 | 0/82 | 1.2 | 2.217 | -0.056 | correct_prefix:67, space:15 |  v:67,  :15 |
| patch_qmark_separator | 82 | 75/82 | 71/82 | 1/82 | 0/82 | 1.1 | 2.236 | -0.012 | correct_prefix:75, space:7 |  v:75,  :7 |
| patch_separator_answer | 82 | 72/82 | 70/82 | 0/82 | 1/82 | 1.1 | 2.280 | -0.030 | correct_prefix:72, space:8, word:1, newline:1 |  v:72,  :8,  o:1,  ?\n\n:1 |
| patch_tail_all | 82 | 75/82 | 72/82 | 0/82 | 0/82 | 1.1 | 2.236 | -0.012 | correct_prefix:75, space:7 |  v:75,  :7 |

### non_target

| mode | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | margin-top | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 174 | 38/174 | 36/174 | 3/174 | 118/174 | 7.3 | -1.282 | -1.589 | newline:118, correct_prefix:38, word:13, explanation:5 |  ?\n\n:118,  v:38,  o:11,  yes:5,  c:2 |
| inline | 174 | 171/174 | 159/174 | 8/174 | 0/174 | 1.0 | 2.362 | -0.001 | correct_prefix:171, space:3 |  v:171,  :3 |
| final_output_inline_to_original | 174 | 171/174 | 154/174 | 12/174 | 0/174 | 1.0 | 2.362 | -0.001 | correct_prefix:171, space:3 |  v:171,  :3 |
| patch_qmark | 174 | 38/174 | 35/174 | 3/174 | 124/174 | 5.8 | -1.227 | -1.444 | newline:124, correct_prefix:38, word:8, explanation:3, space:1 |  ?\n\n:124,  v:38,  o:5,  c:3,  yes:3,  :1 |
| patch_separator | 174 | 168/174 | 153/174 | 11/174 | 0/174 | 1.0 | 2.441 | -0.006 | correct_prefix:168, space:6 |  v:168,  :6 |
| patch_colon | 174 | 166/174 | 150/174 | 11/174 | 1/174 | 1.0 | 2.128 | -0.010 | correct_prefix:166, space:7, newline:1 |  v:166,  :7,  ?\n\n:1 |
| patch_prompt_last | 174 | 165/174 | 148/174 | 11/174 | 0/174 | 1.0 | 2.392 | -0.013 | correct_prefix:165, space:9 |  v:165,  :9 |
| patch_qmark_separator | 174 | 171/174 | 154/174 | 12/174 | 0/174 | 1.0 | 2.362 | -0.001 | correct_prefix:171, space:3 |  v:171,  :3 |
| patch_separator_answer | 174 | 168/174 | 153/174 | 11/174 | 0/174 | 1.0 | 2.441 | -0.006 | correct_prefix:168, space:6 |  v:168,  :6 |
| patch_tail_all | 174 | 171/174 | 158/174 | 9/174 | 0/174 | 1.0 | 2.362 | -0.001 | correct_prefix:171, space:3 |  v:171,  :3 |

### Examples

- sample=0 split=target mode=original top0=' ?\\n\\n'/newline rank=4 exact=False text=' ?\n\nTo solve' ladder=1: ?\n\n[newline], 2: ?\n[newline], 3: [space], 4: v[correct_prefix], 5: c[word]
- sample=0 split=target mode=inline top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: o[word]
- sample=0 split=target mode=final_output_inline_to_original top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: o[word]
- sample=0 split=target mode=patch_qmark top0=' ?\\n\\n'/newline rank=4 exact=False text=' ?\n\nI think' ladder=1: ?\n\n[newline], 2: ?\n[newline], 3: [space], 4: v[correct_prefix], 5: c[word]
- sample=0 split=target mode=patch_separator top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: o[word], 5: c[word]
- sample=0 split=target mode=patch_colon top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: o[word], 5: ?\n[newline]
- sample=0 split=target mode=patch_prompt_last top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: o[word], 5: c[word]
- sample=0 split=target mode=patch_qmark_separator top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: o[word]
- sample=0 split=target mode=patch_separator_answer top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: o[word], 5: c[word]
- sample=0 split=target mode=patch_tail_all top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: o[word]
- sample=1 split=non_target mode=original top0=' ?\\n\\n'/newline rank=30 exact=False text=' ?\n\nTo solve' ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: \n\n[newline], 5: yes[explanation]
- sample=1 split=non_target mode=inline top0=' v'/correct_prefix rank=1 exact=True text=' v48' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: ?\n[newline]
- sample=1 split=non_target mode=final_output_inline_to_original top0=' v'/correct_prefix rank=1 exact=True text=' v48' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: ?\n[newline]
- sample=1 split=non_target mode=patch_qmark top0=' ?\\n\\n'/newline rank=21 exact=False text=' ?\n\nTo solve' ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: \n\n[newline], 5: yes[explanation]
- sample=1 split=non_target mode=patch_separator top0=' v'/correct_prefix rank=1 exact=True text=' v48' ladder=1: v[correct_prefix], 2: [space], 3: ?\n\n[newline], 4: c[word], 5: o[word]
- sample=1 split=non_target mode=patch_colon top0=' v'/correct_prefix rank=1 exact=True text=' v48' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: ?\n[newline]
