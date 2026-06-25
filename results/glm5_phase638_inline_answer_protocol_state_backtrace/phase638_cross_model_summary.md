# Phase 638 Cross-Model Summary

目标：把 inline answer protocol state 回溯到 original prompt 的候选承载位置，测试哪些内部状态能压制 newline prior。

## qwen3

- raw_cases: 256 / target_seen: 17 / cases_written: 256 / mode_rows: 2048
- top_k: 20
- layer_map: `{'prompt_last': 27, 'answer_label': 27, 'question_mark_answer': 27, 'relation_tail': 27, 'question_subject': 27, 'question_all': 27}`
- filtered: `{'not_target': 0, 'group_missing': 0, 'group_len_mismatch': 512, 'empty_patch': 256}`

### target

| mode | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | margin-top | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 17 | 14/17 | 11/17 | 3/17 | 0/17 | 1.2 | 1.272 | -0.118 | correct_prefix:14, space:3 |  v:14,  :3 |
| inline | 17 | 1/17 | 0/17 | 0/17 | 9/17 | 4.8 | -1.471 | -1.566 | newline:9, space:7, correct_prefix:1 |  ?\n\n:9,  :7,  v:1 |
| final_output_inline_to_original | 17 | 1/17 | 0/17 | 0/17 | 9/17 | 4.8 | -1.471 | -1.566 | newline:9, space:7, correct_prefix:1 |  ?\n\n:9,  :7,  v:1 |
| patch_prompt_last | 17 | 2/17 | 1/17 | 0/17 | 7/17 | 4.1 | -1.309 | -1.684 | space:8, newline:7, correct_prefix:2 |  :8,  ?\n\n:7,  v:2 |
| patch_question_mark_answer | 17 | 1/17 | 0/17 | 0/17 | 9/17 | 4.8 | -1.471 | -1.566 | newline:9, space:7, correct_prefix:1 |  ?\n\n:9,  :7,  v:1 |
| patch_relation_tail | 17 | 1/17 | 0/17 | 0/17 | 9/17 | 4.8 | -1.471 | -1.566 | newline:9, space:7, correct_prefix:1 |  ?\n\n:9,  :7,  v:1 |
| patch_question_all | 17 | 1/17 | 0/17 | 0/17 | 9/17 | 4.8 | -1.471 | -1.566 | newline:9, space:7, correct_prefix:1 |  ?\n\n:9,  :7,  v:1 |
| patch_all5 | 17 | 1/17 | 0/17 | 0/17 | 9/17 | 4.8 | -1.471 | -1.566 | newline:9, space:7, correct_prefix:1 |  ?\n\n:9,  :7,  v:1 |

### non_target

| mode | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | margin-top | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 239 | 144/239 | 140/239 | 3/239 | 2/239 | 2.1 | 1.272 | -0.631 | correct_prefix:144, space:92, newline:2, word:1 |  v:144,  :92,  o:1,  \n\n:1,  ?\n\n:1 |
| inline | 239 | 26/239 | 27/239 | 0/239 | 123/239 | 5.3 | -1.721 | -1.970 | newline:123, space:90, correct_prefix:26 |  ?\n\n:123,  :90,  v:26 |
| final_output_inline_to_original | 239 | 26/239 | 27/239 | 0/239 | 123/239 | 5.3 | -1.721 | -1.970 | newline:123, space:90, correct_prefix:26 |  ?\n\n:123,  :90,  v:26 |
| patch_prompt_last | 239 | 39/239 | 39/239 | 0/239 | 66/239 | 4.7 | -1.623 | -2.280 | space:134, newline:66, correct_prefix:39 |  :134,  ?\n\n:66,  v:39 |
| patch_question_mark_answer | 239 | 26/239 | 27/239 | 0/239 | 123/239 | 5.3 | -1.721 | -1.970 | newline:123, space:90, correct_prefix:26 |  ?\n\n:123,  :90,  v:26 |
| patch_relation_tail | 239 | 26/239 | 27/239 | 0/239 | 123/239 | 5.3 | -1.721 | -1.970 | newline:123, space:90, correct_prefix:26 |  ?\n\n:123,  :90,  v:26 |
| patch_question_all | 239 | 26/239 | 27/239 | 0/239 | 123/239 | 5.3 | -1.721 | -1.970 | newline:123, space:90, correct_prefix:26 |  ?\n\n:123,  :90,  v:26 |
| patch_all5 | 239 | 26/239 | 27/239 | 0/239 | 123/239 | 5.3 | -1.721 | -1.970 | newline:123, space:90, correct_prefix:26 |  ?\n\n:123,  :90,  v:26 |

### Examples

- sample=0 split=non_target mode=original top0=' '/space rank=2 exact=False text=' 22' ladder=1: [space], 2: v[correct_prefix], 3: \n\n[newline], 4: ?\n\n[newline], 5: ?\n[newline]
- sample=0 split=non_target mode=inline top0=' '/space rank=6 exact=False text=' 22' ladder=1: [space], 2: ?\n\n[newline], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=0 split=non_target mode=final_output_inline_to_original top0=' '/space rank=6 exact=False text=' 22' ladder=1: [space], 2: ?\n\n[newline], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=0 split=non_target mode=patch_prompt_last top0=' '/space rank=5 exact=False text=' 22' ladder=1: [space], 2: ?\n\n[newline], 3: ?\n[newline], 4: ?[punctuation], 5: v[correct_prefix]
- sample=0 split=non_target mode=patch_question_mark_answer top0=' '/space rank=6 exact=False text=' 22' ladder=1: [space], 2: ?\n\n[newline], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=0 split=non_target mode=patch_relation_tail top0=' '/space rank=6 exact=False text=' 22' ladder=1: [space], 2: ?\n\n[newline], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=0 split=non_target mode=patch_question_all top0=' '/space rank=6 exact=False text=' 22' ladder=1: [space], 2: ?\n\n[newline], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=0 split=non_target mode=patch_all5 top0=' '/space rank=6 exact=False text=' 22' ladder=1: [space], 2: ?\n\n[newline], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=1 split=non_target mode=original top0=' v'/correct_prefix rank=1 exact=True text=' v48' ladder=1: v[correct_prefix], 2: \n\n[newline], 3: ?\n\n[newline], 4: [space], 5: ?\n[newline]
- sample=1 split=non_target mode=inline top0=' ?\\n\\n'/newline rank=6 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=1 split=non_target mode=final_output_inline_to_original top0=' ?\\n\\n'/newline rank=6 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=1 split=non_target mode=patch_prompt_last top0=' ?\\n\\n'/newline rank=4 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: ?[punctuation], 5: v[correct_prefix]
- sample=1 split=non_target mode=patch_question_mark_answer top0=' ?\\n\\n'/newline rank=6 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=1 split=non_target mode=patch_relation_tail top0=' ?\\n\\n'/newline rank=6 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=1 split=non_target mode=patch_question_all top0=' ?\\n\\n'/newline rank=6 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=1 split=non_target mode=patch_all5 top0=' ?\\n\\n'/newline rank=6 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]

## glm4

- raw_cases: 256 / target_seen: 31 / cases_written: 256 / mode_rows: 2048
- top_k: 20
- layer_map: `{'prompt_last': 32, 'answer_label': 32, 'question_mark_answer': 32, 'relation_tail': 32, 'question_subject': 32, 'question_all': 32}`
- filtered: `{'not_target': 0, 'group_missing': 0, 'group_len_mismatch': 512, 'empty_patch': 256}`

### target

| mode | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | margin-top | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 31 | 29/31 | 28/31 | 1/31 | 0/31 | 1.1 | 80.722 | -0.020 | correct_prefix:29, word:2 |  v:29,  c:2 |
| inline | 31 | 27/31 | 26/31 | 1/31 | 0/31 | 1.2 | 71.648 | -0.042 | correct_prefix:27, explanation:3, word:1 |  v:27,  Yes:3,  c:1 |
| final_output_inline_to_original | 31 | 27/31 | 27/31 | 1/31 | 0/31 | 1.2 | 71.648 | -0.042 | correct_prefix:27, explanation:3, word:1 |  v:27,  Yes:3,  c:1 |
| patch_prompt_last | 31 | 26/31 | 25/31 | 1/31 | 0/31 | 1.3 | 99.000 | -0.081 | correct_prefix:26, explanation:3, word:2 |  v:26,  Yes:3,  c:2 |
| patch_question_mark_answer | 31 | 27/31 | 27/31 | 1/31 | 0/31 | 1.2 | 71.648 | -0.042 | correct_prefix:27, explanation:3, word:1 |  v:27,  Yes:3,  c:1 |
| patch_relation_tail | 31 | 27/31 | 27/31 | 1/31 | 0/31 | 1.2 | 71.648 | -0.042 | correct_prefix:27, explanation:3, word:1 |  v:27,  Yes:3,  c:1 |
| patch_question_all | 31 | 27/31 | 27/31 | 1/31 | 0/31 | 1.2 | 71.648 | -0.042 | correct_prefix:27, explanation:3, word:1 |  v:27,  Yes:3,  c:1 |
| patch_all5 | 31 | 27/31 | 27/31 | 1/31 | 0/31 | 1.2 | 71.648 | -0.042 | correct_prefix:27, explanation:3, word:1 |  v:27,  Yes:3,  c:1 |

### non_target

| mode | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | margin-top | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 225 | 209/225 | 183/225 | 23/225 | 0/225 | 1.1 | 78.048 | -0.018 | correct_prefix:209, word:15, explanation:1 |  v:209,  c:15,  No:1 |
| inline | 225 | 203/225 | 177/225 | 23/225 | 0/225 | 1.2 | 65.054 | -0.041 | correct_prefix:203, word:13, explanation:9 |  v:203,  c:13,  Yes:7,  No:2 |
| final_output_inline_to_original | 225 | 203/225 | 177/225 | 23/225 | 0/225 | 1.2 | 65.054 | -0.041 | correct_prefix:203, word:13, explanation:9 |  v:203,  c:13,  Yes:7,  No:2 |
| patch_prompt_last | 225 | 176/225 | 152/225 | 22/225 | 0/225 | 1.4 | 99.000 | -0.092 | correct_prefix:176, word:39, explanation:10 |  v:176,  c:39,  Yes:8,  No:2 |
| patch_question_mark_answer | 225 | 203/225 | 177/225 | 23/225 | 0/225 | 1.2 | 65.054 | -0.041 | correct_prefix:203, word:13, explanation:9 |  v:203,  c:13,  Yes:7,  No:2 |
| patch_relation_tail | 225 | 203/225 | 175/225 | 25/225 | 0/225 | 1.2 | 65.054 | -0.041 | correct_prefix:203, word:13, explanation:9 |  v:203,  c:13,  Yes:7,  No:2 |
| patch_question_all | 225 | 203/225 | 175/225 | 25/225 | 0/225 | 1.2 | 65.054 | -0.041 | correct_prefix:203, word:13, explanation:9 |  v:203,  c:13,  Yes:7,  No:2 |
| patch_all5 | 225 | 203/225 | 175/225 | 25/225 | 0/225 | 1.2 | 65.054 | -0.041 | correct_prefix:203, word:13, explanation:9 |  v:203,  c:13,  Yes:7,  No:2 |

### Examples

- sample=0 split=non_target mode=original top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: c[word], 3: o[word], 4: [space], 5: Yes[explanation]
- sample=0 split=non_target mode=inline top0=' v'/correct_prefix rank=1 exact=False text=' c59' ladder=1: v[correct_prefix], 2: c[word], 3: Yes[explanation], 4: r[word], 5: [space]
- sample=0 split=non_target mode=final_output_inline_to_original top0=' v'/correct_prefix rank=1 exact=False text=' c59' ladder=1: v[correct_prefix], 2: c[word], 3: Yes[explanation], 4: r[word], 5: [space]
- sample=0 split=non_target mode=patch_prompt_last top0=' c'/word rank=2 exact=False text=' c59' ladder=1: c[word], 2: v[correct_prefix], 3: Yes[explanation], 4: r[word], 5: [space]
- sample=0 split=non_target mode=patch_question_mark_answer top0=' v'/correct_prefix rank=1 exact=False text=' c59' ladder=1: v[correct_prefix], 2: c[word], 3: Yes[explanation], 4: r[word], 5: [space]
- sample=0 split=non_target mode=patch_relation_tail top0=' v'/correct_prefix rank=1 exact=False text=' c59' ladder=1: v[correct_prefix], 2: c[word], 3: Yes[explanation], 4: r[word], 5: [space]
- sample=0 split=non_target mode=patch_question_all top0=' v'/correct_prefix rank=1 exact=False text=' c59' ladder=1: v[correct_prefix], 2: c[word], 3: Yes[explanation], 4: r[word], 5: [space]
- sample=0 split=non_target mode=patch_all5 top0=' v'/correct_prefix rank=1 exact=False text=' c59' ladder=1: v[correct_prefix], 2: c[word], 3: Yes[explanation], 4: r[word], 5: [space]
- sample=1 split=non_target mode=original top0=' v'/correct_prefix rank=1 exact=True text=' v48' ladder=1: v[correct_prefix], 2: c[word], 3: Yes[explanation], 4: o[word], 5: No[explanation]
- sample=1 split=non_target mode=inline top0=' c'/word rank=2 exact=False text=' c59' ladder=1: c[word], 2: v[correct_prefix], 3: Yes[explanation], 4: No[explanation], 5: o[word]
- sample=1 split=non_target mode=final_output_inline_to_original top0=' c'/word rank=2 exact=False text=' c59' ladder=1: c[word], 2: v[correct_prefix], 3: Yes[explanation], 4: No[explanation], 5: o[word]
- sample=1 split=non_target mode=patch_prompt_last top0=' c'/word rank=3 exact=False text=' c59' ladder=1: c[word], 2: Yes[explanation], 3: v[correct_prefix], 4: No[explanation], 5: o[word]
- sample=1 split=non_target mode=patch_question_mark_answer top0=' c'/word rank=2 exact=False text=' c59' ladder=1: c[word], 2: v[correct_prefix], 3: Yes[explanation], 4: No[explanation], 5: o[word]
- sample=1 split=non_target mode=patch_relation_tail top0=' c'/word rank=2 exact=False text=' c59' ladder=1: c[word], 2: v[correct_prefix], 3: Yes[explanation], 4: No[explanation], 5: o[word]
- sample=1 split=non_target mode=patch_question_all top0=' c'/word rank=2 exact=False text=' c59' ladder=1: c[word], 2: v[correct_prefix], 3: Yes[explanation], 4: No[explanation], 5: o[word]
- sample=1 split=non_target mode=patch_all5 top0=' c'/word rank=2 exact=False text=' c59' ladder=1: c[word], 2: v[correct_prefix], 3: Yes[explanation], 4: No[explanation], 5: o[word]

## deepseek7b

- raw_cases: 256 / target_seen: 82 / cases_written: 256 / mode_rows: 2048
- top_k: 20
- layer_map: `{'prompt_last': 25, 'answer_label': 21, 'question_mark_answer': 21, 'relation_tail': 23, 'question_subject': 21, 'question_all': 20}`
- filtered: `{'not_target': 0, 'group_missing': 0, 'group_len_mismatch': 512, 'empty_patch': 256}`

### target

| mode | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | margin-top | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 82 | 20/82 | 20/82 | 0/82 | 57/82 | 9.4 | -1.704 | -1.970 | newline:57, correct_prefix:20, word:3, space:1, explanation:1 |  ?\n\n:57,  v:20,  o:2,  c:1,  :1,  yes:1 |
| inline | 82 | 75/82 | 72/82 | 0/82 | 0/82 | 1.1 | 2.236 | -0.012 | correct_prefix:75, space:7 |  v:75,  :7 |
| final_output_inline_to_original | 82 | 75/82 | 72/82 | 0/82 | 0/82 | 1.1 | 2.236 | -0.012 | correct_prefix:75, space:7 |  v:75,  :7 |
| patch_prompt_last | 82 | 67/82 | 63/82 | 1/82 | 0/82 | 1.2 | 2.217 | -0.056 | correct_prefix:67, space:15 |  v:67,  :15 |
| patch_question_mark_answer | 82 | 75/82 | 71/82 | 1/82 | 0/82 | 1.1 | 2.236 | -0.012 | correct_prefix:75, space:7 |  v:75,  :7 |
| patch_relation_tail | 82 | 75/82 | 71/82 | 1/82 | 0/82 | 1.1 | 2.236 | -0.012 | correct_prefix:75, space:7 |  v:75,  :7 |
| patch_question_all | 82 | 75/82 | 72/82 | 0/82 | 0/82 | 1.1 | 2.236 | -0.012 | correct_prefix:75, space:7 |  v:75,  :7 |
| patch_all5 | 82 | 67/82 | 64/82 | 0/82 | 0/82 | 1.2 | 2.217 | -0.056 | correct_prefix:67, space:15 |  v:67,  :15 |

### non_target

| mode | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | margin-top | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 174 | 38/174 | 36/174 | 3/174 | 118/174 | 7.3 | -1.282 | -1.589 | newline:118, correct_prefix:38, word:13, explanation:5 |  ?\n\n:118,  v:38,  o:11,  yes:5,  c:2 |
| inline | 174 | 171/174 | 159/174 | 8/174 | 0/174 | 1.0 | 2.362 | -0.001 | correct_prefix:171, space:3 |  v:171,  :3 |
| final_output_inline_to_original | 174 | 171/174 | 154/174 | 12/174 | 0/174 | 1.0 | 2.362 | -0.001 | correct_prefix:171, space:3 |  v:171,  :3 |
| patch_prompt_last | 174 | 165/174 | 148/174 | 11/174 | 0/174 | 1.0 | 2.392 | -0.013 | correct_prefix:165, space:9 |  v:165,  :9 |
| patch_question_mark_answer | 174 | 171/174 | 154/174 | 12/174 | 0/174 | 1.0 | 2.362 | -0.001 | correct_prefix:171, space:3 |  v:171,  :3 |
| patch_relation_tail | 174 | 171/174 | 154/174 | 12/174 | 0/174 | 1.0 | 2.362 | -0.001 | correct_prefix:171, space:3 |  v:171,  :3 |
| patch_question_all | 174 | 171/174 | 158/174 | 9/174 | 0/174 | 1.0 | 2.362 | -0.001 | correct_prefix:171, space:3 |  v:171,  :3 |
| patch_all5 | 174 | 165/174 | 151/174 | 9/174 | 0/174 | 1.0 | 2.392 | -0.013 | correct_prefix:165, space:9 |  v:165,  :9 |

### Examples

- sample=0 split=target mode=original top0=' ?\\n\\n'/newline rank=4 exact=False text=' ?\n\nTo solve' ladder=1: ?\n\n[newline], 2: ?\n[newline], 3: [space], 4: v[correct_prefix], 5: c[word]
- sample=0 split=target mode=inline top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: o[word]
- sample=0 split=target mode=final_output_inline_to_original top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: o[word]
- sample=0 split=target mode=patch_prompt_last top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: o[word], 5: c[word]
- sample=0 split=target mode=patch_question_mark_answer top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: o[word]
- sample=0 split=target mode=patch_relation_tail top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: o[word]
- sample=0 split=target mode=patch_question_all top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: o[word]
- sample=0 split=target mode=patch_all5 top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: o[word], 5: c[word]
- sample=1 split=non_target mode=original top0=' ?\\n\\n'/newline rank=30 exact=False text=' ?\n\nTo solve' ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: \n\n[newline], 5: yes[explanation]
- sample=1 split=non_target mode=inline top0=' v'/correct_prefix rank=1 exact=True text=' v48' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: ?\n[newline]
- sample=1 split=non_target mode=final_output_inline_to_original top0=' v'/correct_prefix rank=1 exact=True text=' v48' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: ?\n[newline]
- sample=1 split=non_target mode=patch_prompt_last top0=' v'/correct_prefix rank=1 exact=True text=' v48' ladder=1: v[correct_prefix], 2: [space], 3: ?\n\n[newline], 4: c[word], 5: o[word]
- sample=1 split=non_target mode=patch_question_mark_answer top0=' v'/correct_prefix rank=1 exact=True text=' v48' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: ?\n[newline]
- sample=1 split=non_target mode=patch_relation_tail top0=' v'/correct_prefix rank=1 exact=True text=' v48' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: ?\n[newline]
- sample=1 split=non_target mode=patch_question_all top0=' v'/correct_prefix rank=1 exact=True text=' v48' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: ?\n[newline]
- sample=1 split=non_target mode=patch_all5 top0=' v'/correct_prefix rank=1 exact=True text=' v48' ladder=1: v[correct_prefix], 2: [space], 3: ?\n\n[newline], 4: c[word], 5: o[word]
