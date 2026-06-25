# Phase 637 Cross-Model Summary

目标：测试 prompt ablation 是否能压制 newline / format continuation prior，并记录 non-target 副作用。

## qwen3

- raw_cases: 256 / target_seen: 17 / rows: 4608
- top_k: 20

### base_subject / target

| variant | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 17 | 11/17 | 1/17 | 9/17 | 5/17 | 1.9 | 0.316 | correct_prefix:11, newline:5, space:1 |  v:11,  ?\n\n:5,  :1 |
| no_qmark | 17 | 16/17 | 4/17 | 12/17 | 0/17 | 1.1 | 2.949 | correct_prefix:16, word:1 |  v:16,  o:1 |
| period | 17 | 15/17 | 5/17 | 10/17 | 0/17 | 1.1 | 2.522 | correct_prefix:15, word:2 |  v:15,  o:2 |
| inline_answer | 17 | 0/17 | 0/17 | 0/17 | 17/17 | 6.5 | -2.846 | newline:17 |  ?\n\n:17 |
| short_only | 17 | 0/17 | 0/17 | 0/17 | 0/17 | 13.6 | -4.713 | space:17 |  :17 |
| no_explain | 17 | 0/17 | 0/17 | 0/17 | 0/17 | 7.2 | -3.471 | space:17 |  :17 |
| no_qmark_short | 17 | 0/17 | 0/17 | 0/17 | 0/17 | 9.3 | -3.206 | space:17 |  :17 |
| value_label | 17 | 0/17 | 0/17 | 0/17 | 17/17 | 13.5 | -5.757 | newline:17 |  ?\n\n:17 |
| direct_value_label | 17 | 0/17 | 0/17 | 0/17 | 17/17 | 15.9 | -6.007 | newline:17 |  ?\n\n:17 |

### base_subject / non_target

| variant | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 239 | 172/239 | 170/239 | 4/239 | 38/239 | 1.6 | 0.905 | correct_prefix:172, newline:38, space:24, word:5 |  v:172,  ?\n\n:38,  :24,  o:5 |
| no_qmark | 239 | 237/239 | 233/239 | 3/239 | 0/239 | 1.0 | 3.472 | correct_prefix:237, space:2 |  v:237,  :2 |
| period | 239 | 230/239 | 225/239 | 3/239 | 0/239 | 1.0 | 3.184 | correct_prefix:230, word:7, space:2 |  v:230,  o:7,  :2 |
| inline_answer | 239 | 9/239 | 10/239 | 0/239 | 226/239 | 5.7 | -2.463 | newline:226, correct_prefix:9, space:4 |  ?\n\n:226,  v:9,  :4 |
| short_only | 239 | 0/239 | 0/239 | 0/239 | 0/239 | 11.8 | -4.369 | space:239 |  :239 |
| no_explain | 239 | 1/239 | 1/239 | 0/239 | 0/239 | 6.0 | -2.968 | space:238, correct_prefix:1 |  :238,  v:1 |
| no_qmark_short | 239 | 1/239 | 1/239 | 0/239 | 0/239 | 8.5 | -2.745 | space:238, correct_prefix:1 |  :238,  v:1 |
| value_label | 239 | 0/239 | 0/239 | 0/239 | 235/239 | 12.8 | -5.545 | newline:235, space:4 |  ?\n\n:235,  :4 |
| direct_value_label | 239 | 0/239 | 0/239 | 0/239 | 239/239 | 14.7 | -5.738 | newline:239 |  ?\n\n:239 |

### repair_subject / target

| variant | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 17 | 14/17 | 11/17 | 3/17 | 0/17 | 1.2 | 1.272 | correct_prefix:14, space:3 |  v:14,  :3 |
| no_qmark | 17 | 17/17 | 16/17 | 1/17 | 0/17 | 1.0 | 3.971 | correct_prefix:17 |  v:17 |
| period | 17 | 16/17 | 15/17 | 1/17 | 0/17 | 1.1 | 3.699 | correct_prefix:16, space:1 |  v:16,  :1 |
| inline_answer | 17 | 1/17 | 0/17 | 0/17 | 9/17 | 4.8 | -1.471 | newline:9, space:7, correct_prefix:1 |  ?\n\n:9,  :7,  v:1 |
| short_only | 17 | 0/17 | 0/17 | 0/17 | 0/17 | 5.4 | -2.897 | space:17 |  :17 |
| no_explain | 17 | 0/17 | 0/17 | 0/17 | 0/17 | 3.1 | -0.919 | space:17 |  :17 |
| no_qmark_short | 17 | 0/17 | 0/17 | 0/17 | 0/17 | 3.6 | -1.397 | space:17 |  :17 |
| value_label | 17 | 0/17 | 0/17 | 0/17 | 16/17 | 11.3 | -5.478 | newline:16, space:1 |  ?\n\n:16,  :1 |
| direct_value_label | 17 | 0/17 | 0/17 | 0/17 | 15/17 | 11.5 | -5.272 | newline:15, space:2 |  ?\n\n:15,  :2 |

### repair_subject / non_target

| variant | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 239 | 144/239 | 140/239 | 3/239 | 2/239 | 2.1 | 1.272 | correct_prefix:144, space:92, newline:2, word:1 |  v:144,  :92,  o:1,  \n\n:1,  ?\n\n:1 |
| no_qmark | 239 | 213/239 | 208/239 | 3/239 | 0/239 | 1.2 | 3.865 | correct_prefix:213, space:26 |  v:213,  :26 |
| period | 239 | 198/239 | 192/239 | 2/239 | 0/239 | 1.3 | 3.624 | correct_prefix:198, space:41 |  v:198,  :41 |
| inline_answer | 239 | 26/239 | 27/239 | 0/239 | 123/239 | 5.3 | -1.721 | newline:123, space:90, correct_prefix:26 |  ?\n\n:123,  :90,  v:26 |
| short_only | 239 | 2/239 | 2/239 | 0/239 | 0/239 | 7.6 | -3.302 | space:237, correct_prefix:2 |  :237,  v:2 |
| no_explain | 239 | 11/239 | 11/239 | 0/239 | 0/239 | 3.8 | -1.518 | space:228, correct_prefix:11 |  :228,  v:11 |
| no_qmark_short | 239 | 6/239 | 6/239 | 0/239 | 0/239 | 5.3 | -1.971 | space:233, correct_prefix:6 |  :233,  v:6 |
| value_label | 239 | 0/239 | 0/239 | 0/239 | 229/239 | 12.3 | -5.515 | newline:229, space:10 |  ?\n\n:229,  :10 |
| direct_value_label | 239 | 0/239 | 0/239 | 0/239 | 214/239 | 12.1 | -5.142 | newline:214, space:25 |  ?\n\n:214,  :25 |

### Examples

- sample=0 split=non_target mode=base_subject__original top0=' ?\\n\\n'/newline rank=2 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: v[correct_prefix], 3: [space], 4: ?\n[newline], 5: \n\n[newline]
- sample=0 split=non_target mode=base_subject__no_qmark top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: o[word], 3: [space], 4: ?\n\n[newline], 5: ?\n[newline]
- sample=0 split=non_target mode=base_subject__period top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: o[word], 3: [space], 4: ?\n\n[newline], 5: ?\n[newline]
- sample=0 split=non_target mode=base_subject__inline_answer top0=' ?\\n\\n'/newline rank=6 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: ?\n[newline], 3: [space], 4: ?[punctuation], 5: \n\n[newline]
- sample=0 split=non_target mode=base_subject__short_only top0=' '/space rank=7 exact=False text=' 22' ladder=1: [space], 2: \n\n[newline], 3: ?\n\n[newline], 4: ?\n[newline], 5: \n[newline]
- sample=0 split=non_target mode=base_subject__no_explain top0=' '/space rank=3 exact=False text=' 22' ladder=1: [space], 2: \n\n[newline], 3: v[correct_prefix], 4: \n[newline], 5: The[explanation]
- sample=0 split=non_target mode=base_subject__no_qmark_short top0=' '/space rank=5 exact=False text=' 22' ladder=1: [space], 2: \n\n[newline], 3: The[explanation], 4: o[word], 5: v[correct_prefix]
- sample=0 split=non_target mode=base_subject__value_label top0=' ?\\n\\n'/newline rank=12 exact=False text=' ?\n\nTo solve' ladder=1: ?\n\n[newline], 2: ?\n[newline], 3: [space], 4: ?[punctuation], 5: \n\n[newline]
- sample=0 split=non_target mode=base_subject__direct_value_label top0=' ?\\n\\n'/newline rank=9 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: \n\n[newline], 3: ?\n[newline], 4: ?[punctuation], 5: [space]
- sample=0 split=non_target mode=repair_subject__original top0=' '/space rank=2 exact=False text=' 22' ladder=1: [space], 2: v[correct_prefix], 3: \n\n[newline], 4: ?\n\n[newline], 5: ?\n[newline]
- sample=0 split=non_target mode=repair_subject__no_qmark top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: [space], 3: \n\n[newline], 4: ?\n\n[newline], 5: o[word]
- sample=0 split=non_target mode=repair_subject__period top0=' '/space rank=2 exact=False text=' 22' ladder=1: [space], 2: v[correct_prefix], 3: \n\n[newline], 4: ?\n\n[newline], 5: o[word]
- sample=0 split=non_target mode=repair_subject__inline_answer top0=' '/space rank=6 exact=False text=' 22' ladder=1: [space], 2: ?\n\n[newline], 3: ?\n[newline], 4: \n\n[newline], 5: ?[punctuation]
- sample=0 split=non_target mode=repair_subject__short_only top0=' '/space rank=4 exact=False text=' 22' ladder=1: [space], 2: \n\n[newline], 3: \n[newline], 4: v[correct_prefix], 5: The[explanation]

## glm4

- raw_cases: 256 / target_seen: 31 / rows: 4608
- top_k: 20

### base_subject / target

| variant | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 31 | 11/31 | 2/31 | 9/31 | 0/31 | 2.7 | 86.734 | word:17, correct_prefix:11, explanation:3 |  o:14,  v:11,  c:3,  Yes:2,  No:1 |
| no_qmark | 31 | 23/31 | 1/31 | 22/31 | 0/31 | 1.5 | 99.000 | correct_prefix:23, word:8 |  v:23,  c:3,  o:3,  True:1,  False:1 |
| period | 31 | 1/31 | 0/31 | 1/31 | 0/31 | 8.6 | 99.000 | word:30, correct_prefix:1 |  True:11,  False:9,  o:8,  c:2,  v:1 |
| inline_answer | 31 | 6/31 | 1/31 | 3/31 | 0/31 | 6.2 | 99.000 | explanation:18, word:7, correct_prefix:6 |  Yes:15,  v:6,  o:4,  No:3,  c:3 |
| short_only | 31 | 0/31 | 0/31 | 0/31 | 0/31 | 2.1 | 16.018 | space:31 |  :31 |
| no_explain | 31 | 11/31 | 5/31 | 5/31 | 0/31 | 1.7 | 83.761 | space:19, correct_prefix:11, word:1 |  :19,  v:11,  c:1 |
| no_qmark_short | 31 | 3/31 | 1/31 | 2/31 | 0/31 | 2.0 | 25.443 | space:28, correct_prefix:3 |  :28,  v:3 |
| value_label | 31 | 29/31 | 7/31 | 22/31 | 0/31 | 1.1 | 31.798 | correct_prefix:29, word:2 |  v:29,  c:2 |
| direct_value_label | 31 | 4/31 | 0/31 | 3/31 | 0/31 | 2.0 | 28.496 | space:27, correct_prefix:4 |  :27,  v:4 |

### base_subject / non_target

| variant | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 225 | 99/225 | 88/225 | 7/225 | 0/225 | 2.3 | 85.897 | word:103, correct_prefix:99, explanation:23 |  v:99,  o:85,  c:17,  Yes:15,  No:8,  True:1 |
| no_qmark | 225 | 188/225 | 167/225 | 20/225 | 0/225 | 1.3 | 99.000 | correct_prefix:188, word:37 |  v:188,  c:22,  o:8,  True:6,  False:1 |
| period | 225 | 8/225 | 7/225 | 0/225 | 0/225 | 9.0 | 99.000 | word:217, correct_prefix:8 |  True:97,  o:66,  False:35,  c:19,  v:8 |
| inline_answer | 225 | 51/225 | 43/225 | 6/225 | 0/225 | 5.5 | 86.660 | explanation:110, word:64, correct_prefix:51 |  Yes:89,  v:51,  c:33,  o:31,  No:21 |
| short_only | 225 | 6/225 | 5/225 | 0/225 | 0/225 | 2.6 | 14.633 | space:219, correct_prefix:6 |  :219,  v:6 |
| no_explain | 225 | 99/225 | 75/225 | 18/225 | 0/225 | 1.7 | 89.336 | space:124, correct_prefix:99, word:2 |  :124,  v:99,  c:2 |
| no_qmark_short | 225 | 46/225 | 36/225 | 5/225 | 0/225 | 2.2 | 23.001 | space:179, correct_prefix:46 |  :179,  v:46 |
| value_label | 225 | 203/225 | 174/225 | 24/225 | 0/225 | 1.1 | 35.453 | correct_prefix:203, word:17, space:4, explanation:1 |  v:203,  c:17,  :4,  Yes:1 |
| direct_value_label | 225 | 35/225 | 30/225 | 4/225 | 0/225 | 2.1 | 15.766 | space:189, correct_prefix:35, symbol:1 |  :189,  v:35, ##:1 |

### repair_subject / target

| variant | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 31 | 29/31 | 28/31 | 1/31 | 0/31 | 1.1 | 80.722 | correct_prefix:29, word:2 |  v:29,  c:2 |
| no_qmark | 31 | 31/31 | 31/31 | 0/31 | 0/31 | 1.0 | 99.000 | correct_prefix:31 |  v:31 |
| period | 31 | 14/31 | 12/31 | 2/31 | 0/31 | 1.6 | 99.000 | word:17, correct_prefix:14 |  o:17,  v:14 |
| inline_answer | 31 | 27/31 | 26/31 | 1/31 | 0/31 | 1.2 | 71.648 | correct_prefix:27, explanation:3, word:1 |  v:27,  Yes:3,  c:1 |
| short_only | 31 | 0/31 | 0/31 | 0/31 | 0/31 | 2.7 | 34.138 | space:31 |  :31 |
| no_explain | 31 | 7/31 | 7/31 | 0/31 | 0/31 | 1.8 | 86.871 | space:24, correct_prefix:7 |  :24,  v:7 |
| no_qmark_short | 31 | 1/31 | 0/31 | 0/31 | 0/31 | 2.1 | 59.391 | space:30, correct_prefix:1 |  :30,  v:1 |
| value_label | 31 | 29/31 | 24/31 | 4/31 | 0/31 | 1.0 | 16.261 | correct_prefix:29, word:1, space:1 |  v:29,  o:1,  :1 |
| direct_value_label | 31 | 5/31 | 4/31 | 0/31 | 0/31 | 2.0 | 46.983 | space:26, correct_prefix:5 |  :26,  v:5 |

### repair_subject / non_target

| variant | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 225 | 209/225 | 183/225 | 23/225 | 0/225 | 1.1 | 78.048 | correct_prefix:209, word:15, explanation:1 |  v:209,  c:15,  No:1 |
| no_qmark | 225 | 222/225 | 199/225 | 21/225 | 0/225 | 1.0 | 97.751 | correct_prefix:222, word:3 |  v:222,  o:3 |
| period | 225 | 118/225 | 103/225 | 14/225 | 0/225 | 1.6 | 98.153 | correct_prefix:118, word:107 |  v:118,  o:94,  c:11,  True:2 |
| inline_answer | 225 | 203/225 | 177/225 | 23/225 | 0/225 | 1.2 | 65.054 | correct_prefix:203, word:13, explanation:9 |  v:203,  c:13,  Yes:7,  No:2 |
| short_only | 225 | 7/225 | 4/225 | 2/225 | 0/225 | 3.8 | 33.233 | space:218, correct_prefix:7 |  :218,  v:7 |
| no_explain | 225 | 71/225 | 57/225 | 12/225 | 0/225 | 1.9 | 88.032 | space:154, correct_prefix:71 |  :154,  v:71 |
| no_qmark_short | 225 | 31/225 | 25/225 | 4/225 | 0/225 | 2.9 | 71.160 | space:194, correct_prefix:31 |  :194,  v:31 |
| value_label | 225 | 205/225 | 185/225 | 16/225 | 0/225 | 1.1 | 11.545 | correct_prefix:205, space:18, word:2 |  v:205,  :18,  o:2 |
| direct_value_label | 225 | 38/225 | 34/225 | 3/225 | 0/225 | 2.2 | 43.867 | space:187, correct_prefix:38 |  :187,  v:38 |

### Examples

- sample=0 split=non_target mode=base_subject__original top0=' c'/word rank=3 exact=False text=' c59' ladder=1: c[word], 2: o[word], 3: v[correct_prefix], 4: r[word], 5: Yes[explanation]
- sample=0 split=non_target mode=base_subject__no_qmark top0=' c'/word rank=3 exact=False text=' c59' ladder=1: c[word], 2: o[word], 3: v[correct_prefix], 4: belongs[word], 5: r[word]
- sample=0 split=non_target mode=base_subject__period top0=' c'/word rank=10 exact=False text=' c59' ladder=1: c[word], 2: o[word], 3: True[word], 4: r[word], 5: Yes[explanation]
- sample=0 split=non_target mode=base_subject__inline_answer top0=' c'/word rank=6 exact=False text=' c59' ladder=1: c[word], 2: o[word], 3: Yes[explanation], 4: No[explanation], 5: r[word]
- sample=0 split=non_target mode=base_subject__short_only top0=' '/space rank=4 exact=False text=' 22' ladder=1: [space], 2: c[word], 3:22[number], 4: v[correct_prefix], 5: o[word]
- sample=0 split=non_target mode=base_subject__no_explain top0=' '/space rank=5 exact=False text=' 22' ladder=1: [space], 2: c[word], 3: o[word], 4:22[number], 5: v[correct_prefix]
- sample=0 split=non_target mode=base_subject__no_qmark_short top0=' '/space rank=4 exact=False text=' 22' ladder=1: [space], 2: c[word], 3:22[number], 4: v[correct_prefix], 5: o[word]
- sample=0 split=non_target mode=base_subject__value_label top0=' c'/word rank=3 exact=False text=' c59' ladder=1: c[word], 2: [space], 3: v[correct_prefix], 4: r[word], 5: o[word]
- sample=0 split=non_target mode=base_subject__direct_value_label top0=' '/space rank=3 exact=False text=' 22' ladder=1: [space], 2:##[symbol], 3: v[correct_prefix], 4: r[word], 5:Solution[word]
- sample=0 split=non_target mode=repair_subject__original top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: c[word], 3: o[word], 4: [space], 5: Yes[explanation]
- sample=0 split=non_target mode=repair_subject__no_qmark top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: o[word], 3: c[word], 4: belongs[word], 5: [space]
- sample=0 split=non_target mode=repair_subject__period top0=' o'/word rank=3 exact=False text=' o17' ladder=1: o[word], 2: c[word], 3: v[correct_prefix], 4: [space], 5: Yes[explanation]
- sample=0 split=non_target mode=repair_subject__inline_answer top0=' v'/correct_prefix rank=1 exact=False text=' c59' ladder=1: v[correct_prefix], 2: c[word], 3: Yes[explanation], 4: r[word], 5: [space]
- sample=0 split=non_target mode=repair_subject__short_only top0=' '/space rank=6 exact=False text=' 22' ladder=1: [space], 2:22[number], 3:05[number], 4: c[word], 5:48[number]

## deepseek7b

- raw_cases: 256 / target_seen: 82 / rows: 4608
- top_k: 20

### base_subject / target

| variant | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 82 | 0/82 | 0/82 | 0/82 | 81/82 | 92.8 | -6.354 | newline:81, word:1 |  ?\n\n:81,  c:1 |
| no_qmark | 82 | 8/82 | 2/82 | 5/82 | 39/82 | 6.1 | -1.815 | newline:39, word:35, correct_prefix:8 |  ?\n\n:39,  c:33,  v:8,  belongs:2 |
| period | 82 | 6/82 | 1/82 | 2/82 | 38/82 | 8.2 | -1.700 | newline:38, word:38, correct_prefix:6 |  ?\n\n:38,  c:32,  v:6,  belongs:3,  r:2,  o:1 |
| inline_answer | 82 | 33/82 | 5/82 | 30/82 | 45/82 | 1.8 | -0.205 | newline:45, correct_prefix:33, word:3, space:1 |  ?\n\n:45,  v:33,  c:3,  :1 |
| short_only | 82 | 0/82 | 0/82 | 0/82 | 35/82 | 29.1 | -4.559 | space:46, newline:35, word:1 |  :46,  ?\n\n:35,  c:1 |
| no_explain | 82 | 0/82 | 0/82 | 0/82 | 79/82 | 20.9 | -5.206 | newline:79, word:2, space:1 |  ?\n\n:79,  c:2,  :1 |
| no_qmark_short | 82 | 0/82 | 0/82 | 0/82 | 72/82 | 18.5 | -4.370 | newline:72, space:9, word:1 |  ?\n\n:72,  :9,  c:1 |
| value_label | 82 | 0/82 | 0/82 | 0/82 | 82/82 | 3.7 | -2.276 | newline:82 |  ?\n\n:82 |
| direct_value_label | 82 | 0/82 | 0/82 | 0/82 | 82/82 | 39.2 | -7.841 | newline:82 |  ?\n\n:82 |

### base_subject / non_target

| variant | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 174 | 1/174 | 1/174 | 0/174 | 164/174 | 81.9 | -5.994 | newline:164, word:9, correct_prefix:1 |  ?\n\n:164,  c:9,  v:1 |
| no_qmark | 174 | 20/174 | 19/174 | 2/174 | 78/174 | 5.5 | -1.356 | newline:78, word:76, correct_prefix:20 |  ?\n\n:78,  c:69,  v:20,  belongs:7 |
| period | 174 | 24/174 | 22/174 | 2/174 | 69/174 | 7.2 | -1.271 | word:79, newline:69, correct_prefix:24, space:2 |  ?\n\n:69,  c:61,  v:24,  belongs:11,  r:7,  :2 |
| inline_answer | 174 | 78/174 | 75/174 | 10/174 | 88/174 | 1.7 | -0.036 | newline:88, correct_prefix:78, word:5, space:3 |  ?\n\n:88,  v:78,  c:5,  :3 |
| short_only | 174 | 0/174 | 0/174 | 0/174 | 60/174 | 26.2 | -4.258 | space:110, newline:60, word:4 |  :110,  ?\n\n:60,  c:4 |
| no_explain | 174 | 0/174 | 0/174 | 0/174 | 163/174 | 19.6 | -4.915 | newline:163, word:11 |  ?\n\n:163,  c:11 |
| no_qmark_short | 174 | 0/174 | 0/174 | 0/174 | 154/174 | 16.4 | -4.112 | newline:154, space:16, word:4 |  ?\n\n:154,  :16,  c:4 |
| value_label | 174 | 3/174 | 3/174 | 1/174 | 171/174 | 3.5 | -2.049 | newline:171, correct_prefix:3 |  ?\n\n:171,  v:3 |
| direct_value_label | 174 | 0/174 | 0/174 | 0/174 | 174/174 | 41.7 | -7.839 | newline:174 |  ?\n\n:174 |

### repair_subject / target

| variant | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 82 | 20/82 | 20/82 | 0/82 | 57/82 | 9.4 | -1.704 | newline:57, correct_prefix:20, word:3, space:1, explanation:1 |  ?\n\n:57,  v:20,  o:2,  c:1,  :1,  yes:1 |
| no_qmark | 82 | 39/82 | 36/82 | 1/82 | 27/82 | 2.1 | 0.479 | correct_prefix:39, newline:27, word:14, space:2 |  v:39,  ?\n\n:27,  o:14,  :2 |
| period | 82 | 28/82 | 27/82 | 0/82 | 16/82 | 2.8 | 0.393 | word:36, correct_prefix:28, newline:16, space:2 |  o:34,  v:28,  ?\n\n:16,  :2,  c:2 |
| inline_answer | 82 | 75/82 | 72/82 | 0/82 | 0/82 | 1.1 | 2.236 | correct_prefix:75, space:7 |  v:75,  :7 |
| short_only | 82 | 2/82 | 2/82 | 0/82 | 2/82 | 9.0 | -1.454 | space:78, newline:2, correct_prefix:2 |  :78,  ?\n\n:2,  v:2 |
| no_explain | 82 | 3/82 | 3/82 | 0/82 | 53/82 | 6.4 | -1.982 | newline:53, space:26, correct_prefix:3 |  ?\n\n:53,  :26,  v:3 |
| no_qmark_short | 82 | 8/82 | 8/82 | 0/82 | 6/82 | 5.5 | -1.041 | space:68, correct_prefix:8, newline:6 |  :68,  v:8,  ?\n\n:6 |
| value_label | 82 | 3/82 | 3/82 | 0/82 | 79/82 | 2.8 | -1.648 | newline:79, correct_prefix:3 |  ?\n\n:79,  v:3 |
| direct_value_label | 82 | 0/82 | 0/82 | 0/82 | 82/82 | 48.1 | -8.406 | newline:82 |  ?\n\n:82 |

### repair_subject / non_target

| variant | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 174 | 38/174 | 36/174 | 3/174 | 118/174 | 7.3 | -1.282 | newline:118, correct_prefix:38, word:13, explanation:5 |  ?\n\n:118,  v:38,  o:11,  yes:5,  c:2 |
| no_qmark | 174 | 99/174 | 95/174 | 4/174 | 34/174 | 1.9 | 0.713 | correct_prefix:99, word:38, newline:34, space:3 |  v:99,  o:38,  ?\n\n:34,  :3 |
| period | 174 | 63/174 | 62/174 | 2/174 | 20/174 | 2.4 | 0.718 | word:88, correct_prefix:63, newline:20, space:3 |  o:86,  v:63,  ?\n\n:20,  :3,  c:2 |
| inline_answer | 174 | 171/174 | 159/174 | 8/174 | 0/174 | 1.0 | 2.362 | correct_prefix:171, space:3 |  v:171,  :3 |
| short_only | 174 | 2/174 | 2/174 | 0/174 | 4/174 | 5.8 | -0.992 | space:168, newline:4, correct_prefix:2 |  :168,  ?\n\n:4,  v:2 |
| no_explain | 174 | 3/174 | 3/174 | 0/174 | 103/174 | 4.9 | -1.659 | newline:103, space:68, correct_prefix:3 |  ?\n\n:103,  :68,  v:3 |
| no_qmark_short | 174 | 7/174 | 7/174 | 0/174 | 8/174 | 4.1 | -0.688 | space:159, newline:8, correct_prefix:7 |  :159,  ?\n\n:8,  v:7 |
| value_label | 174 | 9/174 | 10/174 | 0/174 | 162/174 | 2.7 | -1.452 | newline:162, correct_prefix:9, space:3 |  ?\n\n:162,  v:9,  :3 |
| direct_value_label | 174 | 0/174 | 0/174 | 0/174 | 174/174 | 46.8 | -8.138 | newline:174 |  ?\n\n:174 |

### Examples

- sample=0 split=target mode=base_subject__original top0=' ?\\n\\n'/newline rank=22 exact=False text=' ?\n\nTo solve' ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: yes[explanation], 5: c[word]
- sample=0 split=target mode=base_subject__no_qmark top0=' ?\\n\\n'/newline rank=4 exact=False text=' ?\n\nI think' ladder=1: ?\n\n[newline], 2: c[word], 3: belongs[word], 4: v[correct_prefix], 5: r[word]
- sample=0 split=target mode=base_subject__period top0=' ?\\n\\n'/newline rank=4 exact=False text=' ?\n\nI think' ladder=1: ?\n\n[newline], 2: r[word], 3: c[word], 4: v[correct_prefix], 5: belongs[word]
- sample=0 split=target mode=base_subject__inline_answer top0=' ?\\n\\n'/newline rank=2 exact=False text=' ?\n\nI think' ladder=1: ?\n\n[newline], 2: v[correct_prefix], 3: [space], 4: c[word], 5: ?\n[newline]
- sample=0 split=target mode=base_subject__short_only top0=' ?\\n\\n'/newline rank=17 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: [space], 3: \n\n[newline], 4: ?\n[newline], 5: The[explanation]
- sample=0 split=target mode=base_subject__no_explain top0=' ?\\n\\n'/newline rank=18 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: ?\n[newline], 3: c[word], 4: \n\n[newline], 5: [space]
- sample=0 split=target mode=base_subject__no_qmark_short top0=' ?\\n\\n'/newline rank=13 exact=False text=' ?\n\nTo solve' ladder=1: ?\n\n[newline], 2: ?\n[newline], 3: [space], 4: c[word], 5: \n\n[newline]
- sample=0 split=target mode=base_subject__value_label top0=' ?\\n\\n'/newline rank=6 exact=False text=' ?\n\nI think' ladder=1: ?\n\n[newline], 2: ?\n[newline], 3: [space], 4: ?[punctuation], 5: r[word]
- sample=0 split=target mode=base_subject__direct_value_label top0=' ?\\n\\n'/newline rank=42 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: ?\n[newline], 3: ?[punctuation], 4: \n\n[newline], 5: [space]
- sample=0 split=target mode=repair_subject__original top0=' ?\\n\\n'/newline rank=4 exact=False text=' ?\n\nTo solve' ladder=1: ?\n\n[newline], 2: ?\n[newline], 3: [space], 4: v[correct_prefix], 5: c[word]
- sample=0 split=target mode=repair_subject__no_qmark top0=' ?\\n\\n'/newline rank=2 exact=False text=' ?\n\nI think' ladder=1: ?\n\n[newline], 2: v[correct_prefix], 3: [space], 4: o[word], 5: ?\n[newline]
- sample=0 split=target mode=repair_subject__period top0=' ?\\n\\n'/newline rank=2 exact=False text=' ?\n\nI think' ladder=1: ?\n\n[newline], 2: v[correct_prefix], 3: o[word], 4: [space], 5: ?\n[newline]
- sample=0 split=target mode=repair_subject__inline_answer top0=' v'/correct_prefix rank=1 exact=True text=' v22' ladder=1: v[correct_prefix], 2: ?\n\n[newline], 3: [space], 4: c[word], 5: o[word]
- sample=0 split=target mode=repair_subject__short_only top0=' ?\\n\\n'/newline rank=6 exact=False text=' ?\n\nOkay,' ladder=1: ?\n\n[newline], 2: [space], 3: ?\n[newline], 4: The[explanation], 5: \n\n[newline]
