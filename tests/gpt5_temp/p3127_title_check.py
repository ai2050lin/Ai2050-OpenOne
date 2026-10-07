import io, json

t = io.open(
    r'D:\AI2050\Ai2050-OpenOne'
    r'\research\gpt5\docs'
    r'\AGI_GPT5_MEMO.md',
    encoding='utf-8').read()
ls = t.splitlines()
idx = max(i for i, l in enumerate(ls)
          if l.startswith('## Phase 3127'))
sec = ls[idx:idx + 26]
V = ('a1_short_range_intrinsic',
     'regen_replay_bit_exact', 'lag46',
     'port_depth_divergent',
     'qwen_write_not', 'coverage_full')
chk = {
    'head': sec[0],
    'n_lines': len(sec),
    'has_xingzhi': any('\u6027\u8d28' in l
                       for l in sec),
    'has_findings': any(
        '\u4e09\u5927\u53d1\u73b0' in l
        for l in sec),
    'has_nums': any('\u5173\u952e\u6570\u503c'
                    in l for l in sec),
    'has_flaws': any('\u786c\u4f24' in l
                     for l in sec),
    'has_puzzle': any('\u673a\u5236\u62fc\u56fe'
                      in l for l in sec),
    'has_prereg': any(
        '3128 \u9884\u6ce8\u518c' in l
        for l in sec),
    'has_artifacts': any(
        'p125_readout.npz' in l
        for l in sec),
    'verdict_bits': sum(
        any(v in l for l in sec) for v in V),
}
io.open(r'D:\AI2050\Ai2050-OpenOne'
        r'\gpt5_temp\p3127_title_check.txt',
        'w', encoding='utf-8').write(
    json.dumps(chk, ensure_ascii=False,
               indent=1))
print('CHECK_DONE')
