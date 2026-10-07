# -*- coding: utf-8 -*-
"""Phase 3015 patch1: correction_note run1 registration
(landed as a phantom edit - redo on real disk)."""
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3015_omega_p2i_k_consumer_heads_qwen.py')
t = io.open(P, encoding='utf-8').read()
old = "            'none; run1 authoritative',"
new = ("            'run1: ran to completion but verdict '\n"
       "            'VOID by protocol (gates fail) - '\n"
       "            'NH_EXPECT was frozen at 32 but the '\n"
       "            'live L3 key cache is GQA with '\n"
       "            'num_kv_heads=8 (32 query heads share '\n"
       "            '8 KV heads), so nh_ok=False, T2b '\n"
       "            'skipped, and T2d crashed on a '\n"
       "            'broadcast (32-slot query-head '\n"
       "            'attention into 8-slot arrays, caught '\n"
       "            'as att_ok=False); fix: NH_EXPECT=8 '\n"
       "            'with explicit GQA group mapping (kv '\n"
       "            'head g serves query heads 4g..4g+3), '\n"
       "            'T2d recast at query-head granularity '\n"
       "            'with group aggregation; no '\n"
       "            'measurement-design change (per-KV-'\n"
       "            'head erasure was already the de facto '\n"
       "            'unit in run1, its T2a shares were '\n"
       "            'computed but voided by the gate); '\n"
       "            'run2: full pass verdict '\n"
       "            'k_consumer_mixed_qwen but its '\n"
       "            'correction_note edit was a phantom '\n"
       "            '(old text persisted on disk) - '\n"
       "            're-registered here; run3: '\n"
       "            'authoritative',")
assert old in t, 'anchor miss'
t = t.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
chk = {
    'note_in': 'run3: ' in t2 and 'authoritative' in t2,
    'old_gone': 'none; run1 authoritative' not in t2,
    'gqa_in': 'GQA' in t2,
    'nh8_in': 'NH_EXPECT = 8' in t2,
    'attgrp_in': "'att_grp'" in t2,
}
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_p15.txt', 'w',
        encoding='utf-8').write(str(chk))
print('patch ok')
