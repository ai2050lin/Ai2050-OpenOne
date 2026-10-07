# -*- coding: utf-8 -*-
"""Phase 3016 patch1: lens dtype fix + note run1."""
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3016_omega_p2j_amplification_trace_qwen.py')
t = io.open(P, encoding='utf-8').read()

old1 = ("            hb = model.model.norm(torch.tensor(\n"
        "                res_b, device='cuda'))\n"
        "            he = model.model.norm(torch.tensor(\n"
        "                res_e, device='cuda'))")
new1 = ("            hb = model.model.norm(torch.tensor(\n"
        "                res_b, device='cuda')) \\\n"
        "                .to(model.lm_head.weight.dtype)\n"
        "            he = model.model.norm(torch.tensor(\n"
        "                res_e, device='cuda')) \\\n"
        "                .to(model.lm_head.weight.dtype)")
assert old1 in t, 'anchor1 miss'
t = t.replace(old1, new1, 1)

old2 = "        'correction_note':\n            'none; run1 authoritative',"
new2 = ("        'correction_note':\n"
        "            'run1: crashed at the logit-lens '\n"
        "            'readout - model.norm returns fp32 '\n"
        "            'while lm_head weights are bf16 '\n"
        "            '(F.linear dtype mismatch '\n"
        "            'double != BFloat16); fix: cast the '\n"
        "            'normed residuals to '\n"
        "            'lm_head.weight.dtype; crash was at '\n"
        "            'the FIRST lens call, after T2a '\n"
        "            'positions were computed but before '\n"
        "            'any verdict - no data validity '\n"
        "            'impact, design unchanged; run2: '\n"
        "            'authoritative',")
assert old2 in t, 'anchor2 miss'
t = t.replace(old2, new2, 1)
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
chk = {
    'cast_in': 'lm_head.weight.dtype' in t2,
    'note_in': 'run2: ' in t2
    and 'authoritative' in t2,
    'old_gone': 'none; run1 authoritative' not in t2,
}
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_p16.txt', 'w',
        encoding='utf-8').write(str(chk))
print('patch ok')
