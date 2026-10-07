# -*- coding: utf-8 -*-
"""Verify deepseek memo artifacts exist on disk + spot-check key numbers."""
import os, re, json, hashlib

BASE = r'D:\AI2050\Ai2050-OpenOne'
TMP = os.path.join(BASE, 'tests', 'gpt5_temp')
OUT = os.path.join(BASE, 'tests', 'gpt5_temp', 'memo_review_20261001', 'artifact_verify_report.txt')

lines = []

def sha8(p):
    try:
        with open(p, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:8]
    except Exception as e:
        return 'ERR:%s' % e

# 1) artifact existence
expected = [
    'N1_design_seal.json', 'N2h1_design_seal.json', 'N3_design_seal.json',
    'e1_embed_probe_20260930.py', 'e1_embed_probe_report.txt',
    'e2_context_conditioned_probe_20261001.py', 'e2_report.txt',
    'n1_v2_main_axis_scan.py', 'n1b_ontology_readout.py', 'n1c_ontology_cloze.py',
    'n2_reconstruction_source.py', 'n2b_robustness.py', 'n2c_slot_commitment.py',
    'n2d_attn_write.py', 'n2e_template_robustness.py', 'n2f_topk_diag.py', 'n2g_critical_layers.py',
    'n2h1_permutation_ablation.py', 'n2h1b_subspace_align.py', 'n2h1c_cross_material.py',
    'n3_subspace_generality.py',
    'n2_report_qwen3-4b.txt',
    'n2h1_report_qwen3-4b.txt',
    'n2h1b_report_qwen2.5-3b-instruct.txt', 'n2h1b_report_glm4-9b-chat-hf.txt',
    'n2h1c_report_qwen3-4b.txt', 'n2h1c_report_glm4-9b-chat-hf.txt',
    'n3_report_qwen3-4b.txt', 'n3_report_qwen2.5-3b-instruct.txt', 'n3_report_glm4-9b-chat-hf.txt',
    'e3_report.txt', 'e3b_report.txt',
]
lines.append('=== 1. artifact existence (tests/gpt5_temp/) ===')
missing = []
for name in expected:
    p = os.path.join(TMP, name)
    ok = os.path.exists(p)
    if not ok:
        missing.append(name)
    lines.append('%-46s %s %s' % (name, 'OK ' if ok else 'MISS', sha8(p) if ok else ''))
lines.append('missing_count=%d %s' % (len(missing), missing))

# also check other-model n1 reports count
n1v2 = [f for f in os.listdir(TMP) if f.startswith('n1v2_report_')]
n1b = [f for f in os.listdir(TMP) if f.startswith('n1b_report_')]
n1c = [f for f in os.listdir(TMP) if f.startswith('n1c_report_')]
n2b = [f for f in os.listdir(TMP) if f.startswith('n2b_report_')]
n2c = [f for f in os.listdir(TMP) if f.startswith('n2c_report_')]
n2d = [f for f in os.listdir(TMP) if f.startswith('n2d_report_')]
n2e = [f for f in os.listdir(TMP) if f.startswith('n2e_report_')]
n2f = [f for f in os.listdir(TMP) if f.startswith('n2f_report_')]
n2g = [f for f in os.listdir(TMP) if f.startswith('n2g_report_')]
lines.append('')
lines.append('report file counts: n1v2=%d n1b=%d n1c=%d n2b=%d n2c=%d n2d=%d n2e=%d n2f=%d n2g=%d'
             % (len(n1v2), len(n1b), len(n1c), len(n2b), len(n2c), len(n2d), len(n2e), len(n2f), len(n2g)))
lines.append('n1v2: %s' % n1v2)
lines.append('n1b: %s' % n1b)
lines.append('n1c: %s' % n1c)

# 2) design seals content
lines.append('')
lines.append('=== 2. design seals (frozen criteria) ===')
for sn in ['N1_design_seal.json', 'N2h1_design_seal.json', 'N3_design_seal.json']:
    p = os.path.join(TMP, sn)
    if not os.path.exists(p):
        lines.append('%s: MISSING' % sn)
        continue
    with open(p, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()
    lines.append('--- %s (%d bytes, sha8 %s) ---' % (sn, len(raw), sha8(p)))
    try:
        d = json.loads(raw)
        def walk(obj, prefix=''):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        walk(v, prefix + k + '.')
                    else:
                        s = str(v)
                        lines.append('  %s%s = %s' % (prefix, k, s[:120]))
            elif isinstance(obj, list):
                for i, v in enumerate(obj[:12]):
                    if isinstance(v, (dict, list)):
                        walk(v, prefix + ('[%d].' % i))
                    else:
                        lines.append('  %s[%d] = %s' % (prefix, i, str(v)[:120]))
        walk(d)
    except Exception as e:
        lines.append('  JSON parse fail: %s' % e)
        lines.append('  raw head: %s' % raw[:500].replace('\n', ' | '))

# 3) spot-check key numbers in reports
lines.append('')
lines.append('=== 3. key-number spot checks ===')
def grep_file(path, patterns, label):
    lines.append('--- %s (%s) ---' % (label, os.path.basename(path)))
    if not os.path.exists(path):
        lines.append('  MISSING')
        return
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        txt = f.read()
    for pat in patterns:
        hits = []
        for ln in txt.splitlines():
            if re.search(pat, ln):
                hits.append(ln.strip()[:160])
        lines.append('  /%s/ -> %d hit(s)' % (pat, len(hits)))
        for h in hits[:4]:
            lines.append('    | ' + h)

grep_file(os.path.join(TMP, 'n2_report_qwen3-4b.txt'),
          [r'share', r'0\.030', r'6\.01', r'5\.15'],
          'N2: distributed verdict / L6 6.01 / I=5.15')

grep_file(os.path.join(TMP, 'n2h1_report_qwen3-4b.txt'),
          [r'10\.6\d', r'B_cat', r'C_resid', r'D_rand', r'E_same'],
          'N2h1 qwen3-4b: B_cat ~10.65 etc')

grep_file(os.path.join(TMP, 'n2h1b_report_glm4-9b-chat-hf.txt'),
          [r'9\.6\d', r'9\.7\d', r'B/A'],
          'N2h1b glm4: ~9.6-9.9')

grep_file(os.path.join(TMP, 'n3_report_qwen3-4b.txt'),
          [r'0\.925', r'0\.829', r'L27', r'color', r'ratio'],
          'N3 qwen3-4b: color B/A 0.925 / loo 0.829')

grep_file(os.path.join(TMP, 'n3_report_glm4-9b-chat-hf.txt'),
          [r'0\.894', r'0\.922', r'L24'],
          'N3 glm4: color values')

grep_file(os.path.join(TMP, 'n3_report_qwen2.5-3b-instruct.txt'),
          [r'1\.040', r'0\.852', r'L33'],
          'N3 qwen2.5-3b: color values')

grep_file(os.path.join(TMP, 'e2_report.txt'),
          [r'1\.0000', r'0\.1758', r'0\.715'],
          'E2: exact zeros/ones + R11')

grep_file(os.path.join(TMP, 'e1_embed_probe_report.txt'),
          [r'0\.0555', r'0\.0241', r'0\.1670'],
          'E1: apple-food / apple-plant / q95')

with open(OUT, 'w', encoding='utf-8') as w:
    w.write('\n'.join(lines))
print('OK ->', OUT)
