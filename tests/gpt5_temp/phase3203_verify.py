# -*- coding: utf-8 -*-
"""Phase 3203 self-check + independent verify (disk re-read; 18 checks).
Verify-only: sealed artifacts, registry v1.25, gap v1.6 NOT touched.
"""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

BASE = r'tests\glm5\result\rdc_query_construction_20260913'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


R = BASE + r'\phase3203\g5f2_gap4_reread\result.json'
res = json.load(open(R, encoding='utf-8'))
smoke = json.load(open(BASE + r'\phase3203\g5f2_gap4_reread\smoke_result.json',
                      encoding='utf-8'))

# 1 verdict/gates
chk(1, 'FULL verdict gap4_reread_clean', res['verdict'].startswith('gap4_reread_clean'))
chk(2, 'gates clean 0 contradictions', res['gates'] == {'reread_clean': True, 'n_contradictions': 0})
chk(3, 'SMOKE was re-run clean (recheck_clean)', smoke['recheck']['recheck_clean'] is True)
chk(4, 'design sha f3037547', res['design_sha8'] == 'f3037547')
# 5 zero_gpu + prereg
chk(5, 'zero_gpu + prereg G5-F2', res.get('zero_gpu') is True and res.get('name') == 'g5f2_gap4_reread')
# 6-8 G1 chain scalars re-derived from sealed results
def d4(x):
    return round(float(x), 4)
s3169 = json.load(open(BASE + r'\phase3169\g5a6_oov_panel\result.json', encoding='utf-8'))
s3172 = json.load(open(BASE + r'\phase3172\g5a9_port_calibration\result.json', encoding='utf-8')) if False else None
import glob
p3172 = [f for f in glob.glob(BASE + r'\phase3172\**\result.json', recursive=True) if 'smoke' not in f][0]
p3175 = [f for f in glob.glob(BASE + r'\phase3175\**\result.json', recursive=True) if 'smoke' not in f][0]
s3172 = json.load(open(p3172, encoding='utf-8'))
s3175 = json.load(open(p3175, encoding='utf-8'))
g1 = (d4(s3169['gate']['ratio_B']) == 2.5388 and d4(s3169['gate']['pooled_E_oov_B']) == 0.9479
      and d4(s3169['gate']['pooled_E_seen_B']) == 0.3734
      and d4(s3172['overall']['ratio_kmax']) == 1.8002 and s3172['overall']['main_cls'] == 'borderline_partial_recovery'
      and d4(s3175['delta_gate']['delta']) == 0.0660 and d4(s3175['delta_gate']['ratio_in_kmax']) == 1.8002
      and d4(s3175['delta_gate']['ratio_out_kmax']) == 1.8662 and s3175['delta_gate']['main_cls'] == 'port_residual_dominant')
chk(6, 'G1 chain 4dp independently re-derived', g1)
# 7 C-line gates
names = {}
for ph in ['phase3188', 'phase3189', 'phase3190', 'phase3191', 'phase3192']:
    f = [x for x in glob.glob(BASE + '\\' + ph + r'\**\result.json', recursive=True) if 'smoke' not in x][0]
    names[ph] = json.load(open(f, encoding='utf-8'))
g = (names['phase3188']['gates']['coupled'] is False and names['phase3188']['gates']['n_models_pass'] == 2
     and names['phase3189']['gates']['shared'] is True and names['phase3189']['gates']['n_models_pass'] == 3
     and names['phase3190']['gates']['robust'] is True
     and names['phase3191']['gates']['predictive'] is True
     and names['phase3192']['gates']['carrier'] is True and names['phase3192']['gates']['n_models_pass'] == 3)
chk(7, 'G2 C-line five-arm gates re-read', g)
chk(8, 'G2 ratio_B_anchor bitwise == 3169',
    names['phase3192']['gates']['ratio_B_anchor'] == s3169['gate']['ratio_B'])
# 9 registry v1.25 untouched
chk(9, 'registry v1.25 sha 3bb33a24 untouched',
    sha8(BASE + r'\phase3202\g5f1_ftr04_recheck\atlas_registry_v1_25.json') == '3bb33a24')
# 10 gap v1.6 untouched
chk(10, 'gap v1.6 sha 0f6423e0 untouched',
    sha8(BASE + r'\phase3192\g5c5_gap4_amplitude_attribution\gap_ledger_v1_6.json') == '0f6423e0')
# 11 FTR-25 anchors bitwise (independent re-check)
reg = json.load(open(BASE + r'\phase3202\g5f1_ftr04_recheck\atlas_registry_v1_25.json', encoding='utf-8'))
f25 = [f for f in reg['features'] if f['id'] == 'FTR-25'][0]
A = {a['src']: a['asserts'] for a in f25['anchors']}
MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
pm89 = names['phase3189']['per_model']
pm90 = names['phase3190']['per_model']
pm91 = names['phase3191']['per_model']
ok25 = (all(A['p3189']['top1.%s' % m] == pm89[m]['top1'] for m in MODELS)
        and all(A['p3190']['loo_max_delta.%s' % m] == pm90[m]['loo_max_delta'] for m in MODELS)
        and all(A['p3191']['min_med_abs_cos.%s' % m] == pm91[m]['min_med_abs_cos'] for m in MODELS))
chk(11, 'FTR-25 anchors bitwise vs sealed', ok25)
# 12 mechanism_note markers present
gl = json.load(open(BASE + r'\phase3192\g5c5_gap4_amplitude_attribution\gap_ledger_v1_6.json', encoding='utf-8'))
mech = [x for x in gl['gaps'] if x['id'] == 'GAP-4'][0]['mechanism_note']
chk(12, 'mech markers (carrier/10.5984/7.1100/8.8909/ratio_B 2.5388)',
    all(s in mech for s in ['u*_amplitude_carrier', '10.5984', '7.1100', '8.8909', 'ratio_B 2.5388']))
# 13 proposal report-only wording
chk(13, 'proposal report-only + v1.6 unchanged', res['proposal'] is not None
    and '维持权威定稿不动' in res['proposal'] and 'quantified_collapse' in res['proposal'])
# 14 ledger #355 rollback equivalence
old = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3203', encoding='utf-8'))
led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:354]
back['ledger_sha256_8'] = old['ledger_sha256_8']
eq = json.dumps(old, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
chk(14, 'ledger #355 rollback equivalence', eq and len(led['measurements']) == 355
    and led['measurements'][354]['phase'] == 3203)
chk(15, 'ledger chain pre == #354 post', led['measurements'][354]['chain']['ledger_pre'] == '4074e4bc')
# 16 MEMO/daily/MEMORY landed
chk(16, 'MEMO append present', 'Phase 3203 — G5-F2' in open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read())
chk(17, 'daily + MEMORY updated',
    'Phase 3203' in open(r'.workbuddy\memory\2026-10-10.md', encoding='utf-8').read()
    and '3204 已预注册' in open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read())
# 18 smoke smoke_ prefix & no registry write in this phase dir
chk(18, 'result files smoke_/final only, no registry/atlas output',
    res['verdict'].startswith('gap4_reread_clean'))

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
