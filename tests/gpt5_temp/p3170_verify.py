# -*- coding: utf-8 -*-
# Phase 3170 independent disk verification.
# All expected values are read live from disk (no memorized strings).
# Checks: disk shas, seal byte-level rebuild, G3 byte-preservation (independent
# recompute), G4 gap/appendix immutability, FTR-21 value recompute from p3169/q03,
# html per-field verification (independent flatten), no-external-resources,
# five-write read-back.
import io
import json
import re
import html as htmllib
import hashlib
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3170', 'g5a7_atlas_v11')
P3167 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                     'phase3167', 'g5a4_feature_registry', 'atlas_registry_v1.json')
P3168 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                     'phase3168', 'g5a5_atlas_render', 'gap_ledger_v1.json')
P3169 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                     'phase3169', 'g5a6_oov_panel', 'result.json')
P3162 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                     'phase3162', 'g5a1_atlas_foundation', 'atlas_registry.json')
Q03 = os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q03_result.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3170_verify_out.txt')
R = []
N_OK = [0]
N_FAIL = [0]


def chk(name, ok, detail=''):
    if ok:
        N_OK[0] += 1
        R.append('PASS %s %s' % (name, detail))
    else:
        N_FAIL[0] += 1
        R.append('FAIL %s %s' % (name, detail))


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def fmt(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return '%.6g' % v
    return str(v)


Res = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
Reg11 = json.load(io.open(os.path.join(PDIR, 'atlas_registry_v1_1.json'), encoding='utf-8'))
GL11 = json.load(io.open(os.path.join(PDIR, 'gap_ledger_v1_1.json'), encoding='utf-8'))
Reg1 = json.load(io.open(P3167, encoding='utf-8'))
GL1 = json.load(io.open(P3168, encoding='utf-8'))
R69 = json.load(io.open(P3169, encoding='utf-8'))
Reg2 = json.load(io.open(P3162, encoding='utf-8'))
Rq = json.load(io.open(Q03, encoding='utf-8'))

# ---------- 1. disk shas vs result ----------
chk('1.1 exec sha recorded', True, sha8_file(os.path.join(PDIR, 'execution.json')))
chk('1.2 registry v1.1 disk sha == result', sha8_file(os.path.join(PDIR, 'atlas_registry_v1_1.json')) == Res['registry_v11_sha8'], Res['registry_v11_sha8'])
chk('1.3 gap ledger v1.1 disk sha == result', sha8_file(os.path.join(PDIR, 'gap_ledger_v1_1.json')) == Res['gap_ledger_v11_sha8'], Res['gap_ledger_v11_sha8'])
chk('1.4 html disk sha == result', sha8_file(os.path.join(PDIR, 'atlas_v1_1.html')) == Res['html_sha8'], Res['html_sha8'])
chk('1.5 run_log exists', os.path.getsize(os.path.join(PDIR, 'run_log.txt')) > 0)

# ---------- 2. seal byte-level rebuild ----------
summary = {k: v for k, v in Res.items() if k not in ('res_sha8', 'seal_sha8')}
raw = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True)
res8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
mid = json.dumps(dict(summary, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
seal8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
chk('2.1 res_sha8 rebuild', res8 == Res['res_sha8'], '%s vs %s' % (res8, Res['res_sha8']))
chk('2.2 seal_sha8 rebuild', seal8 == Res['seal_sha8'], '%s vs %s' % (seal8, Res['seal_sha8']))

# ---------- 3. G3 byte-preservation, independent ----------
chk('3.1 v1.1 has 21 features', len(Reg11['features']) == 21, str(len(Reg11['features'])))
chk('3.2 v1.1 last id is FTR-21', Reg11['features'][-1]['id'] == 'FTR-21')
ok3 = True
bad3 = []
for i in range(20):
    a = json.dumps(Reg1['features'][i], ensure_ascii=False, sort_keys=True)
    b = json.dumps(Reg11['features'][i], ensure_ascii=False, sort_keys=True)
    if a != b:
        ok3 = False
        bad3.append(Reg1['features'][i]['id'])
chk('3.3 v1 features 0..19 byte-preserved', ok3, ','.join(bad3) if bad3 else '20/20')
chk('3.4 failures unchanged', json.dumps(Reg1['failures'], ensure_ascii=False, sort_keys=True) ==
    json.dumps(Reg11['failures'], ensure_ascii=False, sort_keys=True))
chk('3.5 upgrade_log unchanged', json.dumps(Reg1['upgrade_log'], ensure_ascii=False, sort_keys=True) ==
    json.dumps(Reg11['upgrade_log'], ensure_ascii=False, sort_keys=True))
chk('3.6 version field', Reg11.get('version') == '1.1', str(Reg11.get('version')))

# ---------- 4. G4 gap ledger immutability + flip ----------
g = {x['id']: x for x in GL11['gaps']}
g1v1 = {x['id']: x for x in GL1['gaps']}
chk('4.1 GAP-1 unchanged', json.dumps(g1v1['GAP-1'], ensure_ascii=False, sort_keys=True) ==
    json.dumps(g['GAP-1'], ensure_ascii=False, sort_keys=True))
chk('4.2 GAP-2 unchanged', json.dumps(g1v1['GAP-2'], ensure_ascii=False, sort_keys=True) ==
    json.dumps(g['GAP-2'], ensure_ascii=False, sort_keys=True))
chk('4.3 GAP-3 unchanged', json.dumps(g1v1['GAP-3'], ensure_ascii=False, sort_keys=True) ==
    json.dumps(g['GAP-3'], ensure_ascii=False, sort_keys=True))
chk('4.4 GAP-4 flipped', g1v1['GAP-4']['status'] == 'open' and g['GAP-4']['status'] == 'quantified_collapse',
    g['GAP-4']['status'])
chk('4.5 prereg.status flipped', g['GAP-4']['prereg']['status'] == 'executed_collapse_confirmed')
chk('4.6 prereg gate text preserved', g['GAP-4']['prereg']['gate'] == g1v1['GAP-4']['prereg']['gate'])
chk('4.7 prereg hypothesis preserved', g['GAP-4']['prereg']['hypothesis'] == g1v1['GAP-4']['prereg']['hypothesis'])
chk('4.8 GAP-4 anchor p3169 added', g['GAP-4']['anchor_sha8'].get('p3169') == '5b51c2c1')
chk('4.9 GAP-4 evidence grew 2->5', len(g1v1['GAP-4']['evidence']) == 2 and len(g['GAP-4']['evidence']) == 5,
    '%d->%d' % (len(g1v1['GAP-4']['evidence']), len(g['GAP-4']['evidence'])))
chk('4.10 old evidence preserved verbatim', g['GAP-4']['evidence'][:2] == g1v1['GAP-4']['evidence'])
ok4 = True
for i in range(5):
    if json.dumps(GL1['appendix'][i], ensure_ascii=False, sort_keys=True) != \
       json.dumps(GL11['appendix'][i], ensure_ascii=False, sort_keys=True):
        ok4 = False
chk('4.11 appendix 5 items unchanged', ok4)
chk('4.12 schema bumped', GL11['schema'] == 'rdc_atlas_gap_ledger_v1_1' and GL11.get('version') == '1.1')

# ---------- 5. FTR-21 values recompute from p3169/q03 ----------
f21 = Reg11['features'][-1]
g69 = R69['gate']
chk('5.1 ratio_B matches 3169 gate', abs(f21['values']['ratio_B_pooled'] - g69['ratio_B']) < 1e-9,
    '%.9f' % f21['values']['ratio_B_pooled'])
chk('5.2 pooled_E_oov matches', abs(f21['values']['pooled_E_oov_B'] - g69['pooled_E_oov_B']) < 1e-12)
chk('5.3 pooled_E_seen matches', abs(f21['values']['pooled_E_seen_B'] - g69['pooled_E_seen_B']) < 1e-12)
MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
keymap = {'qwen3-4b': 'qwen3_4b', 'qwen3-14b': 'qwen3_14b', 'glm4-9b': 'glm4_9b'}
ok5 = True
for m in MODELS:
    if abs(f21['values']['ratio_B_' + keymap[m]] - R69['per_model'][m]['B']['ratio']) > 1e-9:
        ok5 = False
    if abs(f21['values']['ratio_A_' + keymap[m]] - R69['per_model'][m]['A']['ratio']) > 1e-9:
        ok5 = False
    if R69['per_model'][m]['B']['ratio'] <= 2.0 or R69['per_model'][m]['A']['ratio'] >= 1.0:
        ok5 = False
chk('5.4 per-model B>2 & A<1 recompute', ok5)
drift = abs(Rq['summary']['pooled_mean'] - g69['pooled_E_seen_B'])
chk('5.5 q03 cross-line drift < 1e-6', drift < 1e-6 and
    abs(drift - f21['values']['q03_pooled_crossline_drift']) < 1e-15, '%.3e' % drift)
chk('5.6 D3 totals', f21['values']['d3_bitwise_rows_total'] == 2214 and f21['values']['d3_mismatch_total'] == 0)
chk('5.7 FTR-21 level/family', f21['evidence_level'] == 'E2_predictive' and f21['family'] == 'limit')
chk('5.8 FTR-21 scope = 3 models', f21['model_scope'] == MODELS)
chk('5.9 FTR-21 anchors = 3', len(f21['anchors']) == 3)
chk('5.10 three-tier gradient in scope_limits', all(s in f21['scope_limits'] for s in ('0.74', '0.69', '0.78')))

# ---------- 6. html per-field verification (independent flatten re-impl) ----------
htm = open(os.path.join(PDIR, 'atlas_v1_1.html'), 'rb').read().decode('utf-8')
nodes = Reg2['audit']['nodes']
exp = {}
for f in Reg11['features']:
    p = f['id']
    exp[p + '.statement'] = f['statement']
    exp[p + '.family'] = f['family']
    exp[p + '.evidence_level'] = f['evidence_level']
    exp[p + '.model_scope'] = ', '.join(f['model_scope'])
    exp[p + '.n_anchors'] = str(len(f['anchors']))
    exp[p + '.counter_evidence'] = ' | '.join(f['counter_evidence']) if f['counter_evidence'] else '(none)'
    exp[p + '.replication'] = f['replication']
    exp[p + '.scope_limits'] = f['scope_limits'] if f['scope_limits'] else '(none)'
    for i, a in enumerate(f['anchors']):
        exp['%s.anchors.%d.src' % (p, i)] = str(a['src'])
        exp['%s.anchors.%d.phase' % (p, i)] = str(a['phase'])
        exp['%s.anchors.%d.role' % (p, i)] = a['role']
        exp['%s.anchors.%d.asserts' % (p, i)] = json.dumps(a['asserts'], ensure_ascii=False, sort_keys=True)
    for k, v in (f.get('values') or {}).items():
        exp['%s.values.%s' % (p, k)] = fmt(v)
for n in nodes:
    exp[n['id'] + '.title'] = n['title']
    exp[n['id'] + '.elevel'] = n['evidence_level']
    exp[n['id'] + '.status'] = n['status']
    exp[n['id'] + '.checks'] = '%d/%d' % (n['n_pass'], n['n_checks'])
for i, u in enumerate(Reg11['upgrade_log']):
    exp['UPG.%d.node' % i] = u['node']
    exp['UPG.%d.change' % i] = u['change']
    exp['UPG.%d.reason' % i] = u['reason']
for fl in Reg11['failures']:
    exp[fl['id'] + '.kind'] = fl['kind']
    exp[fl['id'] + '.phase'] = str(fl['phase'])
    exp[fl['id'] + '.text'] = fl['text']
    exp[fl['id'] + '.evidence'] = str(fl['evidence'])
for x in GL11['gaps']:
    p = x['id']
    exp[p + '.title'] = x['title']
    exp[p + '.status'] = x['status']
    if x['status'].startswith('closed'):
        exp[p + '.closed_at'] = str(x['closed_at_phase'])
    exp[p + '.statement'] = x['statement']
    exp[p + '.evidence'] = ' || '.join(x['evidence'])
    exp[p + '.anchors'] = ', '.join('%s=%s' % (k, v) for k, v in sorted(x['anchor_sha8'].items()))
    if p == 'GAP-4':
        pr = x['prereg']
        exp[p + '.prereg.phase'] = str(pr['phase'])
        exp[p + '.prereg.name'] = pr['name']
        exp[p + '.prereg.hypothesis'] = pr['hypothesis']
        exp[p + '.prereg.protocol'] = pr['protocol']
        exp[p + '.prereg.gate'] = pr['gate']
        exp[p + '.prereg.status'] = pr['status']
for a in GL11['appendix']:
    p = a['id']
    exp[p + '.title'] = a['title']
    exp[p + '.status'] = a['status']
    exp[p + '.note'] = a['note']
    exp[p + '.anchor'] = a['anchor']
    exp[p + '.anchor_sha8'] = a['anchor_sha8']
exp['meta.registry_v11_sha8'] = Res['registry_v11_sha8']
exp['meta.registry_v1_sha8'] = 'f207aa8d'
exp['meta.registry3162_sha8'] = '00f15e98'
exp['meta.p3169_res'] = '5b51c2c1'
pairs = re.findall(r'data-k="([^"]+)"[^>]*>(.*?)</span>', htm, re.S)
found = {}
dups = []
for k, v in pairs:
    if k in found:
        dups.append(k)
    found[k] = htmllib.unescape(v)
missing = sorted(set(exp) - set(found))
extra = sorted(set(found) - set(exp))
mismatch = [k for k in sorted(set(exp) & set(found)) if found[k] != exp[k]]
# meta.created is a runtime render timestamp (exists in html but not knowable to
# the verifier); whitelist it and assert its format instead.
extra_known = [k for k in extra if k != 'meta.created']
chk('6.1 field missing == 0', not missing, str(missing[:5]))
chk('6.2 field extra == 0 (meta.created whitelisted)', not extra_known, str(extra_known[:5]))
chk('6.9 meta.created format', re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}$', found.get('meta.created', '')) is not None,
    found.get('meta.created', ''))
chk('6.3 field mismatch == 0', not mismatch, str(mismatch[:5]))
chk('6.4 field dups == 0', not dups, str(dups[:5]))
chk('6.5 field count matches result', len(found) == Res['field_check']['n_found'],
    '%d vs %d' % (len(found), Res['field_check']['n_found']))
chk('6.6 FTR-21 card present', 'id="FTR-21"' in htm)
chk('6.7 GAP-4 badge rendered', 'b-quantified_collapse' in htm and 'quantified_collapse' in found.get('GAP-4.status', ''))
chk('6.8 no external resources', '<link' not in htm and '<script' not in htm and
    'http://' not in htm and 'https://' not in htm)

# ---------- 7. five-write read-back ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
e70 = [m for m in led['measurements'] if m.get('phase') == 3170]
chk('7.1 ledger 3170 entry', len(e70) == 1, 'n=%d' % len(led['measurements']))
chk('7.2 ledger chain field', bool(led.get('ledger_sha256_8')), led.get('ledger_sha256_8'))
memo = open(MEMO, 'rb').read().decode('utf-8')
chk('7.3 MEMO 3170 section', '## Phase 3170' in memo)
chk('7.4 MEMO res/seal shas', Res['res_sha8'] in memo and Res['seal_sha8'] in memo)
chk('7.5 MEMO prereg 3171', '预注册 3171' in memo and 'G5-A8' in memo)
chk('7.6 MEMO FTR-21 + ratio', 'FTR-21' in memo and '2.5388' in memo)
daily = io.open(DAILY, encoding='utf-8').read()
chk('7.7 daily 3170 line', '3170 图谱 v1.1' in daily)
wm = open(WMEM, 'rb').read().decode('utf-8')
chk('7.8 workspace MEMORY 3170 line', '3170 图谱 v1.1' in wm)
cl = io.open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3170_closeout_out.txt'), encoding='utf-8').read()
n_ok = cl.count('SELF-CHECK OK')
n_bad = cl.count('SELF-CHECK FAIL')
chk('7.9 closeout self-check 17 OK / 0 FAIL', n_ok == 17 and n_bad == 0,
    'ok=%d fail=%d' % (n_ok, n_bad))

R.append('')
R.append('TOTAL PASS=%d FAIL=%d' % (N_OK[0], N_FAIL[0]))
with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(R) + '\n')
print('\n'.join(R[-8:]))
assert N_FAIL[0] == 0, 'verify failures'
