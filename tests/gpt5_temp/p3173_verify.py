# -*- coding: utf-8 -*-
# Phase 3173 independent verification. Re-implements (does not import) the
# flatten/seal logic; byte-level seal reconstruction; FTR-22 re-derivation
# from raw sealed sources; immutability of v1.1 features / v1.3 mechanism_note;
# html full re-extract vs independently rebuilt expected dict; five-write
# readback. Any FAIL line means a real inconsistency.
import io
import json
import os
import re
import html as htmlmod
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
S = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(S, 'phase3173', 'g5a10_atlas_v13')
V11DIR = os.path.join(S, 'phase3170', 'g5a7_atlas_v11')
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3173_verify_out.txt')
LINES = []
n_ok = 0


def chk(name, ok, detail=''):
    global n_ok
    if ok:
        n_ok += 1
        LINES.append('OK   %s %s' % (name, detail))
    else:
        LINES.append('FAIL %s %s' % (name, detail))


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def sha8_file(p):
    return sha8(open(p, 'rb').read())


def fmt(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return '%.6g' % v
    return str(v)


# ---------- load everything ----------
R = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
SM = json.load(io.open(os.path.join(PDIR, 'smoke_result.json'), encoding='utf-8'))
reg12 = json.load(io.open(os.path.join(PDIR, 'atlas_registry_v1_2.json'), encoding='utf-8'))
reg11 = json.load(io.open(os.path.join(V11DIR, 'atlas_registry_v1_1.json'), encoding='utf-8'))
gl14 = json.load(io.open(os.path.join(PDIR, 'gap_ledger_v1_4.json'), encoding='utf-8'))
gl13 = json.load(io.open(os.path.join(V11DIR, 'gap_ledger_v1_3.json'), encoding='utf-8'))
r72 = json.load(io.open(os.path.join(S, 'phase3172', 'g5a9_port_calibration', 'result.json'), encoding='utf-8'))
r71 = json.load(io.open(os.path.join(S, 'phase3171', 'g5a8_collapse_mechanism', 'result.json'), encoding='utf-8'))
r69 = json.load(io.open(os.path.join(S, 'phase3169', 'g5a6_oov_panel', 'result.json'), encoding='utf-8'))
rq = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q03_result.json'), encoding='utf-8'))
reg3162 = json.load(io.open(os.path.join(S, 'phase3162', 'g5a1_atlas_foundation', 'atlas_registry.json'), encoding='utf-8'))
html_text = io.open(os.path.join(PDIR, 'atlas_v1_3.html'), encoding='utf-8').read()

# ---------- 1. seal byte-level reconstruction ----------
core = {k: v for k, v in R.items() if k not in ('res_sha8', 'seal_sha8')}
raw = json.dumps(core, ensure_ascii=False, indent=1, sort_keys=True)
res8 = sha8(raw.encode('utf-8'))
mid = json.dumps(dict(core, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
seal8 = sha8(mid.encode('utf-8'))
chk('1.1 res_sha8 reconstructed', res8 == R['res_sha8'], '%s vs %s' % (res8, R['res_sha8']))
chk('1.2 seal_sha8 reconstructed', seal8 == R['seal_sha8'], '%s vs %s' % (seal8, R['seal_sha8']))
chk('1.3 smoke result sealed', SM.get('res_sha8') and SM.get('seal_sha8'), SM.get('res_sha8', ''))
chk('1.4 verdict shape', R['verdict'].startswith('g5a10_atlas_v13|features_22|failures_14|')
    and 'html_fields_637_ok' in R['verdict'] and 'mechanism_note_rendered' in R['verdict'], R['verdict'][:60])

# ---------- 2. FTR-22 independent re-derivation from raw sources ----------
KS = ['0', '1', '2', '4', '8']
rk = {k: r72['pooled'][k]['ratio'] for k in KS}
pf = (rk['0'] - rk['8']) / rk['0']
f22 = reg12['features'][21]
V = f22['values']
chk('2.1 FTR-22 id/family/level', f22['id'] == 'FTR-22' and f22['family'] == 'limit'
    and f22['evidence_level'] == 'E2_predictive', f22['id'])
chk('2.2 curve values match r72', all(abs(V['ratio_k' + k] - rk[k]) < 1e-12 for k in KS),
    ' '.join('%.4f' % rk[k] for k in KS))
chk('2.3 k0 re-3169 drift', abs(rk['0'] - r69['gate']['ratio_B']) < 1e-9
    and abs(V['k0_drift_ratio_vs_3169'] - abs(rk['0'] - r69['gate']['ratio_B'])) < 1e-15,
    '%.2e' % abs(rk['0'] - r69['gate']['ratio_B']))
chk('2.4 k0 E drift', abs(r72['pooled']['0']['E_oov'] - r69['gate']['pooled_E_oov_B']) < 1e-6
    and abs(r72['pooled']['0']['E_seen'] - r69['gate']['pooled_E_seen_B']) < 1e-6, '')
chk('2.5 k8 band [1.5,2) x3', all(1.5 <= r72['per_model'][m]['kcurves']['8']['ratio'] < 2.0
    for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b')), '')
chk('2.6 per-model k8 values', all(abs(V['ratio_k8_' + m.replace('-', '_')] -
    r72['per_model'][m]['kcurves']['8']['ratio']) < 1e-12
    for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b')), '')
chk('2.7 port_frac', abs(V['port_removed_frac'] - pf) < 1e-12 and 0.28 <= pf <= 0.30,
    '%.4f' % pf)
seq = [rk[k] for k in KS]
chk('2.8 curve monotone non-increasing', all(seq[i] >= seq[i + 1] for i in range(4)), '')
en0 = sum(r72['per_model'][m]['kcurves']['0']['E_newent'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b')) / 3.0
en8 = sum(r72['per_model'][m]['kcurves']['8']['E_newent'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b')) / 3.0
chk('2.9 E_newent pooled mean3', abs(V['E_newent_pooled_mean3_k0'] - en0) < 1e-12
    and abs(V['E_newent_pooled_mean3_k8'] - en8) < 1e-12, '%.4f->%.4f' % (en0, en8))
rs = {m: r71['per_model'][m]['slots']['k_main']['ratio_S'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b')}
chk('2.10 p3171 ratios in FTR-22', all(abs(V['p3171_ratio_S_' + m.replace('-', '_')] - rs[m]) < 1e-12
    for m in rs), '%.4f/%.4f/%.4f' % (rs['qwen3-4b'], rs['qwen3-14b'], rs['glm4-9b']))
chk('2.11 p3171 mixed re-derivation', rs['qwen3-4b'] < 0.8 and rs['qwen3-14b'] < 0.8
    and rs['glm4-9b'] > 0.8 and all(v > 0.5 for v in rs.values())
    and r71['overall']['main_cls'] == 'mixed_across_models', '')
chk('2.12 anchors = 3 independent phases', len(f22['anchors']) == 3
    and f22['anchors'][0]['src'] == 'p3172' and f22['anchors'][1]['src'] == 'p3169'
    and f22['anchors'][2]['src'] == 'p3171', '')
a0 = f22['anchors'][0]['asserts']
chk('2.13 anchor p3172 seal', a0['res_sha8'] == '463f42c8' and a0['seal_sha8'] == 'cdb85525'
    and a0['res_sha8'] == r72['res_sha8'] and a0['seal_sha8'] == r72['seal_sha8'], '')
a1 = f22['anchors'][1]['asserts']
chk('2.14 anchor p3169 seal', a1['res_sha8'] == '49430a39' and a1['seal_sha8'] == '018c6024'
    and abs(a1['gate.ratio_B'] - r69['gate']['ratio_B']) < 1e-12, '')
a2 = f22['anchors'][2]['asserts']
chk('2.15 anchor p3171 seal', a2['res_sha8'] == '6a29201c' and a2['seal_sha8'] == 'cc7ccedd'
    and a2['res_sha8'] == r71['res_sha8'] and a2['seal_sha8'] == r71['seal_sha8'], '')
chk('2.16 q03 cross-line drift', abs(rq['summary']['pooled_mean'] - r69['gate']['pooled_E_seen_B']) < 1e-6,
    '%.2e' % abs(rq['summary']['pooled_mean'] - r69['gate']['pooled_E_seen_B']))

# ---------- 3. registry v1.2 structure + immutability ----------
chk('3.1 registry 22 features', len(reg12['features']) == 22, str(len(reg12['features'])))
imp = 0
for i in range(21):
    a = json.dumps(reg11['features'][i], ensure_ascii=False, sort_keys=True)
    b = json.dumps(reg12['features'][i], ensure_ascii=False, sort_keys=True)
    if a == b:
        imp += 1
chk('3.2 v1.1 features byte-for-byte 21/21', imp == 21, '%d/21' % imp)
chk('3.3 failures 12->14 F13/F14', len(reg11['failures']) == 12 and len(reg12['failures']) == 14
    and reg12['failures'][12]['id'] == 'F13' and reg12['failures'][13]['id'] == 'F14', '')
impf = sum(1 for i in range(12)
           if json.dumps(reg11['failures'][i], ensure_ascii=False, sort_keys=True) ==
           json.dumps(reg12['failures'][i], ensure_ascii=False, sort_keys=True))
chk('3.4 first 12 failures byte-for-byte', impf == 12, '%d/12' % impf)
chk('3.5 upgrade_log unchanged 3', len(reg12['upgrade_log']) == 3
    and json.dumps(reg11['upgrade_log'], sort_keys=True) == json.dumps(reg12['upgrade_log'], sort_keys=True), '')
chk('3.6 version/supersedes', reg12['version'] == '1.2'
    and '3170' in reg12['supersedes'] and '1fedbd80' in reg12['supersedes'], reg12['supersedes'][:60])
f13 = reg12['failures'][12]
f14 = reg12['failures'][13]
chk('3.7 F13 values live', '1.8002' in f13['text'] and '29.1%' in f13['text']
    and '463f42c8' in f13['evidence'] and f13['phase'] == 3172, f13['kind'])
chk('3.8 F14 values live', '0.8205' in f14['text'] and '6a29201c' in f14['evidence']
    and f14['phase'] == 3171, f14['kind'])

# ---------- 4. gap ledger v1.4 increment + immutability ----------
g4n = [g for g in gl14['gaps'] if g['id'] == 'GAP-4'][0]
g4o = [g for g in gl13['gaps'] if g['id'] == 'GAP-4'][0]
chk('4.1 statement prefix preserved', g4n['statement'].startswith(g4o['statement'])
    and len(g4n['statement']) > len(g4o['statement']), '+%d chars' % (len(g4n['statement']) - len(g4o['statement'])))
chk('4.2 statement has 3172 verdict', '3172 端口校准定判' in g4n['statement']
    and '1.8002' in g4n['statement'] and '29.1%' in g4n['statement'], '')
chk('4.3 evidence +2', len(g4n['evidence']) == len(g4o['evidence']) + 2
    and g4n['evidence'][:len(g4o['evidence'])] == g4o['evidence'], str(len(g4n['evidence'])))
chk('4.4 anchors +2', g4n['anchor_sha8'] == dict(g4o['anchor_sha8'], p3171='a43cf48c', p3172='d8ddc481'),
    json.dumps(g4n['anchor_sha8'], sort_keys=True))
chk('4.5 mechanism_note byte-identical', g4n['mechanism_note'] == g4o['mechanism_note'],
    '%d chars' % len(g4n['mechanism_note']))
chk('4.6 status unchanged', g4n['status'] == 'quantified_collapse' == g4o['status'], g4n['status'])
chk('4.7 prereg unchanged', g4n['prereg'] == g4o['prereg'], '')
okg = all(json.dumps([g for g in gl13['gaps'] if g['id'] == gid][0], ensure_ascii=False, sort_keys=True) ==
          json.dumps([g for g in gl14['gaps'] if g['id'] == gid][0], ensure_ascii=False, sort_keys=True)
          for gid in ('GAP-1', 'GAP-2', 'GAP-3'))
chk('4.8 GAP-1/2/3 byte-identical', okg, '')
chk('4.9 appendix 5 byte-identical', json.dumps(gl13['appendix'], sort_keys=True) ==
    json.dumps(gl14['appendix'], sort_keys=True) and len(gl14['appendix']) == 5, '')
chk('4.10 version/supersedes', gl14['version'] == '1.4' and '2436ec08' in gl14['supersedes'], gl14['supersedes'][:50])

# ---------- 5. html full re-extract vs independently rebuilt expected dict ----------
nodes = reg3162['audit']['nodes']
exp = {}
for f in reg12['features']:
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
for i, u in enumerate(reg12['upgrade_log']):
    exp['UPG.%d.node' % i] = u['node']
    exp['UPG.%d.change' % i] = u['change']
    exp['UPG.%d.reason' % i] = u['reason']
for fl in reg12['failures']:
    exp['%s.kind' % fl['id']] = fl['kind']
    exp['%s.phase' % fl['id']] = str(fl['phase'])
    exp['%s.text' % fl['id']] = fl['text']
    exp['%s.evidence' % fl['id']] = str(fl['evidence'])
for g in gl14['gaps']:
    p = g['id']
    exp[p + '.title'] = g['title']
    exp[p + '.status'] = g['status']
    if g['status'].startswith('closed'):
        exp[p + '.closed_at'] = str(g['closed_at_phase'])
    exp[p + '.statement'] = g['statement']
    exp[p + '.evidence'] = ' || '.join(g['evidence'])
    exp[p + '.anchors'] = ', '.join('%s=%s' % (k, v) for k, v in sorted(g['anchor_sha8'].items()))
    if p == 'GAP-4':
        pr = g['prereg']
        exp[p + '.prereg.phase'] = str(pr['phase'])
        exp[p + '.prereg.name'] = pr['name']
        exp[p + '.prereg.hypothesis'] = pr['hypothesis']
        exp[p + '.prereg.protocol'] = pr['protocol']
        exp[p + '.prereg.gate'] = pr['gate']
        exp[p + '.prereg.status'] = pr['status']
        exp[p + '.mechanism_note'] = g['mechanism_note']
for a in gl14['appendix']:
    p = a['id']
    exp[p + '.title'] = a['title']
    exp[p + '.status'] = a['status']
    exp[p + '.note'] = a['note']
    exp[p + '.anchor'] = a['anchor']
    exp[p + '.anchor_sha8'] = a['anchor_sha8']
exp['meta.registry_v12_sha8'] = sha8_file(os.path.join(PDIR, 'atlas_registry_v1_2.json'))
exp['meta.registry_v11_sha8'] = '1fedbd80'
exp['meta.registry3162_sha8'] = '00f15e98'
exp['meta.p3171_res'] = 'a43cf48c'
exp['meta.p3172_res'] = 'd8ddc481'

pairs = re.findall(r'data-k="([^"]+)"[^>]*>(.*?)</span>', html_text, re.S)
found = {}
dups = []
for k, v in pairs:
    if k in found:
        dups.append(k)
    found[k] = htmlmod.unescape(v)
missing = sorted(set(exp) - set(found))
extra = sorted(set(found) - set(exp))
mismatch = [k for k in sorted(set(exp) & set(found)) if found[k] != exp[k]]
# meta.created is a runtime render timestamp (same whitelist as 3170): format check only
import datetime as _dt
ts_ok = 'meta.created' in found and bool(
    _dt.datetime.strptime(found['meta.created'], '%Y-%m-%d %H:%M'))
if 'meta.created' in extra and ts_ok:
    extra.remove('meta.created')
chk('5.0 meta.created format', ts_ok, found.get('meta.created', 'absent'))
chk('5.1 html no missing keys', not missing, str(missing[:5]))
chk('5.2 html no extra keys', not extra, str(extra[:5]))
chk('5.3 html no mismatch', not mismatch, str(mismatch[:5]))
chk('5.4 html no dups', not dups, str(dups[:5]))
chk('5.5 html field count matches result', len(found) == R['field_check']['n_found'] == 637,
    str(len(found)))
chk('5.6 mechanism_note rendered in html', 'GAP-4.mechanism_note' in found
    and found['GAP-4.mechanism_note'] == g4n['mechanism_note'], '%d chars' % len(found.get('GAP-4.mechanism_note', '')))
chk('5.7 FTR-22 card rendered', 'FTR-22.statement' in found, '')
chk('5.8 no external resources', '<link' not in html_text and '<script' not in html_text
    and 'http://' not in html_text and 'https://' not in html_text, '')
chk('5.9 html on disk sha', sha8_file(os.path.join(PDIR, 'atlas_v1_3.html')) == R['html_sha8'], R['html_sha8'])

# ---------- 6. five-write readback ----------
led = json.load(io.open(os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json'), encoding='utf-8'))
e3 = [m for m in led['measurements'] if m.get('phase') == 3173]
chk('6.1 ledger 3173 entry', len(e3) == 1, 'n=%d' % len(led['measurements']))
chk('6.2 ledger n=325', len(led['measurements']) >= 325, str(len(led['measurements'])))
chk('6.3 ledger chain sha present', len(led.get('ledger_sha256_8', '')) == 8, led.get('ledger_sha256_8', ''))
memo = open(os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md'), 'rb').read().decode('utf-8')
chk('6.4 MEMO has 3173 + res/seal', '## Phase 3173' in memo and R['res_sha8'] in memo
    and R['seal_sha8'] in memo, '')
chk('6.5 MEMO has prereg 3174', '预注册 3174' in memo, '')
daily = io.open(os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md'), encoding='utf-8').read()
chk('6.6 daily has 3173', '3173 图谱 v1.3 关账' in daily, '')
wm = open(os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md'), 'rb').read().decode('utf-8')
chk('6.7 MEMORY has 3173', '3173 图谱 v1.3 渲染关账' in wm, '')
chk('6.8 execution.json on disk', sha8_file(os.path.join(PDIR, 'execution.json')) != ''
    and json.load(io.open(os.path.join(PDIR, 'execution.json'), encoding='utf-8'))['design_sha8']
    == R['design_sha8'], R['design_sha8'])

LINES.append('TOTAL PASS=%d FAIL=%d' % (n_ok, len([l for l in LINES if l.startswith('FAIL')])))
with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LINES) + '\n')
print('verify written: %d ok, %d fail' % (n_ok, len(LINES) - n_ok - 1))
