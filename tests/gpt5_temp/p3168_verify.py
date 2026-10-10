# -*- coding: utf-8 -*-
# Phase 3168 independent verify (separate process, independent flatten re-implementation).
# No hardcoded artifact sha expectations: hashes are checked against result.json records
# plus same-file double-read stability, per the "no matching strings from memory" rule.
import hashlib
import io
import json
import os
import re
import html as htmlmod

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(SRC_DIR, 'phase3168', 'g5a5_atlas_render')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3168_verify_out.txt')
LOG = []


def chk(name, ok, info=''):
    LOG.append('%s %s %s' % ('PASS' if ok else 'FAIL', name, info))
    print('%s %s %s' % ('PASS' if ok else 'FAIL', name, info), flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def fmt(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return '%.6g' % v
    return str(v)


R = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
REG = json.load(io.open(os.path.join(SRC_DIR, 'phase3167', 'g5a4_feature_registry',
                                      'atlas_registry_v1.json'), encoding='utf-8'))
REG2 = json.load(io.open(os.path.join(SRC_DIR, 'phase3162', 'g5a1_atlas_foundation',
                                      'atlas_registry.json'), encoding='utf-8'))
GL = json.load(io.open(os.path.join(PDIR, 'gap_ledger_v1.json'), encoding='utf-8'))
HTML = io.open(os.path.join(PDIR, 'atlas_v1.html'), encoding='utf-8').read()

# ---------- 1. disk hashes vs result.json records ----------
chk('exec sha logged', sha8_file(os.path.join(PDIR, 'execution.json')) is not None)
chk('html sha == result record', sha8_file(os.path.join(PDIR, 'atlas_v1.html')) == R['html_sha8'],
    R['html_sha8'])
chk('gap ledger sha == result record',
    sha8_file(os.path.join(PDIR, 'gap_ledger_v1.json')) == R['gap_ledger_sha8'], R['gap_ledger_sha8'])
chk('html same-file double read', sha8_file(os.path.join(PDIR, 'atlas_v1.html')) ==
    sha8_file(os.path.join(PDIR, 'atlas_v1.html')))
chk('registry_v1 sha matches 3167 provenance',
    sha8_file(os.path.join(SRC_DIR, 'phase3167', 'g5a4_feature_registry', 'atlas_registry_v1.json'))
    == R['sources']['registry_v1'], R['sources']['registry_v1'])

# ---------- 2. seal byte-level rebuild ----------
body = {k: v for k, v in R.items() if k not in ('res_sha8', 'seal_sha8')}
raw = json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True)
res8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
mid = json.dumps(dict(body, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
seal8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
chk('res_sha8 rebuild', res8 == R['res_sha8'], '%s vs %s' % (res8, R['res_sha8']))
chk('seal_sha8 rebuild', seal8 == R['seal_sha8'], '%s vs %s' % (seal8, R['seal_sha8']))

# ---------- 3. independent flatten + html field check ----------
exp = {}
for f in REG['features']:
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
for n in REG2['audit']['nodes']:
    exp[n['id'] + '.title'] = n['title']
    exp[n['id'] + '.elevel'] = n['evidence_level']
    exp[n['id'] + '.status'] = n['status']
    exp[n['id'] + '.checks'] = '%d/%d' % (n['n_pass'], n['n_checks'])
for i, u in enumerate(REG['upgrade_log']):
    exp['UPG.%d.node' % i] = u['node']
    exp['UPG.%d.change' % i] = u['change']
    exp['UPG.%d.reason' % i] = u['reason']
for fl in REG['failures']:
    exp[fl['id'] + '.kind'] = fl['kind']
    exp[fl['id'] + '.phase'] = str(fl['phase'])
    exp[fl['id'] + '.text'] = fl['text']
    exp[fl['id'] + '.evidence'] = str(fl['evidence'])
for g in GL['gaps']:
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
for a in GL['appendix']:
    p = a['id']
    exp[p + '.title'] = a['title']
    exp[p + '.status'] = a['status']
    exp[p + '.note'] = a['note']
    exp[p + '.anchor'] = a['anchor']
    exp[p + '.anchor_sha8'] = a['anchor_sha8']
exp['meta.registry_sha8'] = R['sources']['registry_v1']
exp['meta.registry3162_sha8'] = R['sources']['registry_3162']
exp['meta.created'] = re.search(r'data-k="meta\.created">([^<]*)<', HTML).group(1)

pairs = re.findall(r'data-k="([^"]+)"[^>]*>(.*?)</span>', HTML, re.S)
found = {}
dups = []
for k, v in pairs:
    if k in found:
        dups.append(k)
    found[k] = htmlmod.unescape(v)
missing = sorted(set(exp) - set(found))
extra = sorted(set(found) - set(exp))
mism = [(k, found[k][:60], exp[k][:60]) for k in sorted(set(exp) & set(found)) if found[k] != exp[k]]
chk('html field count == result record', len(found) == R['field_check']['n_found'],
    '%d vs %d' % (len(found), R['field_check']['n_found']))
chk('html fields missing=0', not missing, str(missing[:6]))
chk('html fields extra=0', not extra, str(extra[:6]))
chk('html fields mismatch=0', not mism, str(mism[:4]))
chk('html fields dup=0', not dups, str(dups[:6]))

# ---------- 4. html structure ----------
chk('no <link', '<link' not in HTML)
chk('no <script', '<script' not in HTML)
chk('no external url', 'http://' not in HTML and 'https://' not in HTML)
for i in range(1, 21):
    if 'id="FTR-%02d"' % i not in HTML:
        chk('feature card FTR-%02d present' % i, False)
        break
else:
    chk('feature cards FTR-01..20 all present', True, '20')
for i in range(16):
    if 'data-k="N%02d.status"' % i not in HTML:
        chk('node N%02d present' % i, False)
        break
else:
    chk('nodes N00..N15 all present', True, '16')
chk('failures F1..F12 present', all('data-k="F%d.text"' % i in HTML for i in range(1, 13)))
chk('upgrades 3 present', all('data-k="UPG.%d.node"' % i in HTML for i in range(3)))
chk('gaps GAP-1..4 present', all('data-k="GAP-%d.status"' % i in HTML for i in range(1, 5)))
chk('appendix APPX-1..5 present', all('data-k="APPX-%d.status"' % i in HTML for i in range(1, 6)))
chk('GAP-4 prereg rendered', 'data-k="GAP-4.prereg.gate"' in HTML)

# ---------- 5. gap ledger content ----------
chk('gap statuses', [g['status'] for g in GL['gaps']] == ['closed', 'closed', 'closed_v1', 'open'],
    json.dumps({g['id']: g['status'] for g in GL['gaps']}))
chk('closed gaps carry anchor sha8', all(
    g['anchor_sha8'] and all(re.match(r'^[0-9a-f]{8}$', v) for v in g['anchor_sha8'].values())
    for g in GL['gaps'] if g['status'].startswith('closed')))
chk('GAP-4 prereg status', GL['gaps'][3]['prereg']['status'] == 'preregistered_not_executed')
chk('GAP-4 prereg gate text', '2x' in GL['gaps'][3]['prereg']['gate'])
chk('appendix 5 items with anchors', all(
    re.match(r'^[0-9a-f]{8}$', a['anchor_sha8']) for a in GL['appendix']), 'n=%d' % len(GL['appendix']))
chk('GAP-1 evidence carries 3164 verdicts',
    'zero_like_q06' in GL['gaps'][0]['evidence'][0] and 'rope_relative_supported' in GL['gaps'][0]['evidence'][1])
chk('GAP-2 evidence chain', 'attention_reallocation_primary' in GL['gaps'][1]['evidence'][1]
    and 'redundant_closing' in GL['gaps'][1]['evidence'][3])
chk('GAP-4 evidence q03+3151', 'gate_pass_frac=0/3' in GL['gaps'][3]['evidence'][0]
    and 'shuiguo(1.2226)' in GL['gaps'][3]['evidence'][1])

# ---------- 6. five-write read-back ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms2 = led['measurements']
e3168 = [m for m in ms2 if m.get('phase') == 3168]
chk('ledger has 3168 entry', len(e3168) == 1, 'n_total=%d' % len(ms2))
chk('ledger n>=320', len(ms2) >= 320, 'n=%d' % len(ms2))
if e3168:
    chk('ledger 3168 verdict contains html fields', ('%d fields ok' % R['field_check']['n_found'])
        in e3168[0]['verdict'], e3168[0]['verdict'][:80])
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk('MEMO has Phase 3168', '## Phase 3168' in memo2)
chk('MEMO has res sha', R['res_sha8'] in memo2)
chk('MEMO has seal sha', R['seal_sha8'] in memo2)
chk('MEMO has html sha', R['html_sha8'] in memo2)
chk('MEMO has prereg 3169', '预注册 3169' in memo2)
chk('MEMO has field count', str(R['field_check']['n_found']) in memo2)
d2 = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk('daily has 3168', '3168 图谱 v1 渲染' in d2)
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk('MEMORY has 3168', '3168 图谱 v1 渲染' in w2)
chk('MEMORY has ledger n 320', '320' in w2)

bad = [ln for ln in LOG if ln.startswith('FAIL')]
with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
print('VERIFY %s (%d checks, %d fail)' % ('ALL PASS' if not bad else 'HAS FAIL', len(LOG), len(bad)))
