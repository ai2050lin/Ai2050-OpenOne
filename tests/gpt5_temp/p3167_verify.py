# -*- coding: utf-8 -*-
"""Phase 3167 independent disk verification. Re-derives everything from sealed files."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3167', 'g5a4_feature_registry')
GLM = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3167_verify_out.txt')
LINES = []
N_OK = [0]
N_FAIL = [0]


def chk(name, ok, note=''):
    if ok:
        N_OK[0] += 1
        LINES.append('PASS %s %s' % (name, note))
    else:
        N_FAIL[0] += 1
        LINES.append('FAIL %s %s' % (name, note))


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def load(p):
    return json.load(io.open(p, encoding='utf-8'))


def getpath(obj, dotted):
    cur = obj
    for part in dotted.split('.'):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


R = load(os.path.join(PDIR, 'result.json'))
REG = load(os.path.join(PDIR, 'atlas_registry_v1.json'))
REG3162 = load(os.path.join(GLM, 'phase3162', 'g5a1_atlas_foundation', 'atlas_registry.json'))
SMOKE = load(os.path.join(PDIR, 'smoke_result.json'))

# ---------- 1. disk sha of artifacts ----------
sha_exec = sha8(os.path.join(PDIR, 'execution.json'))
sha_res = sha8(os.path.join(PDIR, 'result.json'))
sha_reg = sha8(os.path.join(PDIR, 'atlas_registry_v1.json'))
chk('exec disk sha logged', True, sha_exec)
chk('result disk sha logged', True, sha_res)
# expected registry sha read live from the sealed ledger entry (no memory constants)
led0 = load(LEDGER)
det3167 = [m for m in led0['measurements'] if m.get('phase') == 3167]
exp_reg = ''
if det3167:
    dd = det3167[0].get('detail', '')
    marker = 'Registry artifact sha8 '
    if marker in dd:
        exp_reg = dd.split(marker, 1)[1].split()[0].rstrip('.,;')
chk('registry disk sha == ledger detail', sha_reg == exp_reg,
    '%s vs %s' % (sha_reg, exp_reg))
EX = load(os.path.join(PDIR, 'execution.json'))
chk('exec design_sha8', EX['design_sha8'] == '0a2b3120', EX['design_sha8'])
core = {k: v for k, v in EX.items() if k not in ('created', 'design_sha8')}
d8 = hashlib.sha256(json.dumps(core, ensure_ascii=False, indent=1, sort_keys=True)
                    .encode('utf-8')).hexdigest()[:8]
chk('exec design hash re-derives', d8 == EX['design_sha8'], d8)

# ---------- 2. seal byte-level rebuild ----------
raw = json.dumps({k: v for k, v in R.items() if k not in ('res_sha8', 'seal_sha8')},
                 ensure_ascii=False, indent=1, sort_keys=True)
res8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
mid = json.dumps(dict({k: v for k, v in R.items() if k != 'seal_sha8'}, res_sha8=res8),
                 ensure_ascii=False, indent=1, sort_keys=True)
seal8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
chk('res_sha8 rebuild', res8 == R['res_sha8'], '%s vs %s' % (res8, R['res_sha8']))
chk('seal_sha8 rebuild', seal8 == R['seal_sha8'], '%s vs %s' % (seal8, R['seal_sha8']))

# ---------- 3. registry structural gates ----------
feats = REG['features']
chk('features == 20', len(feats) == 20, 'n=%d' % len(feats))
chk('failures == 12', len(REG['failures']) == 12, 'n=%d' % len(REG['failures']))
chk('principles == 4 verbatim', REG['principles'] == REG3162['principles'])
chk('taxonomy verbatim', REG['evidence_level_taxonomy'] == REG3162['evidence_level_taxonomy'])
chk('upgrade_log == 3', len(REG['upgrade_log']) == 3)
ids = [f['id'] for f in feats]
chk('ids contiguous FTR-01..20', ids == ['FTR-%02d' % i for i in range(1, 21)])
schema_ok = all(all(k in f for k in ('id', 'family', 'statement', 'evidence_level', 'model_scope',
                                     'anchors', 'counter_evidence', 'replication', 'values',
                                     'scope_limits')) for f in feats)
chk('8-field schema complete on all', schema_ok)

# ---------- 4. per-feature anchor re-verification ----------
SRC = {}
for pr in REG['provenance']['anchor_files'].values():
    SRC[pr['path']] = pr['sha8']
# rebuild full path map from provenance (stored relative to ROOT+sep)
FMAP = {}
for tagkey, pr in REG['provenance']['anchor_files'].items():
    full = os.path.join(ROOT, pr['path'])
    FMAP[pr['path']] = full
# map feature anchors (they reference provenance keys by sha + phase only in registry? check)
# The registry features store anchors with 'src' keys; provenance keys are those src tags.
PROV = REG['provenance']['anchor_files']
n_asserts = 0
lv_count = {}
for f in feats:
    fid = f['id']
    # G2 independence
    phases = set(str(a['phase']) for a in f['anchors'])
    chk('%s anchors>=2 phases>=2' % fid,
        len(f['anchors']) >= 2 and len(phases) >= 2,
        'anchors=%d phases=%d' % (len(f['anchors']), len(phases)))
    # G3 scope
    chk('%s model_scope explicit' % fid,
        bool(f['model_scope']) and set(f['model_scope']) <= set(REG['models_mainline']),
        ','.join(f['model_scope']))
    # anchor files exist + sha
    for a in f['anchors']:
        src = a['src']
        chk('%s anchor %s in provenance' % (fid, src), src in PROV)
        if src in PROV:
            p = os.path.join(ROOT, PROV[src]['path'])
            chk('%s anchor %s exists' % (fid, src), os.path.exists(p))
            if os.path.exists(p):
                chk('%s anchor %s sha8' % (fid, src), sha8(p) == PROV[src]['sha8'],
                    '%s vs %s' % (sha8(p), PROV[src]['sha8']))
    # G4 asserts re-derived from disk
    for a in f['anchors']:
        src = a['src']
        if src not in PROV:
            continue
        p = os.path.join(ROOT, PROV[src]['path'])
        if not os.path.exists(p):
            continue
        RR = load(p)
        for k, exp in sorted(a['asserts'].items()):
            n_asserts += 1
            try:
                if k.startswith('node:'):
                    rest = k[len('node:'):]
                    nid, field = rest.split('.', 1)
                    node = [n for n in RR['audit']['nodes'] if n['id'] == nid][0]
                    got = getpath(node, field)
                else:
                    got = getpath(RR, k)
            except Exception as e:
                chk('%s %s.%s path' % (fid, src, k), False, str(e))
                continue
            if isinstance(exp, float) and isinstance(got, (int, float)) and not isinstance(got, bool):
                ok = abs(float(got) - exp) <= max(1e-9, 1e-9 * abs(exp) if exp else 0.0)
            elif isinstance(exp, bool) or isinstance(got, bool):
                ok = bool(got) == bool(exp)
            else:
                ok = str(got) == str(exp)
            chk('%s %s.%s' % (fid, src, k), ok, 'got=%r' % (got,))
    # G5 level consistency
    roles = [a.get('role', '') for a in f['anchors']]
    tags_all = [t for a in f['anchors'] for t in a.get('tags', [])]
    lv = f['evidence_level']
    lv_count[lv] = lv_count.get(lv, 0) + 1
    if lv == 'E3_causal_scoped':
        chk('%s E3 intervention anchor' % fid, any('intervention' in r for r in roles))
    if lv == 'E2_predictive':
        chk('%s E2 cross/heldout anchor' % fid,
            any(t in ('cross_model', 'held_out') for t in tags_all))
    chk('%s not E0' % fid, lv != 'E0_candidate')
    # G6 values non-empty for features with values_spec-bearing anchors
    chk('%s values rendered' % fid, isinstance(f.get('values'), dict))

chk('total asserts re-derived == 164', n_asserts == 164, 'n=%d' % n_asserts)
chk('levels E2=12 E1=5 E3=3',
    lv_count.get('E2_predictive') == 12 and lv_count.get('E1_repeatable') == 5
    and lv_count.get('E3_causal_scoped') == 3, json.dumps(lv_count))

# ---------- 5. failure ledger verbatim ----------
f3162 = {x['id']: x for x in REG3162['failures']}
for x in REG['failures']:
    if x['id'] == 'F12':
        chk('F12 appended', '3165->3166' in x['phase'])
    else:
        chk('%s verbatim from 3162' % x['id'],
            f3162.get(x['id'], {}).get('text') == x['text'])

# ---------- 6. key numeric spot checks (independent reads) ----------
q03 = load(os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q03_result.json'))
chk('spot E_read pooled', abs(q03['summary']['pooled_mean'] - 0.37335047125816345) < 1e-12)
chk('spot gate 0/3', q03['summary']['gate_pass_frac'] == '0/3')
q06 = load(os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q06_result.json'))
chk('spot C_steer 0', q06['C_steer_main']['value'] == 0.0)
chk('spot wilson hi', abs(q06['C_steer_main']['wilson'][1] - 0.010113689495831947) < 1e-15)
p66 = load(os.path.join(GLM, 'phase3166', 'g5a3b_logic_direction', 'result.json'))
chk('spot 14b R x K_ent', p66['per_model']['qwen3-14b']['census']['R_logic__K_entity']['top1_deg'] == 26.641)
chk('spot 14b eff=2', p66['per_model']['qwen3-14b']['census']['R_logic__K_entity']['eff_ge05'] == 2)
p65 = load(os.path.join(GLM, 'phase3165', 'g5a3_family_alignment', 'result.json'))
chk('spot 4b KxS_class 67.0', p65['pairwise_4b']['K_readout__S_class']['top1_deg'] == 67.0)
ag = load(os.path.join(ROOT, 'research', 'deepseek', 'atlas', 'a_gate_closure_v1.json'))
chk('spot gate closed', ag['gate_closed'] is True)

# ---------- 7. five-write read-back ----------
led2 = load(LEDGER)
ms2 = led2['measurements']
e3167 = [m for m in ms2 if m.get('phase') == 3167]
chk('ledger has 3167 entry', len(e3167) == 1, 'n=%d' % len(ms2))
if e3167:
    chk('ledger 3167 verdict', '20 features' in e3167[0]['verdict'], e3167[0]['verdict'][:60])
    chk('ledger 3167 res sha', e3167[0].get('detail', '').find('b8e715e0') >= 0)
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk('MEMO Phase 3167', '## Phase 3167' in memo2)
chk('MEMO registry sha', exp_reg and exp_reg in memo2, exp_reg)
chk('MEMO prereg 3168', '预注册 3168' in memo2)
d2 = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk('daily 3167', '3167 图谱 v1 特征登记表' in d2)
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk('MEMORY 3167', '3167 图谱 v1 特征登记表' in w2)
chk('smoke res present', bool(SMOKE.get('res_sha8')), SMOKE.get('res_sha8'))

LINES.append('TOTAL PASS=%d FAIL=%d' % (N_OK[0], N_FAIL[0]))
with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LINES) + '\n')
print('TOTAL PASS=%d FAIL=%d' % (N_OK[0], N_FAIL[0]))
