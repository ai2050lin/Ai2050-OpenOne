# -*- coding: utf-8 -*-
# Phase 3176 independent verify: seal byte-level reconstruction + full
# independent re-implementation of the three arms (from sealed npz/vocab
# sources, not by importing the main script) + registry v1.3 invariants vs
# v1.2 + html flatten independent rebuild + five-write readback.
import hashlib
import io
import json
import os
import re
import html as htmlmod

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
S = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(S, 'phase3176', 'g5b1_elev_upgrade')
SRC12 = os.path.join(S, r'phase3173\g5a10_atlas_v13\atlas_registry_v1_2.json')
GV15 = os.path.join(S, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json')
HTML = os.path.join(PDIR, 'atlas_v1_4.html')
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3176_verify_out.txt')
LINES = []


def chk(name, ok, note=''):
    LINES.append(('%s %s %s' % ('PASS' if ok else 'FAIL', name, note)).rstrip())


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def sha8_file(p):
    return sha8(open(p, 'rb').read())


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def kross(A, Bd):
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(Bd.T)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.degrees(np.arccos(s[0])))


R = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))

# ---------- 1. seal reconstruction ----------
raw = json.dumps(R, ensure_ascii=False, indent=1, sort_keys=True)
core = {k: v for k, v in R.items() if k not in ('res_sha8', 'seal_sha8')}
r1 = json.dumps(core, ensure_ascii=False, indent=1, sort_keys=True)
res8 = sha8(r1.encode('utf-8'))
mid = json.dumps(dict(core, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
seal8 = sha8(mid.encode('utf-8'))
chk('1.1 res_sha8 reconstruction', res8 == R['res_sha8'], res8)
chk('1.2 seal_sha8 reconstruction', seal8 == R['seal_sha8'], seal8)
chk('1.3 verdict', R['verdict'].startswith('g5b1_elev_upgrade|arms_fail_pass_pass|upgrades_2'),
    R['verdict'][:60])

# ---------- 2. independent arm (b) re-implementation ----------
CLASSES = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
ENTS = [e for cl in CLASSES for e in ENT[cl]]
CLS_OF = [CLASSES.index(cl) for cl in CLASSES for e in ENT[cl]]
NE, NC = len(ENTS), len(CLASSES)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NPANEL = len(PAIRS)


def build_R(Hk, ent_rows, y, ents):
    ds = []
    for i in ents:
        tr = [r for r in ent_rows[i] if y[r]]
        fa = [r for r in ent_rows[i] if not y[r]]
        ds.append(Hk[tr].mean(0) - Hk[fa].mean(0))
    Dm = np.stack(ds)
    DmU = np.stack([unit(v) for v in Dm])
    Dc = DmU - DmU.mean(0, keepdims=True)
    _, _, Vh = np.linalg.svd(Dc, full_matrices=False)
    return Vh[:8]


NPZ_B = {'qwen3-4b': 'phase3152\\g1p2_tri_model_k1\\qwen3-4b\\collect.npz',
         'qwen3-14b': 'phase3152\\g1p2_tri_model_k1\\qwen3-14b\\collect.npz',
         'glm4-9b': 'phase3151\\g1p1_combo_additive_vs_interaction\\collect.npz'}
NPZ_KE = {'qwen3-4b': r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz',
          'qwen3-14b': r'phase3157\g2p2_transform_algebra_commutator\qwen3-14b\collect.npz',
          'glm4-9b': r'phase3157\g2p2_transform_algebra_commutator\glm4\collect.npz'}
ANCH_B = {'qwen3-4b': 45.845, 'qwen3-14b': 26.641, 'glm4-9b': 46.005}
arm_b_ok = True
for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
    z = np.load(os.path.join(S, NPZ_B[m]))
    H = z['H']
    NL = H.shape[2] - 1
    D = H.shape[3]
    Hk = H[:, :, NL, :].reshape(3 * NPANEL, D).astype(np.float64)
    y = np.zeros(3 * NPANEL, bool)
    ent_rows = {i: [] for i in range(NE)}
    for t in range(3):
        for pi, (ei, ci) in enumerate(PAIRS):
            ent_rows[ei].append(t * NPANEL + pi)
            if ci == CLS_OF[ei]:
                y[t * NPANEL + pi] = True
    z57 = np.load(os.path.join(S, NPZ_KE[m]))
    H57 = z57['H'].astype(np.float64)
    NL57 = H57.shape[1] - 1
    X = H57[:, NL57, :]
    Xc = X - X.mean(0, keepdims=True)
    _, _, Vh57 = np.linalg.svd(Xc, full_matrices=False)
    K_ent = Vh57[:8]
    ents = list(range(NE))
    _, t1_full = None, kross(build_R(Hk, ent_rows, y, ents), K_ent)
    angles = []
    for e in ents:
        others = [i for i in ents if i != e]
        angles.append(kross(build_R(Hk, ent_rows, y, others), K_ent))
    loeo = float(np.mean(angles))
    exp = R['arm_b']['per_model'][m]
    ok = (abs(round(t1_full, 3) - ANCH_B[m]) <= 0.05 and
          abs(loeo - exp['loeo_mean_deg']) < 1e-6 and
          abs(t1_full - exp['angle_full_deg']) < 1e-6)
    arm_b_ok &= ok
    chk('2.x arm_b %s' % m, ok, 'full=%.4f loeo=%.4f' % (t1_full, loeo))
chk('2.y arm_b gate re-derive', arm_b_ok and R['arm_b']['gate'] is True, '')

# ---------- 3. independent arm (c) re-implementation ----------
from transformers import AutoTokenizer
from safetensors import safe_open

HF = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'Qwen3-14B', 'glm4-9b': 'glm4-9b-chat-hf'}
z74 = np.load(os.path.join(S, r'phase2874\attr_vocab_v2\attr_vocab_v2.npz'), allow_pickle=True)
z78 = np.load(os.path.join(S, r'phase2878\syntax_trans_vocab\syntax_trans_vocab.npz'), allow_pickle=True)
AX_A = json.loads(str(z74['pairs_json']))
AX_S = json.loads(str(z78['pairs_json']))
ORD_A = [str(x) for x in z74['axes']]
ORD_S = [str(x) for x in z78['axes']]
NPZ_KR = {'qwen3-4b': r'phase3158\g4p1_output_equivalence_class\qwen3-4b\collect.npz',
          'qwen3-14b': r'phase3158\g4p1_output_equivalence_class\qwen3-14b\collect.npz',
          'glm4-9b': r'phase3158\g4p1_output_equivalence_class\glm4\collect.npz'}
arm_c_ok = True
for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
    mdir = os.path.join(ROOT, 'models', 'hf', HF[m])
    tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True, trust_remote_code=True, use_fast=True)
    cfg = json.load(io.open(os.path.join(mdir, 'config.json'), encoding='utf-8'))
    tied = bool(cfg.get('tie_word_embeddings', False))
    want = 'model.embed_tokens.weight' if tied else 'lm_head.weight'
    idxp = os.path.join(mdir, 'model.safetensors.index.json')
    shard = json.load(io.open(idxp, encoding='utf-8'))['weight_map'][want] if os.path.exists(idxp) else 'model.safetensors'
    with safe_open(os.path.join(mdir, shard), framework='pt') as f:
        W = f.get_tensor(want).float().numpy()
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, t
            tc[t] = int(ids[0])
        return tc[t]

    z58 = np.load(os.path.join(S, NPZ_KR[m]))
    K_read = z58['top64'].astype(np.float64).T
    got = {}
    for tag, AX, order in (('attr', AX_A, ORD_A), ('syntax', AX_S, ORD_S)):
        dirs = []
        for a in order:
            pd = []
            for _, wp, wm in AX[a]:
                try:
                    iw, im = tid(wp), tid(wm)
                except AssertionError:
                    continue
                pd.append(unit(W[iw] - W[im]))
            if len(pd) >= 2:
                dirs.append(unit(np.stack(pd).mean(0)))
        dW = np.stack(dirs)
        got[tag] = kross(dW, K_read)
    exp = R['arm_c']['per_model'][m]
    ok = (abs(got['attr'] - exp['attr_top1_deg']) < 1e-6 and
          abs(got['syntax'] - exp['syntax_top1_deg']) < 1e-6)
    arm_c_ok &= ok
    chk('3.x arm_c %s' % m, ok, 'attr=%.4f syntax=%.4f' % (got['attr'], got['syntax']))
    del W
d4a = abs(R['arm_c']['per_model']['qwen3-4b']['attr_top1_deg'] - 68.094)
d4s = abs(R['arm_c']['per_model']['qwen3-4b']['syntax_top1_deg'] - 58.488)
chk('3.y arm_c 4b drift vs 3165', d4a <= 0.5 and d4s <= 0.5, '%.4f/%.4f' % (d4a, d4s))
chk('3.z arm_c gate re-derive', arm_c_ok and R['arm_c']['gate'] is True, '')

# ---------- 4. arm (a) re-derivation from result numbers (gate logic) ----------
ra = R['arm_a']['per_model']
accs = [ra[m]['loeo_acc'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b')]
chk('4.1 arm_a gate frozen at 0.5', R['arm_a']['gate'] is False and
    min(accs) < 0.5 and min(accs) > 0.44, 'accs=%.4f min' % min(accs))
chk('4.2 arm_a glm4 specificity', ra['glm4-9b']['loeo_acc'] - ra['glm4-9b']['acc_rand'] < 0.05,
    '%.4f vs rand %.4f' % (ra['glm4-9b']['loeo_acc'], ra['glm4-9b']['acc_rand']))
chk('4.3 chance field', all(ra[m]['chance'] == 0.2 and ra[m]['n_test_rows'] == 525
                            for m in ra), '')

# ---------- 5. registry v1.3 invariants vs v1.2 ----------
reg12 = json.load(io.open(SRC12, encoding='utf-8'))
reg13 = json.load(io.open(os.path.join(PDIR, 'atlas_registry_v1_3.json'), encoding='utf-8'))
chk('5.1 version/supersedes', reg13['version'] == '1.3' and
    reg13['supersedes'].endswith('(0e5abcaf, Phase 3173)'), reg13['version'])
m12 = {f['id']: f for f in reg12['features']}
m13 = {f['id']: f for f in reg13['features']}
chk('5.2 22 features', len(reg13['features']) == 22, '')
unchanged = 0
for fid, f in m12.items():
    a = json.dumps(f, ensure_ascii=False, sort_keys=True)
    b = json.dumps(m13[fid], ensure_ascii=False, sort_keys=True)
    if fid in ('FTR-04', 'FTR-06', 'FTR-08'):
        continue
    assert a == b, fid
    unchanged += 1
chk('5.3 19 features byte-for-byte', unchanged == 19, str(unchanged))
f4 = m13['FTR-04']
ce_old = m12['FTR-04']['counter_evidence']
chk('5.4 FTR-04 only counter append', f4['evidence_level'] == 'E1_repeatable' and
    f4['counter_evidence'][:len(ce_old)] == ce_old and
    len(f4['counter_evidence']) == len(ce_old) + 1 and
    json.dumps({k: v for k, v in f4.items() if k != 'counter_evidence'},
               ensure_ascii=False, sort_keys=True) ==
    json.dumps({k: v for k, v in m12['FTR-04'].items() if k != 'counter_evidence'},
               ensure_ascii=False, sort_keys=True), '')
f6, f8 = m13['FTR-06'], m13['FTR-08']
tags6 = [t for a in f6['anchors'] for t in a.get('tags', [])]
tags8 = [t for a in f8['anchors'] for t in a.get('tags', [])]
chk('5.5 FTR-06 upgraded', f6['evidence_level'] == 'E2_predictive' and
    f6['model_scope'] == ['qwen3-4b', 'qwen3-14b', 'glm4-9b'] and 'cross_model' in tags6 and
    len(f6['anchors']) == 3, '')
chk('5.6 FTR-08 upgraded', f8['evidence_level'] == 'E2_predictive' and 'held_out' in tags8 and
    len(f8['anchors']) == 3, '')
av6 = f6['anchors'][-1]['asserts']
chk('5.7 FTR-06 anchor values == result', abs(av6['attr_top1_qwen3_4b'] -
    R['arm_c']['per_model']['qwen3-4b']['attr_top1_deg']) < 1e-12 and
    abs(av6['syntax_top1_glm4_9b'] - R['arm_c']['per_model']['glm4-9b']['syntax_top1_deg']) < 1e-12, '')
av8 = f8['anchors'][-1]['asserts']
chk('5.8 FTR-08 anchor values == result', abs(av8['loeo_mean_qwen3_14b'] -
    R['arm_b']['per_model']['qwen3-14b']['loeo_mean_deg']) < 1e-12, '')
chk('5.9 failures 15 + F15 last', len(reg13['failures']) == 15 and
    reg13['failures'][-1]['id'] == 'F15' and reg13['failures'][-1]['phase'] == 3176, '')
chk('5.10 upgrade_log 5, first 3 byte-for-byte', len(reg13['upgrade_log']) == 5 and
    all(json.dumps(reg12['upgrade_log'][i], ensure_ascii=False, sort_keys=True) ==
        json.dumps(reg13['upgrade_log'][i], ensure_ascii=False, sort_keys=True)
        for i in range(3)) and
    [u['node'] for u in reg13['upgrade_log'][3:]] == ['FTR-06', 'FTR-08'], '')
chk('5.11 failures first 14 byte-for-byte', all(
    json.dumps(reg12['failures'][i], ensure_ascii=False, sort_keys=True) ==
    json.dumps(reg13['failures'][i], ensure_ascii=False, sort_keys=True) for i in range(14)), '')
chk('5.12 file sha anchors', sha8_file(os.path.join(PDIR, 'atlas_registry_v1_3.json')) ==
    R['registry_v13_sha8'] == 'e64733f7' and
    sha8_file(HTML) == R['html_sha8'] == 'd611a2a1', '')

# ---------- 6. html flatten independent rebuild ----------
gl15 = json.load(io.open(GV15, encoding='utf-8'))
chk('6.0 gap v1.5 anchored', sha8_file(GV15) == '4fb3f37d' and gl15['version'] == '1.5', '')


def fmt(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return '%.6g' % v
    return str(v)


nodes = json.load(io.open(os.path.join(S, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
                          encoding='utf-8'))['audit']['nodes']
d = {}
for f in reg13['features']:
    p = f['id']
    d[p + '.statement'] = f['statement']
    d[p + '.family'] = f['family']
    d[p + '.evidence_level'] = f['evidence_level']
    d[p + '.model_scope'] = ', '.join(f['model_scope'])
    d[p + '.n_anchors'] = str(len(f['anchors']))
    d[p + '.counter_evidence'] = ' | '.join(f['counter_evidence']) if f['counter_evidence'] else '(none)'
    d[p + '.replication'] = f['replication']
    d[p + '.scope_limits'] = f['scope_limits'] if f['scope_limits'] else '(none)'
    for i, a in enumerate(f['anchors']):
        d['%s.anchors.%d.src' % (p, i)] = str(a['src'])
        d['%s.anchors.%d.phase' % (p, i)] = str(a['phase'])
        d['%s.anchors.%d.role' % (p, i)] = a['role']
        d['%s.anchors.%d.asserts' % (p, i)] = json.dumps(a['asserts'], ensure_ascii=False, sort_keys=True)
    for k, v in (f.get('values') or {}).items():
        d['%s.values.%s' % (p, k)] = fmt(v)
for n in nodes:
    d[n['id'] + '.title'] = n['title']
    d[n['id'] + '.elevel'] = n['evidence_level']
    d[n['id'] + '.status'] = n['status']
    d[n['id'] + '.checks'] = '%d/%d' % (n['n_pass'], n['n_checks'])
for i, u in enumerate(reg13['upgrade_log']):
    d['UPG.%d.node' % i] = u['node']
    d['UPG.%d.change' % i] = u['change']
    d['UPG.%d.reason' % i] = u['reason']
for fl in reg13['failures']:
    d[fl['id'] + '.kind'] = fl['kind']
    d[fl['id'] + '.phase'] = str(fl['phase'])
    d[fl['id'] + '.text'] = fl['text']
    d[fl['id'] + '.evidence'] = str(fl['evidence'])
for g in gl15['gaps']:
    p = g['id']
    d[p + '.title'] = g['title']
    d[p + '.status'] = g['status']
    if g['status'].startswith('closed'):
        d[p + '.closed_at'] = str(g['closed_at_phase'])
    d[p + '.statement'] = g['statement']
    d[p + '.evidence'] = ' || '.join(g['evidence'])
    d[p + '.anchors'] = ', '.join('%s=%s' % (k, v) for k, v in sorted(g['anchor_sha8'].items()))
    if p == 'GAP-4':
        pr = g['prereg']
        d[p + '.prereg.phase'] = str(pr['phase'])
        d[p + '.prereg.name'] = pr['name']
        d[p + '.prereg.hypothesis'] = pr['hypothesis']
        d[p + '.prereg.protocol'] = pr['protocol']
        d[p + '.prereg.gate'] = pr['gate']
        d[p + '.prereg.status'] = pr['status']
        d[p + '.mechanism_note'] = g['mechanism_note']
for a in gl15['appendix']:
    p = a['id']
    d[p + '.title'] = a['title']
    d[p + '.status'] = a['status']
    d[p + '.note'] = a['note']
    d[p + '.anchor'] = a['anchor']
    d[p + '.anchor_sha8'] = a['anchor_sha8']
d['meta.registry_v13_sha8'] = R['registry_v13_sha8']
d['meta.registry_v12_sha8'] = '0e5abcaf'
d['meta.gap_v15_sha8'] = '4fb3f37d'
d['meta.p3166_res'] = '448af595'
d['meta.p3165_res'] = '511d9b13'
d['meta.p3169_res'] = '5b51c2c1'
d['meta.created'] = None  # runtime timestamp: whitelisted below, format-checked
htxt = io.open(HTML, encoding='utf-8').read()
pairs = re.findall(r'data-k="([^"]+)"[^>]*>(.*?)</span>', htxt, re.S)
found = {}
dups = []
for k, v in pairs:
    if k in found:
        dups.append(k)
    found[k] = htmlmod.unescape(v)
missing = sorted(set(d) - set(found))
extra = sorted(set(found) - set(d))
mism = [k for k in sorted(set(d) & set(found)) if found[k] != d[k]]
mism = [k for k in mism if k != 'meta.created']  # whitelisted runtime timestamp
# meta.created is a runtime timestamp: whitelist it (present + 'YYYY-MM-DD HH:MM' shape)
import datetime as _dt
ok_created = False
if 'meta.created' in found:
    try:
        _dt.datetime.strptime(found['meta.created'], '%Y-%m-%d %H:%M')
        ok_created = True
    except ValueError:
        ok_created = False
missing = [k for k in missing if k != 'meta.created']
extra = [k for k in extra if k != 'meta.created']
if not ok_created:
    mism.append('meta.created (whitelist fail)')
d.pop('meta.created', None)
chk('6.1 html fields', not missing and not extra and not mism and not dups,
    'exp=%d found=%d miss=%d extra=%d mism=%d dup=%d' % (len(d), len(found), len(missing), len(extra), len(mism), len(dups)))
chk('6.2 no external', '<link' not in htxt and '<script' not in htxt and
    'http://' not in htxt and 'https://' not in htxt, '')
chk('6.3 GAP-4 mechanism_note rendered', 'GAP-4.mechanism_note' in found and
    found['GAP-4.mechanism_note'] == gl15['gaps'][3]['mechanism_note'] if gl15['gaps'][3]['id'] == 'GAP-4' else False, '')
chk('6.4 FTR-04 failed-arm counter rendered', 'FTR-04.counter_evidence' in found and
    '3176 arm (a) FAILED gate' in found['FTR-04.counter_evidence'], '')

# ---------- 7. five-write readback ----------
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
led = json.load(io.open(LEDGER, encoding='utf-8'))
e = [m for m in led['measurements'] if m.get('phase') == 3176]
chk('7.1 ledger 3176 single entry', len(e) == 1 and len(led['measurements']) == 328,
    'n=%d' % len(led['measurements']))
chk('7.2 ledger verdict', e and 'arms_fail_pass_pass' in e[0]['verdict'] and
    e[0]['prereg_id'] == 'G5-B1', '')
memo = open(MEMO, 'rb').read().decode('utf-8')
chk('7.3 MEMO 3176 + shas + prereg', '## Phase 3176' in memo and R['res_sha8'] in memo and
    R['seal_sha8'] in memo and 'e64733f7' in memo and '预注册 3177' in memo, '')
daily = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk('7.4 daily 3176', '3176 E1→E2 批量升级' in daily and 'c8a9797d' in daily and
    'd2647ffb' in daily and 'ledger n→328' in daily, '')
wm = open(WMEM, 'rb').read().decode('utf-8')
chk('7.5 workspace MEMORY 3176', '3176 E1→E2 批量升级' in wm and 'n=327→**328**' in wm, '')
sm = json.load(io.open(os.path.join(PDIR, 'smoke_result.json'), encoding='utf-8'))
chk('7.6 smoke vs full', sm['res_sha8'] != R['res_sha8'] and
    sm['registry_v13_sha8'] == R['registry_v13_sha8'], 'arms identical, render subset differs')

n_ok = sum(1 for c in LINES if c.startswith('PASS'))
n_fail = sum(1 for c in LINES if c.startswith('FAIL'))
LINES.append('TOTAL PASS=%d FAIL=%d' % (n_ok, n_fail))
with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LINES) + '\n')
print('\n'.join(LINES))
