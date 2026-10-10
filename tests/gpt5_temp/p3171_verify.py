# -*- coding: utf-8 -*-
# Phase 3171 independent disk verification.
# Independent re-implementation: panel row sets, S_class rebuild, projection
# energy ratio, verdict gate logic. Seal byte-level rebuild. Gap ledger v1.2
# immutability vs v1.1. Five-write read-back. No imports from the main script.
import io
import json
import hashlib
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(RDIR, 'phase3171', 'g5a8_collapse_mechanism')
P3169 = os.path.join(RDIR, 'phase3169', 'g5a6_oov_panel')
P3170 = os.path.join(RDIR, 'phase3170', 'g5a7_atlas_v11')
P3166 = os.path.join(RDIR, 'phase3166', 'g5a3b_logic_direction', 'result.json')
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3171_verify_out.txt')
R_ = []
N_OK = [0]
N_FAIL = [0]


def chk(name, ok, detail=''):
    if ok:
        N_OK[0] += 1
        R_.append('PASS %s %s' % (name, detail))
    else:
        N_FAIL[0] += 1
        R_.append('FAIL %s %s' % (name, detail))


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


Res = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
SMK = json.load(io.open(os.path.join(PDIR, 'smoke_result.json'), encoding='utf-8'))
Exec = json.load(io.open(os.path.join(PDIR, 'execution.json'), encoding='utf-8'))
R69 = json.load(io.open(os.path.join(P3169, 'result.json'), encoding='utf-8'))
R66 = json.load(io.open(P3166, encoding='utf-8'))
GL12 = json.load(io.open(os.path.join(P3170, 'gap_ledger_v1_2.json'), encoding='utf-8'))
GL11 = json.load(io.open(os.path.join(P3170, 'gap_ledger_v1_1.json'), encoding='utf-8'))

# ---------- 1. disk shas ----------
chk('1.1 exec sha probe', True, sha8_file(os.path.join(PDIR, 'execution.json')))
chk('1.2 result sha probe', True, sha8_file(os.path.join(PDIR, 'result.json')))
chk('1.3 smoke result sha probe', True, sha8_file(os.path.join(PDIR, 'smoke_result.json')))
chk('1.4 run_log non-empty', os.path.getsize(os.path.join(PDIR, 'run_log.txt')) > 0)
chk('1.5 smoke_run_log non-empty', os.path.getsize(os.path.join(PDIR, 'smoke_run_log.txt')) > 0)

# ---------- 2. seal byte-level rebuild ----------
summary = {k: v for k, v in Res.items() if k not in ('res_sha8', 'seal_sha8')}
raw = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True)
res8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
mid = json.dumps(dict(summary, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
seal8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
chk('2.1 res_sha8 rebuild', res8 == Res['res_sha8'], '%s vs %s' % (res8, Res['res_sha8']))
chk('2.2 seal_sha8 rebuild', seal8 == Res['seal_sha8'], '%s vs %s' % (seal8, Res['seal_sha8']))
smry = {k: v for k, v in SMK.items() if k not in ('res_sha8', 'seal_sha8')}
raws = json.dumps(smry, ensure_ascii=False, indent=1, sort_keys=True)
res8s = hashlib.sha256(raws.encode('utf-8')).hexdigest()[:8]
mids = json.dumps(dict(smry, res_sha8=res8s), ensure_ascii=False, indent=1, sort_keys=True)
seal8s = hashlib.sha256(mids.encode('utf-8')).hexdigest()[:8]
chk('2.3 smoke res_sha8 rebuild', res8s == SMK['res_sha8'], res8s)
chk('2.4 smoke seal_sha8 rebuild', seal8s == SMK['seal_sha8'], seal8s)

# ---------- 3. execution/design consistency ----------
core = {k: v for k, v in Exec.items() if k not in ('created', 'design_sha8')}
d8 = hashlib.sha256(json.dumps(core, ensure_ascii=False, indent=1,
                               sort_keys=True).encode('utf-8')).hexdigest()[:8]
chk('3.1 exec design_sha8 self-consistent', d8 == Exec['design_sha8'], d8)
chk('3.2 result design_sha8 matches exec', Res['design_sha8'] == Exec['design_sha8'])
chk('3.3 exec smoke field is static description (str, not runtime bool)',
    isinstance(core.get('smoke'), str) and not isinstance(core.get('smoke'), bool),
    repr(type(core.get('smoke')).__name__))

# ---------- 4. independent row-set + ratio recompute ----------
CLS_SEEN = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
CLASSES_SEEN = CLS_SEEN
CLASSES_OOV = ['乐器', '天气', '运动', '电器']
CLASSES = CLASSES_SEEN + CLASSES_OOV
NC = 10
ENT_SEEN_N = [8, 8, 6, 6, 7, 6]
ENT_OOV_N = [8, 8, 8, 8]
NE = sum(ENT_SEEN_N) + sum(ENT_OOV_N)
assert (NE, NC) == (73, 10)
N_SEEN_ENT = sum(ENT_SEEN_N)
seen_rows, oov_rows, oovpure_rows, newent_rows = set(), set(), set(), set()
for t in range(3):
    for i in range(NE):
        for c in range(NC):
            r = t * NE * NC + i * NC + c
            if c >= 6:
                oov_rows.add(r)
                if i >= N_SEEN_ENT:
                    oovpure_rows.add(r)
            elif i >= N_SEEN_ENT:
                newent_rows.add(r)
            else:
                seen_rows.add(r)
chk('4.1 row counts', (len(seen_rows), len(oov_rows), len(newent_rows),
                       len(oovpure_rows)) == (738, 876, 576, 384),
    '%d/%d/%d/%d' % (len(seen_rows), len(oov_rows), len(newent_rows), len(oovpure_rows)))


def build_sclass(mdir):
    """Independent S_class rebuild (same 3166 recipe, separate code path)."""
    from safetensors import safe_open
    cfgm = json.load(io.open(os.path.join(mdir, 'config.json'), encoding='utf-8'))
    want = 'model.embed_tokens.weight' if cfgm.get('tie_word_embeddings') else 'lm_head.weight'
    W = None
    for sh in sorted(os.listdir(mdir)):
        if sh.endswith('.safetensors'):
            with safe_open(os.path.join(mdir, sh), framework='pt') as f:
                if want in set(f.keys()):
                    W = f.get_tensor(want).float().numpy().astype(np.float64)
                    break
    assert W is not None, mdir
    e2806 = json.load(io.open(os.path.join(
        RDIR, 'phase2806', 'qwen4_hierarchy', 'execution.json'), encoding='utf-8'))
    CATS = e2806['cats']
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True,
                                        trust_remote_code=True, use_fast=True)
    cents = []
    for cat in ['fruit', 'animal', 'metal', 'vehicle', 'country',
                'food', 'nature', 'furniture', 'tool', 'clothing']:
        vecs = []
        for w in CATS[cat]:
            ids = tok(' ' + w, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(w, add_special_tokens=False)['input_ids']
            if len(ids) == 1:
                vecs.append(W[int(ids[0])])
        assert len(vecs) >= 8, (cat, len(vecs))
        # 3166 recipe: centroid uses ALL single-token words of the class
        # (the [:8] cut only applies to class_targets/class_words, not S_class)
        cents.append(np.stack(vecs).mean(0))
    Cm = np.stack(cents)
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    S = np.stack([dW[i] / np.linalg.norm(dW[i]) for i in range(10)])
    return S.astype(np.float32).astype(np.float64)


MDIRS = {'qwen3-4b': os.path.join(ROOT, 'models', 'hf', 'qwen3-4b'),
         'qwen3-14b': os.path.join(ROOT, 'models', 'hf', 'Qwen3-14B'),
         'glm4-9b': os.path.join(ROOT, 'models', 'hf', 'glm4-9b-chat-hf')}
NPZS = {'qwen3-4b': os.path.join(P3169, 'collect_qwen3-4b.npz'),
        'qwen3-14b': os.path.join(P3169, 'collect_qwen3-14b.npz'),
        'glm4-9b': os.path.join(P3169, 'collect_glm4-9b.npz')}
max_rel = 0.0
for mk in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
    z = np.load(NPZS[mk])
    H16 = z['H']
    NTz, NPz, NH, D = H16.shape
    NL = NH - 1
    k_main = NL - 1
    chk('4.2 %s slot vs 3169 readout' % mk, k_main == int(R69['per_model'][mk]['readout']),
        'k_main=%d readout=%s' % (k_main, R69['per_model'][mk]['readout']))
    Y = H16[:, :, k_main, :].reshape(NTz * NPz, D).astype(np.float32).astype(np.float64)
    S = build_sclass(MDIRS[mk])
    QS, _ = np.linalg.qr(S.T)
    def pfrac(rows):
        sub = Y[sorted(rows)]
        num = ((sub @ QS) ** 2).sum(1)
        den = (sub ** 2).sum(1)
        return float((num / den).mean())
    f_seen, f_oov = pfrac(seen_rows), pfrac(oov_rows)
    ratio = f_oov / f_seen
    got = Res['per_model'][mk]['slots']['k_main']
    rel = abs(ratio - got['ratio_S']) / got['ratio_S']
    max_rel = max(max_rel, rel)
    chk('4.3 %s ratio_S recompute' % mk, rel < 1e-9,
        'rec=%.9f vs res=%.9f rel=%.2e' % (ratio, got['ratio_S'], rel))
    rel2 = abs(f_seen - got['f_S_seen']) / got['f_S_seen']
    chk('4.4 %s f_S_seen recompute' % mk, rel2 < 1e-9, 'rel=%.2e' % rel2)
    # verdict gate logic re-derivation
    cls = ('encoding_missing' if ratio < 0.5 else
           'readout_missing' if ratio > 0.8 else 'mixed')
    chk('4.5 %s cls re-derive' % mk, cls == got['cls'], cls)
    del z, H16, Y
clses = [Res['per_model'][m]['slots']['k_main']['cls'] for m in
         ('qwen3-4b', 'qwen3-14b', 'glm4-9b')]
main = clses[0] if len(set(clses)) == 1 else 'mixed_across_models'
chk('4.6 main_cls re-derive', main == Res['overall']['main_cls'], main)

# ---------- 5. crosscheck anchors ----------
for mk in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
    mk66 = {'glm4-9b': 'glm4'}.get(mk, mk)
    anc = float(R66['per_model'][mk66]['census']['K_readout__S_class']['top1_deg'])
    cc = Res['crosscheck'][mk]
    chk('5.1 %s crosscheck anchor matches 3166' % mk, abs(cc['anchor_deg'] - anc) < 1e-9,
        '%.3f' % anc)
    chk('5.2 %s crosscheck drift within gate' % mk, cc['drift_deg'] < 1e-3,
        '%.2e' % cc['drift_deg'])
    chk('5.3 %s drift consistent' % mk,
        abs(abs(cc['recomputed_deg'] - cc['anchor_deg']) - cc['drift_deg']) < 1e-12)

# ---------- 6. gap ledger v1.2 immutability ----------
chk('6.1 version 1.2', GL12['version'] == '1.2', GL12['version'])
for k in ('schema', 'provenance', 'created', 'appendix'):
    chk('6.2 %s byte-identical' % k,
        json.dumps(GL11[k], ensure_ascii=False, sort_keys=True) ==
        json.dumps(GL12[k], ensure_ascii=False, sort_keys=True))
for a, b in zip(GL11['gaps'], GL12['gaps']):
    kb = {k: v for k, v in b.items() if k != 'mechanism_note'}
    same = json.dumps(a, ensure_ascii=False, sort_keys=True) == \
        json.dumps(kb, ensure_ascii=False, sort_keys=True)
    chk('6.3 %s unchanged' % a['id'], same)
g4 = [g for g in GL12['gaps'] if g['id'] == 'GAP-4'][0]
chk('6.4 GAP-4 mechanism_note present', 'mechanism_note' in g4)
chk('6.5 mechanism_note carries ratio', '0.7511/0.7167/0.8205' in g4['mechanism_note'])
chk('6.6 mechanism_note carries verdict', 'mixed_across_models' in g4['mechanism_note'])
chk('6.7 GAP-4 main fields unchanged', g4['status'] == 'quantified_collapse' and
    g4['prereg']['status'] == 'executed_collapse_confirmed')

# ---------- 7. five-write read-back ----------
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
led = json.load(io.open(LEDGER, encoding='utf-8'))
e3171 = [m for m in led['measurements'] if m.get('phase') == 3171]
chk('7.1 ledger 3171 single entry', len(e3171) == 1, 'n=%d' % len(led['measurements']))
chk('7.2 ledger n>=323', len(led['measurements']) >= 323)
chk('7.3 ledger verdict has ratio', rat_ok if (rat_ok := ('0.7511/0.7167/0.8205' in
    e3171[0]['verdict'])) else False)
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk('7.4 MEMO Phase 3171', '## Phase 3171' in memo2)
chk('7.5 MEMO res sha', Res['res_sha8'] in memo2)
chk('7.6 MEMO seal sha', Res['seal_sha8'] in memo2)
chk('7.7 MEMO prereg 3172', '预注册 3172' in memo2)
d2 = io.open(DAILY, encoding='utf-8').read()
chk('7.8 daily 3171', '3171 谱外崩塌机制定位' in d2)
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk('7.9 MEMORY 3171', '3171 谱外崩塌机制定位' in w2)
chk('7.10 MEMO gap ledger v1.2 sha', '68c2bf21' in memo2)

# ---------- 8. smoke structure ----------
chk('8.1 smoke is smoke', SMK['smoke'] is True and Res['smoke'] is False)
chk('8.2 smoke 4b only', list(SMK['per_model'].keys()) == ['qwen3-4b'])
chk('8.3 smoke ratio finite', 0.0 < SMK['per_model']['qwen3-4b']['slots']['k_main']['ratio_S'] < 10)

txt = '\n'.join(R_) + '\nTOTAL PASS=%d FAIL=%d' % (N_OK[0], N_FAIL[0])
with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write(txt + '\n')
print(txt[-600:])
assert N_FAIL[0] == 0, ('verify failures', N_FAIL[0])
