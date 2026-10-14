# -*- coding: utf-8 -*-
"""Phase 3234 (G5-G30): GAP-5 upstream-lever scan — compute + seal arm.

Per prereg in MEMO Phase 3233 tail (design d74b6484, frozen before any
observation; default arm (a)): layer-wise residual patch h_L += dv at
the OUTPUT of block L (hook on model.model.layers[L], last token),
FULL downstream recomputation (blocks L+1..NL-1 + final norm + W_U via
the normal forward), judgment J = sign(margin isa-set minus hasa-set)
on exact recomputed logits, flip toward R = [J0 == not R] and
[J_after == R].

Clarifications logged BEFORE any observation (design gate texts verbatim):
  C1 layer semantics: hook at block-L OUTPUT = design "patch h_L, then
     recompute layers > L". Grid L in {0..NL-1}. The 3228 post-norm
     pre-logit slot (NH-1) is NOT re-scanned (F20 measured it); it
     serves as the m0 anchor source instead.
  C2 injection scale: dv = unit(u) * alpha * rms(h_row^L), h_row^L =
     the baseline block-L output state of that row (per-layer relative
     dose; 3228 used the pre-logit-state rms at the last layer — that
     case is not re-run here).
  C3 14B precision: bf16 resident is infeasible (29.55 GB > 16 GB
     VRAM; CPU offload infeasible, RAM 33.7 GB — Q05/Q12 precedent)
     => 14B runs nf4 (bitsandbytes). Double-precision bridge: 4B runs
     the SAME protocol twice (bf16 + nf4) on the coarse peak layer, so
     the nf4 perturbation on flip rates is bounded on-model and the
     bound travels to 14B (Q05 lesson 47 adapted: same-model bridge
     impossible on 14B, cross-instance 4B bridge carries the bound).
     4B bf16 baseline H is asserted BITWISE vs the sealed caches; 14B
     baseline agreement is recorded (descriptive), not asserted.
  C4 forward modes: 4B bf16 row-by-row (no padding — bitwise anchor
     path, identical to the collector); 14B nf4 row-by-row too
     (uniform semantics; speed measured at SMOKE).
  C5 peak layer: L*_dir = argmax_L pooled(4 folds) flip_rate(dir, a=1).
     Fine band = {L*-1, L*, L*+1} per direction.
  C6 nulls at the peak layer ONLY (design budget line "x11 on coarse
     argmax layer only"; the design D3 parenthetical "per layer" is
     downgraded accordingly, recorded): random norm-matched x10 pinned
     seeds + orthogonalized x2, at L*_dir.
  C7 candidate scope: coarse scan = C1 u_R only (2 dirs, design
     budget "NL x 2 dirs"); fine scan adds C2 u* (UL_amp) and C3 dR /
     C4 dW (peak layer, alpha=1, descriptive); C5 k*-band projection
     is DEFERRED (3152 subspace basis not sealed in npz — recorded as
     scope limit).
  C8 fine-scan rows = 3169 template t=0 (design "seed-0 rows"; 3228
     C1: first dim is 3 TEMPLATES).
Gate APPLICATION is deferred to 3235 per the 3233 chain; this arm
seals flip matrices + nulls + amplitude data only.
SMOKE (P3234_SMOKE=1): qwen3-4b bf16 only; coarse subgrid L step 6;
fine band collapsed to {L*}; all device anchors fully run.
"""
import datetime
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3194_g5d2_ftr26_heldout_upgrade as p94  # sealed recipes

ROOT = p94.ROOT
SRC_DIR = p94.SRC_DIR
NAME = 'g5g30_gap5_upstream_lever_scan'
OUTDIR = os.path.join(SRC_DIR, 'phase3234', NAME)
SMOKE = os.environ.get('P3234_SMOKE') == '1'
PFX = 'smoke_' if SMOKE else ''
MODELS = ['qwen3-4b'] if SMOKE else ['qwen3-4b', 'qwen3-14b']

LOG = []


def log(s):
    LOG.append(s)
    print(s, flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def csha8(obj):
    return hashlib.sha256(json.dumps(
        obj, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    ).hexdigest()[:8]


os.makedirs(OUTDIR, exist_ok=True)
bad = []

ALPHAS_FINE = (0.5, 1.0, 2.0)
NULL_SEEDS = [20260400 + i for i in range(10)]
TOK_ISA = 101158
TOK_HASA = 18830
CLS_NAMES = ['水果', '动物', '交通工具', '家具', '金属', '颜色',
             '乐器', '天气', '运动', '电器']
HUB = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
WU_SOURCES = {
    'qwen3-4b': dict(
        path=os.path.join(ROOT, 'models', 'hf', 'qwen3-4b',
                          'model-00001-of-00003.safetensors'),
        key='model.embed_tokens.weight', sha8='328a91d3'),
    'qwen3-14b': dict(
        path=glob.glob(os.path.join(HUB, 'models--Qwen--Qwen3-14B',
                                    'snapshots', '*'))[0]
             + os.sep + 'model-00008-of-00008.safetensors',
        key='lm_head.weight', sha8='0d6b9229'),
}
SNAP = {'qwen3-4b': glob.glob(os.path.join(HUB, 'models--Qwen--Qwen3-4B-Base',
                                           'snapshots', '*'))[0],
        'qwen3-14b': glob.glob(os.path.join(HUB, 'models--Qwen--Qwen3-14B',
                                            'snapshots', '*'))[0]}
TOKPATH = {'qwen3-4b': os.path.join(ROOT, 'models', 'hf', 'qwen3-4b',
                                    'tokenizer.json'),
           'qwen3-14b': os.path.join(ROOT, 'models', 'hf', 'Qwen3-14B',
                                     'tokenizer.json')}
MODEL_DIR = {'qwen3-4b': os.path.join(ROOT, 'models', 'hf', 'qwen3-4b'),
             'qwen3-14b': os.path.join(ROOT, 'models', 'hf',
                                       'Qwen3-14B-bnb-nf4')}
CACHE_3154 = {'qwen3-4b': 'bb99438f', 'qwen3-14b': '6d142b9d'}
CACHE_3169 = {'qwen3-4b': '36eb4ff0', 'qwen3-14b': '97f98575'}
CACHE_3157 = {'qwen3-4b': '95a25965', 'qwen3-14b': '9552086d'}
ANCHORS_3230_NPZ = 'd430d29e'
ANCHORS_3225_NPZ = 'ac5d4d74'
ANCHORS_3232_NPZ = '0bbe1f42'

# ---------- 3154 prompt reconstruction (MAT verbatim from 3154) ----------
TOPICS = ['天气', '运动', '饮食', '学习', '植物', '市场']
TOPICS_HO = ['睡眠', '冷链']
LANGS = ['zh', 'en']
STYLES = ['formal', 'casual']
LOGICS = ['con', 'contra']
CUT_FRACS = (0.20, 0.45, 0.70, 0.90)
HO_CUT_IDX = 1
MAT = {
    ('天气', 'zh', 'formal'): ('研究区经历了长达一周的强降雨过程', '土壤墒情明显改善', '土壤墒情明显恶化'),
    ('天气', 'zh', 'casual'): ('这雨下了一整个礼拜都没停过', '地里的水管够了', '地里旱得裂口了'),
    ('天气', 'en', 'formal'): ('The region recorded persistent heavy rainfall over the past week',
                              'and soil moisture improved markedly', 'and soil moisture declined markedly'),
    ('天气', 'en', 'casual'): ('it rained hard all week long',
                              'so the yard is damp and soft', 'so the yard is dry and hard'),
    ('运动', 'zh', 'formal'): ('运动员完成了为期三个月的高原集训', '有氧耐力显著提升', '有氧耐力显著下降'),
    ('运动', 'zh', 'casual'): ('他天天早上五点起来跑步', '体力比以前好多了', '体力比以前差多了'),
    ('运动', 'en', 'formal'): ('The athletes completed a three-month high-altitude training camp',
                              'and their aerobic endurance improved', 'and their aerobic endurance declined'),
    ('运动', 'en', 'casual'): ('she trains at the gym every single day',
                              'so she got stronger and faster', 'so she got slower and weaker'),
    ('饮食', 'zh', 'formal'): ('受试者连续四周坚持低盐饮食方案', '血压水平稳步回落', '血压水平持续攀升'),
    ('饮食', 'zh', 'casual'): ('他这一个月顿顿都吃清水菜', '秤上的数字降了些', '秤上的数字涨了些'),
    ('饮食', 'en', 'formal'): ('The participants followed a low-salt diet for four consecutive weeks',
                              'and their blood pressure decreased', 'and their blood pressure increased'),
    ('饮食', 'en', 'casual'): ('he skipped sugary drinks for a month',
                              'so his jeans fit him again', 'so his jeans got way tighter'),
    ('学习', 'zh', 'formal'): ('实验组采用了间隔重复的记忆策略', '词汇测验成绩提高', '词汇测验成绩下滑'),
    ('学习', 'zh', 'casual'): ('她每天睡前背二十个单词', '听写基本都能过关', '听写回回都不及格'),
    ('学习', 'en', 'formal'): ('The experimental group used a spaced-repetition memory strategy',
                              'and their vocabulary scores rose', 'and their vocabulary scores fell'),
    ('学习', 'en', 'casual'): ('he crammed all night before the quiz',
                              'and he passed it easily', 'and he bombed it completely'),
    ('植物', 'zh', 'formal'): ('温室保持了恒定的光照与湿度', '幼苗长势整齐健壮', '幼苗成片枯萎发黄'),
    ('植物', 'zh', 'casual'): ('阳台那盆绿萝天天浇水施肥', '叶子绿得直发亮', '叶子黄得直掉渣'),
    ('植物', 'en', 'formal'): ('The greenhouse maintained constant light and humidity levels',
                              'and the seedlings grew vigorously', 'and the seedlings withered rapidly'),
    ('植物', 'en', 'casual'): ('she repotted the old ficus last spring',
                              'and it grew lots of new leaves', 'and it lost lots of old leaves'),
    ('市场', 'zh', 'formal'): ('节前冷链物流运力全面提升', '生鲜价格保持平稳', '生鲜价格大幅上涨'),
    ('市场', 'zh', 'casual'): ('早市新来了几家卖菜的摊位', '菜价比往常便宜些', '菜价比往常贵得多'),
    ('市场', 'en', 'formal'): ('Cold-chain capacity expanded ahead of the holiday season',
                              'and produce prices stayed stable', 'and produce prices rose sharply'),
    ('市场', 'en', 'casual'): ('the night market added a bunch of stalls',
                              'so the lines got even longer', 'so the lines got much shorter'),
    ('睡眠', 'zh', 'formal'): ('受试者连续两周保持规律作息', '白天精神状态饱满', '白天哈欠连绵不断'),
    ('睡眠', 'zh', 'casual'): ('他这两天十点就爬上床睡了', '早上起来特有劲', '早上起来特犯困'),
    ('睡眠', 'en', 'formal'): ('The participants kept a regular sleep schedule for two weeks',
                              'and their daytime alertness improved', 'and their daytime alertness dropped'),
    ('睡眠', 'en', 'casual'): ('she finally quit late-night scrolling',
                              'and woke up feeling refreshed', 'and woke up feeling groggy'),
    ('冷链', 'zh', 'formal'): ('新冷库投用后全程温控不断链', '疫苗批次全部合格', '疫苗批次大量报废'),
    ('冷链', 'zh', 'casual'): ('那家店进了台新的冰柜', '雪糕再也不化了', '雪糕化得软塌塌'),
    ('冷链', 'en', 'formal'): ('The new cold storage kept the temperature stable throughout',
                              'and the vaccine batches passed', 'and the vaccine batches failed'),
    ('冷链', 'en', 'casual'): ('the delivery van got a freezer upgrade',
                              'so the fish arrived still fresh', 'so the fish arrived half spoiled'),
}


def variant_texts(topic, lang, style, tail):
    A, Bcon, Bcot = MAT[(topic, lang, style)]
    B = tail
    out = []
    if lang == 'zh':
        n = len(A)
        for frac in CUT_FRACS:
            c = int(round(frac * n))
            assert 1 <= c < n
            pre = A[:c]
            prompt = A[:c] + '\uff0c' + A[c:] + B
            out.append((prompt, pre))
    else:
        ws = A.split(' ')
        n = len(ws)
        for frac in CUT_FRACS:
            w = int(round(frac * n))
            assert 1 <= w < n
            pre = ' '.join(ws[:w])
            prompt = pre + ',' + ' ' + ' '.join(ws[w:]) + ' ' + B
            out.append((prompt, pre))
    return out


def build_rows_3154():
    rows = []
    for ti, tp in enumerate(TOPICS):
        for l, lg in enumerate(LANGS):
            for s, st in enumerate(STYLES):
                A_, Bcon_, Bcot_ = MAT[(tp, lg, st)]
                for g, gc in enumerate(LOGICS):
                    vts = variant_texts(tp, lg, st, Bcon_ if g == 0 else Bcot_)
                    for ci, (prompt, pre) in enumerate(vts):
                        rows.append(dict(ti=ti, l=l, s=s, g=g, ci=ci, ho=0,
                                         prompt=prompt, prefix=pre))
    base_ti = len(TOPICS)
    for ti, tp in enumerate(TOPICS_HO):
        for l, lg in enumerate(LANGS):
            for s, st in enumerate(STYLES):
                A_, Bcon_, Bcot_ = MAT[(tp, lg, st)]
                for g, gc in enumerate(LOGICS):
                    vts = variant_texts(tp, lg, st, Bcon_ if g == 0 else Bcot_)
                    prompt, pre = vts[HO_CUT_IDX]
                    rows.append(dict(ti=base_ti + ti, l=l, s=s, g=g,
                                     ci=HO_CUT_IDX, ho=1,
                                     prompt=prompt, prefix=pre))
    return rows


ROWS_3154 = build_rows_3154()
assert len(ROWS_3154) == 208, len(ROWS_3154)

# ---------- token-level verification of the reconstruction (both models) ----------
from tokenizers import Tokenizer
TOK = {}
for m in MODELS:
    tk = Tokenizer.from_file(TOKPATH[m])
    TOK[m] = tk
    assert tk.encode('是一种', add_special_tokens=False).ids == [TOK_ISA]
    assert tk.encode('有', add_special_tokens=False).ids == [TOK_HASA]
log('STEP0 tokenizers OK (%s)' % MODELS)

for m in MODELS:
    z54 = np.load(os.path.join(SRC_DIR, r'phase3154\g1p4_mfd_multifactor_disentangle',
                               m, 'collect.npz'))
    tags = {k: z54[k].astype(int) for k in ('g', 'l', 's', 'ti', 'ci', 'ho', 'd')}
    n_tag_bad = 0
    n_d_bad = 0
    for i, r in enumerate(ROWS_3154):
        if (tags['g'][i] != r['g'] or tags['l'][i] != r['l'] or
                tags['s'][i] != r['s'] or tags['ti'][i] != r['ti'] or
                tags['ci'][i] != r['ci'] or tags['ho'][i] != r['ho']):
            n_tag_bad += 1
        npt = len(TOK[m].encode(r['prompt'], add_special_tokens=False).ids)
        npp = len(TOK[m].encode(r['prefix'], add_special_tokens=False).ids)
        if tags['d'][i] != npt - npp:
            n_d_bad += 1
    log('3154 reconstruction %s: tag mismatches %d/208, d mismatches %d/208'
        % (m, n_tag_bad, n_d_bad))
    assert n_tag_bad == 0, ('3154 row-order/tags', m, n_tag_bad)
    assert n_d_bad == 0, ('3154 d column (token count)', m, n_d_bad)
    del z54
log('3154 prompt reconstruction VERIFIED (tags + d column, both models)')

# ---------- freeze BEFORE any observation ----------
COARSE_STEP = 6 if SMOKE else 1
fz = {
    'phase': '3234', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3233 tail (design d74b6484; default '
              'arm (a) compute+seal, GPU layer scan)',
    'design_basis': {'design_doc': 'g5g30_upstream_lever_design_v1.json',
                     'design_content_sha8': 'd74b6484',
                     'design_file_sha8': '6a53616c',
                     'design_freeze_sha8': '7e201c78'},
    'models': MODELS,
    'precision': {'qwen3-4b': 'bf16 (LOCAL collector weights, bitwise '
                               'anchor path)',
                  'qwen3-14b': 'nf4 (bitsandbytes; bf16 resident '
                               'infeasible 29.55GB>16GB VRAM, offload '
                               'infeasible RAM 33.7GB — Q05/Q12; 4B '
                               'bf16-vs-nf4 bridge bounds the quantization '
                               'perturbation, bound travels to 14B)'},
    'clarifications': {
        'C1': 'hook at block-L OUTPUT (design "patch h_L, recompute '
              'layers > L"); grid L in {0..NL-1}; 3228 post-norm slot '
              'NH-1 not re-scanned (F20), serves as m0 anchor',
        'C2': 'dv = unit(u)*alpha*rms(h_row^L), h_row^L = baseline '
              'block-L output state (per-layer relative dose)',
        'C3': '4B bf16-vs-nf4 bridge on the coarse peak layer (Q05 '
              'lesson 47 adapted); R1 revision: 14B runs FULL bf16 '
              '(models/hf/Qwen3-14B, the 3169 collection path verbatim) '
              'so A1 is a hard bitwise anchor for BOTH models — the '
              'first FULL attempt crashed at 14B load because the hub '
              'snapshot holds shard 8 only (3225/3228 CPU-readout '
              'assumption leaked into this arm)',
        'C4': 'row-by-row forward, no padding, both models (collector-'
              'identical path)',
        'C5': 'peak layer L*_dir = argmax_L pooled flip_rate(dir, a=1); '
              'fine band {L*-1, L*, L*+1}',
        'C6': 'nulls at peak layer only (budget line; D3 "per layer" '
              'downgraded, recorded): random norm-matched x10 pinned '
              'seeds + orthogonalized x2',
        'C7': 'coarse = C1 u_R 2 dirs; fine adds C2 u* (UL_amp) + C3 dR '
              '/ C4 dW (peak, a=1, descriptive); C5 k*-band projection '
              'DEFERRED (3152 subspace basis not sealed)',
        'C8': 'fine rows = 3169 template t=0, 730 rows',
        'C9': 'R1 load-path revision (registered BEFORE any gate '
              'observation; first FULL attempt freeze 81c99f99 aborted '
              'at 14B from_pretrained, no sealed artifacts produced): '
              "MODEL_DIR['qwen3-14b'] hub snapshot -> local full bf16; "
              'nf4 bridge retained as 4B-only record',
        'C10': 'R2 load-path revision SUPERSEDING C9 (registered before '
               'any gate observation; second FULL attempt freeze '
               '71e39697 segfaulted at 14B bf16 weight load): machine '
               'RAM 31.4 GB / VRAM 16 GB makes 14B bf16 (~28.8 GB) '
               'physically infeasible locally; 14B runs the 3161/3164 '
               'nf4-pre checkpoint (Qwen3-14B-bnb-nf4, quantization '
               'baked in config) => A1 hard bitwise only for 4B, 14B '
               'anchors recorded descriptively (3164 family, same-'
               'precision self-consistency precedent); J0 anchors stay '
               'cache-based bitwise for BOTH models; 4B nf4 bridge '
               'retained (quantifies nf4-vs-bf16 perturbation family '
               'that applies to the 14B data)',
    },
    'intervention': 'residual patch at block-L output, last token; full '
                    'downstream recomputation (blocks L+1..NL-1 + norm '
                    '+ W_U) via normal forward with forward hook',
    'judgment': {'isa_set': [TOK_ISA], 'hasa_set': [TOK_HASA],
                 'formula': 'J = sign(m_isa - m_hasa) on recomputed '
                            'logits; flip toward R = [J0==not R] & '
                            '[J_after==R]; J0 from cached NH-1 slot '
                            'linear map (3228 m0 anchor)'},
    'directions': {'u_R': 'T2 recipe verbatim (3228 code path: 3157 '
                          'cache dC at kout, dCn, unit_mean by relation)',
                   'ustar': 'sealed 3225 npz %s, key {m}_ustar'
                            % ANCHORS_3225_NPZ,
                   'dR': 'sealed 3230 npz %s, key {m}_dR_unit'
                         % ANCHORS_3230_NPZ,
                   'dW': 'unit(W_U[isa] - W_U[hasa]) from C9 sources'},
    'grids': {'coarse': 'L in {0..NL-1} step %d, alpha=1, 3154 panel '
                        '208 rows, dirs {u_isa, u_hasa}' % COARSE_STEP,
              'fine': 'band {L*-1, L*, L*+1} x alpha {0.5,1,2} x dirs '
                      '{u_isa,u_hasa} + u* (band, alpha=1) + dR/dW '
                      '(peak, alpha=1), 3169 t=0 730 rows',
              'bridge': '4B nf4 full coarse grid (same as 4B bf16)',
              'nulls': 'peak layer: 10 random norm-matched seeds + 2 '
                       'orthogonalized, alpha=1, both panels'},
    'null_seeds': NULL_SEEDS,
    'device_gates': {
        'a_4b_bitwise': '4B bf16 baseline hidden_states fp16 == sealed '
                        'cache H bitwise (3154 all 208 rows all slots + '
                        '3169 t=0 all 730 rows all slots)',
        'a_3230_m0': 'baseline margins from cache == 3230 npz seen_m0 '
                     'bitwise for BOTH models (same H x WT product, '
                     'cache-based)',
        'a_14b_recorded': '14B nf4-pre (3161/3164 family) baseline vs '
                          'cache: recorded descriptively (argmax '
                          'agreement, max|d|), not asserted '
                          '(C10: bf16 infeasible on this machine)',
        'a_c6v3_4b': '4B direct-map class readout vs sealed 3169 marg '
                     'argmax agreement >= 0.98; 14B same readout '
                     'recorded descriptively',
        'a_nf4_bridge_4b': '4B nf4 on-the-fly bridge at peak layers: '
                           'recorded descriptively (quantization '
                           'perturbation quantification, 3164 family)',
    },
    'gate_application': 'DEFERRED to 3235 per 3233 chain; this arm seals '
                        'matrices + nulls + amplitude data only',
    'ledger_state': [385, '86f04998'],
    'sources': {'caches_3154': CACHE_3154, 'caches_3169': CACHE_3169,
                'caches_3157': CACHE_3157,
                'npz_3225': ANCHORS_3225_NPZ, 'npz_3230': ANCHORS_3230_NPZ,
                'npz_3232': ANCHORS_3232_NPZ},
    'smoke_note': ('SMOKE: 4b only, coarse step 6, fine band {L*} only, '
                   'all anchors fully run' if SMOKE else 'FULL'),
}
fz['sha8'] = csha8(dict(fz))
with open(os.path.join(OUTDIR, PFX + 'execution.json'), 'w',
          encoding='utf-8', newline='') as f:
    json.dump(fz, f, ensure_ascii=False, indent=1)
log('== Phase 3234 %s %s == freeze %s (before any observation)'
    % (NAME, 'SMOKE' if SMOKE else 'FULL', fz['sha8']))

# ---------- source sha asserts ----------
for tag, rel, exp in [
    ('p3154_4b', r'phase3154\g1p4_mfd_multifactor_disentangle\qwen3-4b\collect.npz', 'bb99438f'),
    ('p3154_14b', r'phase3154\g1p4_mfd_multifactor_disentangle\qwen3-14b\collect.npz', '6d142b9d'),
    ('p3169_4b', r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz', '36eb4ff0'),
    ('p3169_14b', r'phase3169\g5a6_oov_panel\collect_qwen3-14b.npz', '97f98575'),
    ('p3157_4b', r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz', '95a25965'),
    ('p3157_14b', r'phase3157\g2p2_transform_algebra_commutator\qwen3-14b\collect.npz', '9552086d'),
    ('p3225', r'phase3225\g5g25_a1_behavioral_band_compute\a1_behavioral_band.npz', 'ac5d4d74'),
    ('p3230', r'phase3230\g5g27_relcontrast_efficacy\e3_relcontrast_flip.npz', 'd430d29e'),
    ('p3232', r'phase3232\g5g29_gd_amp_class_robustness\gd_amp_class_expansion.npz', '0bbe1f42'),
]:
    got = sha8_file(os.path.join(SRC_DIR, rel))
    assert got == exp, (tag, got, exp)
led = json.load(open(os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json'),
                     encoding='utf-8'))
assert (len(led['measurements']), led['ledger_sha256_8']) == (385, '86f04998')
log('source sha anchors x9 + ledger n=385 post=86f04998 OK')

# ---------- 3230 m0 anchor data ----------
Z30 = np.load(os.path.join(SRC_DIR, r'phase3230\g5g27_relcontrast_efficacy'
                           r'\e3_relcontrast_flip.npz'))
Z25 = np.load(os.path.join(SRC_DIR, r'phase3225\g5g25_a1_behavioral_band_compute'
                           r'\a1_behavioral_band.npz'))
Z32 = np.load(os.path.join(SRC_DIR, r'phase3232\g5g29_gd_amp_class_robustness'
                           r'\gd_amp_class_expansion.npz'))
E3228 = json.load(open(os.path.join(SRC_DIR, r'phase3228\g5g26_e3_compute_seal'
                                    r'\result.json'), encoding='utf-8'))
E_CLS = {m: E3228['models'][m]['gd_amp']['E_cls'] for m in
         ('qwen3-4b', 'qwen3-14b')}

npz_payload = {}
out = {}


def row_rms(h):
    return float(np.sqrt((h.astype(np.float32) ** 2).mean()))


for m in MODELS:
    log('=== %s ===' % m)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, \
        BitsAndBytesConfig
    torch.manual_seed(0)
    cfg = json.load(open(os.path.join(MODEL_DIR[m], 'config.json'),
                         encoding='utf-8'))
    NL = int(cfg['num_hidden_layers'])
    D = int(cfg['hidden_size'])
    NH = NL + 1
    fin = NH - 1

    tk = AutoTokenizer.from_pretrained(MODEL_DIR[m], trust_remote_code=True)

    def ids_of(text):
        return tk(text, add_special_tokens=False)['input_ids']

    # --- directions (3228 code path verbatim) ---
    A57 = p94.ANCH_3157[p94.ANCH_KEY[m]]
    z57 = np.load(p94.SRC[p94.NPZ57_KEY[m]])
    H57 = z57['H'].astype(np.float32)
    assert H57.shape == (128, NH, D)
    EI = z57['ei'].astype(int)
    RL = z57['rel'].astype(int)
    PL = z57['pol'].astype(int)
    CX = z57['ctx'].astype(int)
    IDX = {(int(EI[i]), int(RL[i]), int(PL[i]), int(CX[i])): i
           for i in range(128)}
    dC = []
    for ei in range(16):
        for rl in range(2):
            for pl in range(2):
                dC.append(H57[IDX[(ei, rl, pl, 1)], A57['kout']]
                          - H57[IDX[(ei, rl, pl, 0)], A57['kout']])
    dC = np.array(dC)
    dCn = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-18)
    cosM = dCn @ dCn.T
    tri = cosM[np.triu_indices(len(dC), 1)]
    assert abs(float(tri.mean()) - A57['tc_mean_cos']) < 1e-9
    RL_rows = np.array([(i // 4, (i % 4) // 2, i % 2) for i in range(64)])
    v_isa = p94.unit_mean(dCn[RL_rows[:, 1] == 0])
    v_hasa = p94.unit_mean(dCn[RL_rows[:, 1] == 1])
    E83 = json.load(open(p94.SRC['p3183_result'], encoding='utf-8'))['per_model'][m]
    assert max(abs(float(v_isa @ v_hasa) - E83['cos_rl']),
               abs(float(np.linalg.norm(v_isa - v_hasa)) - E83['norm_dv'])) < 1e-9
    ustar = Z25[m + '_ustar'].astype(np.float32)
    dR = Z30[m + '_dR_unit'].astype(np.float32)
    log('  directions OK: v_isa/v_hasa (tc+3183 anchors), u*, dR')

    # --- weights (isa/hasa rows only, C9 sources) ---
    from safetensors import safe_open
    src = WU_SOURCES[m]
    assert sha8_file(src['path']) == src['sha8']
    with safe_open(src['path'], framework='pt') as f:
        W_U2 = f.get_slice(src['key'])[[TOK_ISA, TOK_HASA]].float().numpy()
    dW = W_U2[0] - W_U2[1]
    dW = dW / (float(np.linalg.norm(dW)) + 1e-18)
    WT = W_U2.astype(np.float32)
    del W_U2

    # --- caches ---
    z54 = np.load(os.path.join(SRC_DIR, r'phase3154\g1p4_mfd_multifactor_disentangle',
                               m, 'collect.npz'))
    H54 = z54['H']
    tags54 = {k: z54[k].astype(int) for k in ('g', 'l', 's', 'ti', 'ci', 'ho')}
    del z54
    z69 = np.load(os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel',
                               'collect_%s.npz' % ('qwen3-4b' if m == 'qwen3-4b'
                                                   else 'qwen3-14b')))
    H69 = z69['H']
    MARG69 = z69['marg']
    del z69
    assert H54.shape == (208, NH, D) and H69.shape[0] == 3
    folds = [np.where((tags54['ci'] == c) & (tags54['ho'] == 0))[0]
             for c in range(4)]
    assert all(len(s) == 48 for s in folds)

    # J0 from cached NH-1 slot (3228 m0 anchor)
    H54f = H54[:, fin, :].astype(np.float32)
    m0_54 = H54f @ WT.T
    J0_54 = (m0_54[:, 0] > m0_54[:, 1]).astype(np.int8)
    m0_ref = Z30[m + '_seen_m0'].astype(np.float32)
    d_m0 = float(np.abs(m0_54 - m0_ref).max())
    assert d_m0 == 0.0, ('3230 m0 anchor', m, d_m0)
    H69f = H69[0, :, fin, :].astype(np.float32)     # template t=0
    m0_69 = H69f @ WT.T
    J0_69 = (m0_69[:, 0] > m0_69[:, 1]).astype(np.int8)
    log('  J0 anchors OK: 3230 seen_m0 bitwise (drift %.2e)' % d_m0)

    # --- checkpoint resume (crash-resilience, C10 family; keyed on
    # freeze so any design change invalidates) ---
    ckp = os.path.join(OUTDIR, 'ckpt_%s.json' % m)
    if not SMOKE and os.path.exists(ckp):
        c = json.load(open(ckp, encoding='utf-8'))
        if c['freeze'] == fz['sha8']:
            out[m] = c['out']
            zc = np.load(os.path.join(OUTDIR, 'ckpt_%s.npz' % m))
            for k in zc.files:
                npz_payload[k] = zc[k]
            log('  RESUMED %s from per-model checkpoint (freeze match)'
                % m)
            continue

    # --- model load (C10: 4B bf16 .to(cuda); 14B nf4-pre 3161/3164 path) ---
    if m == 'qwen3-4b':
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR[m], dtype=torch.bfloat16,
            trust_remote_code=True).to('cuda').eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR[m], dtype=torch.bfloat16,
            trust_remote_code=True).eval()
    assert model.config.num_hidden_layers == NL
    layers = model.model.layers
    log('  model loaded (%s)' % m)

    # --- baseline forwards + bitwise anchor (row-by-row) ---
    def baseline_row(ids):
        ii = torch.tensor([ids], device='cuda')
        with torch.no_grad():
            o = model(input_ids=ii, output_hidden_states=True)
        hs = np.stack([h[0, -1].float().detach().cpu().numpy()
                       for h in o.hidden_states], 0)
        lg = o.logits[0, -1].float().detach().cpu().numpy()
        del o
        return hs.astype(np.float16), lg[[TOK_ISA, TOK_HASA]]

    prompts_54 = [r['prompt'] for r in ROWS_3154]
    ids_54 = [ids_of(p) for p in prompts_54]
    ids_69 = [ids_of(p94.TPL[0].format(e=p94.ENTS[i], c=p94.CLASSES[c]))
              for (i, c) in p94.PAIRS]
    assert len(ids_69) == 730

    hs54 = np.zeros((208, NH, D), np.float16)
    lg54 = np.zeros((208, 2), np.float32)
    for i in range(208):
        hs54[i], lg54[i] = baseline_row(ids_54[i])
    if m == 'qwen3-4b':
        n_bad = int((hs54 != H54).any(axis=(1, 2)).sum())
        assert n_bad == 0, ('bitwise 3154', m, n_bad)
        log('  A1 4b bitwise 3154: 208/208 rows ALL slots equal')
    else:
        dd = np.abs(hs54[:, fin, :].astype(np.float32)
                    - H54[:, fin, :].astype(np.float32))
        m0_rec = hs54[:, fin].astype(np.float32) @ WT.T
        am_agree = float((m0_rec.argmax(1) == m0_54.argmax(1)).mean())
        log('  A2 14b nf4-pre vs cache (recorded, C10): final-slot '
            'max|d|=%.4f, baseline J agreement=%.4f'
            % (dd.max(), am_agree))
        out.setdefault('nf4_record', {})[m] = dict(
            final_slot_maxd=float(dd.max()), j_agree=float(am_agree))
    hs69 = np.zeros((730, NH, D), np.float16)
    lg69 = np.zeros((730, 2), np.float32)
    for i in range(730):
        hs69[i], lg69[i] = baseline_row(ids_69[i])
    if m == 'qwen3-4b':
        n_bad = int((hs69 != H69[0]).any(axis=(1, 2)).sum())
        assert n_bad == 0, ('bitwise 3169 t0', m, n_bad)
        log('  A1 4b bitwise 3169 t=0: 730/730 rows ALL slots equal')
    del lg54, lg69

    # C6v3 anchor: direct-map class readout vs sealed marg
    with safe_open(src['path'], framework='pt') as f:
        CLS_TOK = [tk.encode(c, add_special_tokens=False)[0]
                   for c in CLS_NAMES]
        assert len(set(CLS_TOK)) == 10
        WUC = f.get_slice(src['key'])[CLS_TOK].float().numpy()
    c0 = H69[0, :, fin, :].astype(np.float32) @ WUC.T
    cached = MARG69[0]
    agree = float((c0.argmax(1) == cached.argmax(1)).mean())
    if m == 'qwen3-4b':
        assert agree >= 0.98, ('C6v3', m, agree)
    log('  C6v3 anchor %s: argmax agreement %.4f%s'
        % (m, agree, '' if m == 'qwen3-4b'
           else ' (recorded, nf4-pre vs bf16 W_U)'))
    out.setdefault('c6v3', {})[m] = agree
    del WUC, c0

    # --- patch forward machinery ---
    def patch_row(ids, L, dv_row):
        h = torch.tensor([ids], device='cuda')
        dv_t = torch.tensor(dv_row, dtype=torch.bfloat16,
                            device='cuda')[None, None, :]
        handle = [None]

        def hook(mod, inp, o):
            hh = o[0] if isinstance(o, tuple) else o
            hh = hh.clone()
            hh[0, -1, :] += dv_t[0, 0]
            if isinstance(o, tuple):
                return (hh,) + o[1:]
            return hh

        handle[0] = layers[L].register_forward_hook(hook)
        try:
            with torch.no_grad():
                o = model(input_ids=h)
            lg = o.logits[0, -1].float().detach().cpu().numpy()
            del o
        finally:
            handle[0].remove()
        return lg

    # rms per layer per row from baseline hs (block-L output = slot L+1)
    def rms_L(hs_rows, L):
        # block-L output state = hidden_states[L+1] (slot L+1); for
        # L = NL-1 the slot NH-1 is post-norm (recorded caveat C1)
        slot = min(L + 1, fin)
        return np.sqrt((hs_rows[:, slot, :].astype(np.float32) ** 2)
                       .mean(axis=1))

    def flip_stats(J1, J0, folds):
        res = {}
        for fi, sl in enumerate(folds):
            for R, Rn in ((1, 'isa'), (0, 'hasa')):
                fl = (J0[sl] == (1 - R)) & (J1[sl].astype(np.int8) == R)
                res['%s|cut%d' % (Rn, fi)] = dict(
                    n=int(fl.sum()), rows=int(len(sl)),
                    rate=float(fl.sum() / len(sl)))
        return res

    DIRS = {'u_isa': v_isa, 'u_hasa': v_hasa}

    # ---------- coarse scan (3154, alpha=1) ----------
    LGRID = list(range(0, NL, COARSE_STEP))
    if LGRID[-1] != NL - 1:
        LGRID.append(NL - 1)
    coarse = {}
    for dname, u in DIRS.items():
        for L in LGRID:
            rmsL = rms_L(hs54, L)
            dv = (u[None, :] * (1.0 * rmsL)[:, None]).astype(np.float32)
            J1 = np.zeros(208, np.int8)
            for i in range(208):
                lg = patch_row(ids_54[i], L, dv[i])
                J1[i] = int(lg[TOK_ISA] > lg[TOK_HASA])
            key = '%s|L%d' % (dname, L)
            coarse[key] = flip_stats(J1, J0_54, folds)
            npz_payload['%s_%s_J1_L%d' % (m, dname, L)] = J1
            if L == LGRID[len(LGRID) // 2] or L == LGRID[0]:
                pooled_isa = float(np.mean([coarse[key]['isa|cut%d' % f]['rate']
                                            for f in range(4)]))
                log('  coarse %s: pooled isa flip=%.4f' % (key, pooled_isa))
    npz_payload[m + '_coarse_J0'] = J0_54

    # pooled rates -> peak layers
    pooled = {}
    for dname in DIRS:
        Rn = 'isa' if dname == 'u_isa' else 'hasa'
        pr = {}
        for L in LGRID:
            key = '%s|L%d' % (dname, L)
            pr[str(L)] = float(np.mean([coarse[key]['%s|cut%d' % (Rn, f)]['rate']
                                        for f in range(4)]))
        Lstar = max(pr, key=pr.get)
        pr['Lstar'] = int(Lstar)
        pooled[dname] = pr
        log('  peak %s: L*=%s pooled=%.4f' % (dname, Lstar, pr[Lstar]))

    # ---------- 14b nf4 record: none at coarse (bridge below) ----------

    # ---------- fine scan (3169 t=0, 730 rows) ----------
    if SMOKE:
        bands = {d: [pooled[d]['Lstar']] for d in DIRS}
        fine_alphas = (1.0,)
    else:
        bands = {d: sorted(set(x for x in (pooled[d]['Lstar'] - 1,
                                           pooled[d]['Lstar'],
                                           pooled[d]['Lstar'] + 1)
                                if 0 <= x <= NL - 1)) for d in DIRS}
        fine_alphas = ALPHAS_FINE
    fine = {}
    for dname, u in DIRS.items():
        for L in bands[dname]:
            rmsL = rms_L(hs69, L)
            for a in fine_alphas:
                dv = (u[None, :] * (a * rmsL)[:, None]).astype(np.float32)
                J1 = np.zeros(730, np.int8)
                mg = np.zeros((730, 2), np.float32)
                for i in range(730):
                    lg = patch_row(ids_69[i], L, dv[i])
                    mg[i] = lg[[TOK_ISA, TOK_HASA]]
                    J1[i] = int(lg[TOK_ISA] > lg[TOK_HASA])
                key = '%s|L%d|a%s' % (dname, L, a)
                nflip = int(((J0_69 == 0) & (J1 == 1)).sum()
                            + ((J0_69 == 1) & (J1 == 0)).sum())
                fine[key] = dict(flip={'tpl0': dict(n=nflip, rows=730,
                                                    rate=nflip / 730.0)},
                                 dD=float(((mg[:, 0] - mg[:, 1])
                                           - (m0_69[:, 0] - m0_69[:, 1])).mean()))
                npz_payload['%s_fine_%s_L%d_a%s_J1' % (m, dname, L, a)] = J1
                npz_payload['%s_fine_%s_L%d_a%s_marg' % (m, dname, L, a)] = mg
    npz_payload[m + '_fine_J0'] = J0_69

    # u* channel (UL_amp): band layers, alpha=1, per-class margin shift
    ustar_band = {}
    for L in bands['u_isa']:
        rmsL = rms_L(hs69, L)
        dv = (ustar[None, :] * (1.0 * rmsL)[:, None]).astype(np.float32)
        mg = np.zeros((730, 2), np.float32)
        for i in range(730):
            lg = patch_row(ids_69[i], L, dv[i])
            mg[i] = lg[[TOK_ISA, TOK_HASA]]
        d_shift = (mg[:, 0] - mg[:, 1]) - (m0_69[:, 0] - m0_69[:, 1])
        cls_of_row = np.array([p94.PAIRS[i][1] for i in range(730)])
        per_cls = [float(np.median(d_shift[cls_of_row == c]))
                   for c in range(10)]
        ustar_band[str(L)] = dict(per_class_med_shift=per_cls)
        npz_payload['%s_ustar_L%d_marg' % (m, L)] = mg
        log('  u* L%d per-class med shift: %s' % (
            L, '/'.join('%.3f' % x for x in per_cls)))
    # gate statistic (4 OOV classes, 3228 E_cls) — sealed for 3235
    ul_amp = {}
    for L, rec in ustar_band.items():
        oov_cls = [6, 7, 8, 9]
        amps = [rec['per_class_med_shift'][c] for c in oov_cls]
        ec = E_CLS[m]
        rx = np.argsort(np.argsort(amps)).astype(np.float64)
        ry = np.argsort(np.argsort(ec)).astype(np.float64)
        rho = float(np.corrcoef(rx, ry)[0, 1])
        ul_amp[L] = dict(spearman_4oov=rho, amps=amps, E_cls=ec)
        log('  UL_amp L%s: spearman(4 OOV)=%.4f' % (L, rho))
    # 10-class report-only vs 3232 collapse
    collapse10 = Z32[m + '_collapse_amp'].astype(np.float64)
    for L, rec in ustar_band.items():
        rx = np.argsort(np.argsort(rec['per_class_med_shift'])).astype(np.float64)
        ry = np.argsort(np.argsort(collapse10)).astype(np.float64)
        ul_amp[L]['spearman_10cls_report_only'] = float(
            np.corrcoef(rx, ry)[0, 1])

    # dR / dW at peak (descriptive, alpha=1)
    extra = {}
    for tag_name, u in (('dR', dR), ('dW', dW)):
        L = pooled['u_isa']['Lstar']
        rmsL = rms_L(hs69, L)
        dv = (u[None, :] * (1.0 * rmsL)[:, None]).astype(np.float32)
        J1 = np.zeros(730, np.int8)
        for i in range(730):
            lg = patch_row(ids_69[i], L, dv[i])
            J1[i] = int(lg[TOK_ISA] > lg[TOK_HASA])
        nf = int(((J0_69 == 0) & (J1 == 1)).sum()
                 + ((J0_69 == 1) & (J1 == 0)).sum())
        extra[tag_name] = dict(L=L, flip_n=nf, rate=nf / 730.0)
        log('  %s peak L%d: flip rate %.4f' % (tag_name, L, nf / 730.0))

    # ---------- nulls at peak layers (both panels) ----------
    nulls = {}
    for dname in DIRS:
        L = pooled[dname]['Lstar']
        u = DIRS[dname]
        # orthogonalized direction (3228 C5)
        unh = u / (float(np.linalg.norm(u)) + 1e-18)
        proj = (H54f @ unh)[:, None] * unh[None, :]
        wo = H54f - proj
        wo = wo / (np.linalg.norm(wo, axis=1, keepdims=True) + 1e-18)
        null_dirs = [('orth', wo)]
        for s in NULL_SEEDS:
            rng = np.random.RandomState(s)
            v = rng.randn(D).astype(np.float32)
            v /= float(np.linalg.norm(v))
            null_dirs.append(('rand_%d' % s,
                              np.broadcast_to(v, (208, D)).copy()))
        for ndname, nd in null_dirs:
            rmsL = rms_L(hs54, L)
            dv = (nd * (1.0 * rmsL)[:, None]).astype(np.float32)
            J1 = np.zeros(208, np.int8)
            for i in range(208):
                lg = patch_row(ids_54[i], L, dv[i])
                J1[i] = int(lg[TOK_ISA] > lg[TOK_HASA])
            st = flip_stats(J1, J0_54, folds)
            nulls['%s|%s|L%d' % (dname, ndname, L)] = st
            npz_payload['%s_null_%s_L%d_J1' % (m, ndname, L)] = J1

    out[m] = dict(coarse=coarse, pooled={d: {str(k): v for k, v in pooled[d].items()}
                                         for d in DIRS},
                  fine=fine, ustar_band=ustar_band, ul_amp=ul_amp,
                  extra=extra, nulls=nulls,
                  n_layers=NL, D=D)
    # per-model checkpoint (crash-resilience; keyed on freeze)
    if not SMOKE:
        with open(os.path.join(OUTDIR, 'ckpt_%s.json' % m), 'w',
                  encoding='utf-8', newline='') as f:
            json.dump({'freeze': fz['sha8'], 'out': out[m]}, f,
                      ensure_ascii=False, indent=1)
        np.savez_compressed(os.path.join(OUTDIR, 'ckpt_%s.npz' % m),
                            **{k: v for k, v in npz_payload.items()
                               if k.startswith(m)})
        log('  checkpoint saved: %s (%d npz keys)'
            % (m, sum(1 for k in npz_payload if k.startswith(m))))
    del model, H54, H54f, H69, H69f
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    log('  %s done (%.1fs)' % (m, time.time() - T0))

# ---------- 4B nf4 bridge (FULL only, 4b model) ----------
if not SMOKE and 'qwen3-4b' in out:
    log('=== 4B nf4 bridge (peak layer) ===')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, \
        BitsAndBytesConfig
    torch.manual_seed(0)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR['qwen3-4b'], quantization_config=bnb,
        trust_remote_code=True).eval()
    tk = AutoTokenizer.from_pretrained(MODEL_DIR['qwen3-4b'],
                                       trust_remote_code=True)
    layers = model.model.layers
    NL = model.config.num_hidden_layers

    def ids_of(text):
        return tk(text, add_special_tokens=False)['input_ids']

    ids_54 = [ids_of(r['prompt']) for r in ROWS_3154]
    z54 = np.load(os.path.join(SRC_DIR, r'phase3154\g1p4_mfd_multifactor_disentangle',
                               'qwen3-4b', 'collect.npz'))
    H54f = z54['H'][:, -1, :].astype(np.float32)
    del z54
    A57 = p94.ANCH_3157['qwen3-4b']
    z57 = np.load(p94.SRC[p94.NPZ57_KEY['qwen3-4b']])
    H57 = z57['H'].astype(np.float32)
    EI = z57['ei'].astype(int)
    RL = z57['rel'].astype(int)
    PL = z57['pol'].astype(int)
    CX = z57['ctx'].astype(int)
    IDX = {(int(EI[i]), int(RL[i]), int(PL[i]), int(CX[i])): i
           for i in range(128)}
    dC = np.array([H57[IDX[(ei, rl, pl, 1)], A57['kout']]
                   - H57[IDX[(ei, rl, pl, 0)], A57['kout']]
                   for ei in range(16) for rl in range(2) for pl in range(2)])
    dCn = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-18)
    RL_rows = np.array([(i // 4, (i % 4) // 2, i % 2) for i in range(64)])
    v_isa = p94.unit_mean(dCn[RL_rows[:, 1] == 0])
    v_hasa = p94.unit_mean(dCn[RL_rows[:, 1] == 1])
    del z57, H57, dC, dCn
    DIRS = {'u_isa': v_isa, 'u_hasa': v_hasa}
    J0 = out['qwen3-4b']['coarse'].get('__J0__')
    # rebuild J0 from 3230 anchor npz
    Z30l = np.load(os.path.join(SRC_DIR, r'phase3230\g5g27_relcontrast_efficacy'
                                r'\e3_relcontrast_flip.npz'))
    m0 = Z30l['qwen3-4b_seen_m0'].astype(np.float32)
    J0 = (m0[:, 0] > m0[:, 1]).astype(np.int8)
    tags54 = None
    z54b = np.load(os.path.join(SRC_DIR, r'phase3154\g1p4_mfd_multifactor_disentangle',
                                'qwen3-4b', 'collect.npz'))
    tags54 = {k: z54b[k].astype(int) for k in ('ci', 'ho')}
    del z54b
    folds = [np.where((tags54['ci'] == c) & (tags54['ho'] == 0))[0]
             for c in range(4)]

    def patch_row(ids, L, dv_row):
        h = torch.tensor([ids], device='cuda')
        dv_t = torch.tensor(dv_row, dtype=torch.bfloat16,
                            device='cuda')[None, None, :]

        def hook(mod, inp, o):
            hh = o[0] if isinstance(o, tuple) else o
            hh = hh.clone()
            hh[0, -1, :] += dv_t[0, 0]
            return (hh,) + o[1:] if isinstance(o, tuple) else hh

        hnd = layers[L].register_forward_hook(hook)
        try:
            with torch.no_grad():
                o = model(input_ids=h)
            lg = o.logits[0, -1].float().detach().cpu().numpy()
            del o
        finally:
            hnd.remove()
        return lg

    bridge = {}
    for dname, u in DIRS.items():
        L = int(out['qwen3-4b']['pooled'][dname]['Lstar'])
        zc = np.load(os.path.join(SRC_DIR, r'phase3154\g1p4_mfd_multifactor_disentangle',
                                  'qwen3-4b', 'collect.npz'))
        rmsL = np.sqrt((zc['H'][:, min(L + 1, NL), :].astype(np.float32) ** 2)
                       .mean(axis=1))
        del zc
        dv = (u[None, :] * (1.0 * rmsL)[:, None]).astype(np.float32)
        J1 = np.zeros(208, np.int8)
        for i in range(208):
            lg = patch_row(ids_54[i], L, dv[i])
            J1[i] = int(lg[TOK_ISA] > lg[TOK_HASA])
        rates = {}
        for fi, sl in enumerate(folds):
            for R, Rn in ((1, 'isa'), (0, 'hasa')):
                fl = (J0[sl] == (1 - R)) & (J1[sl].astype(np.int8) == R)
                rates['%s|cut%d' % (Rn, fi)] = float(fl.sum() / len(sl))
        bridge[dname] = dict(L=L, rates=rates)
        log('  bridge nf4 %s L%d: %s' % (dname, L, rates))
    out['nf4_bridge_4b'] = bridge
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# ---------- SEAL ----------
npz_payload = {k: v for k, v in npz_payload.items() if v is not None}
npz_path = os.path.join(OUTDIR, PFX + 'upstream_lever_scan.npz')
np.savez_compressed(npz_path, **npz_payload)
clean = not bad
verdict = 'upstream_lever_scan_sealed' if clean else \
    'upstream_lever_scan_sealed_dirty(%d)' % len(bad)
res = {
    'phase': 3234, 'name': NAME,
    'arm': 'compute+seal (default (a), GPU layer scan)',
    'prereg_authorization_record': '"好的，继续" = preregistered-default-arm trigger',
    'freeze_sha8': fz['sha8'],
    'design_basis': fz['design_basis'],
    'clarifications': ['C1 ' + fz['clarifications']['C1'],
                       'C2 ' + fz['clarifications']['C2'],
                       'C3 ' + fz['clarifications']['C3'],
                       'C4 ' + fz['clarifications']['C4'],
                       'C5 ' + fz['clarifications']['C5'],
                       'C6 ' + fz['clarifications']['C6'],
                       'C7 ' + fz['clarifications']['C7'],
                       'C8 ' + fz['clarifications']['C8'],
                       'C9 ' + fz['clarifications']['C9']],
    'models': out,
    'sealed_npz': {'file': PFX + 'upstream_lever_scan.npz',
                   'file_sha8': sha8_file(npz_path)},
    'gate_application': 'DEFERRED to 3235 per 3233 chain',
    'contradictions': bad,
    'verdict': verdict,
    'res_sha8': None,
}
res['res_sha8'] = csha8(dict(res))
with open(os.path.join(OUTDIR, PFX + 'result.json'), 'w',
          encoding='utf-8', newline='') as f:
    f.write(json.dumps(res, ensure_ascii=False, indent=1))
with open(os.path.join(OUTDIR, PFX + 'run_log.txt'), 'w',
          encoding='utf-8', newline='') as f:
    f.write('\n'.join(LOG) + '\n')
log('RECHECK: %s (%d contradictions)' % ('CLEAN' if clean else 'DIRTY',
                                         len(bad)))
log('verdict: %s | res %s | npz %s'
    % (verdict, res['res_sha8'], res['sealed_npz']['file_sha8']))
