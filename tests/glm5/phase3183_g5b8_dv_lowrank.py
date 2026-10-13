# -*- coding: utf-8 -*-
"""Phase 3183 (G5-B8): 跨关系偏移向量低秩检验（T2 前置臂，零 GPU）。

Per prereg in MEMO Phase 3182 tail, section "接续：预注册 3183" (frozen before
any observation):
  Test whether the cross-relation offset is low-rank: Delta-v = v_isa - v_hasa
  (per model, from the sealed 3157 dC panel).  Freeze-time protocol pinning:
  - v_rl = unit_mean(dCn[dC_rows(rl)]) over that relation's own 32 cells
    (3178 dC recipe verbatim).  Pinning note: prereg text parenthetically
    wrote "v_isa（hasa 32 细胞 unit_mean）"; the self-relation definition is
    pinned here (offset between the two relation directions), deviation
    registered honestly in MEMO.
  - Aggregate: cos(v_isa, v_hasa), ||Delta-v||, energy split of Delta-v along
    v_full (unit_mean of all 64 dCn rows) vs orthogonal complement.
  - Entity level: Delta-v_e = unit(mean(isa 2 cells of e) - mean(hasa 2 cells
    of e)), 16 vectors; SVD top-1 PC share s1^2/sum(s^2); 120 pairwise cos;
    alignment cos(u_e, unit(Delta-v)).
  - GATE (pinned pre-observation): LOW-RANK iff pc_share >= 0.4 x3 models AND
    median pairwise cos >= 0.2 x3 models => FTR-24 candidate registered
    (E1_repeatable, family=context, registry v1.9 -> v1.10 features append,
    23 legacy byte-preserved), atlas_v1_11.html.  Otherwise diffuse =>
    F18 (low_rank_hypothesis_rejected) into failures + FTR-23
    counter_evidence quantified append; T2 route revised; stays no new
    feature; atlas re-rendered likewise.
  - SMOKE (env P3183_SMOKE=1): zero-GPU => identical numbers; smoke_ prefix.
"""
import copy
import datetime
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
import phase3179_g5b4_ftr23_registration as p79

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g5b8_dv_lowrank'
OUTDIR = os.path.join(SRC_DIR, 'phase3183', NAME)
SMOKE = os.environ.get('P3183_SMOKE') == '1'
LOG = []
N_E = 16
MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
ANCH_KEY = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4-9b': 'glm4'}
NPZ_KEY = {'qwen3-4b': 'p3157_4b', 'qwen3-14b': 'p3157_14b', 'glm4-9b': 'p3157_glm4'}
RES_KEY = {'qwen3-4b': 'p3157_res_4b', 'qwen3-14b': 'p3157_res_14b', 'glm4-9b': 'p3157_res_glm4'}


def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    LOG.append(ln)
    try:
        print(ln, flush=True)
    except Exception:
        pass


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def sha8_file(p):
    return sha8(open(p, 'rb').read())


SRC = {
    'p3157_4b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz'),
    'p3157_14b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-14b\collect.npz'),
    'p3157_glm4': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\glm4\collect.npz'),
    'p3157_res_4b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\result.json'),
    'p3157_res_14b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-14b\result.json'),
    'p3157_res_glm4': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\glm4\result.json'),
    'p3157_summary': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\summary\result_summary.json'),
    'p3178_result': os.path.join(SRC_DIR, r'phase3178\g5b3_ftr14_a2_attribution\result.json'),
    'p3182_result': os.path.join(SRC_DIR, r'phase3182\g5b7_ftr23_heldout_loro\result.json'),
    'registry_v19': os.path.join(SRC_DIR, r'phase3182\g5b7_ftr23_heldout_loro\atlas_registry_v1_9.json'),
    'gap_v15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p3157_res_4b': 'ca863153', 'p3157_res_14b': 'e797adf9', 'p3157_res_glm4': '4030081f',
    'p3157_summary': '71959b6a', 'p3178_result': 'abae296e', 'p3182_result': '15b16f37',
    'registry_v19': '59a91512', 'gap_v15': '4fb3f37d', 'registry_3162': '00f15e98',
}
CONTENT_SHA_3178 = ('4f521d9b', '1f166645')
CONTENT_SHA_3182 = ('33d73196', 'ed51ca22')

ANCH_3157 = {
    'qwen3-4b': dict(nl=36, kout=35, res_sha8='c0db3455', seal_sha8='41916ed8',
                     tc_mean_cos=0.5629908442497253),
    'qwen3-14b': dict(nl=40, kout=39, res_sha8='1270c921', seal_sha8='98c06ab0',
                      tc_mean_cos=0.5614360570907593),
    'glm4': dict(nl=40, kout=39, res_sha8='a1d9a98a', seal_sha8='e8be1026',
                 tc_mean_cos=0.5640953183174133),
}

DEV_TOL = 1e-9
G_PC = 0.4
G_COS = 0.2
GATE = {'pc_share_min': G_PC, 'median_cos_min': G_COS, 'n_entities': 16}

DESIGN = {
    'phase': '3183', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3182 tail G5-B8 (frozen before any observation)',
    'data': 'sealed 3157 collect npz x3 + sealed 3178/3182 result.json; zero GPU',
    'dv_definition': 'v_rl = unit_mean(dCn[dC_rows(rl)]) over that relation\'s own 32 '
                     'cells (3178 dC recipe verbatim); Delta-v = v_isa - v_hasa. '
                     'Pinning note: prereg parenthetical said v_isa from hasa cells; '
                     'self-relation definition pinned, deviation registered in MEMO',
    'measurements': 'aggregate cos(v_isa,v_hasa), ||Delta-v||, energy split along '
                    'v_full vs orth complement; entity Delta-v_e = unit(mean(isa 2 '
                    'cells) - mean(hasa 2 cells)) x16; SVD top-1 PC share; 120 pairwise '
                    'cos; alignment cos(u_e, unit(Delta-v)); g_mean report-only from '
                    'sealed 3178',
    'gate': 'LOW-RANK iff pc_share >= 0.4 x3 AND median pairwise cos >= 0.2 x3 => '
            'FTR-24 candidate E1_repeatable (registry v1.9->v1.10 features append, '
            '23 legacy byte-preserved, atlas_v1_11.html); else diffuse => F18 + '
            'FTR-23 counter_evidence quantified append, T2 route revised',
    'sources': SHA_ANCHOR,
    'gates': GATE,
}


def freeze():
    eblob = json.dumps(DESIGN, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    dsha = hashlib.sha256(eblob).hexdigest()
    os.makedirs(OUTDIR, exist_ok=True)
    exe_p = os.path.join(OUTDIR, 'execution.json')
    if os.path.exists(exe_p) and not SMOKE:
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha8'] == dsha[:8], 'DESIGN DRIFT'
        log('freeze: existing execution.json OK (%s)' % dsha[:8])
    else:
        json.dump({'phase': 3183, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


# ----------------- verbatim 3178 helpers -----------------
def dC_rows(rel_filter):
    out = []
    for ei in range(N_E):
        for rl in range(2):
            if rel_filter is not None and rl != rel_filter:
                continue
            for pl in range(2):
                out.append(ei * 4 + rl * 2 + pl)
    return out


def unit_mean(dCn):
    v = dCn.mean(0)
    return v / (float(np.linalg.norm(v)) + 1e-18)


def unit(v):
    return v / (float(np.linalg.norm(v)) + 1e-18)


def run_arm(sealed_3178):
    out = {}
    for m in MODELS:
        ak = ANCH_KEY[m]
        A = ANCH_3157[ak]
        z = np.load(SRC[NPZ_KEY[m]])
        H = z['H'].astype(np.float32)
        NROWS, NH, D = H.shape
        NL = NH - 1
        KOUT = NL - 1
        assert NROWS == 128, ('rows', m, NROWS)
        assert (NL, KOUT) == (A['nl'], A['kout']), ('layer anchor', m, NL, KOUT)
        EI = z['ei'].astype(int)
        RL = z['rel'].astype(int)
        PL = z['pol'].astype(int)
        CX = z['ctx'].astype(int)
        IDX = {(int(EI[i]), int(RL[i]), int(PL[i]), int(CX[i])): i for i in range(NROWS)}
        assert len(IDX) == 128, ('cells', m, len(IDX))

        # device gate: tc_full recompute verbatim
        dC = []
        for ei in range(N_E):
            for rl in range(2):
                for pl in range(2):
                    dC.append(H[IDX[(ei, rl, pl, 1)], KOUT] - H[IDX[(ei, rl, pl, 0)], KOUT])
        dC = np.array(dC)
        assert np.isfinite(dC).all(), ('nonfinite dC', m)
        dCn = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-18)
        cosM = dCn @ dCn.T
        tri = cosM[np.triu_indices(len(dC), 1)]
        tc_full = float(tri.mean())
        assert abs(tc_full - A['tc_mean_cos']) < DEV_TOL, ('tc drift', m)

        # relation directions + aggregate offset
        v_isa = unit_mean(dCn[dC_rows(0)])
        v_hasa = unit_mean(dCn[dC_rows(1)])
        cos_rl = float(v_isa @ v_hasa)
        dvec = v_isa - v_hasa
        norm_dv = float(np.linalg.norm(dvec))
        v_full = unit_mean(dCn)
        frac_full = float((dvec @ v_full) ** 2 / max(norm_dv ** 2, 1e-300))
        frac_orth = 1.0 - frac_full

        # entity-level Delta-v_e (unit), SVD rank structure
        U = []
        for e in range(N_E):
            a_e = dCn[[e * 4 + 0 * 2 + p for p in (0, 1)]].mean(0)
            b_e = dCn[[e * 4 + 1 * 2 + p for p in (0, 1)]].mean(0)
            U.append(unit(a_e - b_e))
        U = np.stack(U)
        assert np.isfinite(U).all(), ('nonfinite U', m)
        s = np.linalg.svd(U, compute_uv=False)
        pc_share = float(s[0] ** 2 / (s ** 2).sum())
        Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-18)
        C = Un @ Un.T
        iu = np.triu_indices(N_E, 1)
        pairs = C[iu]
        med_cos = float(np.median(pairs))
        min_cos = float(pairs.min())
        frac_pos = float((pairs > 0).mean())
        dv_unit = unit(dvec)
        align = Un @ dv_unit
        med_align = float(np.median(align))

        # report-only: g_mean from sealed 3178
        r18 = sealed_3178['arm']['per_model'][m]
        g_mean = float((r18['resid']['isa']['gap'] + r18['resid']['hasa']['gap']) / 2.0)
        px_mean = float(np.mean([r18['resid'][r]['p'] for r in ('isa', 'hasa')])) \
            if 'p' in r18['resid']['isa'] else None

        lowrank = bool(pc_share >= G_PC and med_cos >= G_COS)
        out[m] = dict(kout=KOUT, nl=NL, tc_full=tc_full,
                      cos_rl=cos_rl, norm_dv=norm_dv,
                      frac_full=frac_full, frac_orth=frac_orth,
                      pc_share=pc_share, med_cos=med_cos, min_cos=min_cos,
                      frac_pos=frac_pos, med_align=med_align,
                      g_mean_report=g_mean, px_mean_report=px_mean,
                      lowrank=lowrank)
        log('%s: cos(v_isa,v_hasa)=%.4f ||dv||=%.4f (g_mean=%.4f, px_mean=%s) | '
            'frac_full=%.4f | pc_share=%.4f med_cos=%.4f min=%.4f pos=%.2f align=%.4f -> %s'
            % (m, cos_rl, norm_dv, g_mean,
               ('%.4f' % px_mean) if px_mean is not None else 'n/a',
               frac_full, pc_share, med_cos, min_cos, frac_pos, med_align,
               'LOWRANK' if lowrank else 'DIFFUSE'))
    low_ok = all(out[m]['lowrank'] for m in MODELS)
    log('GATE low-rank (pc_share>=%.2f AND med_cos>=%.2f x3): %s' % (G_PC, G_COS, low_ok))
    return out, low_ok


# ----------------- registry v1.9 -> v1.10 + atlas_v1_11 -----------------
CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
span = p79.span


def finish_registry(summary, low_ok, res8, seal8, design_sha):
    reg19 = json.loads(open(SRC['registry_v19'], 'rb').read().decode('utf-8'))
    gl15 = json.loads(open(SRC['gap_v15'], encoding='utf-8').read())
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg19['version'] == '1.9' and len(reg19['features']) == 23
    reg20 = copy.deepcopy(reg19)
    reg20['version'] = '1.10'
    reg20['supersedes'] = 'atlas_registry_v1_9.json (59a91512, Phase 3182)'
    pm = summary['per_model']
    fmt3 = lambda k, f: '/'.join('%s' % ('%.4f' % pm[m][k]) for m in MODELS)
    if low_ok:
        f24 = {
            'id': 'FTR-24',
            'family': 'context',
            'statement': ('跨关系偏移向量低秩（3183 G5-B8 定判）：Δv = v_isa − v_hasa '
                          '（各自关系 32 细胞 unit_mean，dC 面板 3178 配方 verbatim）'
                          '在实体级呈单方向结构——16 个实体级 Δv_e 的 SVD 主成分占比 '
                          + fmt3('pc_share', None) + '、120 对两两 cos 中位数 '
                          + fmt3('med_cos', None) + '（≥门 0.4/0.2 ×3 模型）；'
                          '聚合 ‖Δv‖ ' + fmt3('norm_dv', None) + '、cos(v_isa,v_hasa) '
                          + fmt3('cos_rl', None) + '，沿 v_full 能量占比 '
                          + fmt3('frac_full', None) + '（正交补主导）。与 3182 cross-null '
                          '罚 p_x（0.13–0.16）量级一致——ΔM_rel = T_C + R_rel 低秩假设'
                          '获得首条直接证据：关系偏移集中于单一方向，参数化为低秩 R_rel '
                          '路线开放（T2）。'),
            'evidence_level': 'E1_repeatable',
            'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
            'anchors': [{
                'src': 'p3183', 'phase': 3183,
                'role': 'dv_lowrank_structure',
                'tags': ['cross_model'],
                'asserts': {
                    'pc_share.qwen3-4b': pm['qwen3-4b']['pc_share'],
                    'pc_share.qwen3-14b': pm['qwen3-14b']['pc_share'],
                    'pc_share.glm4-9b': pm['glm4-9b']['pc_share'],
                    'med_cos.qwen3-4b': pm['qwen3-4b']['med_cos'],
                    'med_cos.qwen3-14b': pm['qwen3-14b']['med_cos'],
                    'med_cos.glm4-9b': pm['glm4-9b']['med_cos'],
                    'norm_dv.qwen3-4b': pm['qwen3-4b']['norm_dv'],
                    'norm_dv.qwen3-14b': pm['qwen3-14b']['norm_dv'],
                    'norm_dv.glm4-9b': pm['glm4-9b']['norm_dv'],
                    'gate_pc_min': G_PC, 'gate_cos_min': G_COS,
                    'res_sha8': res8, 'seal_sha8': seal8}}],
            'counter_evidence': [],
            'replication': 'zero-GPU sealed-panel recompute; SMOKE/FULL byte-identical; '
                           'device tc_full drift < 1e-9 vs sealed 3157',
            'scope_limits': '双关系面板（isa/hasa）内检验；低秩判据为本面板内实体级结构，'
                            '不外推为任意关系对的普适低秩性',
            'values': None,
        }
        reg20['features'] = reg19['features'] + [f24]
        disposition = 'v1.9_to_v1.10_ftr24_registered_E1'
    else:
        f23 = [f for f in reg20['features'] if f['id'] == 'FTR-23'][0]
        ce = ('3183 低秩检验定判 DIFFUSE：Δv 实体级 SVD 主成分占比 '
              + fmt3('pc_share', None) + '、两两 cos 中位数 ' + fmt3('med_cos', None)
              + '（门 0.4/0.2 ×3 不全满足）——ΔM_rel 低秩假设在本面板被拒，'
              'T2 路线修正（R_rel 非单一方向）；F18 入失败账本。')
        f23['counter_evidence'] = f23['counter_evidence'] + [ce]
        reg20['failures'] = reg19['failures'] + [{
            'id': 'F18', 'kind': 'low_rank_hypothesis_rejected', 'phase': 3183,
            'text': 'Delta-v entity-level structure diffuse (pc_share/median-cos below '
                    'gate): relation offset not low-rank in this panel',
            'evidence': 'p3183 result %s/seal %s' % (res8, seal8)}]
        disposition = 'v1.9_to_v1.10_f18_diffuse'
    # preservation asserts
    for f in reg19['features']:
        if (not low_ok) and f['id'] == 'FTR-23':
            continue
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg20['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        assert a == b, ('feature not preserved', f['id'])
    for k in ('upgrade_log', 'evidence_level_taxonomy', 'principles', 'schema',
              'models_mainline', 'models_pending', 'provenance', 'created',
              'phase', 'name', 'design_sha8'):
        assert json.dumps(reg19.get(k), ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg20.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    if low_ok:
        assert json.dumps(reg19['failures'], ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg20['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'
    else:
        assert len(reg20['failures']) == len(reg19['failures']) + 1
        assert norm_prefix(reg19['failures'], reg20['failures'])
    rp = os.path.join(OUTDIR, 'atlas_registry_v1_10.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg20, ensure_ascii=False, indent=1))
    reg20_sha = sha8_file(rp)
    log('registry v1.10 written sha8 %s (%s)' % (reg20_sha, disposition))

    meta_sha = {'reg20': reg20_sha}
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.11 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.11 &mdash; 语义机制图谱（registry v1.10 渲染）</h1>')
    H.append('<div class="sub">Phase 3183 G5-B8 &middot; 跨关系偏移向量低秩检验（%s） &middot; '
             'registry v1.10 sha8=%s &middot; v1.9 sha8=%s &middot; v1.8 sha8=%s &middot; v1.7 sha8=%s &middot; v1.6 sha8=%s'
             ' &middot; gap ledger v1.5 sha8=%s（3175 定稿，原样渲染）'
             ' &middot; 源 p3157 collect %s/%s/%s + p3178 %s + p3182 %s &middot; 渲染于 %s</div>' % (
                 'FTR-24 立项 E1' if low_ok else 'F18 diffuse',
                 span('meta.registry_v110_sha8', reg20_sha),
                 span('meta.registry_v19_sha8', '59a91512'),
                 span('meta.registry_v18_sha8', 'd08e9064'),
                 span('meta.registry_v17_sha8', 'f3ad070f'),
                 span('meta.registry_v16_sha8', 'bebdf38d'),
                 span('meta.gap_v15_sha8', SHA_ANCHOR['gap_v15']),
                 span('meta.p3157_4b', SHA_ANCHOR['p3157_4b']),
                 span('meta.p3157_14b', SHA_ANCHOR['p3157_14b']),
                 span('meta.p3157_glm4', SHA_ANCHOR['p3157_glm4']),
                 span('meta.p3178_result', SHA_ANCHOR['p3178_result']),
                 span('meta.p3182_result', SHA_ANCHOR['p3182_result']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg20['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg20['principles']):
        H.append('<div class="kv"><b>原则%d</b>%s</div>' % (i + 1, p79.esc(pr)))
    H.append('</div>')
    H.append('<h2>图谱节点基座（16 节点，3162 audit disk_verified）</h2>')
    H.append('<table><tr><th>id</th><th>title</th><th>evidence</th><th>checks</th></tr>')
    for n in nodes:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td><span class="badge b-%s">%s</span>%s</td>'
                 '<td class="mono">%s</td></tr>' % (
                     n['id'], span(n['id'] + '.title', n['title']),
                     n['evidence_level'], span(n['id'] + '.elevel', n['evidence_level']),
                     span(n['id'] + '.status', n['status']),
                     span(n['id'] + '.checks', '%d/%d' % (n['n_pass'], n['n_checks']))))
    H.append('</table>')
    fams = {}
    for f in reg20['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.10，%d 条）</h2>' % len(reg20['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg20['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg20['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg20['failures']:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td class="mono">%s</td><td>%s</td><td class="mono">%s</td></tr>' % (
            fl['id'], span(fl['id'] + '.kind', fl['kind']), span(fl['id'] + '.phase', fl['phase']),
            span(fl['id'] + '.text', fl['text']), span(fl['id'] + '.evidence', fl['evidence'])))
    H.append('</table>')
    H.append('<h2>缺口账本 v1.5（GAP-4 quantified_collapse 定稿）</h2>')
    for g in gl15['gaps']:
        closed_at = str(g['closed_at_phase']) if g.get('closed_at_phase') is not None else '-'
        badge_closed = '<span class="badge">closed@%s</span>' % span(g['id'] + '.closed_at', closed_at) \
            if g['status'].startswith('closed') else ''
        H.append('<div class="card"><h3>%s &middot; %s <span class="badge b-%s">%s</span>%s</h3>' % (
            g['id'], span(g['id'] + '.title', g['title']), g['status'], span(g['id'] + '.status', g['status']), badge_closed))
        H.append('<div class="kv"><b>statement</b>%s</div>' % span(g['id'] + '.statement', g['statement']))
        H.append('<div class="evid">evidence:</div>')
        H.append('<div class="kv">%s</div>' % span(g['id'] + '.evidence', ' || '.join(g['evidence'])))
        H.append('<div class="anchorbox"><b>anchors</b> %s</div>' % span(
            g['id'] + '.anchors', ', '.join('%s=%s' % (k, v) for k, v in sorted(g['anchor_sha8'].items()))))
        if g['id'] == 'GAP-4':
            pr = g['prereg']
            H.append('<div class="prereg"><div class="kv"><b>prereg</b>Phase %s %s（%s）</div>' % (
                span('GAP-4.prereg.phase', pr['phase']), span('GAP-4.prereg.name', pr['name']),
                span('GAP-4.prereg.status', pr['status'])))
            H.append('<div class="kv"><b>hypothesis</b>%s</div>' % span('GAP-4.prereg.hypothesis', pr['hypothesis']))
            H.append('<div class="kv"><b>protocol</b>%s</div>' % span('GAP-4.prereg.protocol', pr['protocol']))
            H.append('<div class="kv"><b>gate</b>%s</div></div>' % span('GAP-4.prereg.gate', pr['gate']))
            H.append('<div class="mech"><div class="kv"><b>mechanism_note</b>%s</div></div>' % span(
                'GAP-4.mechanism_note', g['mechanism_note']))
        H.append('</div>')
    H.append('<h2>附录挂账（%d 项，逐条证据锚）</h2><div class="card">' % len(gl15['appendix']))
    for a in gl15['appendix']:
        H.append('<div class="kv"><b>%s</b>%s <span class="badge b-%s">%s</span>'
                 '<span class="mono">anchor=%s(%s)</span><div class="evid">%s</div></div>' % (
                     a['id'], span(a['id'] + '.title', a['title']), a['status'],
                     span(a['id'] + '.status', a['status']),
                     span(a['id'] + '.anchor', a['anchor']),
                     span(a['id'] + '.anchor_sha8', a['anchor_sha8']),
                     span(a['id'] + '.note', a['note'])))
    H.append('</div>')
    H.append('<div class="footer">RDC/LPF atlas v1.11 &middot; 渲染自 atlas_registry_v1_10.json (%s)'
             ' + gap ledger v1.5 (%s) &middot; 增量来源: Phase 3183 G5-B8 跨关系偏移向量低秩检验'
             '（数据源 p3157 collect %s / %s / %s + p3178 %s + p3182 %s）&middot; 本文件为 Phase 3183 封存产物，'
             '字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg20_sha, SHA_ANCHOR['gap_v15'],
                 SHA_ANCHOR['p3157_4b'], SHA_ANCHOR['p3157_14b'], SHA_ANCHOR['p3157_glm4'],
                 SHA_ANCHOR['p3178_result'], SHA_ANCHOR['p3182_result']))
    H.append('</body></html>')
    html_text = '\n'.join(H)
    hp = os.path.join(OUTDIR, 'atlas_v1_11.html')
    with open(hp, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(hp)
    expected = flatten_v111(reg20, nodes, gl15, meta_sha)
    v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
    log('atlas_v1_11.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
        html_sha, v['n_expected'], v['n_found'], len(v['missing']),
        len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
    final = {
        'phase': 3183, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'verdict': summary['verdict'], 'low_ok': bool(low_ok),
                'per_model': summary['per_model']},
        'counts': {'features': len(reg20['features']), 'failures': len(reg20['failures']),
                   'upgrades': len(reg20['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': disposition,
        'registry_v110_file': 'atlas_registry_v1_10.json', 'registry_v110_sha8': reg20_sha,
        'html_file': 'atlas_v1_11.html', 'html_sha8': html_sha,
        'field_check': {'n_expected': v['n_expected'], 'n_found': v['n_found'],
                        'missing': len(v['missing']), 'extra': len(v['extra']),
                        'mismatch': len(v['mismatch']), 'dups': len(v['dups'])},
        'sources': SHA_ANCHOR, 'res_sha8': res8, 'seal_sha8': seal8,
    }
    raw = json.dumps(final, ensure_ascii=False, indent=1, sort_keys=True)
    f_res8 = sha8(raw.encode('utf-8'))
    mid = json.dumps(dict(final, res_sha8=f_res8), ensure_ascii=False, indent=1, sort_keys=True)
    f_seal8 = sha8(mid.encode('utf-8'))
    with open(os.path.join(OUTDIR, 'final_result.json'), 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(dict(final, res_sha8=f_res8, seal_sha8=f_seal8),
                           ensure_ascii=False, indent=1, sort_keys=True))
    log('final result res=%s seal=%s' % (f_res8, f_seal8))


def norm_prefix(old, new):
    return json.dumps(old, ensure_ascii=False, sort_keys=True) == \
        json.dumps(new[:len(old)], ensure_ascii=False, sort_keys=True)


def flatten_v111(reg, nodes, gl, meta_sha):
    d = {}
    for f in reg['features']:
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
            d['%s.values.%s' % (p, k)] = p79.fmt(v) if hasattr(p79, 'fmt') else str(v)
    for n in nodes:
        d[n['id'] + '.title'] = n['title']
        d[n['id'] + '.elevel'] = n['evidence_level']
        d[n['id'] + '.status'] = n['status']
        d[n['id'] + '.checks'] = '%d/%d' % (n['n_pass'], n['n_checks'])
    for i, u in enumerate(reg['upgrade_log']):
        d['UPG.%d.node' % i] = u['node']
        d['UPG.%d.change' % i] = u['change']
        d['UPG.%d.reason' % i] = u['reason']
    for fl in reg['failures']:
        d[fl['id'] + '.kind'] = fl['kind']
        d[fl['id'] + '.phase'] = str(fl['phase'])
        d[fl['id'] + '.text'] = fl['text']
        d[fl['id'] + '.evidence'] = str(fl['evidence'])
    for g in gl['gaps']:
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
    for a in gl['appendix']:
        p = a['id']
        d[p + '.title'] = a['title']
        d[p + '.status'] = a['status']
        d[p + '.note'] = a['note']
        d[p + '.anchor'] = a['anchor']
        d[p + '.anchor_sha8'] = a['anchor_sha8']
    d['meta.registry_v110_sha8'] = meta_sha['reg20']
    d['meta.registry_v19_sha8'] = '59a91512'
    d['meta.registry_v18_sha8'] = 'd08e9064'
    d['meta.registry_v17_sha8'] = 'f3ad070f'
    d['meta.registry_v16_sha8'] = 'bebdf38d'
    d['meta.gap_v15_sha8'] = SHA_ANCHOR['gap_v15']
    d['meta.p3157_4b'] = SHA_ANCHOR['p3157_4b']
    d['meta.p3157_14b'] = SHA_ANCHOR['p3157_14b']
    d['meta.p3157_glm4'] = SHA_ANCHOR['p3157_glm4']
    d['meta.p3178_result'] = SHA_ANCHOR['p3178_result']
    d['meta.p3182_result'] = SHA_ANCHOR['p3182_result']
    d['meta.created'] = CREATED_STR
    return d


def main():
    design_sha = freeze()
    log('== Phase 3183 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    # double-lock 3157 anchors
    for m in MODELS:
        ak = ANCH_KEY[m]
        r57 = json.load(open(SRC[RES_KEY[m]], encoding='utf-8'))
        A = ANCH_3157[ak]
        assert r57['res_sha8'] == A['res_sha8'] and r57['seal_sha8'] == A['seal_sha8'], \
            ('3157 content sha', m)
        assert r57['tc_mean_cos'] == A['tc_mean_cos'], ('3157 tc', m)
    log('3157 anchors double-locked (3 per-model)')
    # double-lock 3178 + 3182 sealed results
    r18s = json.load(open(SRC['p3178_result'], encoding='utf-8'))
    assert r18s['res_sha8'] == CONTENT_SHA_3178[0] and r18s['seal_sha8'] == CONTENT_SHA_3178[1]
    assert r18s['arm']['calibration_pass'] is False
    r18b = json.load(open(SRC['p3182_result'], encoding='utf-8'))
    assert r18b['res_sha8'] == CONTENT_SHA_3182[0] and r18b['seal_sha8'] == CONTENT_SHA_3182[1]
    assert r18b['gates']['held_ok'] is False
    log('3178 (%s/%s) + 3182 (%s/%s) sealed results double-locked'
        % (CONTENT_SHA_3178 + CONTENT_SHA_3182))

    out, low_ok = run_arm(r18s)
    if low_ok:
        verdict = ('dv_lowrank_confirmed (pc_share/median-cos gate x3; FTR-24 '
                   'candidate registered E1_repeatable, registry v1.9 -> v1.10)')
    else:
        verdict = ('dv_diffuse_lowrank_rejected (F18 registered; FTR-23 '
                   'counter_evidence quantified append; T2 route revised)')
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3183, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True,
        'gates': {'pc_share_min': G_PC, 'median_cos_min': G_COS,
                  'device_3157': True, 'low_ok': bool(low_ok)},
        'per_model': out,
        'verdict': verdict,
    }
    raw = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True)
    res8 = sha8(raw.encode('utf-8'))
    mid = json.dumps(dict(summary, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
    seal8 = sha8(mid.encode('utf-8'))
    rp = os.path.join(OUTDIR, suffix + 'result.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(dict(summary, res_sha8=res8, seal_sha8=seal8),
                           ensure_ascii=False, indent=1, sort_keys=True))
    log('result written res=%s seal=%s' % (res8, seal8))
    if not SMOKE:
        finish_registry(summary, low_ok, res8, seal8, design_sha)
    with open(os.path.join(OUTDIR, suffix + 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
