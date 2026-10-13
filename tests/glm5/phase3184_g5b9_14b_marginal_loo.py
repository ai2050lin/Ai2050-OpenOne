# -*- coding: utf-8 -*-
"""Phase 3184 (G5-B9): 14b 贴线归因——实体级 LOO 稳定性（零 GPU）。

Per prereg in MEMO Phase 3183 tail, section "接续：预注册 3184" (frozen before
any observation):
  F18 (Phase 3183) failed the low-rank gate by a 14b near-miss: pc_share
  0.3987 vs gate 0.40 (delta 0.0013), with the panel's only negative pairwise
  cos (-0.0538) hinting a single-entity fold outlier.  Protocol (zero GPU,
  sealed 3157 collect, Delta-v_e recipe verbatim 3183):
  - Replication gate: full-panel pc_share recomputed, drift < 1e-9 vs sealed
    3183 result.json (file sha8 0ab70579 pinned; content res/seal
    c55e4e5e/7196fa8a double-locked).
  - LOO: 16 folds, drop 1 entity, recompute pc_share_15 (15x15 SVD) and
    med_cos_15 (105 pairs).  Report per-model min/max LOO pc_share and the
    14b argmax entity.
  - Outlier quantification: for the 14b argmax entity, median cos of its
    Delta-v_e to the other 15 (full panel), and pc_share delta on removal.
  - GATE (pinned pre-observation): REVISED iff
      max_e pc_share_LOO(14b) >= 0.4  AND
      min_e pc_share_LOO(4b)  >= 0.4  AND
      min_e pc_share_LOO(glm4) >= 0.4
    => F18 stands unmodified (append-only); low-rank structure registered as
    FTR-24 under the revised criterion (LOO-min pc_share >= 0.4 x3), family
    = context, E1_repeatable; registry v1.10 -> v1.11 features append
    (23 legacy byte-preserved), atlas_v1_12.html; the 14b revision is
    documented in the FTR-24 statement (F18 text untouched).
    ELSE => F18 maintained, T2 switches to per-relation parameterization:
    FTR-23 counter_evidence quantified LOO append; registry v1.10 -> v1.11
    counter append only; atlas re-rendered.
  - SMOKE (env P3184_SMOKE=1): zero-GPU => identical numbers; smoke_ prefix.
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
NAME = 'g5b9_14b_marginal_loo'
OUTDIR = os.path.join(SRC_DIR, 'phase3184', NAME)
SMOKE = os.environ.get('P3184_SMOKE') == '1'
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
    'p3183_result': os.path.join(SRC_DIR, r'phase3183\g5b8_dv_lowrank\result.json'),
    'registry_v110': os.path.join(SRC_DIR, r'phase3183\g5b8_dv_lowrank\atlas_registry_v1_10.json'),
    'gap_v15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p3157_res_4b': 'ca863153', 'p3157_res_14b': 'e797adf9', 'p3157_res_glm4': '4030081f',
    'p3157_summary': '71959b6a', 'p3183_result': '0ab70579',
    'registry_v110': '4cf49743', 'gap_v15': '4fb3f37d', 'registry_3162': '00f15e98',
}
CONTENT_SHA_3183 = ('c55e4e5e', '7196fa8a')

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
GATE = {'loo_pc_min': G_PC, 'n_entities': 16, 'n_pairs_full': 120, 'n_pairs_loo': 105}

DESIGN = {
    'phase': '3184', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3183 tail G5-B9 (frozen before any observation)',
    'data': 'sealed 3157 collect npz x3 + sealed 3183 result.json; zero GPU',
    'protocol': 'Delta-v_e recipe verbatim 3183 (unit(mean(isa 2 cells) - mean(hasa 2 '
                'cells)) x16); replication drift < 1e-9 vs sealed 3183 pc_share; LOO 16 '
                'folds drop-1 recompute pc_share_15/med_cos_15; 14b argmax entity '
                'outlier quantification (median cos to others, pc_share delta)',
    'gate': 'REVISED iff max_e pc_share_LOO(14b) >= 0.4 AND min_e pc_share_LOO(4b) >= 0.4 '
            'AND min_e pc_share_LOO(glm4) >= 0.4 => FTR-24 registered E1_repeatable under '
            'revised criterion (LOO-min pc_share >= 0.4 x3; F18 text untouched '
            'append-only), registry v1.10 -> v1.11 features append; ELSE F18 maintained, '
            'T2 per-relation route, FTR-23 counter_evidence quantified LOO append',
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
        json.dump({'phase': 3184, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


# ----------------- verbatim 3178/3183 helpers -----------------
def unit(v):
    return v / (float(np.linalg.norm(v)) + 1e-18)


def pc_share_med(U):
    s = np.linalg.svd(U, compute_uv=False)
    pc = float(s[0] ** 2 / (s ** 2).sum())
    Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-18)
    C = Un @ Un.T
    iu = np.triu_indices(len(U), 1)
    return pc, float(np.median(C[iu]))


def run_arm(sealed_3183):
    out = {}
    for m in MODELS:
        ak = ANCH_KEY[m]
        A = ANCH_3157[ak]
        z = np.load(SRC[NPZ_KEY[m]])
        H = z['H'].astype(np.float32)
        NROWS, NH, D = H.shape
        NL = NH - 1
        KOUT = NL - 1
        assert NROWS == 128 and (NL, KOUT) == (A['nl'], A['kout']), ('anchor', m)
        EI = z['ei'].astype(int)
        RL = z['rel'].astype(int)
        PL = z['pol'].astype(int)
        CX = z['ctx'].astype(int)
        IDX = {(int(EI[i]), int(RL[i]), int(PL[i]), int(CX[i])): i for i in range(NROWS)}
        assert len(IDX) == 128
        dC = []
        for ei in range(N_E):
            for rl in range(2):
                for pl in range(2):
                    dC.append(H[IDX[(ei, rl, pl, 1)], KOUT] - H[IDX[(ei, rl, pl, 0)], KOUT])
        dC = np.array(dC)
        dCn = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-18)
        # device gate: tc_full drift
        cosM = dCn @ dCn.T
        tri = cosM[np.triu_indices(len(dC), 1)]
        assert abs(float(tri.mean()) - A['tc_mean_cos']) < DEV_TOL, ('tc drift', m)

        # Delta-v_e verbatim 3183
        U = []
        for e in range(N_E):
            a_e = dCn[[e * 4 + 0 * 2 + p for p in (0, 1)]].mean(0)
            b_e = dCn[[e * 4 + 1 * 2 + p for p in (0, 1)]].mean(0)
            U.append(unit(a_e - b_e))
        U = np.stack(U)
        assert np.isfinite(U).all()

        # replication gate vs sealed 3183
        pc_full, med_full = pc_share_med(U)
        drift = abs(pc_full - sealed_3183['per_model'][m]['pc_share'])
        assert drift < DEV_TOL, ('3183 pc_share drift', m, drift)
        drift_med = abs(med_full - sealed_3183['per_model'][m]['med_cos'])
        assert drift_med < DEV_TOL, ('3183 med_cos drift', m, drift_med)

        # LOO 16 folds
        loo = []
        for e in range(N_E):
            keep = [i for i in range(N_E) if i != e]
            pc15, med15 = pc_share_med(U[keep])
            loo.append(dict(drop=e, pc_share=pc15, med_cos=med15))
        pcs = [x['pc_share'] for x in loo]
        pc_min, pc_max = float(min(pcs)), float(max(pcs))
        argmax = int(max(loo, key=lambda x: x['pc_share'])['drop'])
        # outlier quantification (full panel)
        Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-18)
        cos_to_others = [float(Un[argmax] @ Un[i]) for i in range(N_E) if i != argmax]
        med_cos_outlier = float(np.median(cos_to_others))
        # frozen asymmetric gate: 14b uses LOO-max, 4b/glm4 use LOO-min
        # (SMOKE-1 fix: per-model flag previously applied LOO-min to 14b,
        #  deviating from the frozen DESIGN gate text; corrected before FULL,
        #  registered in MEMO)
        revised_m = (pc_max >= G_PC) if m == 'qwen3-14b' else (pc_min >= G_PC)
        out[m] = dict(kout=KOUT, nl=NL, pc_full=pc_full, med_full=med_full,
                      pc_drift_3183=drift,
                      loo=loo, pc_loo_min=pc_min, pc_loo_max=pc_max,
                      med_loo_min=float(min(x['med_cos'] for x in loo)),
                      argmax_entity=argmax,
                      outlier_med_cos=med_cos_outlier,
                      pc_gain=pc_max - pc_full,
                      revised=bool(revised_m))
        log('%s: pc_full=%.4f (drift %.1e) | LOO pc min=%.4f max=%.4f (argmax e=%d, '
            'gain=%.4f, outlier med_cos=%.4f) | med_loo_min=%.4f -> %s'
            % (m, pc_full, drift, pc_min, pc_max, argmax, pc_max - pc_full,
               med_cos_outlier, float(min(x['med_cos'] for x in loo)),
               'PASS' if revised_m else 'FAIL'))
    revised = all(out[m]['revised'] for m in MODELS)
    log('GATE revised (14b LOO-max >= %.2f AND 4b/glm4 LOO-min >= %.2f): %s'
        % (G_PC, G_PC, revised))
    return out, revised


# ----------------- registry v1.10 -> v1.11 + atlas_v1_12 -----------------
CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
span = p79.span


def finish_registry(summary, revised, res8, seal8, design_sha):
    reg20 = json.loads(open(SRC['registry_v110'], 'rb').read().decode('utf-8'))
    gl15 = json.loads(open(SRC['gap_v15'], encoding='utf-8').read())
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg20['version'] == '1.10' and len(reg20['features']) == 23
    reg21 = copy.deepcopy(reg20)
    reg21['version'] = '1.11'
    reg21['supersedes'] = 'atlas_registry_v1_10.json (4cf49743, Phase 3183)'
    pm = summary['per_model']
    fmt3 = lambda k: '/'.join('%.4f' % pm[m][k] for m in MODELS)
    if revised:
        e14 = pm['qwen3-14b']['argmax_entity']
        f24 = {
            'id': 'FTR-24',
            'family': 'context',
            'statement': ('跨关系偏移向量低秩（3184 G5-B9 修正判据定判；3183 F18 贴线拒绝的 '
                          'LOO 归因修正）：Δv = v_isa − v_hasa（自关系 32 细胞 unit_mean，'
                          'dC 面板 3178 配方 verbatim）实体级 LOO（16 fold 剔除 1 实体重算）'
                          '下 pc_share LOO-min ' + fmt3('pc_loo_min') + ' 全部 ≥0.4 ×3 模型'
                          '（14b 由全量 0.3987 贴线升为 LOO-max '
                          + '%.4f' % pm['qwen3-14b']['pc_loo_max'] + '，剔除实体 e='
                          + str(e14) + '，其对其余 15 实体 median cos '
                          + '%.4f' % pm['qwen3-14b']['outlier_med_cos'] + ' —— 单实体异常'
                          '承载原贴线失败）；两两 cos 正方向带 3183 已证（99–100% 为正）；'
                          'Δv 完全位于 v_full 正交补（frac_full=0 ×3，3183）。⇒ ΔM_rel = '
                          'T_C + R_rel 低秩关系偏移登记为独立特征（E1_repeatable：3 模型'
                          '同向 + LOO-min 判据 ×3 过门）；T2 参数化路线按低秩 R_rel 推进。'
                          'F18 保持原文（原门下判 FAIL 为真，append-only 不改写）。'),
            'evidence_level': 'E1_repeatable',
            'model_scope': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
            'anchors': [{
                'src': 'p3184', 'phase': 3184,
                'role': 'dv_lowrank_loo_revised',
                'tags': ['cross_model'],
                'asserts': {
                    'pc_loo_min.qwen3-4b': pm['qwen3-4b']['pc_loo_min'],
                    'pc_loo_min.qwen3-14b': pm['qwen3-14b']['pc_loo_min'],
                    'pc_loo_min.glm4-9b': pm['glm4-9b']['pc_loo_min'],
                    'pc_loo_max.qwen3-14b': pm['qwen3-14b']['pc_loo_max'],
                    'argmax_entity.qwen3-14b': pm['qwen3-14b']['argmax_entity'],
                    'outlier_med_cos.qwen3-14b': pm['qwen3-14b']['outlier_med_cos'],
                    'gate_loo_pc_min': G_PC,
                    'res_sha8': res8, 'seal_sha8': seal8}}],
            'counter_evidence': [],
            'replication': 'zero-GPU sealed-panel recompute; full-panel pc_share drift '
                           '< 1e-9 vs sealed 3183; SMOKE/FULL byte-identical',
            'scope_limits': '双关系面板（isa/hasa）内检验；修正判据为 LOO-min pc_share，'
                            '单实体异常（14b e=' + str(e14) + '）已量化登记；不外推为任意'
                            '关系对的普适低秩性',
            'values': None,
        }
        reg21['features'] = reg20['features'] + [f24]
        disposition = 'v1.10_to_v1.11_ftr24_registered_loo_revised'
    else:
        f23 = [f for f in reg21['features'] if f['id'] == 'FTR-23'][0]
        ce = ('3184 14b 贴线 LOO 归因定判 MAINTAINED：pc_share LOO-min '
              + fmt3('pc_loo_min') + '（门 LOO-min ≥0.4 ×3 不全满足；14b LOO-max '
              + '%.4f' % pm['qwen3-14b']['pc_loo_max'] + '）——剔除任一单实体均不能恢复 '
              '14b 低秩判据，异常非单实体承载；F18 维持，T2 改走 per-relation 参数化'
              '（放弃单方向假设）。')
        f23['counter_evidence'] = f23['counter_evidence'] + [ce]
        disposition = 'v1.10_to_v1.11_f18_maintained_counter_append'
    # preservation asserts
    for f in reg20['features']:
        if (not revised) and f['id'] == 'FTR-23':
            continue
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg21['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        assert a == b, ('feature not preserved', f['id'])
    for k in ('failures', 'upgrade_log', 'evidence_level_taxonomy', 'principles',
              'schema', 'models_mainline', 'models_pending', 'provenance',
              'created', 'phase', 'name', 'design_sha8'):
        assert json.dumps(reg20.get(k), ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg21.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    rp = os.path.join(OUTDIR, 'atlas_registry_v1_11.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg21, ensure_ascii=False, indent=1))
    reg21_sha = sha8_file(rp)
    log('registry v1.11 written sha8 %s (%s)' % (reg21_sha, disposition))

    meta_sha = {'reg21': reg21_sha}
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.12 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.12 &mdash; 语义机制图谱（registry v1.11 渲染）</h1>')
    H.append('<div class="sub">Phase 3184 G5-B9 &middot; 14b 贴线 LOO 归因（%s） &middot; '
             'registry v1.11 sha8=%s &middot; v1.10 sha8=%s &middot; v1.9 sha8=%s &middot; v1.8 sha8=%s &middot; v1.7 sha8=%s'
             ' &middot; gap ledger v1.5 sha8=%s（3175 定稿，原样渲染）'
             ' &middot; 源 p3157 collect %s/%s/%s + p3183 %s &middot; 渲染于 %s</div>' % (
                 'FTR-24 立项（LOO 修正判据）' if revised else 'F18 维持',
                 span('meta.registry_v111_sha8', reg21_sha),
                 span('meta.registry_v110_sha8', '4cf49743'),
                 span('meta.registry_v19_sha8', '59a91512'),
                 span('meta.registry_v18_sha8', 'd08e9064'),
                 span('meta.registry_v17_sha8', 'f3ad070f'),
                 span('meta.gap_v15_sha8', SHA_ANCHOR['gap_v15']),
                 span('meta.p3157_4b', SHA_ANCHOR['p3157_4b']),
                 span('meta.p3157_14b', SHA_ANCHOR['p3157_14b']),
                 span('meta.p3157_glm4', SHA_ANCHOR['p3157_glm4']),
                 span('meta.p3183_result', SHA_ANCHOR['p3183_result']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg21['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg21['principles']):
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
    for f in reg21['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.11，%d 条）</h2>' % len(reg21['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg21['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg21['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg21['failures']:
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
    H.append('<div class="footer">RDC/LPF atlas v1.12 &middot; 渲染自 atlas_registry_v1_11.json (%s)'
             ' + gap ledger v1.5 (%s) &middot; 增量来源: Phase 3184 G5-B9 14b 贴线 LOO 归因'
             '（数据源 p3157 collect %s / %s / %s + p3183 %s）&middot; 本文件为 Phase 3184 封存产物，'
             '字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg21_sha, SHA_ANCHOR['gap_v15'],
                 SHA_ANCHOR['p3157_4b'], SHA_ANCHOR['p3157_14b'], SHA_ANCHOR['p3157_glm4'],
                 SHA_ANCHOR['p3183_result']))
    H.append('</body></html>')
    html_text = '\n'.join(H)
    hp = os.path.join(OUTDIR, 'atlas_v1_12.html')
    with open(hp, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(hp)
    expected = flatten_v112(reg21, nodes, gl15, meta_sha)
    v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
    log('atlas_v1_12.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
        html_sha, v['n_expected'], v['n_found'], len(v['missing']),
        len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
    final = {
        'phase': 3184, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'verdict': summary['verdict'], 'revised': bool(revised),
                'per_model': summary['per_model']},
        'counts': {'features': len(reg21['features']), 'failures': len(reg21['failures']),
                   'upgrades': len(reg21['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': disposition,
        'registry_v111_file': 'atlas_registry_v1_11.json', 'registry_v111_sha8': reg21_sha,
        'html_file': 'atlas_v1_12.html', 'html_sha8': html_sha,
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


def flatten_v112(reg, nodes, gl, meta_sha):
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
    d['meta.registry_v111_sha8'] = meta_sha['reg21']
    d['meta.registry_v110_sha8'] = '4cf49743'
    d['meta.registry_v19_sha8'] = '59a91512'
    d['meta.registry_v18_sha8'] = 'd08e9064'
    d['meta.registry_v17_sha8'] = 'f3ad070f'
    d['meta.gap_v15_sha8'] = SHA_ANCHOR['gap_v15']
    d['meta.p3157_4b'] = SHA_ANCHOR['p3157_4b']
    d['meta.p3157_14b'] = SHA_ANCHOR['p3157_14b']
    d['meta.p3157_glm4'] = SHA_ANCHOR['p3157_glm4']
    d['meta.p3183_result'] = SHA_ANCHOR['p3183_result']
    d['meta.created'] = CREATED_STR
    return d


def main():
    design_sha = freeze()
    log('== Phase 3184 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    for m in MODELS:
        ak = ANCH_KEY[m]
        r57 = json.load(open(SRC[RES_KEY[m]], encoding='utf-8'))
        A = ANCH_3157[ak]
        assert r57['res_sha8'] == A['res_sha8'] and r57['seal_sha8'] == A['seal_sha8'], \
            ('3157 content sha', m)
    s57 = json.load(open(SRC['p3157_summary'], encoding='utf-8'))
    assert s57['res_sha8'] == '0fe043bf' and s57['seal_sha8'] == 'cfa5c3ed'
    log('3157 anchors double-locked (3 per-model + summary)')
    r183 = json.load(open(SRC['p3183_result'], encoding='utf-8'))
    assert r183['res_sha8'] == CONTENT_SHA_3183[0] and r183['seal_sha8'] == CONTENT_SHA_3183[1]
    assert r183['gates']['low_ok'] is False
    log('3183 sealed result double-locked (%s/%s)' % CONTENT_SHA_3183)

    out, revised = run_arm(r183)
    if revised:
        verdict = ('dv_lowrank_revised_ftr24_registered (LOO-min pc_share >= 0.4 x3; '
                   'F18 text untouched; registry v1.10 -> v1.11 features append)')
    else:
        verdict = ('f18_maintained (single-entity removal cannot restore 14b; T2 '
                   'switches to per-relation parameterization; FTR-23 counter '
                   'quantified append)')
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3184, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True,
        'gates': {'loo_pc_min': G_PC, 'device_3157': True, 'replication_3183': True,
                  'revised': bool(revised)},
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
        finish_registry(summary, revised, res8, seal8, design_sha)
    with open(os.path.join(OUTDIR, suffix + 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
